from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from tianshou.data import ReplayBuffer, to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import SACPolicy
from tianshou.utils.net.continuous import ActorProb


class DSACPolicy(SACPolicy):
    """Implementation of DSAC algorithm. arXiv:2004.14547.
    Based on https://github.com/xtma/dsac/tree/master
    """

    def __init__(
        self,
        actor: ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        n_taus: int = 16,
        huber_threshold: float = 1.0,
        risk_type: str = 'neutral', # ['neutral', 'std', 'var', 'wang', 'cvar', 'cpw']
        risk_initial_param: float = 1.,
        risk_final_param: float = 0.,
        risk_schedule_timesteps: int = 1e6,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau,
            gamma,
            alpha,
            **kwargs
        )
        self.device = device
        assert risk_type in ['neutral', 'std', 'var', 'wang', 'cvar', 'cpw'], f"risk_type should be one of ['neutral', 'std', 'var', 'wang', 'cvar', 'cpw'] was given {risk_type}"
        assert n_taus > 1, "n_taus should be greater than 1"
        self._n_taus = n_taus  # for policy eval
        self._huber_threshold = huber_threshold
        self._risk_type = risk_type
        self._risk_param = risk_initial_param
        self._risk_initial_param = risk_initial_param
        self._risk_final_param = risk_final_param
        self._risk_schedule_timesteps = risk_schedule_timesteps
        self._train_step = 0

    # TODO implement compute_n_return
    def train(self, mode: bool = True) -> "DSACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def process_fn(
        self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray
    ) -> RolloutBatchProtocol:
        return batch

    def _update_risk_param(self) -> None:
        self._risk_schedule_timesteps += 1
        fraction = min(float(self._train_step) / self._risk_schedule_timesteps, 1.0)
        self._risk_param = self._risk_initial_param + fraction * (self._risk_final_param - self._risk_initial_param)

    @staticmethod
    def _get_taus(batch_size, n_taus, device, dtype):
        presum_taus = (
            torch.rand(
                batch_size,
                n_taus,
                dtype=dtype,
                device=device,
            )
            + 0.1
        )
        presum_taus /= presum_taus.sum(dim=-1, keepdim=True)
        taus = torch.cumsum(presum_taus, dim=1)
        taus_hat = torch.zeros_like(taus).to(device)
        taus_hat[:, 0:1] = taus[:, 0:1] / 2.0
        taus_hat[:, 1:] = (taus[:, 1:] + taus[:, :-1]) / 2.0
        return taus_hat, presum_taus

    @staticmethod
    def _quantile_regression_loss(input, target, tau, weight):
        """
        input: (N, T)
        target: (N, T)
        tau: (N, T)
        """
        input = input.unsqueeze(-1)
        target = target.detach().unsqueeze(-2)
        tau = tau.detach().unsqueeze(-1)
        weight = weight.detach().unsqueeze(-2)
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        L = F.smooth_l1_loss(
            expanded_input, expanded_target, reduction="none"
        )  # (bsz, n_taus, n_taus)
        sign = torch.sign(expanded_input - expanded_target) / 2.0 + 0.5
        rho = torch.abs(tau - sign) * L * weight
        return rho.sum(dim=-1).mean()

    @staticmethod
    def _normal_cdf(value, loc=0., scale=1.):
        return 0.5 * (1 + torch.erf((value - loc) / scale / np.sqrt(2)))

    @staticmethod
    def _normal_icdf(value, loc=0., scale=1.):
        return loc + scale * torch.erfinv(2 * value - 1) * np.sqrt(2)

    @staticmethod
    def _normal_pdf(value, loc=0., scale=1.):
        return torch.exp(-(value - loc)**2 / (2 * scale**2)) / scale / np.sqrt(2 * np.pi)    

    def _distortion_de(self, taus, mode="neutral", param=0., eps=1e-8):
        # Derivative of Risk distortion function
        taus = taus.clamp(0., 1.)
        if param >= 0:
            if mode == "neutral":
                taus_ = torch.ones_like(taus)
            elif mode == "wang":
                taus_ = self._normal_pdf(self._normal_icdf(taus) + param) / (self._normal_pdf(self._normal_icdf(taus)) + eps)
            elif mode == "cvar":
                taus_ = (1. / param) * (taus < param)
            elif mode == "cpw":
                g = taus**param
                h = (taus**param + (1 - taus)**param)**(1 / param)
                g_ = param * taus**(param - 1)
                h_ = (taus**param + (1 - taus)**param)**(1 / param - 1) * (taus**(param - 1) - (1 - taus)**(param - 1))
                taus_ = (g_ * h - g * h_) / (h**2 + eps)
            else:
                raise f"risk type {mode} not in supported list: ['neutral', 'std', 'var', 'wang', 'cvar', 'cpw']"
            return taus_.clamp(0., 5.).to(taus.device)

        else:
            return self._distortion_de(1 - taus, mode, -param)

    def _target_z(self, batch: RolloutBatchProtocol) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch)
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        taus_hat, presum_taus = self._get_taus(
            batch_size, self._n_taus, batch.obs.device, batch.obs.dtype
        )
        next_z1 = self.critic1_old(batch.obs_next, act_, taus_hat)  # (bsz, n_taus)
        next_z2 = self.critic2_old(batch.obs_next, act_, taus_hat)
        log_prob = obs_next_result.log_prob.repeat((1, self._n_taus))
        target_z = (
            batch.rew.unsqueeze(-1).repeat(1, self._n_taus)
            + (1.0 - batch.done.unsqueeze(-1).repeat(1, self._n_taus))
            * self._gamma
            * torch.min(
                next_z1,
                next_z2,
            )
            - self._alpha * log_prob
        )
        return target_z, presum_taus

    def _critic_loss(
        self, batch: RolloutBatchProtocol, critic: torch.nn.Module
        ) -> torch.Tensor:
        batch_size = len(batch)
        taus_hat_j, _ = self._get_taus(
            batch_size, self._n_taus, batch.obs.device, batch.obs.dtype
        )
        current_z = critic(batch.obs, batch.act, taus_hat_j)
        with torch.no_grad():
            target_z, presum_taus_i = self._target_z(batch)

        loss = self._quantile_regression_loss(
            current_z, target_z, taus_hat_j, presum_taus_i
        )
        return loss
    
    def _q_risk(
        self, batch: RolloutBatchProtocol, act: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        # get Q for risk type
        if self._risk_type == 'var':
            taus = torch.ones_like(batch.rew, device=batch.rew.device) * self._risk_param
            q1a = self.critic1(batch.obs, act, taus)
            q2a = self.critic2(batch.obs, act, taus)
        else:
            taus_hat, presum_taus = self._get_taus(
                len(batch), self._n_taus, batch.obs.device, batch.obs.dtype
            )
            z1a = self.critic1(batch.obs, act, taus_hat)
            z2a = self.critic2(batch.obs, act, taus_hat)
            if self._risk_type in ['neutral', 'std']:
                q1a = torch.sum(presum_taus * z1a, dim=1, keepdim=True)
                q2a = torch.sum(presum_taus * z2a, dim=1, keepdim=True)
                if self._risk_type == 'std':
                    q1a_std = presum_taus * (z1a - q1a).pow(2)
                    q2a_std = presum_taus * (z2a - q2a).pow(2)
                    q1a -= self._risk_param * q1a_std.sum(dim=1, keepdims=True).sqrt()
                    q2a -= self._risk_param * q2a_std.sum(dim=1, keepdims=True).sqrt()
            else:
                risk_weights = self._distortion_de(taus_hat, self._risk_type, self._risk_param)
                q1a = torch.sum(risk_weights * presum_taus * z1a, dim=1, keepdim=True)
                q2a = torch.sum(risk_weights* presum_taus * z2a, dim=1, keepdim=True)
        qa = torch.min(q1a, q2a)
        return qa

    def learn(self, batch: RolloutBatchProtocol, **kwargs: Any) -> Dict[str, float]:
        batch: RolloutBatchProtocol = to_torch(batch, dtype=torch.float, device=self.device)
        # critic 1&2
        critic1_loss = self._critic_loss(batch, self.critic1)
        critic2_loss = self._critic_loss(batch, self.critic2)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # actor
        obs_result = self(batch)
        act = obs_result.act
        qa = self._q_risk(batch, act)
        actor_loss = (self._alpha * obs_result.log_prob.flatten() - qa).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()
        self._update_risk_param()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
