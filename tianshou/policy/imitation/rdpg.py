from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import torch
from torch.nn import functional as F

from tianshou.data import ReplayBuffer, to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import DDPGPolicy
from tianshou.utils.net.continuous import Actor, QuantileMlp


class RDPGPolicy(DDPGPolicy):
    """Implementation of Regularized Distributional Policy Gradient.
    """

    def __init__(
        self,
        actor: Actor,
        actor_optim: Optional[torch.optim.Optimizer],
        critic: QuantileMlp,
        critic_optim: Optional[torch.optim.Optimizer],
        action_classifier: torch.nn.Module,
        regularization_weight: float = 1.0,
        q_marge: float = 10.0,
        n_taus: int = 16,
        huber_threshold: float = 1.0,
        risk_type: str = 'neutral', # ['neutral', 'std', 'var', 'wang', 'cvar', 'cpw']
        risk_initial_param: float = 1.,
        risk_final_param: float = 0.,
        risk_schedule_timesteps: int = 1e6,
        distortion_param: float = 0.1,
        tau: float = 0.005,
        gamma: float = 0.99,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor,
            actor_optim,
            critic,
            critic_optim,
            tau,
            gamma,
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
        self._distortion_param = distortion_param
        self._action_classifier = action_classifier
        self._regularization_weight = regularization_weight
        self._q_marge = q_marge

    # TODO implement compute_n_return
    def train(self, mode: bool = True) -> "RDPGPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def process_fn(
        self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray
    ) -> RolloutBatchProtocol:
        next_indices = [i+1 if i+1 < len(buffer) else i for i in indices]
        batch.act_next = buffer.act[next_indices]
        return batch

    def _update_risk_param(self) -> None:
        self._risk_schedule_timesteps += 1
        fraction = min(float(self._train_step) / self._risk_schedule_timesteps, 1.)
        self._risk_param = self._risk_initial_param + fraction * (self._risk_final_param - self._risk_initial_param)

    def _get_taus(self, batch_size, n_taus, dtype):
        presum_taus = torch.rand(
            batch_size,
            n_taus,
            dtype=dtype,
            device=self.device,
        ) + 0.1
        presum_taus /= presum_taus.sum(dim=-1, keepdim=True)
        taus = torch.cumsum(presum_taus, dim=1)
        taus_hat = torch.zeros_like(taus, device=self.device)
        taus_hat[:, 0:1] = taus[:, 0:1] / 2.
        taus_hat[:, 1:] = (taus[:, 1:] + taus[:, :-1]) / 2.
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

    def _distortion_de(self, taus, mode="neutral", param_coef=1, eps=1e-8):
        # Derivative of Risk distortion function
        param = self._distortion_param * param_coef
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
            return self._distortion_de(1 - taus, mode, -1)

    def _target_z(self, batch: RolloutBatchProtocol) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch)
        act_next = batch.act_next
        taus_hat, presum_taus = self._get_taus(
            batch_size, self._n_taus, batch.obs.dtype
        )
        next_z = self.critic_old(batch.obs_next, act_next, taus_hat)  # (bsz, n_taus)
        target_z = (
            batch.rew.unsqueeze(-1).repeat(1, self._n_taus)
            + (1. - batch.done.unsqueeze(-1).repeat(1, self._n_taus))
            * self._gamma
            * next_z
        )
        return target_z, presum_taus

    def _critic_loss(
        self, batch: RolloutBatchProtocol, taus_hat_j: torch.Tensor
        ) -> torch.Tensor:
        current_z = self.critic(batch.obs, batch.act, taus_hat_j)
        with torch.no_grad():
            target_z, presum_taus_i = self._target_z(batch)

        qr_loss = self._quantile_regression_loss(
            current_z, target_z, taus_hat_j, presum_taus_i
        )

        act_pi = self(batch, input="obs").act
        current_z_pi = self.critic(batch.obs, act_pi, taus_hat_j)
        in_data = self._action_classifier(torch.cat([batch.obs, act_pi], dim=-1))
        re_loss = self._regularization_weight * ((1 - in_data) * torch.max(current_z_pi - current_z + self._q_marge, torch.zeros_like(current_z_pi))).mean()

        return qr_loss + re_loss, current_z, in_data, qr_loss, re_loss
    
    def _q_risk(
        self, batch: RolloutBatchProtocol, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get Q for risk type
        if self._risk_type == 'var':
            taus = torch.ones_like(batch.rew, device=self.device) * self._risk_param
            qa = self.critic(batch.obs, act, taus)
        else:
            taus_hat, presum_taus = self._get_taus(
                len(batch), self._n_taus, batch.obs.dtype
            )
            za = self.critic(batch.obs, act, taus_hat)
            if self._risk_type in ['neutral', 'std']:
                qa = torch.sum(presum_taus * za, dim=1, keepdim=True)
                if self._risk_type == 'std':
                    qa_std = presum_taus * (za - qa).pow(2)
                    qa -= self._risk_param * qa_std.sum(dim=1, keepdims=True).sqrt()
            else:
                risk_weights = self._distortion_de(taus_hat, self._risk_type, self._risk_param)
                qa = torch.sum(risk_weights * presum_taus * za, dim=1, keepdim=True)
        return qa

    def learn(self, batch: RolloutBatchProtocol, **kwargs: Any) -> Dict[str, float]:
        batch: RolloutBatchProtocol = to_torch(batch, dtype=torch.float, device=self.device)

        # actor
        obs_result = self(batch)
        act = obs_result.act
        qa = self._q_risk(batch, act)
        actor_loss = (- qa).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # critic
        taus_hat_j, _ = self._get_taus(
            len(batch), self._n_taus, batch.obs.dtype
        )
        critic_loss, current_z, in_data, qr_loss, re_loss = self._critic_loss(batch, taus_hat_j)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.sync_weight()
        self._update_risk_param()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "q_dataset": current_z.mean().item(),
            "in_data": in_data.mean().item(),
            "loss/quantile": qr_loss.item(),
            "loss/regularization": re_loss.item(),
        }

        return result
