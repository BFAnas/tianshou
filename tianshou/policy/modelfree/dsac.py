from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import SACPolicy
from tianshou.utils.net.continuous import ActorProb   

class DSACPolicy(SACPolicy):
    """Implementation of DSAC algorithm. arXiv:2004.14547."""

    def __init__(
        self,
        actor: ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        n_taus: int = 32,
        online_n_taus: int = 8,
        target_n_taus: int = 8,
        huber_threshold: float = 1.0,
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
        assert n_taus > 1, "n_taus should be greater than 1"
        assert online_n_taus > 1, "online_n_taus should be greater than 1"
        assert target_n_taus > 1, "target_n_taus should be greater than 1"
        self._n_taus = n_taus  # for policy eval
        self._online_n_taus = online_n_taus
        self._target_n_taus = target_n_taus
        self._huber_threshold = huber_threshold

    def train(self, mode: bool = True) -> "DSACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        return batch

    @staticmethod
    def _get_taus(batch_size, n_taus, device, dtype):
        presum_taus = torch.rand(
            batch_size,
            n_taus,
            dtype=dtype,
            device=device,
        ) + 0.1
        presum_taus /= presum_taus.sum(dim=-1, keepdim=True)
        taus = torch.cumsum(presum_taus, dim=1)
        with torch.no_grad():
            taus_hat = torch.zeros_like(taus).to(device)
            taus_hat[:, 0:1] = taus[:, 0:1] / 2.
            taus_hat[:, 1:] = (taus[:, 1:] + taus[:, :-1]) / 2.
        return taus, taus_hat, presum_taus
    
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
        L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
        sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
        rho = torch.abs(tau - sign) * L * weight
        return rho.sum(dim=-1).mean()

    def _target_q(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch)
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        taus, taus_hat, presum_taus = self._get_taus(batch_size, self._n_taus, batch.obs.device, batch.obs.dtype)
        next_q1 = self.critic1_old(
            batch.obs_next, act_, taus_hat
        )  # (bsz, n_taus)
        next_q2 = self.critic2_old(batch.obs_next, act_, taus_hat)
        log_prob = obs_next_result.log_prob.repeat((1, self._n_taus))
        target_q = (
            batch.rew.unsqueeze(-1).repeat(1, self._n_taus)
            + (1. - batch.done.unsqueeze(-1).repeat(1, self._n_taus)) * self._gamma
            * torch.min(
                next_q1,
                next_q2,
            )
            - self._alpha * log_prob
        )
        return target_q, presum_taus

    def _huber_optimizer(
        self, batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        batch_size = len(batch)
        taus_j, taus_hat_j, presum_taus_j = self._get_taus(batch_size, self._n_taus, batch.obs.device, batch.obs.dtype)
        current_q = critic(batch.obs, batch.act, taus_hat_j)
        with torch.no_grad():
            target_q, presum_taus_i = self._target_q(batch)

        critic_loss = self._quantile_regression_loss(current_q, target_q, taus_hat_j, presum_taus_i)
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return critic_loss, presum_taus_i

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        batch: Batch = to_torch(batch, dtype=torch.float, device=self.device)
        # critic 1&2
        critic1_loss, taus = self._huber_optimizer(
            batch, self.critic1, self.critic1_optim
        )
        critic2_loss, _ = self._huber_optimizer(batch, self.critic2, self.critic2_optim)

        # actor
        obs_result = self(batch)
        act = obs_result.act
        with torch.no_grad():
            taus, taus_hat, presum_taus = self._get_taus(len(batch), self._n_taus, batch.obs.device, batch.obs.dtype)
        current_q1a = self.critic1(batch.obs, act, taus_hat)
        current_q2a = self.critic2(batch.obs, act, taus_hat)
        q1a = torch.sum(presum_taus * current_q1a, dim=1, keepdim=True)
        q2a = torch.sum(presum_taus * current_q2a, dim=1, keepdim=True)
        qa = torch.min(q1a, q2a)
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

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
