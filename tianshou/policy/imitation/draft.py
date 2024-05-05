from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import SACPolicy
from tianshou.utils.net.continuous import ActorProb


class RBVEPolicy(SACPolicy):
    """Implementation Regularized Behavior Value Estimation.
    """

    def __init__(
        self,
        actor: ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        id_model: torch.nn.Module,
        cql_alpha_lr: float = 1e-4,
        cql_weight: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        temperature: float = 1.0,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
        num_repeat_actions: int = 10,
        alpha_min: float = 0.0,
        alpha_max: float = 1e6,
        clip_grad: float = 1.0,
        calibrated: bool = False,
        q_marge: float = 5.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, tau,
            gamma, alpha, **kwargs
        )
        self.id_model = id_model
        # There are _target_entropy, _log_alpha, _alpha_optim in SACPolicy.
        self.device = device
        self.temperature = temperature
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold

        self.cql_weight = cql_weight

        self.cql_log_alpha = torch.tensor([0.0], requires_grad=True)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)
        self.cql_log_alpha = self.cql_log_alpha.to(device)

        self.min_action = min_action
        self.max_action = max_action

        self.num_repeat_actions = num_repeat_actions

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.clip_grad = clip_grad

        self.calibrated = calibrated
        self.shape = (-1, self.num_repeat_actions, 1)
        self.q_marge = q_marge

    def train(self, mode: bool = True) -> "RBVEPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def actor_pred(self, obs: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        batch = Batch(obs=obs, info=None)
        obs_result = self(batch)
        return obs_result.act, obs_result.log_prob

    def calc_actor_loss(self, batch: RolloutBatchProtocol) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        act_pred, log_pi = self.actor_pred(batch.obs)
        action_size  = act_pred.shape[-1]
        obs_size = batch.obs.shape[-1]
        random_actions = torch.FloatTensor(
            len(act_pred) * self.num_repeat_actions, action_size
        ).uniform_(self.min_action, self.max_action).to(self.device)
        all_act = torch.cat([act_pred, random_actions, batch.act], 0)
        all_obs = batch.obs.repeat_interleave(self.num_repeat_actions + 2, 0)
        q1 = self.critic1(all_obs, all_act)
        q2 = self.critic2(all_obs, all_act)
        q1 = q1[:, None, :].view(-1, self.num_repeat_actions + 2, obs_size)
        q2 = q2[:, None, :].view(-1, self.num_repeat_actions + 2, obs_size)
        q1_max_indices = torch.argmax(q1, 1)
        q2_max_indices = torch.argmax(q2, 1)
        actor_loss = (act_pred - all_act[q1_max_indices])**2 + (act_pred - all_act[q2_max_indices])**2
        # actor_loss.shape: (), log_pi.shape: (batch_size, 1)
        return actor_loss, log_pi

    def calc_pi_values(self, obs_pi: torch.Tensor, obs_to_pred: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        act_pred, log_pi = self.actor_pred(obs_pi)

        q1 = self.critic1(obs_to_pred, act_pred)
        q2 = self.critic2(obs_to_pred, act_pred)

        return q1 - log_pi.detach(), q2 - log_pi.detach(), act_pred

    def calc_random_values(self, obs: torch.Tensor, act: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        random_value1 = self.critic1(obs, act)
        random_log_prob1 = np.log(0.5**act.shape[-1])

        random_value2 = self.critic2(obs, act)
        random_log_prob2 = np.log(0.5**act.shape[-1])

        return random_value1 - random_log_prob1, random_value2 - random_log_prob2

    def process_fn(
        self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray
    ) -> RolloutBatchProtocol:
        # TODO: mypy rightly complains here b/c the design violates
        #   Liskov Substitution Principle
        #   DDPGPolicy.process_fn() results in a batch with returns but
        #   CQLPolicy.process_fn() doesn't add the returns.
        #   Should probably be fixed!
        next_indices = [i+1 if i+1 < len(buffer) else i for i in indices]
        batch.act_next = buffer.act[next_indices]
        return batch

    def learn(self, batch: RolloutBatchProtocol, *args: Any,
              **kwargs: Any) -> Dict[str, float]:
        batch: Batch = to_torch(batch, dtype=torch.float, device=self.device)
        obs, act, rew, obs_next, act_next = batch.obs, batch.act, batch.rew, batch.obs_next, batch.act_next
        batch_size = obs.shape[0]

        # compute actor loss and update actor
        actor_loss, log_pi = self.calc_actor_loss(batch)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # compute alpha loss
        if self._is_auto_alpha:
            log_pi = log_pi + self._target_entropy
            alpha_loss = -(self._log_alpha * log_pi.detach()).mean()
            self._alpha_optim.zero_grad()
            # update log_alpha
            alpha_loss.backward()
            self._alpha_optim.step()
            # update alpha
            self._alpha = self._log_alpha.detach().exp()

        # compute target_Q
        with torch.no_grad():
            # act_next, new_log_pi = self.actor_pred(obs_next)

            target_Q1 = self.critic1_old(obs_next, act_next)
            target_Q2 = self.critic2_old(obs_next, act_next)
            target_Q1 = \
                rew + self._gamma * (1 - batch.done) * target_Q1.flatten()
            target_Q2 = \
                rew + self._gamma * (1 - batch.done) * target_Q2.flatten()

            # target_Q = torch.min(target_Q1, target_Q2) - self._alpha * new_log_pi

            # target_Q = \
            #     rew + self._gamma * (1 - batch.done) * target_Q.flatten()
            # shape: (batch_size)

        # compute critic loss
        current_Q1 = self.critic1(obs, act).flatten()
        current_Q2 = self.critic2(obs, act).flatten()
        # shape: (batch_size)

        mean_current_Q = 0.5 * (current_Q1.mean() + current_Q2.mean())

        _critic1_loss = F.mse_loss(current_Q1, target_Q1)
        _critic2_loss = F.mse_loss(current_Q2, target_Q2)

        # CQL
        # random_actions = torch.FloatTensor(
        #     batch_size * self.num_repeat_actions, act.shape[-1]
        # ).uniform_(self.min_action, self.max_action).to(self.device)

        obs_len = len(obs.shape)
        repeat_size = [1, self.num_repeat_actions] + [1] * (obs_len - 1)
        view_size = [batch_size * self.num_repeat_actions] + list(obs.shape[1:])
        tmp_obs = obs.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        view_size = [batch_size * self.num_repeat_actions] + list(act.shape[1:])
        # tmp_act = act.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        # tmp_obs_next = obs_next.unsqueeze(1).repeat(*repeat_size).view(*view_size)
        # tmp_obs & tmp_obs_next: (batch_size * num_repeat, state_dim)

        current_pi_value1, current_pi_value2, act_pred = self.calc_pi_values(tmp_obs, tmp_obs)
        # next_pi_value1, next_pi_value2 = self.calc_pi_values(tmp_obs_next, tmp_obs)
        # random_value1, random_value2 = self.calc_random_values(tmp_obs, random_actions)

        # mean_random_value = 0.5 * (random_value1.mean() + random_value2.mean())
        # mean_current_pi_value = 0.5 * (current_pi_value1.mean() + current_pi_value2.mean())

        # cat q values
        # cat_q1 = torch.cat([current_pi_value1.reshape(self.shape)], 1)
        # cat_q2 = torch.cat([current_pi_value2.reshape(self.shape)], 1)
        # shape: (batch_size, 3 * num_repeat, 1)

        # check if (obs, act) is in the dataset
        in_dist = self.id_model(torch.cat([tmp_obs, act_pred], 1))

        cql1_scaled_loss = \
            (self.cql_weight * (1- in_dist) * (torch.max(current_pi_value1.squeeze() - current_Q1.repeat_interleave(self.num_repeat_actions) + self.q_marge, torch.zeros_like(current_pi_value1.squeeze())))).mean()
        cql2_scaled_loss = \
            (self.cql_weight * (1- in_dist) * (torch.max(current_pi_value2.squeeze() - current_Q2.repeat_interleave(self.num_repeat_actions) + self.q_marge, torch.zeros_like(current_pi_value2.squeeze())))).mean()
        # shape: (1)

        if self.with_lagrange:
            cql_alpha = torch.clamp(
                self.cql_log_alpha.exp(),
                self.alpha_min,
                self.alpha_max,
            )
            cql1_scaled_loss = \
                cql_alpha * (cql1_scaled_loss - self.lagrange_threshold)
            cql2_scaled_loss = \
                cql_alpha * (cql2_scaled_loss - self.lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(cql1_scaled_loss + cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()

        critic1_loss = _critic1_loss + cql1_scaled_loss
        critic2_loss = _critic2_loss + cql2_scaled_loss

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        # clip grad, prevent the vanishing gradient problem
        # It doesn't seem necessary
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad)
        self.critic2_optim.step()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": _critic1_loss.item(),
            "loss/critic2": _critic2_loss.item(),
            "loss/cql1": cql1_scaled_loss.item(),
            "loss/cql2": cql2_scaled_loss.item(),
            "q_dataset": mean_current_Q.item(),
            "in_dist": in_dist.sum().item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore
        if self.with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        return result
