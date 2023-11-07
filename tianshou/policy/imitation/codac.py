from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn.utils import clip_grad_norm_

from tianshou.data import to_torch, Batch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import DSACPolicy
from tianshou.utils.net.continuous import ActorProb


class CODACPolicy(DSACPolicy):
    """Implementation of the distributed CQL algorithm a.k.a CODAC. arXiv:2107.06106.
    """

    def __init__(
        self,
        actor: ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
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
        behavioral_critic: Optional[torch.nn.Module] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, tau=tau,
            gamma=gamma, alpha=alpha, **kwargs
        )
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

        self.behavioral_critic = behavioral_critic

        # const tensors
        self.random_log_prob = torch.log(torch.tensor(0.5**self.action_space.shape[-1], device=self.device)).repeat(self._n_taus).reshape(1, self._n_taus)
        self.zero_tensor = torch.tensor([0], device=self.device)


    def train(self, mode: bool = True) -> "CODACPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def actor_pred(self, obs: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        batch = Batch(obs=obs, info=None)
        obs_result = self(batch)
        return obs_result.act, obs_result.log_prob
    
    def _critic_loss(
        self, batch: RolloutBatchProtocol, critic: torch.nn.Module, taus_hat_j
        ) -> (torch.Tensor, torch.Tensor):
        current_z = critic(batch.obs, batch.act, taus_hat_j)
        with torch.no_grad():
            target_z, presum_taus_i = self._target_z(batch)

        loss = self._quantile_regression_loss(
            current_z, target_z, taus_hat_j, presum_taus_i
        )
        return current_z, loss

    def repeat_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_len = len(tensor.shape)
        repeat_size = [1, self.num_repeat_actions] + [1] * (tensor_len - 1)
        view_size = [tensor.shape[0] * self.num_repeat_actions] + list(tensor.shape[1:])
        return tensor.unsqueeze(1).repeat(*repeat_size).view(*view_size)  

    def _calc_pi_values(self, obs_pi: torch.Tensor, obs_to_pred: torch.Tensor, taus_hat_j) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        act_pred, log_pi = self.actor_pred(obs_pi)
        log_pi = log_pi.repeat((1, self._n_taus))

        z1 = self.critic1(obs_to_pred, act_pred, taus_hat_j)
        z2 = self.critic2(obs_to_pred, act_pred, taus_hat_j)

        if self.behavioral_critic:
            mc_returns = self.behavioral_critic(obs_to_pred, act_pred, taus_hat_j)
            z1 = torch.max(z1, mc_returns)
            z2 = torch.max(z2, mc_returns)

        return z1 - log_pi.detach(), z2 - log_pi.detach()

    def _calc_random_values(self, obs: torch.Tensor, act: torch.Tensor, taus_hat_j) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        random_value1 = self.critic1(obs, act, taus_hat_j)
        random_value2 = self.critic2(obs, act, taus_hat_j)

        if self.behavioral_critic:
            mc_returns = self.behavioral_critic(obs, act, taus_hat_j)
            random_value1 = torch.max(random_value1, mc_returns)
            random_value2 = torch.max(random_value2, mc_returns)

        return random_value1 - self.random_log_prob, random_value2 - self.random_log_prob

    def learn(self, batch: RolloutBatchProtocol, *args: Any,
              **kwargs: Any) -> Dict[str, float]:

        batch: RolloutBatchProtocol = to_torch(batch, dtype=torch.float, device=self.device)
        obs, act, rew, obs_next = batch.obs, batch.act, batch.rew, batch.obs_next
        batch_size = len(batch)
        taus_hat_j, _ = self._get_taus(
            batch_size, self._n_taus, batch.obs.device, batch.obs.dtype
        )

        # compute actor loss and update actor
        obs_result = self(batch)
        act = obs_result.act
        qa = self._q_risk(batch, act)
        log_pi = obs_result.log_prob
        actor_loss = (self._alpha * log_pi.flatten() - qa).mean()
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

        # compute critics loss
        current_z1, critic1_loss = self._critic_loss(batch, self.critic1, taus_hat_j)
        current_z2, critic2_loss = self._critic_loss(batch, self.critic2, taus_hat_j)

        random_actions = torch.FloatTensor(
            batch_size * self.num_repeat_actions, act.shape[-1]
        ).uniform_(-self.min_action, self.max_action).to(self.device)

        tmp_obs = self.repeat_tensor(obs)
        taus_hat_j = self.repeat_tensor(taus_hat_j)
        # tmp_obs: (batch_size * num_repeat, state_dim)
        # taus_hat_j: (batch_size * num_repeat, n_taus)

        current_pi_value1, current_pi_value2 = self._calc_pi_values(tmp_obs, tmp_obs, taus_hat_j)
        random_value1, random_value2 = self._calc_random_values(tmp_obs, random_actions, taus_hat_j)

        current_pi_value1 = current_pi_value1.reshape(batch_size, self.num_repeat_actions, self._n_taus)
        current_pi_value2 = current_pi_value2.reshape(batch_size, self.num_repeat_actions, self._n_taus)
        random_value1 = random_value1.reshape(batch_size, self.num_repeat_actions, self._n_taus)
        random_value2 = random_value2.reshape(batch_size, self.num_repeat_actions, self._n_taus)

        # cat q values
        cat_q1 = torch.cat([random_value1, current_pi_value1], 1)
        cat_q2 = torch.cat([random_value2, current_pi_value2], 1)
        # shape: (batch_size, 2 * num_repeat, n_taus)

        cql1_scaled_loss = \
            torch.logsumexp(cat_q1 / self.temperature, dim=1) * \
                self.temperature - current_z1
        cql1_scaled_loss = cql1_scaled_loss.mean()
        cql1_scaled_loss *= self.cql_weight

        cql2_scaled_loss = \
            torch.logsumexp(cat_q2 / self.temperature, dim=1) * \
                self.temperature - current_z2
        cql2_scaled_loss = cql2_scaled_loss.mean()
        cql2_scaled_loss *= self.cql_weight

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

        critic1_loss = critic1_loss + cql1_scaled_loss
        critic2_loss = critic2_loss + cql2_scaled_loss

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
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore
        if self.with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        return result
