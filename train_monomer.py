import re
import os
import argparse
import datetime
import sys
import pprint
import torch
import numpy as np
import gymnasium as gym
from gym.spaces import Space
from types import SimpleNamespace
from sklearn.neighbors import KernelDensity

from tianshou.trainer import OffpolicyTrainer
from examples.offline.utils import load_buffer_d4rl
from tianshou.policy import DSACPolicy, BasePolicy
from tianshou.data.buffer.vecbuf import ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, Batch, to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, QuantileMlp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.env.venv_wrappers import MyVectorEnvNormObs

import numpy as np
from env.dompc_poly_env import DoMPC_Poly_env
from env.template_model import template_model
from env.template_mpc import template_mpc
from env.template_simulator import template_simulator

class suppress_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def policy_rollout(expert_policy, base_policy, init_state, length=10, device="cuda"):
    with suppress_print():
        _venv = MyVectorEnvNormObs(SubprocVectorEnv([lambda: DoMPC_Poly_env(hard_constraint=False, penalties=[], randomize=False) for _ in range(2)]))
    expert_policy.eval()
    base_policy.eval()
    obs, info = _venv.reset(init_state=init_state)
    states = []
    for i in range(length):
        states.append(_venv.denormalize_obs(obs).squeeze())
        expert_batch = Batch(obs=torch.from_numpy(obs[0]).reshape((1, -1)).to(device), info=info)
        base_batch = Batch(obs=torch.from_numpy(obs[1]).reshape((1, -1)).to(device), info=info)
        with torch.no_grad():
            expert_result = expert_policy(expert_batch)
            base_result = base_policy(base_batch)
        expert_act = expert_result.act.cpu().numpy()
        expert_act = expert_policy.map_action(expert_act)
        base_act = base_result.act.cpu().numpy()
        base_act = base_policy.map_action(base_act)
        observation, _, _, _, info  = _venv.step(np.concatenate([expert_act, base_act]))
        obs = observation
        if i == 0:
            expert_action = expert_act
            base_action = base_act
    expert_states = np.array(states)[:, 0, :]
    base_states = np.array(states)[:, 1, :]
    return expert_states, base_states, expert_action, base_action

def check_constraints(env, states):
    return all(abs(states[:, 3] - env.desired_temp) <= 2) and all(states[:, -1] <= 382.15)

def get_act(state, expert_policy, base_policy, length=1):
    expert_states, base_states, expert_action, base_action = policy_rollout(expert_policy, base_policy, state, length=length)

    if check_constraints(base_states):
        if abs(base_states[-1, 2] - 20680) < abs(expert_states[-1, 2] - 20680) - 50:
            return base_action, [False]
    
    return expert_action, [True]
    
class MixedPolicy(BasePolicy):
    def __init__(self, venv: MyVectorEnvNormObs, base_policy: BasePolicy, expert_policy: BasePolicy, action_space, device):
        super().__init__(action_space=action_space, action_scaling=False)
        self.venv = venv
        self.base_policy = base_policy
        self.expert_policy = expert_policy
        self.device = device

    def forward(self, batch: RolloutBatchProtocol, state=None, **kwargs):
        obs = batch.obs
        state = self.venv.denormalize_obs(obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs)
        action, cede_ctrl = get_act(state, self.expert_policy, self.base_policy)
        return Batch(**{'act': to_torch(action, device=self.device), 'policy': Batch({'cede_ctrl': to_torch(cede_ctrl, device=self.device, dtype=torch.bool)})})

    def train(self, mode: bool = True) -> "MixedPolicy":
        self.base_policy.train(mode)
        return self
    
    def process_fn(self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray) -> RolloutBatchProtocol:
        return self.base_policy.process_fn(batch, buffer, indices)

    def learn(self, batch, **kwargs):
        cede_ctrl = batch.policy.cede_ctrl.cpu().squeeze()
        batch = batch[~cede_ctrl]
        info = self.base_policy.learn(batch)
        return info

def get_dsac_args(env, device):
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.tensor([np.log(0.1)], requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=0.0003)
    alpha = (target_entropy, log_alpha, alpha_optim)
    args = argparse.Namespace(
        task="DoMPC",
        risk_type="neutral",
        buffer_size=1000000,
        hidden_sizes=[256, 256, 256],
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=1.,
        tau=0.005,
        alpha=alpha,
        alpha_lr=0.0003,
        start_timesteps=1,
        epoch=200,
        step_per_epoch=5000,
        step_per_collect=1,
        update_per_step=1,
        batch_size=256,
        training_num=1,
        test_num=10,
        distortion_param=0.75,
    )
    return args

def load_policy(env, args, path, device):
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=args.state_shape[0] + args.action_shape[0], device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=args.state_shape[0] + args.action_shape[0], device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.tensor([np.log(0.065)], requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    args.alpha = (target_entropy, log_alpha, alpha_optim)
    policy = DSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        risk_type=args.risk_type,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        action_space=env.action_space,
        device=device,
        distortion_param=args.distortion_param,
        action_scaling=True,
    )
    policy.load_state_dict(torch.load(path, map_location=device))
    print("Loaded agent from: ", path)
    return policy

def get_return(venv, policy, device, return_cede_ctrl=False):
    policy.eval()
    obs, info = venv.reset()
    reward_list = []
    cede_ctrl_list = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        batch = Batch(obs=torch.from_numpy(obs).to(device), info=info)
        with torch.no_grad():
            result = policy(batch)
        if return_cede_ctrl:
            cede_ctrl_list.append(result.policy.cede_ctrl.cpu().numpy())
        act = result.act.cpu().numpy()
        act = policy.map_action(act)
        print(act)
        observation, reward, terminated, truncated, info  = venv.step(act)
        obs = observation
        reward_list.append(reward)
    if return_cede_ctrl:
        return sum(reward_list), sum(cede_ctrl_list)/len(cede_ctrl_list)
    return sum(reward_list)

def main():
    parser = argparse.ArgumentParser(description="Training base policy while executing mixed actions")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--task', type=str, default="DoMPC", help='Task name')
    parser.add_argument('--risk-type', type=str, default="neutral", help='Risk type')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Buffer size')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256, 256], help='Hidden layer sizes')
    parser.add_argument('--actor-lr', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma value')
    parser.add_argument('--tau', type=float, default=0.005, help='Tau value')
    parser.add_argument('--alpha', type=float, default=0.065, help='Alpha value')
    parser.add_argument('--auto-alpha', action='store_true', help='Auto alpha flag')
    parser.add_argument('--exploration', action='store_true', help='Exploration flag')
    parser.add_argument('--alpha-lr', type=float, default=3e-4, help='Alpha learning rate')
    parser.add_argument('--start-timesteps', type=int, default=10000, help='Start timesteps')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--step-per-epoch', type=int, default=1000, help='Steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=1, help='Steps per collect')
    parser.add_argument('--update-per-step', type=int, default=1, help='Updates per step')
    parser.add_argument('--n-step', type=int, default=1, help='N step')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--training-num', type=int, default=1, help='Training number')
    parser.add_argument('--test-num', type=int, default=5, help='Test number')
    parser.add_argument('--logdir', type=str, default="log", help='Log directory')
    parser.add_argument('--device', type=str, default="cuda:2", help='Device to use')
    parser.add_argument('--distortion-param', type=float, default=-0.75, help='Distortion parameter')
    parser.add_argument('--bandwidth', type=float, default=1., help='KDE bandwidth')
    parser.add_argument('--percentile', type=float, default=1., help='KDE percentile')
    parser.add_argument('--learn-type', type=str, default="mixed10", help='Learn type options: "on-policy" "all" "mixed*int*"')
    parser.add_argument('--cede-ctrl-decay', type=float, default=0.99999, help='Cede control decay')
    parser.add_argument('--cede-ctrl-type', type=str, default="smooth", help='Cede control type options: "normal" "smooth"')
    parser.add_argument('--cede-ctrl-nature', type=str, default="NR", help='Cede control nature options: "N" for novel, "R" for risky, "NR" for both, "Never" for never ceding control and "Always" for always ceding control')
    parser.add_argument('--expert-policy-path', type=str, default="/data/user/R901105/dev/log/Hopper-v4/dsac/cvar/0/230831-103319", help='Expert policy path')
    parser.add_argument('--base-policy-path', type=str, default="/data/user/R901105/dev/log/Hopper-v2/codac_bc/neutral/0/231102-150037", help='Base policy path')
    args = parser.parse_args()

    env = gym.make(args.task)
 
    base_policy = get_model(args.base_policy_path, args)
    if "Hopper" in args.task:
        expert_policy = get_model(args.expert_policy_path, args, hidden_sizes=[256, 256])
    else: 
        expert_policy = get_model(args.expert_policy_path, args)

    offline_data = load_buffer_d4rl(args.data_task)
    test_envs = SubprocVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    test_envs.seed(args.seed)

    offline_batch, _ = offline_data.sample(5000)
    mixed_policy = MixedPolicy(base_policy, 
                                expert_policy, 
                                env.action_space, 
                                offline_batch.obs, 
                                args.device, 
                                args.bandwidth, 
                                args.percentile, 
                                args.learn_type,
                                args.cede_ctrl_type,
                                args.cede_ctrl_nature)

    test_buffer = VectorReplayBuffer(1000*args.test_num, args.test_num)
    train_collector = Collector(mixed_policy, env, offline_data)
    test_collector = Collector(mixed_policy, test_envs, test_buffer)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(args.task, "SafeRL", now)
    args.log_path = os.path.join(args.logdir, log_name)
    print(f"log path: {args.log_path}")
    writer = SummaryWriter(args.log_path)
    logger = TensorboardLogger(writer)
    writer.add_text("args", str(args))

    n = 1
    expert_returns = get_returns(expert_policy, n, env, args.device)
    def test_fn(num_epoch: int, step_idx: int):        
        returns, cede_ctrl, novel, risky = get_returns(mixed_policy, n, env, args.device, True)
        risky_returns = get_returns(base_policy, n, env, args.device)
        if mixed_policy.cede_ctrl_type == "normal":
            id_data = offline_data.sample(5000)[0].obs
            mixed_policy.update_kde(id_data)
        print("Mixed return: ", returns, "Suboptimal return: ", risky_returns, "Expert return: ", expert_returns, "Cede Control: ", cede_ctrl)
        # Log data to TensorBoard
        writer.add_scalar("Stats/Mixed return", returns, global_step=step_idx)
        writer.add_scalar("Stats/Suboptimal return", risky_returns, global_step=step_idx)
        writer.add_scalar("Stats/Expert return", expert_returns, global_step=step_idx)
        writer.add_scalar("Stats/Cede Control", cede_ctrl, global_step=step_idx)
        writer.add_scalar("Stats/Novel", novel, global_step=step_idx)
        writer.add_scalar("Stats/Risky", risky, global_step=step_idx)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.log_path, "best.pth"))

    result = OffpolicyTrainer(
        policy=mixed_policy,
        train_collector=train_collector,
        test_collector=test_collector,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    ).run()

    pprint.pprint(result)

if __name__ == "__main__":
    main()