#!/usr/bin/env python3

import argparse
import datetime
import os
import re
import pprint

import gymnasium as gym
import numpy as np
from tianshou.data.buffer.base import ReplayBuffer
import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector
from tianshou.policy import RDSACPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, QuantileMlp

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            for i in range(len(hidden_sizes) - 1)
        ])
        self.hidden_norms = nn.ModuleList([
            nn.LayerNorm(hidden_sizes[i+1])
            for i in range(len(hidden_sizes) - 1)
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)  # 1 output for energy score
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = torch.relu(x)
        for hidden_layer, hidden_norm in zip(self.hidden_layers, self.hidden_norms):
            x = hidden_layer(x)
            x = hidden_norm(x)
            x = torch.relu(x)
        output = self.output_layer(x)
        return torch.sigmoid(output)
    
class RewardHighVelocity(gym.RewardWrapper):
    """Wrapper to modify environment rewards of 'Cheetah','Walker' and
    'Hopper'.

    Penalizes with certain probability if velocity of the agent is greater
    than a predefined max velocity.
    Parameters
    ----------
    kwargs: dict
    with keys:
    'prob_vel_penal': prob of penalization
    'cost_vel': cost of penalization
    'max_vel': max velocity

    Methods
    -------
    step(action): next_state, reward, done, info
    execute a step in the environment.
    """

    def __init__(self, env, **kwargs):
        super(RewardHighVelocity, self).__init__(env)
        self.penal_v_distr = Bernoulli(kwargs['prob_vel_penal'])
        self.penal = kwargs['cost_vel']
        self.max_vel = kwargs['max_vel']
        self.max_step = kwargs['max_step']
        self.step_count = 0
        allowed_envs = ['Cheetah', 'Hopper', 'Walker']
        assert(any(e in self.env.unwrapped.spec.id for e in allowed_envs)), \
            'Env {self.env.unwrapped.spec.id} not allowed for RewardWrapper'

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        vel = self.env.sim.data.qvel[0]
        info['risky_state'] = vel > self.max_vel
        info['angle'] = self.env.sim.data.qpos[2]
        self.step_count += 1

        if self.step_count >= self.max_step:
            terminated = True

        if 'Cheetah' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                     terminated, truncated, info)
        if 'Walker' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                     terminated, truncated, info)
        if 'Hopper' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                     terminated, truncated, info)

    def new_reward(self, reward, info):
        if 'Cheetah' in self.env.unwrapped.spec.id:
            forward_reward = info['reward_run']
        else:
            forward_reward = info['x_velocity']

        penal = info['risky_state'] * \
            self.penal_v_distr.sample().item() * self.penal

        # If penalty applied, substract the forward_reward from total_reward
        # original_reward = rew_healthy + forward_reward - cntrl_cost
        new_reward = penal + reward + (penal != 0) * (-forward_reward)
        return new_reward

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    @property
    def name(self):
        return f'{self.__class__.__name__}{self.env}'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="HalfCheetah-v3")
    parser.add_argument("--expert-data-task", type=str, default="/data/user/R901105/dev/my_fork/tianshou/tianshou_buffer_halfcheetah-medium-v0_prob0.05_vel4_cost-70.hdf5")
    parser.add_argument("--max-step", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ac-path", type=str, default="/data/user/R901105/dev/log/HalfCheetah-v3/action_classification/0/240504-201628/model.pt")
    parser.add_argument("--ac-hidden-sizes", type=int, nargs="*", default=[512, 512, 512])
    parser.add_argument("--regularization-weight", type=float, default=0.1)
    parser.add_argument("--q-marge", type=float, default=5.0)
    parser.add_argument("--risk-type", type=str, default="neutral")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gpu-to-use", type=int, default=0)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def test_rdsac(args=get_args()):
    args = get_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_to_use)
    env = gym.make(args.task)
    match = re.search(r"_prob(\d+\.\d+)_vel(\d+)_cost-(\d+)", args.expert_data_task)
    if match:
        prob_vel_penal = float(match.group(1))
        max_vel = int(match.group(2))
        cost_vel = -int(match.group(3))
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]  # float
    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    args.state_dim = args.state_shape[0]
    args.action_dim = args.action_shape[0]
    print("Max_action", args.max_action)

    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: RewardHighVelocity(gym.make(args.task), prob_vel_penal=prob_vel_penal, cost_vel=cost_vel, max_vel=max_vel, max_step=args.max_step) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    critic1 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=args.state_shape[0] + args.action_shape[0], device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=args.state_shape[0] + args.action_shape[0], device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # action classifier
    action_classifier = MyModel(args.state_dim + args.action_dim, args.ac_hidden_sizes).to(args.device)
    action_classifier.load_state_dict(torch.load(args.ac_path))
    print("Loaded AC model from:", args.ac_path)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = RDSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        action_classifier,
        regularization_weight=args.regularization_weight,
        q_marge=args.q_marge,
        risk_type=args.risk_type,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
        device=args.device,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "rdsac"
    log_name = os.path.join(args.task, args.algo_name, args.risk_type, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def watch():
        if args.resume_path is None:
            args.resume_path = os.path.join(log_path, "policy.pth")

        policy.load_state_dict(
            torch.load(args.resume_path, map_location=torch.device("cpu"))
        )
        policy.eval()
        collector = Collector(policy, env)
        collector.collect(n_episode=1, render=1 / 35)

    if not args.watch:
        replay_buffer = ReplayBuffer.load_hdf5(args.expert_data_task)
        # trainer
        result = OfflineTrainer(
            policy=policy,
            buffer=replay_buffer,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
        ).run()
        pprint.pprint(result)
    else:
        watch()

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")


if __name__ == "__main__":
    test_rdsac()