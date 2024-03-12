#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
from typing import Optional, Sequence, Type, Union

import numpy as np
from numpy import ndarray
from tianshou.data.buffer.vecbuf import VectorReplayBuffer
from tianshou.env.venvs import SubprocVectorEnv
import torch
from torch import nn
from torch._C import device
from torch._tensor import Tensor
from torch.nn.modules import Linear, ReLU
from torch.utils.tensorboard import SummaryWriter
from env.risky_pointmass import PointMass

from tianshou.data import Collector
from tianshou.policy.modelfree.qrsac import QRSACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ArgsType, ModuleType, Net, MLP
from tianshou.utils.net.continuous import ActorProb, Critic

class MyNet(MLP):
    def __init__(self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,) -> None:
        super().__init__(input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation, act_args, device, linear_layer, flatten_input)

    def forward(self, obs: ndarray | Tensor, act: ndarray | Tensor) -> Tensor:
        both = torch.concat([obs, act], dim=1)
        return super().forward(both)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PointMass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--risk-penalty", type=float, default=10.0)
    parser.add_argument("--risk-prob", type=float, default=0.9)
    parser.add_argument("--risk-type", type=str, default="neutral")
    parser.add_argument("--num-quantiles", type=int, default=200)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--exploration", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--norm-layer", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def test_sac(args=get_args()):
    env = PointMass(risk_penalty=args.risk_penalty, risk_prob=args.risk_prob, stochastic=True)
    train_envs = SubprocVectorEnv([lambda: PointMass(risk_penalty=args.risk_penalty, risk_prob=args.risk_prob, stochastic=True) for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: PointMass(risk_penalty=args.risk_penalty, risk_prob=args.risk_prob, stochastic=True) for _ in range(args.test_num)])
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    net_c1 = MyNet(
        args.state_shape[0] + args.action_shape[0],
        args.num_quantiles,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        norm_layer=nn.LayerNorm if args.norm_layer else None
    )
    net_c2 = MyNet(
        args.state_shape[0] + args.action_shape[0],
        args.num_quantiles,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        norm_layer=nn.LayerNorm if args.norm_layer else None
    )
    critic1 = net_c1.to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = net_c2.to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = QRSACPolicy(
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
        estimation_step=args.n_step,
        action_space=env.action_space,
        device=args.device,
    )

    # collector
    buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=args.exploration)
    test_buffer = VectorReplayBuffer(1000*args.test_num, args.test_num)
    test_collector = Collector(policy, test_envs, test_buffer)
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dsac"
    log_name = os.path.join(args.task, args.algo_name, args.risk_type, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    print(f"log path: {log_path}")

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    ).run()
    pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_sac()
    