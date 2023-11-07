#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from examples.offline.utils import load_buffer_d4rl
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import CODACPolicy
from tianshou.trainer import OfflineTrainer, OffpolicyTrainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, QuantileMlp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--task", type=str, default="Hopper-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-data-task", type=str, default="hopper-medium-v2")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-taus", type=int, default=16)
    parser.add_argument("--risk-type", type=str, default='neutral')
    parser.add_argument("--calibrated", default=False, action="store_true")
    parser.add_argument("--train-num", type=int, default=1)
    parser.add_argument("--exploration", default=False, action="store_true")
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--with-lagrange", default=False, action="store_true")
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 35)
    parser.add_argument("--gpu-to-use", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--behavioral-critic-path", type=str, default="/data/user/R901105/dev/log/Hopper-v4/qr/231102-133240/model.pth")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_d4rl.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()

def test_cql():
    args = get_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_to_use)
    env = gym.make(args.task)
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
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # model
    # actor network
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

    if args.calibrated:
        behavioral_critic = QuantileMlp(
            input_size=args.state_shape[0] + args.action_shape[0],
            hidden_sizes=[256, 256],
            device=args.device,
        ).to(args.device)
        behavioral_critic.load_state_dict(torch.load(args.behavioral_critic_path, map_location=args.device))
        print("Loaded behavioral critic from: ", args.behavioral_critic_path)


    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = CODACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
        risk_type=args.risk_type,
        n_taus=args.n_taus,
        cql_alpha_lr=args.cql_alpha_lr,
        cql_weight=args.cql_weight,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        temperature=args.temperature,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        min_action=np.min(env.action_space.low),
        max_action=np.max(env.action_space.high),
        behavioral_critic=behavioral_critic if args.calibrated else None,
        device=args.device,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
        dirname = os.path.dirname(args.resume_path)
        torch.save(policy.actor.state_dict(), os.path.join(dirname, 'actor.pth'))
        torch.save(policy.critic1.state_dict(), os.path.join(dirname, 'critic1.pth'))
        torch.save(policy.critic2.state_dict(), os.path.join(dirname, 'critic2.pth'))

    # collector
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "codac_bc" if args.calibrated else "codac"
    if args.mode == "online":
        args.algo_name = "online_" + args.algo_name
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
        replay_buffer = load_buffer_d4rl(args.expert_data_task)
        if args.mode == "online":
            policy.behavioral_critic = None
            train_envs = SubprocVectorEnv(
                [lambda: gym.make(args.task) for _ in range(args.train_num)]
            ) 
            train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=args.exploration)
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
        else:
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
    test_cql()
