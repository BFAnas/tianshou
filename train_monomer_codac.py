#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
from tianshou.data.batch import Batch
from tianshou.data.buffer.vecbuf import VectorReplayBuffer
from tianshou.env.venv_wrappers import MyVectorEnvNormObs
from tianshou.env.venvs import DummyVectorEnv, ShmemVectorEnv
import torch
from torch.utils.tensorboard import SummaryWriter

from env.dompc_poly_env import DoMPC_Poly_env
from tianshou.data import Collector
from tianshou.policy import CODACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, QuantileMlp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DoMPC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--risk-type", type=str, default="neutral")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
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

    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--penalties", type=int, nargs="*", default=[])
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--randomize", default=False, action="store_true")
    parser.add_argument("--hard-constraint", default=False, action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--behavioral-critic-path", type=str, default="/data/user/R901105/dev/log/DoMPC/qr/240312-151726/model.pth")
    parser.add_argument("--resume-path", type=str, default=None)
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


def test_sac(args=get_args()):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_to_use)
    env = DoMPC_Poly_env(render=args.render, randomize=args.randomize, hard_constraint=args.hard_constraint, penalties=args.penalties)
    train_envs = MyVectorEnvNormObs(ShmemVectorEnv([lambda: DoMPC_Poly_env(render=args.render, randomize=args.randomize, hard_constraint=args.hard_constraint, penalties=args.penalties) for _ in range(args.training_num)]))
    test_envs = MyVectorEnvNormObs(ShmemVectorEnv([lambda: DoMPC_Poly_env(render=args.render, randomize=args.randomize, hard_constraint=args.hard_constraint, penalties=args.penalties) for _ in range(args.test_num)]))
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

    critic1 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=args.state_shape[0] + args.action_shape[0], device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=args.state_shape[0] + args.action_shape[0], device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    if args.calibrated:
        behavioral_critic = QuantileMlp(
            input_size=args.state_shape[0] + args.action_shape[0],
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)
        behavioral_critic.load_state_dict(torch.load(args.behavioral_critic_path, map_location=args.device))
        print("Loaded behavioral critic from: ", args.behavioral_critic_path)

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
        venv = MyVectorEnvNormObs(DummyVectorEnv([lambda: env]))
        policy.eval()
        obs, info = venv.reset()
        reward_list = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            batch = Batch(obs=torch.from_numpy(np.expand_dims(obs, 0)).to(args.device), info=info)
            with torch.no_grad():
                result = policy(batch)
            act = result.act.cpu().numpy()
            act = policy.map_action(act)
            observation, reward, terminated, truncated, info  = venv.step(act)
            obs = observation
            
            reward_list.append(reward)
        venv.render()
        fig = venv.venv.get_env_attr("fig")[0]
        fig.savefig(os.path.join(log_path, "best_policy.png"))

    def test_fn(num_epoch: int, step_idx: int):
        if num_epoch > 50:
            return
        if num_epoch % 10 == 0:
            torch.save(policy.state_dict(), os.path.join(log_path, f"policy_{num_epoch}.pth"))


    if not args.watch:
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
            test_fn=test_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    # Save buffer
    buffer.save_hdf5(os.path.join(log_path, "buffer.hdf5"))
    
    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_sac()
