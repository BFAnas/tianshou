#!/data/user/R901105/.conda/envs/dev/bin python

import argparse
import datetime
import os
import pprint
import random
import re

import numpy as np
from copy import deepcopy
from tianshou.data.buffer.vecbuf import VectorReplayBuffer, ReplayBuffer
from tianshou.env.venvs import SubprocVectorEnv
import torch
from torch.utils.tensorboard import SummaryWriter
from env.risky_pointmass import PointMass
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import gymnasium as gym
from gym.spaces import Space

from tianshou.data import Collector, Batch, to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import DSACPolicy, BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, QuantileMlp

class MixedPolicy(BasePolicy):
    def __init__(self, 
                 base_policy: BasePolicy, 
                 expert_policy: BasePolicy, 
                 action_space: Space, 
                 id_data: np.ndarray, 
                 device: str, 
                 bandwidth: float, 
                 percentile: float, 
                 n_repeat: int, 
                 reduce_action: str, 
                 learn_type: str, 
                 cede_ctrl_type: str,
                 cede_ctrl_nature: str,
                 cede_ctrl_decay: float):
        super().__init__(action_space=action_space, action_scaling=True)
        self.base_policy = base_policy
        self.expert_policy = expert_policy
        self.device = device
        self.bandwidth = bandwidth
        self.percentile = percentile
        self.n_repeat = n_repeat
        self.reduce_action = reduce_action
        self.learn_type = learn_type
        self.cede_ctrl_type = cede_ctrl_type
        self.cede_ctrl_nature = cede_ctrl_nature
        self.cede_ctrl_decay = cede_ctrl_decay
        self.cede_ctrl_prob = 1
        self.update_kde(id_data)

    def update_kde(self, id_data):
        self.kde = KernelDensity(kernel='epanechnikov', bandwidth=self.bandwidth).fit(id_data)
        self.density_threshold = np.percentile(self.kde.score_samples(id_data), self.percentile)  

    def get_riksy(self, batch: RolloutBatchProtocol):
        bsz = len(batch.obs)
        self.base_policy.eval()
        self.expert_policy.eval()

        with torch.no_grad():
            if self.n_repeat > 1:
                _batch = Batch(**{'obs': deepcopy(batch.obs.repeat(self.n_repeat, 1)), 'info': None})
                expert_result = self.expert_policy(_batch)
                base_result = self.base_policy(_batch)
                repeated_expert_qvalues1 = self.expert_policy.critic1(_batch.obs, expert_result.act).reshape(self.n_repeat, bsz, -1)
                repeated_expert_qvalues2 = self.expert_policy.critic2(_batch.obs, expert_result.act).reshape(self.n_repeat, bsz, -1)
                expert_qvalues1 = repeated_expert_qvalues1.mean(0)
                expert_qvalues2 = repeated_expert_qvalues2.mean(0)
                expert_qvalues1_std = repeated_expert_qvalues1.std(0)
                expert_qvalues2_std = repeated_expert_qvalues2.std(0)
                expert_qvalues_std = (expert_qvalues1_std + expert_qvalues2_std) * .5
                expert_qvalues = torch.minimum(expert_qvalues1, expert_qvalues2)
                repeated_base_qvalues1 = self.base_policy.critic1(_batch.obs, base_result.act).reshape(self.n_repeat, bsz, -1)
                repeated_base_qvalues2 = self.base_policy.critic2(_batch.obs, base_result.act).reshape(self.n_repeat, bsz, -1)
                base_qvalues1 = repeated_base_qvalues1.mean(0)
                base_qvalues2 = repeated_base_qvalues2.mean(0)
                base_qvalues1_std = repeated_base_qvalues1.std(0)
                base_qvalues2_std = repeated_base_qvalues2.std(0)
                base_qvalues = torch.minimum(base_qvalues1, base_qvalues2)
                base_qvalues_std = (base_qvalues1_std + base_qvalues2_std) * .5
                risky = base_qvalues[:, 0] - base_qvalues_std[:, 0] < expert_qvalues[:, 0] - expert_qvalues_std[:, 0]
                if self.reduce_action ==  "mean":
                    expert_actions = expert_result.act.mean(0)
                    base_actions = base_result.act.mean(0)
                elif self.reduce_action == "first":
                    expert_actions = expert_result.act[0]
                    base_actions = base_result.act[0] 
                else:
                    raise "Not supported"                  
                return risky, expert_actions, base_actions
            else:
                expert_result = self.expert_policy(batch)
                base_result = self.base_policy(batch)
                expert_qvalues1 = self.expert_policy.critic1(batch.obs, expert_result.act)
                expert_qvalues2 = self.expert_policy.critic2(batch.obs, expert_result.act)
                expert_qvalues = torch.minimum(expert_qvalues1, expert_qvalues2)
                base_qvalues1 = self.base_policy.critic1(batch.obs, base_result.act)
                base_qvalues2 = self.base_policy.critic2(batch.obs, base_result.act)
                base_qvalues = torch.minimum(base_qvalues1, base_qvalues2)
                risky = base_qvalues[:, 0] < expert_qvalues[:, 0]
                expert_actions = expert_result.act
                base_actions = base_result.act
                return risky, expert_actions, base_actions

    def forward(self, batch: RolloutBatchProtocol, state=None, **kwargs):
        batch = to_torch(batch, dtype=torch.float32, device=self.device)
        risky, expert_actions, base_actions = self.get_riksy(batch)
        log_dens = self.kde.score_samples(batch.obs.cpu().numpy())
        novel = log_dens < self.density_threshold
        novel = to_torch(novel, dtype=torch.bool, device=self.device)
        if self.cede_ctrl_nature == "R":
            cede_ctrl = risky
        elif self.cede_ctrl_nature == "N":
            cede_ctrl = novel
        elif self.cede_ctrl_nature == "NR":
            cede_ctrl = torch.logical_or(risky, novel)
        elif self.cede_ctrl_nature == "Never":
            cede_ctrl = torch.zeros_like(novel).bool()
        elif self.cede_ctrl_nature == "Always":
            cede_ctrl = torch.ones_like(novel).bool()
        else:
            raise "Not supported"
        if self.cede_ctrl_type == "smooth":
            not_cede_ctrl = ~cede_ctrl * (torch.rand(1, device=self.device) >= self.cede_ctrl_prob)
            cede_ctrl = ~not_cede_ctrl
            self.cede_ctrl_prob *= self.cede_ctrl_decay
        elif self.cede_ctrl_type == "normal":
            pass
        else:
            raise "Not supported"
        cede_ctrl = cede_ctrl.unsqueeze(-1)
        actions = torch.where(cede_ctrl, expert_actions, base_actions)
        return Batch(**{'act': actions, 'policy': Batch({'cede_ctrl': cede_ctrl, 'novel': novel, 'risky': risky})})

    def train(self, mode: bool = True) -> "MixedPolicy":
        self.base_policy.train(mode)
        return self
    
    def process_fn(self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray) -> RolloutBatchProtocol:
        return self.base_policy.process_fn(batch, buffer, indices)

    def learn(self, batch, **kwargs):
        cede_ctrl = batch.policy.cede_ctrl.cpu().squeeze()
        not_cede_ctrl = ~cede_ctrl
        if self.learn_type == "on-policy":
            batch = batch[not_cede_ctrl]
        elif self.learn_type == "all":
            pass
        elif "mixed" in self.learn_type:
            percentage = int(re.findall(r'\d+', self.learn_type)[0])
            false_indices = np.where(not_cede_ctrl == False)[0]
            num_to_replace = min(len(false_indices), int(len(batch)*percentage/100))
            indices_to_replace = random.sample(list(false_indices), num_to_replace)
            not_cede_ctrl[indices_to_replace] = True
            batch = batch[not_cede_ctrl]
        else:
            raise "Not supported"

        info = self.base_policy.learn(batch)
        return info

def load_policy(type, env, args, distortion_param=None):
    if type == "expert":
        path = args.expert_policy_path
    elif type == "base":
        path = args.base_policy_path
    else:
        raise "Not supported"
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net_a = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net_a, action_shape, device=args.device, unbounded=True, conditioned_sigma=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    critic1 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=state_shape[0] + action_shape[0], device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = QuantileMlp(hidden_sizes=args.hidden_sizes, input_size=state_shape[0] + action_shape[0], device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.tensor([np.log(args.alpha)], requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

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
        alpha=alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
        distortion_param= distortion_param if distortion_param else args.distortion_param,
        device=args.device,
    )
    policy.load_state_dict(torch.load(path, map_location=args.device))
    print("Loaded agent from: ", path)
    return policy

def get_init_states(env, n=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x_grid = np.linspace(0.4, 1.5, 101)
    y_grid = np.linspace(0.4, 1.5, 101)
    init_states = np.stack([(x, y) for x in x_grid for y in y_grid])
    init_states = np.array([(x, y) for x, y in init_states if np.linalg.norm(np.array((x,y))-env.center) > env.r])
    init_states = init_states[np.random.choice(len(init_states), n)]
    return init_states   

def plot_trajectories(policy, env, args, figname):
    policy.eval()
    init_states = get_init_states(env)
    trajectories = []
    for state in init_states:
        obs, info = env.reset(init_state=state)
        done = False
        cum_reward = 0
        actions = []
        while not done:
            batch = Batch(obs=torch.from_numpy(np.expand_dims(obs, 0)).to(args.device), info=info)
            with torch.no_grad():
                result = policy(batch)
            act = result.act.cpu().squeeze().numpy()
            act = policy.map_action(act)
            obs, reward, terminated, truncated, info = env.step(act)
            actions.append(act)
            cum_reward += reward
            done = terminated or truncated 
        trajectories.append(env.trajectory)

    # Create a figure for plotting
    plt.figure(figsize=(10, 10))

    # Novel zone coordinates
    x = [1, 1, 0, 0, 1.5, 1.5, 1]  
    y = [0, 1, 1, 1.5, 1.5, 0, 0]

    # Plot the risk area
    circle = plt.Circle((0.5, 0.5), 0.3, color='red', alpha=0.3, label="Risk zone")
    plt.gca().add_patch(circle)

    # Plot the novel area
    plt.plot(x, y, alpha=0.3)
    plt.fill(x, y, alpha=0.3, label='Novel zone')  

    # Plot the goal
    circle = plt.Circle(env.goal, 0.01, color='green', label="Goal")
    plt.gca().add_patch(circle)

    # Plot the start states
    plt.scatter(*zip(*init_states), color='blue', marker='o', s=10, label='Start States')

    # Plot the trajectories
    for traj in trajectories:
        plt.plot(*zip(*traj), color='gray', linewidth=1, alpha=0.3)

    plt.legend()
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Trajectories from Start States to Goal')
    fig_path = os.path.join(args.log_path, figname)
    plt.savefig(fig_path)
    print(f"Saved figure : {fig_path}") 

def get_returns(policy, env, args, init_states, return_cede_ctrl=False):
    policy.eval()
    returns = []
    cede_ctrl = []
    novel = []
    risky = []
    for state in init_states:
        obs, info = env.reset(init_state=state)
        done = False
        cum_reward = 0
        actions = []
        if return_cede_ctrl:
            cctrl_ep = []
            novel_ep = []
            risky_ep = []
        while not done:
            batch = Batch(obs=torch.from_numpy(np.expand_dims(obs, 0)).to(args.device), info=info)
            with torch.no_grad():
                result = policy(batch)
            act = result.act.cpu().squeeze().numpy()
            act = policy.map_action(act)
            if return_cede_ctrl:
                cctrl_ep.append(result.policy.cede_ctrl.cpu().squeeze().numpy())
                novel_ep.append(result.policy.novel.cpu().squeeze().numpy())
                risky_ep.append(result.policy.risky.cpu().squeeze().numpy())
            obs, reward, terminated, truncated, info = env.step(act)
            actions.append(act)
            cum_reward += reward
            done = terminated or truncated
        returns.append(cum_reward)
        if return_cede_ctrl:
            cede_ctrl.append(np.array(cctrl_ep).mean())
            novel.append(np.array(novel_ep).mean())
            risky.append(np.array(risky_ep).mean())
    if return_cede_ctrl:
        return np.array(returns),  np.array(cede_ctrl).mean(), np.array(novel).mean(), np.array(risky).mean()
    return np.array(returns)

def plot_returns(expert_returns, base_returns, mixed_returns, args, figname):
    percentile_10_expert = np.percentile(expert_returns, 10)
    percentile_90_expert = np.percentile(expert_returns, 90)
    percentile_10_base = np.percentile(base_returns, 10)
    percentile_90_base = np.percentile(base_returns, 90)
    percentile_10_mixed = np.percentile(mixed_returns, 10)
    percentile_90_mixed = np.percentile(mixed_returns, 90)

    plt.figure(figsize=(10, 6))
    # Histograms with different styles
    plt.hist(expert_returns, bins=30, alpha=0.6, label='expert policy', histtype='step')
    plt.hist(base_returns, bins=30, alpha=0.6, label='base policy', histtype='stepfilled')
    plt.hist(mixed_returns, bins=30, alpha=0.6, label='Mixed policy', histtype='barstacked')

    # Percentile lines
    plt.axvline(percentile_10_expert, color='blue', linestyle='-', linewidth=1)
    plt.axvline(percentile_90_expert, color='blue', linestyle='-', linewidth=1)
    plt.axvline(percentile_10_base, color='orange', linestyle='--', linewidth=1)
    plt.axvline(percentile_90_base, color='orange', linestyle='--', linewidth=1)
    plt.axvline(percentile_10_mixed, color='green', linestyle=':', linewidth=1)
    plt.axvline(percentile_90_mixed, color='green', linestyle=':', linewidth=1)

    # Titles and labels
    plt.title('Returns Distributions with 0.1 and 0.9 percentiles')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    fig_path = os.path.join(args.log_path, figname)
    plt.savefig(fig_path)
    print(f"Saved figure : {fig_path}")

def main():
    parser = argparse.ArgumentParser(description="Training base policy while executing mixed actions")
    parser.add_argument('--task', type=str, default="PointMass", help='Task name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--risk-type', type=str, default="wang", help='Risk type')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Buffer size')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256, 256], help='Hidden layer sizes')
    parser.add_argument('--actor-lr', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma value')
    parser.add_argument('--tau', type=float, default=0.005, help='Tau value')
    parser.add_argument('--alpha', type=float, default=0.02, help='Alpha value')
    parser.add_argument('--auto-alpha', action='store_true', help='Auto alpha flag')
    parser.add_argument('--exploration', action='store_true', help='Exploration flag')
    parser.add_argument('--alpha-lr', type=float, default=3e-4, help='Alpha learning rate')
    parser.add_argument('--start-timesteps', type=int, default=10000, help='Start timesteps')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--step-per-epoch', type=int, default=500, help='Steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=1, help='Steps per collect')
    parser.add_argument('--update-per-step', type=int, default=1, help='Updates per step')
    parser.add_argument('--n-step', type=int, default=1, help='N step')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--training-num', type=int, default=1, help='Training number')
    parser.add_argument('--test-num', type=int, default=10, help='Test number')
    parser.add_argument('--logdir', type=str, default="log", help='Log directory')
    parser.add_argument('--device', type=str, default="cuda:2", help='Device to use')
    parser.add_argument('--distortion-param', type=float, default=-0.75, help='Distortion parameter')
    parser.add_argument('--risk-penalty', type=int, default=20, help='Risk penalty')
    parser.add_argument('--risk-prob', type=float, default=0.95, help='Risk probability')
    parser.add_argument('--high-state', type=float, default=1.5, help='High state value')
    parser.add_argument('--bandwidth', type=float, default=0.1, help='KDE bandwidth')
    parser.add_argument('--percentile', type=float, default=1., help='KDE percentile')
    parser.add_argument('--n-repeat', type=int, default=100, help='Number of repeats')
    parser.add_argument('--reduce-action', type=str, default="first", help='Reduce action options: "mean" "first"')
    parser.add_argument('--learn-type', type=str, default="mixed10", help='Learn type options: "on-policy" "all" "mixed*int*"')
    parser.add_argument('--cede-ctrl-decay', type=float, default=0.99999, help='Cede control decay')
    parser.add_argument('--cede-ctrl-type', type=str, default="normal", help='Cede control type options: "normal" "smooth"')
    parser.add_argument('--cede-ctrl-nature', type=str, default="NR", help='Cede control nature options: "N" for novel, "R" for risky, "NR" for both, "Never" for never ceding control and "Always" for always ceding control')
    parser.add_argument('--expert-policy-path', type=str, default="/data/user/R901105/dev/my_fork/tianshou/log/PointMass_prob0.95_pen20_hs1.5/dsac/wang0.75/0/240111-114541/policy.pth", help='Expert policy path')
    parser.add_argument('--base-policy-path', type=str, default="/data/user/R901105/dev/my_fork/tianshou/log/PointMass_prob0.95_pen20/dsac/wang-0.75/0/240116-111416/policy.pth", help='Base policy path')
    parser.add_argument('--buffer-path', type=str, default="/data/user/R901105/dev/my_fork/tianshou/log/PointMass_prob0.95_pen20/dsac/wang-0.75/0/240116-111416/buffer.hdf5", help='Base policy training buffer path')
    args = parser.parse_args()

    # Environment setup
    train_envs = SubprocVectorEnv([lambda: PointMass(risk_penalty=args.risk_penalty, risk_prob=args.risk_prob, high_state=args.high_state) for _ in range(args.training_num)])

    env = PointMass(risk_penalty=args.risk_penalty, risk_prob=args.risk_prob, high_state=args.high_state)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_policy = load_policy("expert", env, args)
    base_policy = load_policy("base", env, args, 0.75)
    id_data = np.random.uniform(0, 1, (1000, 2))
    mixed_policy = MixedPolicy(base_policy, 
                                expert_policy, 
                                env.action_space, 
                                id_data, 
                                args.device, 
                                args.bandwidth, 
                                args.percentile, 
                                args.n_repeat, 
                                args.reduce_action,
                                args.learn_type,
                                args.cede_ctrl_type,
                                args.cede_ctrl_nature,
                                args.cede_ctrl_decay)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "SafeRL"
    log_name = os.path.join(args.task+f"_prob{args.risk_prob}"+f"_pen{args.risk_penalty}"+f"_hs{args.high_state}", algo_name, str(args.seed), now)
    args.log_path = os.path.join(args.logdir, log_name)
    print(f"log path: {args.log_path}")

    # logger
    writer = SummaryWriter(args.log_path)
    logger = TensorboardLogger(writer)
    writer.add_text("args", str(args))

    plot_trajectories(expert_policy, env, args, "Expert_trajectories.png")
    plot_trajectories(base_policy, env, args, "Base_trajectories_before.png")
    plot_trajectories(mixed_policy, env, args, "Mixed_trajectories.png")

    buffer = VectorReplayBuffer.load_hdf5(args.buffer_path)
    train_collector = Collector(mixed_policy, train_envs, buffer)
    test_buffer = VectorReplayBuffer(1000*args.test_num, args.test_num)
    test_collector = Collector(mixed_policy, train_envs, test_buffer)

    states = get_init_states(env, 100, 0)
    def test_fn(num_epoch: int, step_idx: int):        
        returns, cede_ctrl, novel, risky = get_returns(mixed_policy, env, args, states, True)
        base_returns = get_returns(base_policy, env, args, states)
        percentile_10 = np.percentile(returns, 10)
        base_percentile_10 = np.percentile(base_returns, 10)
        expert_percentile_10 = np.percentile(expert_returns, 10)
        id_data = buffer.sample(5000)[0].obs
        mixed_policy.update_kde(id_data)
        print("Percentile 10: ", percentile_10, "Base Percentile 10: ", base_percentile_10, "Expert Percentile 10: ", expert_percentile_10, "Cede Control: ", cede_ctrl)
        # Log data to TensorBoard
        writer.add_scalar("Stats/Percentile 10", percentile_10, global_step=step_idx)
        writer.add_scalar("Stats/Base Percentile 10", base_percentile_10, global_step=step_idx)
        writer.add_scalar("Stats/Expert Percentile 10", expert_percentile_10, global_step=step_idx)
        writer.add_scalar("Stats/Cede Control", cede_ctrl, global_step=step_idx)
        writer.add_scalar("Stats/Novel", novel, global_step=step_idx)
        writer.add_scalar("Stats/Risky", risky, global_step=step_idx)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.log_path, "best.pth"))

    init_states = get_init_states(env, 500)
    expert_returns = get_returns(expert_policy, env, args, init_states)
    base_returns = get_returns(base_policy, env, args, init_states)
    mixed_returns = get_returns(mixed_policy, env, args, init_states)
    plot_returns(expert_returns, base_returns, mixed_returns, args, "Returns_before.png")

    result = OffpolicyTrainer(
        policy=mixed_policy,
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

    expert_returns = get_returns(expert_policy, env, args, states)
    base_returns = get_returns(base_policy, env, args, states)
    mixed_returns = get_returns(mixed_policy, env, args, states)
    plot_returns(expert_returns, base_returns, mixed_returns, args, "Returns_after.png")
    plot_trajectories(base_policy, env, args, "Base_trajectories_after.png")

if __name__ == "__main__":
    main()