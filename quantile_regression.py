import os
import argparse
import datetime
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from tianshou.data import ReplayBuffer, Batch
from env.dompc_poly_env import DoMPC_Poly_env
from examples.offline.utils import load_buffer_d4rl
from tianshou.utils.net.continuous import QuantileMlp
from torch.utils.tensorboard.writer import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Hopper-v4")
    parser.add_argument("--expert-data-task", type=str, default="hopper-medium-v2")
    parser.add_argument("--expert-data-path", type=str, default="/data/user/R901105/dev/log/DoMPC/dsac/neutral/0/240311-152608/buffer.hdf5")
    parser.add_argument("--penalties", type=int, nargs="*", default=[])
    parser.add_argument("--randomize", default=False, action="store_true")
    parser.add_argument("--hard-constraint", default=False, action="store_true")
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-taus", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--logdir", type=str, default="log")
    return parser.parse_args()

def add_returns(buffer: ReplayBuffer, gamma: float = 0.99) -> ReplayBuffer:
    """Adds the returns for a given ReplayBuffer.

    Args:
        buffer: The ReplayBuffer to compute returns for.
        gamma: The discount factor.

    Returns:
        A new ReplayBuffer with the returns.
    """
    data_dict = buffer._meta.__dict__
    start_idx = np.concatenate([np.array([0]), np.where(data_dict["done"])[0] + 1])
    end_idx = np.concatenate(
        [np.where(data_dict["done"])[0] + 1, np.array([len(data_dict["done"])])]
    )
    ep_rew = [data_dict["rew"][i:j] for i, j in zip(start_idx, end_idx)]
    ep_ret = []
    for i in range(len(ep_rew)):
        episode_rewards = ep_rew[i]
        disc_returns = [0] * len(episode_rewards)
        discounted_return = 0
        for j in range(1, len(episode_rewards) + 1):
            discounted_return = (
                episode_rewards[len(episode_rewards) - j] + gamma * discounted_return
            )
            disc_returns[len(episode_rewards) - j] = discounted_return
        ep_ret.append(disc_returns)

    new_data_dict = data_dict.copy()
    ep_rets = np.concatenate(ep_ret)
    new_data_dict["calibration_returns"] = ep_rets
    new_batch = Batch(**new_data_dict)
    buffer._meta = new_batch
    return buffer

def get_taus(batch_size, n_taus, device):
    presum_taus = torch.rand(
        batch_size,
        n_taus,
        device=device,
    )
    presum_taus /= presum_taus.sum(dim=-1, keepdim=True)
    taus = torch.cumsum(presum_taus, dim=1)
    taus_hat = torch.zeros_like(taus).to(device)
    taus_hat[:, 0:1] = taus[:, 0:1] / 2.0
    taus_hat[:, 1:] = (taus[:, 1:] + taus[:, :-1]) / 2.0
    return taus_hat, presum_taus

def quantile_regression_loss(input, target, tau, weight):
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

# read the data
args = get_args()
if args.task == "DoMPC":
    replay_buffer = ReplayBuffer(1000000)
    replay_buffer = replay_buffer.load_hdf5(args.expert_data_path)
    replay_buffer = replay_buffer.from_data(replay_buffer.obs, replay_buffer.act, replay_buffer.rew, replay_buffer.terminated, replay_buffer.truncated, replay_buffer.done, replay_buffer.obs_next)
else:
    replay_buffer = load_buffer_d4rl(args.expert_data_task)
replay_buffer = add_returns(replay_buffer, args.gamma)
# train QuantileMlp
if args.task == "DoMPC":
    env = DoMPC_Poly_env(hard_constraint=args.hard_constraint, penalties=args.penalties, randomize=args.randomize)
else:
    env = gym.make(args.task)
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
net = QuantileMlp(
    input_size=args.state_shape[0] + args.action_shape[0],
    hidden_sizes=args.hidden_sizes,
    device=args.device,
).to(args.device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# logging
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
args.algo_name = "qr"
log_name = os.path.join(args.task, args.algo_name, now)
log_path = os.path.join(args.logdir, log_name)
writer = SummaryWriter(log_path)
writer.add_text("args", str(args))

for i in tqdm(range(args.epoch)):
    for j in tqdm(range(int(len(replay_buffer)/args.batch_size))):
        optimizer.zero_grad()
        batch, _ = replay_buffer.sample(args.batch_size)
        batch.to_torch(device=args.device)
        taus_hat, presum_taus = get_taus(args.batch_size, args.n_taus, args.device)
        z_returns = net(batch.obs.to(torch.float32), batch.act, taus_hat) 
        returns = batch.calibration_returns.unsqueeze(-1).repeat(1, args.n_taus)
        loss = quantile_regression_loss(z_returns, returns, taus_hat, presum_taus)
        loss.backward()
        optimizer.step()
    
    writer.add_scalar('training_loss', loss.item(), i)

print("Training complete!")
writer.close()

# Save model's state_dict
torch.save(net.state_dict(), os.path.join(log_path , 'model.pth'))
print("Model saved")