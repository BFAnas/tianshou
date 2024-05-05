from typing import Tuple

import d4rl
import d4rl.gym_mujoco
import gym
import h5py
import numpy as np

from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd


def load_buffer_d4rl(expert_data_task: str) -> ReplayBuffer:
    dataset = d4rl.qlearning_dataset(gym.make(expert_data_task))
    replay_buffer = ReplayBuffer.from_data(
        obs=dataset["observations"],
        act=dataset["actions"],
        rew=dataset["rewards"],
        done=dataset["terminals"],
        obs_next=dataset["next_observations"],
        terminated=dataset["terminals"],
        truncated=np.zeros(len(dataset["terminals"]))
    )
    return replay_buffer


def load_buffer(buffer_path: str) -> ReplayBuffer:
    with h5py.File(buffer_path, "r") as dataset:
        buffer = ReplayBuffer.from_data(
            obs=dataset["observations"],
            act=dataset["actions"],
            rew=dataset["rewards"],
            done=dataset["terminals"],
            obs_next=dataset["next_observations"],
            terminated=dataset["terminals"],
            truncated=np.zeros(len(dataset["terminals"]))
        )
    return buffer


def normalize_all_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer
) -> Tuple[ReplayBuffer, RunningMeanStd]:
    # compute obs mean and var
    obs_rms = RunningMeanStd()
    obs_rms.update(replay_buffer.obs)
    _eps = np.finfo(np.float32).eps.item()
    # normalize obs
    replay_buffer._meta["obs"] = (replay_buffer.obs -
                                  obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    replay_buffer._meta["obs_next"] = (replay_buffer.obs_next -
                                       obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    return replay_buffer, obs_rms


def stochastic_reward(
        env_name: str,
        env_dataset_name: str,
        penalty: float, 
        probability: float,
        replay_buffer: ReplayBuffer) -> ReplayBuffer:
    if "halfcheetah" in env_name.lower():
        if ("medium" in env_dataset_name) or ("mixed" in env_dataset_name):
            velocity_threshold = 4 
        elif "expert" in env_dataset_name:
            velocity_threshold = 10
        else:
            raise ValueError("Unknown dataset name")
        velocity = replay_buffer.obs[:, 8]
        risky_state = velocity > velocity_threshold
        prob_mask = np.random.rand(len(velocity)) < probability
        stochastic_penalty = risky_state * prob_mask * penalty
        

    replay_buffer.rew -= penalty
    return replay_buffer