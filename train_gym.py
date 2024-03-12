import re
import os
import argparse
import datetime
import random
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
    
class MixedPolicy(BasePolicy):
    def __init__(self, 
                 base_policy: BasePolicy, 
                 expert_policy: BasePolicy, 
                 action_space: Space, 
                 id_data: np.ndarray, 
                 device: str, 
                 bandwidth: float, 
                 percentile: float, 
                 learn_type: str, 
                 cede_ctrl_type: str,
                 cede_ctrl_nature: str):
        super().__init__(action_space=action_space, action_scaling=True)
        self.base_policy = base_policy
        self.expert_policy = expert_policy
        self.device = device
        self.bandwidth = bandwidth
        self.percentile = percentile
        self.learn_type = learn_type
        self.cede_ctrl_type = cede_ctrl_type
        self.cede_ctrl_nature = cede_ctrl_nature
        self.t = 0
        self.update_kde(id_data)

    def sigmoid_scheduler(self, t_max=4e5, initial_value=1., final_value=0., param=20):
        """
        Sigmoid-like scheduler.

        :param t: Current step
        :param t_max: Maximum number of steps
        :param initial_value: Initial value
        :param final_value: Final value
        :return: Value at current step
        """
        return initial_value + (final_value - initial_value) * (1 / (1 + np.exp(-param * (self.t / t_max - 0.5))))

    def update_kde(self, id_data):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(id_data)
        self.density_threshold = np.percentile(self.kde.score_samples(id_data), self.percentile)  

    def get_results(self, batch):
        self.base_policy.eval()
        self.expert_policy.eval()
        with torch.no_grad():
            expert_result = self.expert_policy(batch)
            base_result = self.base_policy(batch)
        return expert_result, base_result

    def get_riksy(self, batch: RolloutBatchProtocol, expert_result, base_result):
        with torch.no_grad():
            expert_qvalues1 = self.expert_policy.critic1(batch.obs, expert_result.act)
            expert_qvalues2 = self.expert_policy.critic2(batch.obs, expert_result.act)
            expert_qvalues = torch.minimum(expert_qvalues1, expert_qvalues2)
            base_qvalues1 = self.base_policy.critic1(batch.obs, base_result.act)
            base_qvalues2 = self.base_policy.critic2(batch.obs, base_result.act)
            base_qvalues = torch.minimum(base_qvalues1, base_qvalues2)
        risky = base_qvalues[:, 0] < expert_qvalues[:, 0]
        return risky

    def get_cede_ctrl(self, batch, expert_result, base_result):
        # if self.cede_ctrl_type == "smooth":
        #     probs_tensor = torch.tensor(self.sigmoid_scheduler(), device=self.device)
        #     binomial_dist = torch.distributions.Binomial(1, probs_tensor)
        #     l = len(batch.obs)
        #     cede_ctrl = binomial_dist.sample((l,)).bool()
        #     risky = torch.zeros(l, device=self.device).bool()
        #     novel = torch.zeros(l, device=self.device).bool()
        # elif self.cede_ctrl_type == "normal":
        risky = self.get_riksy(batch, expert_result, base_result)
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
        cede_ctrl = cede_ctrl.unsqueeze(-1)
        # else:
        #     raise "Not supported"
        if self.cede_ctrl_type == "smooth":
            probs_tensor = torch.tensor(self.sigmoid_scheduler(), device=self.device)
            binomial_dist = torch.distributions.Binomial(1, probs_tensor)
            l = len(batch.obs)
            cede_ctrl = torch.where(~cede_ctrl, binomial_dist.sample((l,)).bool(), cede_ctrl)
        return cede_ctrl, risky, novel
    
    def forward(self, batch: RolloutBatchProtocol, state=None, **kwargs):
        batch = to_torch(batch, dtype=torch.float32, device=self.device)
        expert_result, base_result = self.get_results(batch)
        cede_ctrl, risky, novel = self.get_cede_ctrl(batch, expert_result, base_result)
        actions = torch.where(cede_ctrl, expert_result.act, base_result.act)
        self.t += 1
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

def parse_value(value):
    # Convert simple types (int, float, bool, None)
    if value.isdigit():
        return int(value)
    elif re.match(r'^\d+\.\d+$', value):
        return float(value)
    elif value == "True":
        return True
    elif value == "False":
        return False
    elif value == "None":
        return None
    elif value.startswith("[") and value.endswith("]"):
        # Convert the list items
        items = re.split(r',(?=[^\]]*(?:\[|$))', value[1:-1])
        return [parse_value(item.strip()) for item in items]
    elif value.startswith("(") and value.endswith(")"):
        # Convert the tuple items
        items = re.split(r',(?=[^\)]*(?:\(|$))', value[1:-1])
        # Special case for single-item tuple
        if len(items) == 2 and items[0].strip() != '':
            return (parse_value(items[0].strip()),)
        return tuple(parse_value(item.strip()) for item in items)
    elif value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    # Else, return the value as-is
    return value

def get_args(event_file, device):
    ea = EventAccumulator(event_file)
    ea.Reload()  # Load the file
    # Get the text data
    texts = ea.Tags()["tensors"]
    # Extract the actual text content
    text_data = {}
    for tag in texts:
        events = ea.Tensors(tag)
        for event in events:
            # You can extract the wall_time and step if needed
            # wall_time, step, value = event.wall_time, event.step, event.text
            text_data[tag] = event.tensor_proto.string_val
    data = text_data['args/text_summary'][0]
    # Convert bytes to string
    data_str = data.decode('utf-8')
    # Remove the "Namespace(" prefix and the trailing ")"
    data_str = data_str[len("Namespace("):-1]
    # Split into key-value pairs
    key_values = re.split(r',(?=\s\w+=)', data_str)
    # Parse each key-value pair
    args_dict = {}
    for kv in key_values:
        key, value = kv.split('=', 1)
        key = key.strip()
        args_dict[key] = parse_value(value)
    args = SimpleNamespace(**args_dict)
    try:
        env = gym.make(args.task)
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)
    except Exception:
        pass
    return args

def load_policy(args, path, device, hidden_sizes=None):
    if hidden_sizes:
        args.hidden_sizes = hidden_sizes
    env = gym.make(args.task)
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
    alpha = args.alpha
    if args.auto_alpha:
        log_alpha = torch.tensor([np.log(args.alpha)], requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    policy = DSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        risk_type='wang',
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        action_space=env.action_space,
        device=device,
        distortion_param=0.75,
    )
    dirname = os.path.dirname(path)
    if os.path.isfile(os.path.join(dirname, "actor.pth")):
        policy.actor.load_state_dict(torch.load(os.path.join(dirname, "actor.pth"), map_location=device))
        print("Loaded actor from: ", os.path.join(dirname, "actor.pth"))
    if os.path.isfile(os.path.join(dirname, "critic1.pth")):
        policy.critic1.load_state_dict(torch.load(os.path.join(dirname, "critic1.pth"), map_location=device))
        policy.critic1_old.load_state_dict(torch.load(os.path.join(dirname, "critic1.pth"), map_location=device))
        print("Loaded critic1 from: ", os.path.join(dirname, "critic1.pth"))
    if os.path.isfile(os.path.join(dirname, "critic2.pth")):
        policy.critic2.load_state_dict(torch.load(os.path.join(dirname, "critic2.pth"), map_location=device))
        policy.critic2_old.load_state_dict(torch.load(os.path.join(dirname, "critic2.pth"), map_location=device))
        print("Loaded critic2 from: ", os.path.join(dirname, "critic2.pth"))
    else:
        policy.load_state_dict(torch.load(path, map_location=device))
        print("Loaded agent from: ", path)
    return policy

def load_behavioral_crtitic(args, path, device):
    behavioral_critic = QuantileMlp(
        input_size=args.state_shape[0] + args.action_shape[0],
        hidden_sizes=args.hidden_sizes,
        device=device,
    ).to(device)
    behavioral_critic.load_state_dict(torch.load(path, map_location=device))
    return behavioral_critic

def get_model(log_path, sac_args, type=None, hidden_sizes=None):
    if type == "behavioral":
        files = os.listdir(log_path)
        event_file = [f for f in files if f.startswith('event')][0]
        full_path = os.path.join(log_path, event_file)
        args = get_args(full_path, sac_args.device)
        resume_path = os.path.join(log_path, 'model.pth')
        policy = load_behavioral_crtitic(args, resume_path, sac_args.device)
    elif type == "codac":
        files = os.listdir(log_path)
        event_file = [f for f in files if f.startswith('event')][0]
        full_path = os.path.join(log_path, event_file)
        args = get_args(full_path, sac_args.device)
        resume_path = os.path.join(log_path, 'policy.pth')
        policy = load_policy(args, resume_path, sac_args.device)
    else:
        resume_path = os.path.join(log_path, 'policy.pth')
        policy = load_policy(sac_args, resume_path, sac_args.device, hidden_sizes)
    return policy

def get_returns(policy, n, env, device, return_cede_ctrl=False):
    policy.eval()
    returns = []
    cede_ctrl = []
    novel = []
    risky = []
    for _ in range(n):
        obs, info = env.reset()
        done = False
        cum_reward = 0
        actions = []
        if return_cede_ctrl:
            cctrl_ep = []
            novel_ep = []
            risky_ep = []
        while not done:
            batch = Batch(obs=torch.from_numpy(np.expand_dims(obs, 0)).to(device), info=info)
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
        return np.array(returns).squeeze().mean(),  np.array(cede_ctrl).mean(), np.array(novel).mean(), np.array(risky).mean()
    return np.array(returns).squeeze().mean()

def main():
    parser = argparse.ArgumentParser(description="Training base policy while executing mixed actions")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--task', type=str, default="HalfCheetah-v2", help='Task name')
    parser.add_argument("--data-task", type=str, default="halfcheetah-medium-v2")
    parser.add_argument('--risk-type', type=str, default="wang", help='Risk type')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Buffer size')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256, 256], help='Hidden layer sizes')
    parser.add_argument('--actor-lr', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma value')
    parser.add_argument('--tau', type=float, default=0.005, help='Tau value')
    parser.add_argument('--alpha', type=float, default=0.4, help='Alpha value')
    parser.add_argument('--auto-alpha', action='store_true', help='Auto alpha flag')
    parser.add_argument('--exploration', action='store_true', help='Exploration flag')
    parser.add_argument('--alpha-lr', type=float, default=3e-4, help='Alpha learning rate')
    parser.add_argument('--start-timesteps', type=int, default=10000, help='Start timesteps')
    parser.add_argument('--epoch', type=int, default=300, help='Number of epochs')
    parser.add_argument('--step-per-epoch', type=int, default=500, help='Steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=1, help='Steps per collect')
    parser.add_argument('--update-per-step', type=int, default=1, help='Updates per step')
    parser.add_argument('--n-step', type=int, default=1, help='N step')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--training-num', type=int, default=1, help='Training number')
    parser.add_argument('--test-num', type=int, default=5, help='Test number')
    parser.add_argument('--logdir', type=str, default="log", help='Log directory')
    parser.add_argument('--device', type=str, default="cuda:2", help='Device to use')
    parser.add_argument('--distortion-param', type=float, default=-0.75, help='Distortion parameter')
    parser.add_argument('--risk-penalty', type=int, default=20, help='Risk penalty')
    parser.add_argument('--risk-prob', type=float, default=0.95, help='Risk probability')
    parser.add_argument('--high-state', type=float, default=1.5, help='High state value')
    parser.add_argument('--bandwidth', type=float, default=5, help='KDE bandwidth')
    parser.add_argument('--percentile', type=float, default=1., help='KDE percentile')
    parser.add_argument('--learn-type', type=str, default="mixed10", help='Learn type options: "on-policy" "all" "mixed*int*"')
    parser.add_argument('--cede-ctrl-decay', type=float, default=0.99999, help='Cede control decay')
    parser.add_argument('--cede-ctrl-type', type=str, default="smooth", help='Cede control type options: "normal" "smooth"')
    parser.add_argument('--cede-ctrl-nature', type=str, default="NR", help='Cede control nature options: "N" for novel, "R" for risky, "NR" for both, "Never" for never ceding control and "Always" for always ceding control')
    parser.add_argument('--expert-policy-path', type=str, default="/data/user/R901105/dev/halfcheetah_expert", help='Expert policy path')
    parser.add_argument('--base-policy-path', type=str, default="/data/user/R901105/dev/log/HalfCheetah-v4/codac_bc/neutral/0/231127-113302", help='Base policy path')
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

    test_buffer = VectorReplayBuffer(5000, args.test_num)
    train_collector = Collector(mixed_policy, env, offline_data)
    test_collector = Collector(mixed_policy, env, test_buffer)

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