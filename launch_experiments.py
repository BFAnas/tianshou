import itertools
import subprocess
from multiprocessing import Pool
from time import sleep
import time

# Define your command arguments
learn_types = ['on-policy', 'all', 'mixed5']
cede_ctrl_types = ['normal', 'smooth']
cede_ctrl_natures = ['N', 'R', 'NR', 'Never', 'Always']

# Create all combinations
all_combinations = itertools.product(learn_types, cede_ctrl_types, cede_ctrl_natures)

# Function to check if a combination is valid
def is_valid_combination(combination):
    learn_type, cede_ctrl_type, cede_ctrl_nature = combination
    if cede_ctrl_type == 'smooth' and cede_ctrl_nature in ['Always', 'Never']:
        return False
    if learn_type in ['on-policy', 'mixed5'] and cede_ctrl_nature in ['Always', 'Never']:
        return False
    return True

# Filter out invalid combinations
valid_combinations = [c for c in all_combinations if is_valid_combination(c)]

# Create a list of (index, combination) pairs from the valid combinations
indexed_combinations = list(enumerate(valid_combinations))

# Function to run a command on a specific GPU
def run_command(combination, gpu_id):
    cmd = f"/data/user/R901105/.conda/envs/dev/bin/python /data/user/R901105/dev/my_fork/tianshou/train_risky_novel.py --learn-type {combination[0]} --cede-ctrl-type {combination[1]} --cede-ctrl-nature {combination[2]} --device cuda:{gpu_id} --auto-alpha"
    subprocess.run(cmd, shell=True)

# Function to allocate GPU and run commands
def allocate_gpu(indexed_combination):
    index, combination = indexed_combination
    gpu_id = index % 4  # Use the index to allocate a GPU
    time.sleep(index * 5)
    run_command(combination, gpu_id)

# Use a Pool to run 4 commands in parallel
with Pool(4) as pool:
    pool.map(allocate_gpu, indexed_combinations)

