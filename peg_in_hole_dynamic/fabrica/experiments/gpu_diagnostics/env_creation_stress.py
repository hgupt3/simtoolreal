"""
Stress test: create envs with the real training config on 4 GPUs.
Runs the actual training entrypoint but exits right after env creation
(before any training loop) by monkey-patching the runner.
Also logs timing and memory at each stage.
"""
import os
import sys
import time

# IsaacGym must be imported first
from isaacgym import gymapi, gymtorch

import torch
import torch.distributed as dist

local_rank = int(os.getenv("LOCAL_RANK", "0"))
global_rank = int(os.getenv("RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

def log(msg):
    mem = torch.cuda.memory_allocated(local_rank) / 1e9 if torch.cuda.is_available() else 0
    mem_reserved = torch.cuda.memory_reserved(local_rank) / 1e9 if torch.cuda.is_available() else 0
    print(f"[RANK {global_rank} t={time.time()-T0:.1f}s mem={mem:.2f}GB reserved={mem_reserved:.2f}GB] {msg}", flush=True)

T0 = time.time()

log(f"Starting. local_rank={local_rank} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Use the real training entrypoint but intercept after env creation
# We'll use hydra + the isaacgymenvs.train module approach
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import isaacgymenvs

config_dir = os.path.join(os.path.dirname(isaacgymenvs.__file__), "cfg")

# Number of envs to test - configurable via ENV_TEST_NUM_ENVS
num_envs = int(os.environ.get("ENV_TEST_NUM_ENVS", "6144"))

log(f"Will create {num_envs} envs")

with initialize_config_dir(config_dir=config_dir):
    cfg = compose(config_name="config", overrides=[
        "task=FabricaEnvLSTMAsymmetric",
        f"task.env.numEnvs={num_envs}",
        "headless=True",
        "multi_gpu=True",
        "task.env.multiPart=True",
        # Use a small set of parts to isolate memory vs part-count issues
        "task.env.objectNames={}".format(os.environ.get("ENV_TEST_OBJECTS", '["beam_2_coacd"]'))
    ])

log("Config loaded")

# Now create the env using rlgames_utils
from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

env_creator = get_rlgames_env_creator(
    seed=0,
    task_config=OmegaConf.to_container(cfg.task, resolve=True),
    task_name=cfg.task_name,
    sim_device=f"cuda:{local_rank}",
    rl_device=f"cuda:{local_rank}",
    graphics_device_id=local_rank,
    headless=True,
    multi_gpu=True,
)

log("About to create env (this calls create_sim + create_envs)")

# Init process group before env creation — the serialization fix in rlgames_utils.py
# uses dist.barrier() which requires the process group to be initialized.
# In real training, a2c_common.py does this before calling the env creator.
if not dist.is_initialized():
    dist.init_process_group("gloo")

env = env_creator()
log("Env created successfully!")

# Report GPU memory
for i in range(world_size):
    if global_rank == i:
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(local_rank) / 1e9
            mem_reserved = torch.cuda.memory_reserved(local_rank) / 1e9
            max_mem = torch.cuda.max_memory_allocated(local_rank) / 1e9
            log(f"Final: allocated={mem:.2f}GB reserved={mem_reserved:.2f}GB peak={max_mem:.2f}GB")
    if dist.is_initialized():
        dist.barrier()

log("Test PASSED - all ranks created envs successfully")

if dist.is_initialized():
    dist.destroy_process_group()
