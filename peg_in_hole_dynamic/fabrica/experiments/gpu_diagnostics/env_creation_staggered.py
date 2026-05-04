"""
Same as env_creation_stress.py but staggers rank startup to reduce contention.
Each rank waits (local_rank * STAGGER_SECONDS) before creating its env.
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

T0 = time.time()

def log(msg):
    mem = torch.cuda.memory_allocated(local_rank) / 1e9 if torch.cuda.is_available() else 0
    mem_reserved = torch.cuda.memory_reserved(local_rank) / 1e9 if torch.cuda.is_available() else 0
    print(f"[RANK {global_rank} t={time.time()-T0:.1f}s mem={mem:.2f}GB reserved={mem_reserved:.2f}GB] {msg}", flush=True)

log(f"Starting. local_rank={local_rank} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Stagger init
stagger_secs = int(os.environ.get("STAGGER_SECONDS", "60"))
wait_time = local_rank * stagger_secs
log(f"Staggering: waiting {wait_time}s before env creation")
time.sleep(wait_time)
log(f"Stagger wait done, proceeding with env creation")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import isaacgymenvs

config_dir = os.path.join(os.path.dirname(isaacgymenvs.__file__), "cfg")
num_envs = int(os.environ.get("ENV_TEST_NUM_ENVS", "6144"))
object_names = os.environ.get("ENV_TEST_OBJECTS", '["beam_2_coacd"]')

log(f"Will create {num_envs} envs")

with initialize_config_dir(config_dir=config_dir):
    cfg = compose(config_name="config", overrides=[
        "task=FabricaEnvLSTMAsymmetric",
        "task.env.numEnvs={}".format(num_envs),
        "headless=True",
        "multi_gpu=True",
        "task.env.multiPart=True",
        "task.env.objectNames={}".format(object_names),
    ])

log("Config loaded")

from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

env_creator = get_rlgames_env_creator(
    seed=0,
    task_config=OmegaConf.to_container(cfg.task, resolve=True),
    task_name=cfg.task_name,
    sim_device="cuda:{}".format(local_rank),
    rl_device="cuda:{}".format(local_rank),
    graphics_device_id=local_rank,
    headless=True,
    multi_gpu=True,
)

t_create = time.time()
log("About to create env")

# Init process group before env creation — the serialization fix in rlgames_utils.py
# uses dist.barrier() which requires the process group to be initialized.
if not dist.is_initialized():
    dist.init_process_group("gloo")

env = env_creator()
create_elapsed = time.time() - t_create
log(f"Env created in {create_elapsed:.1f}s (wall clock for this rank's env creation only)")

# Wait for all ranks
# Need to init NCCL for the barrier since we haven't yet
if not dist.is_initialized():
    dist.init_process_group("nccl")
dist.barrier()

log("All ranks done. Test PASSED")
dist.destroy_process_group()
