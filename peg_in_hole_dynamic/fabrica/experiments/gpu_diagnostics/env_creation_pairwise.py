"""
Test pairwise env creation: ranks 0,1 create envs, then ranks 2,3.
This tests whether existing PhysX contexts on finished ranks interfere
with new env creation on later ranks.
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
    print("[RANK {} t={:.1f}s mem={:.2f}GB reserved={:.2f}GB] {}".format(
        global_rank, time.time()-T0, mem, mem_reserved, msg), flush=True)

log("Starting. local_rank={} CUDA_VISIBLE_DEVICES={}".format(
    local_rank, os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import isaacgymenvs

config_dir = os.path.join(os.path.dirname(isaacgymenvs.__file__), "cfg")
num_envs = int(os.environ.get("ENV_TEST_NUM_ENVS", "6144"))
object_names = os.environ.get("ENV_TEST_OBJECTS", '["beam_2_coacd"]')

log("Will create {} envs".format(num_envs))

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
from isaacgymenvs.tasks import isaacgym_task_map

# Init process group
if not dist.is_initialized():
    dist.init_process_group("gloo")

task_config = OmegaConf.to_container(cfg.task, resolve=True)

# We bypass get_rlgames_env_creator's serialization and do our own here
# to test different serialization strategies.
_sim_device = "cuda:{}".format(local_rank)
_rl_device = "cuda:{}".format(local_rank)
task_config['rank'] = local_rank
task_config['rl_device'] = _rl_device

strategy = os.environ.get("SERIALIZE_STRATEGY", "sequential")
log("Using serialization strategy: {}".format(strategy))

if strategy == "sequential":
    # One at a time (what rlgames_utils.py now does)
    for turn in range(world_size):
        if global_rank == turn:
            log("Creating env (sequential, turn {}/{})".format(turn, world_size))
            t0 = time.time()
            env = isaacgym_task_map[cfg.task_name](
                cfg=task_config, rl_device=_rl_device, sim_device=_sim_device,
                graphics_device_id=local_rank, headless=True,
                virtual_screen_capture=False, force_render=False,
            )
            log("Env created in {:.1f}s".format(time.time() - t0))
        dist.barrier()

elif strategy == "pairwise":
    # Two at a time: ranks 0,1 together, then ranks 2,3
    pair = global_rank // 2  # 0 for ranks 0,1; 1 for ranks 2,3
    for pair_turn in range(2):
        if pair == pair_turn:
            log("Creating env (pairwise, pair {}, turn {})".format(pair, pair_turn))
            t0 = time.time()
            env = isaacgym_task_map[cfg.task_name](
                cfg=task_config, rl_device=_rl_device, sim_device=_sim_device,
                graphics_device_id=local_rank, headless=True,
                virtual_screen_capture=False, force_render=False,
            )
            log("Env created in {:.1f}s".format(time.time() - t0))
        dist.barrier()

elif strategy == "simultaneous":
    # All at once (original behavior, for comparison)
    log("Creating env (simultaneous)")
    t0 = time.time()
    env = isaacgym_task_map[cfg.task_name](
        cfg=task_config, rl_device=_rl_device, sim_device=_sim_device,
        graphics_device_id=local_rank, headless=True,
        virtual_screen_capture=False, force_render=False,
    )
    log("Env created in {:.1f}s".format(time.time() - t0))

dist.barrier()
log("All ranks done. Test PASSED")
dist.destroy_process_group()
