"""Minimal test: can each rank successfully call gym.create_sim()?"""
import os
import time

# IsaacGym MUST be imported before torch
from isaacgym import gymapi

import torch
import torch.distributed as dist

local_rank = int(os.getenv("LOCAL_RANK", "0"))
global_rank = int(os.getenv("RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

print(f"[RANK {global_rank}] local_rank={local_rank} world_size={world_size} "
      f"CUDA_VISIBLE_DEVICES={cvd}", flush=True)

# Init torch distributed
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)

# Test PyTorch CUDA works on this rank
t = torch.zeros(1, device=f"cuda:{local_rank}")
print(f"[RANK {global_rank}] PyTorch cuda:{local_rank} OK - {torch.cuda.get_device_name(local_rank)}", flush=True)

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

compute_device = local_rank
graphics_device = local_rank

print(f"[RANK {global_rank}] calling gym.create_sim(compute={compute_device}, graphics={graphics_device})", flush=True)
t0 = time.time()
sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
elapsed = time.time() - t0

if sim is None:
    print(f"[RANK {global_rank}] create_sim FAILED (returned None) after {elapsed:.1f}s", flush=True)
else:
    print(f"[RANK {global_rank}] create_sim SUCCESS in {elapsed:.1f}s", flush=True)
    gym.destroy_sim(sim)

dist.barrier()
print(f"[RANK {global_rank}] all ranks reached barrier, test PASSED", flush=True)
dist.destroy_process_group()
