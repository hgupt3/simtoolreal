"""Benchmark TiledCamera vs MultiMeshRayCasterCamera at n=512 envs.

Reports for each backend:
  - sustained env-step throughput (steps/sec)
  - peak torch CUDA memory after warm-up
  - total GPU memory used (sampled via nvidia-smi)
  - mean GPU utilization during the timed window (sampled)

Usage:
    .venv_isaacsim/bin/python isaacsimenvs/tests/bench_raycaster_vs_tiled.py --backend tiled
    .venv_isaacsim/bin/python isaacsimenvs/tests/bench_raycaster_vs_tiled.py --backend raycaster
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def _spawn_gpu_sampler(stop_evt: threading.Event, samples: list[tuple[float, int, int]]):
    """Poll `nvidia-smi` every 0.5 s while `stop_evt` is unset, appending
    (timestamp, mem_used_mib, util_pct) tuples to `samples`."""
    def _run():
        while not stop_evt.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                )
                line = out.strip().splitlines()[0]
                mem_used_mib, util_pct = (int(x.strip()) for x in line.split(","))
                samples.append((time.time(), mem_used_mib, util_pct))
            except Exception:
                pass
            stop_evt.wait(0.5)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tiled", "raycaster"), required=True)
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument("--problem", default="Lpeg.tol0p5mm")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--bench-steps", type=int, default=100)
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Override student-cam capture width. Default: yaml value.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Override student-cam capture height. Default: yaml value.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import yaml

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    spec = gym.spec(args.task)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    with Path(spec.kwargs["env_cfg_yaml_entry_point"]).open() as f:
        cfg.from_dict(yaml.safe_load(f) or {})

    cfg.scene.num_envs = int(args.num_envs)
    cfg.peg_in_hole.problem = args.problem
    cfg.peg_in_hole.goal_mode = "preInsertAndFinal"

    # Identical env config across the two backends: DR off, no crop, no
    # depth-aug / camera-pose-rand. We're benchmarking just the camera pipeline.
    cfg.student_obs.camera_backend = args.backend
    cfg.student_obs.horizontal_aperture_offset = 0.0
    cfg.student_obs.vertical_aperture_offset = 0.0
    cfg.student_obs.use_camera_delay = False
    cfg.student_obs.use_camera_pose_rand = False
    cfg.student_obs.use_depth_aug = False
    cfg.student_obs.crop_enabled = False
    if args.image_width is not None:
        cfg.student_obs.image_width = int(args.image_width)
    if args.image_height is not None:
        cfg.student_obs.image_height = int(args.image_height)
    cfg.student_obs.image_input_width = int(cfg.student_obs.image_width)
    cfg.student_obs.image_input_height = int(cfg.student_obs.image_height)
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False

    print(
        f"[bench] backend={args.backend} num_envs={args.num_envs} "
        f"cam_size={cfg.student_obs.image_width}x{cfg.student_obs.image_height} "
        f"warmup={args.warmup_steps} bench={args.bench_steps}",
        flush=True,
    )

    env = gym.make(args.task, cfg=cfg)
    inner = env.unwrapped
    actions = torch.zeros(
        (inner.num_envs, cfg.action_space), device=inner.device, dtype=torch.float32
    )
    env.reset()

    for _ in range(int(args.warmup_steps)):
        env.step(actions)
        _ = inner.get_student_obs()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Sample GPU every 0.5 s during the timed window.
    stop_evt = threading.Event()
    samples: list[tuple[float, int, int]] = []
    _spawn_gpu_sampler(stop_evt, samples)

    t0 = time.perf_counter()
    for step in range(int(args.bench_steps)):
        env.step(actions)
        _ = inner.get_student_obs()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop_evt.set()

    peak_torch_mib = float(torch.cuda.max_memory_allocated()) / 1024**2
    fps_env = args.bench_steps / max(elapsed, 1e-9)
    fps_total = fps_env * inner.num_envs

    if samples:
        peak_total_mib = max(s[1] for s in samples)
        mean_util_pct = sum(s[2] for s in samples) / len(samples)
    else:
        peak_total_mib = float("nan")
        mean_util_pct = float("nan")

    result = {
        "backend": args.backend,
        "num_envs": int(args.num_envs),
        "image_width": int(cfg.student_obs.image_width),
        "image_height": int(cfg.student_obs.image_height),
        "bench_steps": int(args.bench_steps),
        "elapsed_s": float(elapsed),
        "env_steps_per_s": float(fps_env),
        "total_env_steps_per_s": float(fps_total),
        "peak_torch_mib": float(peak_torch_mib),
        "peak_total_gpu_mib": float(peak_total_mib),
        "mean_gpu_util_pct": float(mean_util_pct),
        "gpu_samples": int(len(samples)),
    }
    print(f"[bench] result: {json.dumps(result, indent=2)}", flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[bench] wrote {args.out_json}", flush=True)

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
