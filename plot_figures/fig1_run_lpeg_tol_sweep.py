#!/usr/bin/env python3
"""Sweep L-peg matched-mass contact tolerances for Fig 1.

Two modes:

  **Orchestrator** (default): loops over (method, tolerance), launches a
  fresh worker subprocess per eval, collects per-tolerance JSON, writes
  the aggregated ``sim_results.json``.

  **Worker** (``--worker --method M --tolerance-mm T``): runs a single
  eval — fresh AppLauncher, env build, rollout, JSON write — then exits.

Subprocess-per-eval is necessary because IsaacSim's USD stage does not
cleanly reset across env.close() / rebuild within a single process. Each
worker pays ~25 s AppLauncher startup but produces deterministic state.

Per-tolerance pipeline:
  - Build the Isaaclab PegInHole env at problem Lpeg_matchedmass.tol<X>mm.
  - Load the rl-games SAPG teacher checkpoint.
  - Roll out 500 envs in a single batch for up to --max-steps steps.
  - Record per-env first-episode success and step count. ``env._successes``
    is monotonic non-decreasing within an episode and is zeroed by reset,
    so we read ``env._prev_episode_successes`` after a reset to get the
    just-ended episode's final count.
  - Compute per-robot wall-clock throughput:
        throughput = mean_first_succ * 60 / (mean_first_steps * control_dt)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CHECKPOINTS = {
    "play2win": (
        "train_dir/teachers_final/play2win_peg_insertion/"
        "lpeg_tol0p5mm_finetune_rgf0_dr_2026-05-15_21-56-40/"
        "0_lpeg_tol0p5mm_finetune_rgf0_dr_2026-05-15_21-56-40/last/model.pth"
    ),
    "play_only": "pretrained_policy/model.pth",
}

# Per-method problem family override. Play-only uses the ``_dense`` family
# (3-goal sequence: 15 cm pre-insert + 5 cm pre-insert + final) for finer-
# grained subgoal guidance during insertion, combined with the looser
# ``insertion_success_tolerance`` (see METHOD_INSERTION_SUCCESS_TOL_M). Play2Win
# was trained on the standard 2-goal sequence so we leave it on
# the non-dense family. Earlier we tried a ``_playnatural`` 4-goal variant
# with a lift-at-start waypoint, but the fixed-receptive-frame lift target
# was inconsistent with the env's per-episode peg-start randomization.
METHOD_PROBLEM_FAMILY = {
    "play2win": None,
    "play_only": "Lpeg_matchedmass_dense",
}

# Per-method success criterion (env's ``insertion_success_tolerance`` in
# meters; multiplied by ``keypoint_scale=1.5`` inside the env). Play2Win
# uses the clearance-scaled criterion (see
# ``_scaled_insertion_success_tol_m``); Play-only is held at a fixed
# loose 30 mm keypoint criterion (cfg value 0.020) so its numbers are
# directly comparable to the v3 sweep.
METHOD_INSERTION_SUCCESS_TOL_M: dict = {
    "play_only": 0.020,
}


# Clearance-scaled success criterion. The keypoint criterion (= cfg value
# x keypoint_scale=1.5) must always exceed the mechanical play in the hole;
# otherwise success can never fire at wide clearances. Floor stays at the
# strict 10 mm cfg value (15 mm keypoint) up to 1 mm per-side clearance
# (2 mm diameter), then grows 1:1 with clearance:
#   succ_tol_m = 0.010 + max(0, tol_per_side_m - 0.001)
# So at per-side 20 mm (40 mm diameter), the cfg value is 0.029 and the
# keypoint criterion is 43.5 mm. Same formula applied to both methods so
# the cross-method comparison is exact at every tolerance.
def _scaled_insertion_success_tol_m(tol_per_side_mm: float) -> float:
    tol_per_side_m = float(tol_per_side_mm) * 1e-3
    return 0.010 + max(0.0, tol_per_side_m - 0.001)

DEFAULT_WORKER_PYTHON = str(REPO_ROOT / ".venv_isaacsim" / "bin" / "python")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PegInHole-Direct-v0")
    p.add_argument("--problem-family", default="Lpeg_matchedmass")
    p.add_argument(
        "--tolerances",
        type=str,
        default=None,
        help="Comma-separated **per-side** tolerance values in mm. Default: "
             "geomspace(0.1, 5.0, 10) (which is diameter geomspace(0.2, 10, 10)).",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="play2win,play_only",
        help="Comma list. Each must be a key in CHECKPOINTS.",
    )
    p.add_argument("--num-envs", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument(
        "--keep-dr",
        type=int,
        default=0,
        choices=(0, 1),
        help="0: disable DR knobs for clean per-tolerance reading. 1: keep training DR on.",
    )
    p.add_argument(
        "--hole-yaw-range-deg",
        type=float,
        default=10.0,
        help="Per-episode hole yaw randomization, applied symmetrically about "
             "world +Z (yaw ~ U(-range, +range) deg). Default 10.0 matches the "
             "Play2Win training overlay.",
    )
    p.add_argument(
        "--control-dt",
        type=float,
        default=1.0 / 60.0,
        help="Seconds per env step (decimation already folded in). Matches "
             "CONTROL_DT in peg_in_hole_dynamic/eval_isaacsim.py. Used for "
             "the per-robot throughput conversion.",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "plot_figures" / "inputs" / "fig1"),
    )
    p.add_argument(
        "--worker-python",
        default=DEFAULT_WORKER_PYTHON,
        help="Python interpreter used to launch per-eval workers. Defaults to "
             ".venv_isaacsim/bin/python.",
    )
    p.add_argument(
        "--init-fail-step-thresh",
        type=int,
        default=100,
        help="Envs whose first episode terminates strictly before this step "
             "count are treated as unstable-init failures and excluded from "
             "the filtered stats (the raw stats include them). Successes "
             "empirically never happen in <100 steps, so this drops only "
             "fall-at-init artifacts.",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip running evals; just re-aggregate from existing raw JSONs.",
    )

    # Worker-mode flags.
    p.add_argument(
        "--worker",
        action="store_true",
        help="Run as a single-eval worker (one method, one tolerance).",
    )
    p.add_argument("--method", default=None, help="Worker-mode: which method to eval.")
    p.add_argument(
        "--tolerance-mm",
        type=float,
        default=None,
        help="Worker-mode: which tolerance (in mm) to eval.",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Worker-mode: where to write the per-eval JSON.",
    )
    p.add_argument(
        "--insertion-success-tol-m",
        type=float,
        default=None,
        help="Worker-mode override for ``insertion_success_tolerance`` in "
             "meters. If set, takes precedence over the per-method default "
             "in METHOD_INSERTION_SUCCESS_TOL_M. Used by the wide-tolerance "
             "patch for play2win (re-run at per-side >= 6.16 mm with "
             "0.020 m to match Play-only's looser criterion).",
    )
    return p.parse_args()


def _fmt_tol(t: float) -> str:
    if float(t).is_integer():
        return str(int(t))
    s = f"{t:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _resolve_tolerances(arg: str | None) -> list[float]:
    import numpy as np

    if arg:
        return [float(x) for x in arg.split(",") if x.strip()]
    # Default: per-side geomspace 0.1 -> 20.0 mm in 10 steps (equivalently
    # diameter geomspace 0.2 -> 40.0 mm — matches the paper x-axis).
    return [round(float(x), 2) for x in np.geomspace(0.1, 20.0, 10)]


# ----------------------------------------------------------------------------
# Worker (single-eval) implementation.
# ----------------------------------------------------------------------------


def _reseed(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _coerce_obs(o):
    if isinstance(o, tuple):
        o = o[0]
    if isinstance(o, dict):
        o = o.get("obs", o)
    return o


def _run_one_eval(args: argparse.Namespace) -> int:
    """Single eval, fresh AppLauncher, writes one JSON, then exits."""
    if not args.method or args.tolerance_mm is None or not args.output_json:
        print("[worker] --method, --tolerance-mm, --output-json all required", flush=True)
        return 2
    if args.method not in CHECKPOINTS:
        print(f"[worker] unknown method {args.method!r}", flush=True)
        return 2

    ckpt_rel = CHECKPOINTS[args.method]
    ckpt = (REPO_ROOT / ckpt_rel).resolve()
    if not ckpt.is_file():
        print(f"[worker] checkpoint not found: {ckpt}", flush=True)
        # Write a stub JSON so the orchestrator can detect the failure.
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({
            "method": args.method,
            "tolerance_mm": float(args.tolerance_mm),
            "error": f"checkpoint not found: {ckpt}",
        }, indent=2))
        return 3

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = False
    app = AppLauncher(launcher_args).app

    import torch  # noqa: E402

    import isaacsimenvs  # noqa: F401, E402  triggers gym.register side effects
    import peg_in_hole_dynamic.peg.problems  # noqa: F401, E402  registers Lpeg_matchedmass.*
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env  # noqa: E402
    from peg_in_hole_dynamic.eval_isaacsim import (  # noqa: E402
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
        _set_attr_path,
    )
    from rl_games.torch_runner import Runner, _load_checkpoint_weights  # noqa: E402

    # DR knobs we explicitly disable at eval (mirrors EVAL_DR_OFF_OVERRIDES
    # minus the reset_dof_* / table_reset_z_range entries — we leave reset
    # noise at training defaults so the eval samples the start distribution
    # the policy was trained under).
    DR_KNOBS_OFF = {
        "env.domain_randomization.use_obs_delay": False,
        "env.domain_randomization.use_action_delay": False,
        "env.domain_randomization.use_object_state_delay_noise": False,
        "env.domain_randomization.object_state_xyz_noise_std": 0.0,
        "env.domain_randomization.object_state_rotation_noise_degrees": 0.0,
        "env.domain_randomization.object_scale_noise_multiplier_range": [1.0, 1.0],
        "env.domain_randomization.joint_velocity_obs_noise_std": 0.0,
        "env.domain_randomization.force_scale": 0.0,
        "env.domain_randomization.torque_scale": 0.0,
    }

    tol_tag = _fmt_tol(args.tolerance_mm)
    # Per-method problem family override (e.g. play_only uses the dense
    # multi-waypoint sequence). Falls back to --problem-family when unset.
    method_family = METHOD_PROBLEM_FAMILY.get(args.method)
    problem_family = method_family if method_family else args.problem_family
    problem = f"{problem_family}.tol{tol_tag}mm"
    print(
        f"[worker] method={args.method}  tol={args.tolerance_mm}mm  "
        f"problem={problem}  num_envs={args.num_envs}  max_steps={args.max_steps}",
        flush=True,
    )

    # Success criterion: scaled with hole clearance by default (so the
    # keypoint criterion always exceeds the mechanical play); per-method
    # dict and CLI flag are escape hatches if set.
    if args.insertion_success_tol_m is not None:
        success_tol_m = float(args.insertion_success_tol_m)
    elif args.method in METHOD_INSERTION_SUCCESS_TOL_M:
        success_tol_m = float(METHOD_INSERTION_SUCCESS_TOL_M[args.method])
    else:
        success_tol_m = _scaled_insertion_success_tol_m(args.tolerance_mm)

    cfg = _load_env_cfg(args.task)
    _apply_env_overrides(
        cfg,
        problem=problem,
        goal_mode="preInsertAndFinal",
        random_goal_fraction=0.0,
        insertion_success_tolerance=float(success_tol_m),
        retract_success_tolerance=0.005,
        num_envs=int(args.num_envs),
        sim_device=args.sim_device,
        sdf=False,
        keep_dr=True,
        extra_overrides={},
    )
    if not bool(args.keep_dr):
        for _k, _v in DR_KNOBS_OFF.items():
            _set_attr_path(cfg, _k, _v)
    _set_attr_path(cfg, "env.peg_in_hole.hole_yaw_range_deg", float(args.hole_yaw_range_deg))
    # `success_steps=1`: a goal counts as hit after 1 consecutive frame in
    # tolerance. The training default for play2win was also 1; we mirror it
    # so eval matches. This also helps in very wide holes where the peg may
    # only momentarily settle in the keypoint tolerance.
    _set_attr_path(cfg, "env.termination.success_steps", 1)
    # `enable_retract=False`: episode exits as soon as `_successes >=
    # env_max_goals` (i.e. all insertion subgoals hit, no retract phase).
    # Retract is impossible at wide clearances (peg has too much play to
    # be lifted cleanly), so this lets us measure insertion alone.
    _set_attr_path(cfg, "env.peg_in_hole.enable_retract", False)

    env = _instantiate_env(args.task, cfg)

    agent_cfg = _configure_agent(
        args.task,
        "rl_games_sapg_cfg_entry_point",
        rl_device=args.rl_device,
        num_envs=int(args.num_envs),
        deterministic=True,
        games=1,
        extra_overrides={},
    )
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(
        env, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions,
    )

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    weights = _load_checkpoint_weights(player, str(ckpt))
    player.set_weights(weights)
    player.has_batch_dimension = True

    _reseed(args.seed)
    player.reset()
    obs = _coerce_obs(player.env_reset(wrapped))

    num_envs = int(args.num_envs)
    device = args.rl_device
    done_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_succ = torch.zeros(num_envs, dtype=torch.float, device=device)
    first_steps = torch.full(
        (num_envs,), int(args.max_steps), dtype=torch.long, device=device,
    )

    t0 = time.time()
    for step in range(int(args.max_steps)):
        with torch.no_grad():
            action = player.get_action(obs, is_deterministic=True)
        obs, _rew, dones, _infos = player.env_step(wrapped, action)
        obs = _coerce_obs(obs)

        done_now = dones.bool().view(-1).to(device)
        newly_done = done_now & (~done_once)
        if bool(newly_done.any().item()):
            # After env_step, auto-reset has already written _prev_episode_successes
            # (see isaacsimenvs/tasks/simtoolreal/utils/reset_utils.py:342). Read
            # that to get the just-ended episode's success count.
            prev_succ = env._prev_episode_successes
            prev_mg = env.prev_episode_env_max_goals.clamp_min(1)
            succ_bool = (prev_succ >= prev_mg).float()
            first_succ[newly_done] = succ_bool[newly_done]
            first_steps[newly_done] = step + 1
        done_once = done_once | done_now
        if bool(done_once.all().item()):
            break
    wall = time.time() - t0

    first_succ_rate = float(first_succ.mean().item())
    first_steps_mean = float(first_steps.float().mean().item())
    first_finished = float(done_once.float().mean().item())
    mean_ep_s = first_steps_mean * float(args.control_dt)
    throughput = first_succ_rate * 60.0 / mean_ep_s if mean_ep_s > 1e-9 else 0.0

    res = {
        "method": args.method,
        "checkpoint": str(ckpt_rel),
        "problem": problem,
        "tolerance_mm": float(args.tolerance_mm),
        "num_envs": num_envs,
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "control_dt_seconds": float(args.control_dt),
        "keep_dr": bool(args.keep_dr),
        "hole_yaw_range_deg": float(args.hole_yaw_range_deg),
        "insertion_success_tolerance_m": float(success_tol_m),
        "first_episode_success_rate": first_succ_rate,
        "first_episode_mean_steps": first_steps_mean,
        "first_episode_finished_frac": first_finished,
        "mean_episode_seconds": float(mean_ep_s),
        "throughput_per_min": float(throughput),
        "rollout_seconds": float(wall),
        "first_succ_per_env": first_succ.cpu().tolist(),
        "first_steps_per_env": first_steps.cpu().tolist(),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(
        f"[worker] succ={first_succ_rate:.3f}  "
        f"mean_steps={first_steps_mean:.1f}  "
        f"finished={first_finished:.3f}  "
        f"thr={throughput:.2f}/min  "
        f"wall={wall:.1f}s  -> {out_path.name}",
        flush=True,
    )

    try:
        env.close()
    except Exception:
        pass
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ----------------------------------------------------------------------------
# Filtered stat computation (init-fall failures discarded).
# ----------------------------------------------------------------------------


def _filtered_stats_from_raw(raw_json: dict, thresh: int, control_dt: float) -> dict:
    """Given a per-tolerance raw JSON, compute success_rate / throughput
    after discarding envs whose first episode terminated in fewer than
    ``thresh`` steps (treated as unstable-init failures).

    Returns a dict with keys: success_rate, mean_episode_steps,
    mean_episode_seconds, throughput_per_min, n_kept, n_discarded.
    """
    import numpy as np

    succ = np.asarray(raw_json["first_succ_per_env"], dtype=float)
    steps = np.asarray(raw_json["first_steps_per_env"], dtype=float)
    keep = steps >= int(thresh)
    n_kept = int(keep.sum())
    n_discarded = int((~keep).sum())
    if n_kept == 0:
        return {
            "success_rate": None,
            "mean_episode_steps": None,
            "mean_episode_seconds": None,
            "throughput_per_min": None,
            "n_kept": 0,
            "n_discarded": n_discarded,
        }
    succ_rate = float(succ[keep].mean())
    steps_mean = float(steps[keep].mean())
    mean_ep_s = steps_mean * control_dt
    thr = succ_rate * 60.0 / mean_ep_s if mean_ep_s > 1e-9 else 0.0
    return {
        "success_rate": succ_rate,
        "mean_episode_steps": steps_mean,
        "mean_episode_seconds": mean_ep_s,
        "throughput_per_min": thr,
        "n_kept": n_kept,
        "n_discarded": n_discarded,
    }


def _aggregate_from_raw_jsons(args: argparse.Namespace) -> int:
    """Re-aggregate sim_results.json from existing per-tolerance raw JSONs.

    For each (method, tolerance) emits both ``raw`` and ``filtered`` series.
    ``filtered`` drops envs that reset within the first
    ``--init-fail-step-thresh`` steps (unstable-init failures).
    """
    tolerances = _resolve_tolerances(args.tolerances)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"

    thresh = int(args.init_fail_step_thresh)
    control_dt = float(args.control_dt)

    aggregate = {
        "problem_family": args.problem_family,
        "task": args.task,
        "num_envs": int(args.num_envs),
        "max_steps_per_episode": int(args.max_steps),
        "seed": int(args.seed),
        "control_dt_seconds": control_dt,
        "keep_dr": bool(args.keep_dr),
        "hole_yaw_range_deg": float(args.hole_yaw_range_deg),
        "init_fail_step_thresh": thresh,
        "tolerances_mm": tolerances,
    }

    for method in methods:
        per = {
            "checkpoint": str(CHECKPOINTS.get(method, "")),
            "success_rate": [],
            "mean_episode_seconds": [],
            "throughput_per_min": [],
            "rollout_seconds": [],
            "first_episode_finished_frac": [],
            "first_episode_mean_steps": [],
            "filtered": {
                "success_rate": [],
                "mean_episode_seconds": [],
                "throughput_per_min": [],
                "mean_episode_steps": [],
                "n_kept": [],
                "n_discarded": [],
            },
            "raw_paths": [],
            "errors": {},
        }
        for tol_mm in tolerances:
            tol_tag = _fmt_tol(tol_mm)
            raw_path = raw_dir / f"{method}_tol{tol_tag}mm.json"
            if not raw_path.exists():
                per["errors"][tol_tag] = "raw JSON missing"
                for col in ("success_rate", "mean_episode_seconds",
                            "throughput_per_min", "rollout_seconds",
                            "first_episode_finished_frac",
                            "first_episode_mean_steps"):
                    per[col].append(None)
                for col in ("success_rate", "mean_episode_seconds",
                            "throughput_per_min", "mean_episode_steps"):
                    per["filtered"][col].append(None)
                per["filtered"]["n_kept"].append(0)
                per["filtered"]["n_discarded"].append(0)
                per["raw_paths"].append(None)
                continue

            d = json.loads(raw_path.read_text())
            if "error" in d:
                per["errors"][tol_tag] = d["error"]
                for col in ("success_rate", "mean_episode_seconds",
                            "throughput_per_min", "rollout_seconds",
                            "first_episode_finished_frac",
                            "first_episode_mean_steps"):
                    per[col].append(None)
                for col in ("success_rate", "mean_episode_seconds",
                            "throughput_per_min", "mean_episode_steps"):
                    per["filtered"][col].append(None)
                per["filtered"]["n_kept"].append(0)
                per["filtered"]["n_discarded"].append(0)
                per["raw_paths"].append(None)
                continue

            per["success_rate"].append(d.get("first_episode_success_rate"))
            per["mean_episode_seconds"].append(d.get("mean_episode_seconds"))
            per["throughput_per_min"].append(d.get("throughput_per_min"))
            per["rollout_seconds"].append(d.get("rollout_seconds"))
            per["first_episode_finished_frac"].append(d.get("first_episode_finished_frac"))
            per["first_episode_mean_steps"].append(d.get("first_episode_mean_steps"))
            per["raw_paths"].append(str(raw_path.relative_to(REPO_ROOT)))

            fs = _filtered_stats_from_raw(d, thresh, control_dt)
            per["filtered"]["success_rate"].append(fs["success_rate"])
            per["filtered"]["mean_episode_seconds"].append(fs["mean_episode_seconds"])
            per["filtered"]["throughput_per_min"].append(fs["throughput_per_min"])
            per["filtered"]["mean_episode_steps"].append(fs["mean_episode_steps"])
            per["filtered"]["n_kept"].append(fs["n_kept"])
            per["filtered"]["n_discarded"].append(fs["n_discarded"])

        aggregate[method] = per

    summary_path = output_dir / "sim_results.json"
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(f"=> wrote {summary_path}  (init_fail_step_thresh={thresh})", flush=True)
    # Compact per-method per-tolerance summary print.
    for method in methods:
        per = aggregate[method]
        print(f"\n[{method}]  raw / filtered (n_discarded)")
        for i, tol in enumerate(tolerances):
            s_raw = per["success_rate"][i]
            s_fil = per["filtered"]["success_rate"][i]
            n_dis = per["filtered"]["n_discarded"][i]
            t_raw = per["throughput_per_min"][i]
            t_fil = per["filtered"]["throughput_per_min"][i]
            if s_raw is None:
                print(f"  {tol:>5.2f} mm: missing/error")
            else:
                print(
                    f"  {tol:>5.2f} mm: succ {s_raw:.3f} / {s_fil:.3f}  "
                    f"thr {t_raw:.2f} / {t_fil:.2f}  "
                    f"(discarded {n_dis})"
                )
    return 0


# ----------------------------------------------------------------------------
# Orchestrator (subprocess-per-eval).
# ----------------------------------------------------------------------------


def _shared_worker_args(args: argparse.Namespace) -> list[str]:
    return [
        "--task", str(args.task),
        "--problem-family", str(args.problem_family),
        "--num-envs", str(int(args.num_envs)),
        "--max-steps", str(int(args.max_steps)),
        "--seed", str(int(args.seed)),
        "--rl-device", str(args.rl_device),
        "--sim-device", str(args.sim_device),
        "--keep-dr", str(int(args.keep_dr)),
        "--hole-yaw-range-deg", str(float(args.hole_yaw_range_deg)),
        "--control-dt", str(float(args.control_dt)),
    ]


def _orchestrate(args: argparse.Namespace) -> int:
    tolerances = _resolve_tolerances(args.tolerances)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in CHECKPOINTS:
            print(f"[err] unknown method {m!r}; valid: {list(CHECKPOINTS)}", flush=True)
            return 2

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    per_method: dict = {
        m: {
            "checkpoint": str(CHECKPOINTS[m]),
            "success_rate": [],
            "mean_episode_seconds": [],
            "throughput_per_min": [],
            "rollout_seconds": [],
            "first_episode_finished_frac": [],
            "first_episode_mean_steps": [],
            "raw_paths": [],
            "errors": {},
        }
        for m in methods
    }

    print(
        f"=> sweep: methods={methods}  tolerances_mm={tolerances}  "
        f"num_envs={args.num_envs}  max_steps={args.max_steps}  seed={args.seed}",
        flush=True,
    )

    shared = _shared_worker_args(args)
    worker_py = args.worker_python

    for method in methods:
        for tol_mm in tolerances:
            tol_tag = _fmt_tol(tol_mm)
            problem = f"{args.problem_family}.tol{tol_tag}mm"
            raw_path = raw_dir / f"{method}_tol{tol_tag}mm.json"
            print(
                f"\n=== method={method}  tol={tol_mm} mm  problem={problem} ===",
                flush=True,
            )

            cmd = [
                worker_py, str(Path(__file__).resolve()),
                "--worker",
                "--method", method,
                "--tolerance-mm", str(float(tol_mm)),
                "--output-json", str(raw_path),
                *shared,
            ]
            t_start = time.time()
            try:
                completed = subprocess.run(cmd, check=False)
            except KeyboardInterrupt:
                print("[orchestrator] interrupted", flush=True)
                return 130
            t_wall = time.time() - t_start

            if completed.returncode != 0:
                err = f"worker exit code {completed.returncode}"
                print(f"[err] {method} tol={tol_mm}mm: {err}", flush=True)
                per_method[method]["errors"][tol_tag] = err
                per_method[method]["success_rate"].append(None)
                per_method[method]["mean_episode_seconds"].append(None)
                per_method[method]["throughput_per_min"].append(None)
                per_method[method]["rollout_seconds"].append(None)
                per_method[method]["first_episode_finished_frac"].append(None)
                per_method[method]["first_episode_mean_steps"].append(None)
                per_method[method]["raw_paths"].append(None)
                continue

            if not raw_path.exists():
                err = "worker exited 0 but no JSON was written"
                print(f"[err] {method} tol={tol_mm}mm: {err}", flush=True)
                per_method[method]["errors"][tol_tag] = err
                per_method[method]["success_rate"].append(None)
                per_method[method]["mean_episode_seconds"].append(None)
                per_method[method]["throughput_per_min"].append(None)
                per_method[method]["rollout_seconds"].append(None)
                per_method[method]["first_episode_finished_frac"].append(None)
                per_method[method]["first_episode_mean_steps"].append(None)
                per_method[method]["raw_paths"].append(None)
                continue

            res = json.loads(raw_path.read_text())
            if "error" in res:
                err = res["error"]
                print(f"[err] {method} tol={tol_mm}mm: {err}", flush=True)
                per_method[method]["errors"][tol_tag] = err
                per_method[method]["success_rate"].append(None)
                per_method[method]["mean_episode_seconds"].append(None)
                per_method[method]["throughput_per_min"].append(None)
                per_method[method]["rollout_seconds"].append(None)
                per_method[method]["first_episode_finished_frac"].append(None)
                per_method[method]["first_episode_mean_steps"].append(None)
                per_method[method]["raw_paths"].append(None)
                continue

            print(
                f"  succ={res['first_episode_success_rate']:.3f}  "
                f"mean_steps={res['first_episode_mean_steps']:.1f}  "
                f"finished={res['first_episode_finished_frac']:.3f}  "
                f"thr={res['throughput_per_min']:.2f}/min  "
                f"wall={t_wall:.1f}s  -> {raw_path.name}",
                flush=True,
            )
            per_method[method]["success_rate"].append(res["first_episode_success_rate"])
            per_method[method]["mean_episode_seconds"].append(res["mean_episode_seconds"])
            per_method[method]["throughput_per_min"].append(res["throughput_per_min"])
            per_method[method]["rollout_seconds"].append(res["rollout_seconds"])
            per_method[method]["first_episode_finished_frac"].append(
                res["first_episode_finished_frac"]
            )
            per_method[method]["first_episode_mean_steps"].append(
                res["first_episode_mean_steps"]
            )
            per_method[method]["raw_paths"].append(
                str(raw_path.relative_to(REPO_ROOT))
            )

    aggregate = {
        "problem_family": args.problem_family,
        "task": args.task,
        "num_envs": int(args.num_envs),
        "max_steps_per_episode": int(args.max_steps),
        "seed": int(args.seed),
        "control_dt_seconds": float(args.control_dt),
        "keep_dr": bool(args.keep_dr),
        "hole_yaw_range_deg": float(args.hole_yaw_range_deg),
        "tolerances_mm": tolerances,
    }
    for method in methods:
        aggregate[method] = per_method[method]

    summary_path = output_dir / "sim_results.json"
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\n=> wrote {summary_path}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _run_one_eval(args)
    if args.aggregate_only:
        return _aggregate_from_raw_jsons(args)
    rc = _orchestrate(args)
    if rc == 0:
        # Always re-aggregate at the end so sim_results.json carries both raw
        # and filtered series (orchestrator's inline write is raw-only).
        _aggregate_from_raw_jsons(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
