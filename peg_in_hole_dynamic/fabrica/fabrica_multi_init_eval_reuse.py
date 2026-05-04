"""Persistent-worker batch evaluation of Fabrica policies on multi-init scenes.

Keeps one IsaacGym subprocess alive per part and reuses it across all
policies, goal modes, and scenes -- mutating `trajectory_states` /
`object_init_state` between episodes instead of spawning a fresh
subprocess per task. ~10x faster than fabrica_multi_init_eval_all.py on
the same workload because IsaacGym startup + env creation is amortized.

Usage:
    python fabrica/fabrica_multi_init_eval_reuse.py \\
        --policies-dir hardware_rollouts/Apr16_experiments \\
        --goal-modes dense final_only pre_insert_and_final \\
        --split both --num-scenes-train 10 --num-scenes-val 10 \\
        --num-repeats 1 --parallel 5
"""

import argparse
import json
import multiprocessing
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from fabrica.fabrica_eval_all import (
    ASSETS_DIR,
    BASE_OVERRIDES,
    N_ACT,
    OBS_DIM,
    REPO_ROOT,
    TABLE_Z,
    _create_fabrica_env,
    _table_urdf_rel,
    parse_overrides,
)
from fabrica.fabrica_multi_init_eval import (
    GOAL_MODES,
    SPLIT_FILES,
    _load_scenes,
    _scene_start_and_goals,
    apply_goal_mode,
)


ALL_ASSEMBLIES_DEFAULT = ["beam"]

# Noise keys users might toggle via --override to get meaningful K>1 variance.
_NOISE_OVERRIDE_KEYS = (
    "task.env.resetPositionNoiseX",
    "task.env.resetPositionNoiseY",
    "task.env.resetPositionNoiseZ",
    "task.env.randomizeObjectRotation",
    "task.env.resetDofPosRandomIntervalFingers",
    "task.env.resetDofPosRandomIntervalArm",
    "task.env.resetDofVelRandomInterval",
)


# ===================================================================
# Worker subprocess
# ===================================================================

def persistent_eval_worker(conn, assembly, part_id, collision_method,
                           scenes_by_split, extra_overrides, seed_config_path):
    """Long-lived worker process.

    Owns one env for one part; handles pipe commands:
      ("load_policy", config_path, checkpoint_path)
         -> ("policy_ready",) | ("error", tb)
      ("run", split, scene_idx, mode)
         -> ("ok", goal_pct, steps, retract_ok, wall_s) | ("error", tb)
      ("quit",)
         -> exits the loop
    """
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
        import torch
        from deployment.rl_player import RlPlayer
        import fabrica.objects  # noqa: F401

        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj_suffix = {"vhacd": "", "sdf": "_sdf", "coacd": "_coacd"}[collision_method]
        object_name = f"{assembly}_{part_id}{obj_suffix}"
        table_urdf = _table_urdf_rel(assembly, part_id, collision_method)

        # Seed env with a representative scene's start/goals from the first
        # available split -- everything is overwritten on each "run" command.
        split0 = next(iter(scenes_by_split))
        seed_start, seed_goals = _scene_start_and_goals(
            scenes_by_split[split0], 0, part_id
        )

        env = _create_fabrica_env(
            config_path=str(seed_config_path), headless=True, device=device,
            overrides={
                **BASE_OVERRIDES,
                "task.env.objectName": object_name,
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": seed_goals,
                "task.env.asset.table": table_urdf,
                "task.env.tableResetZ": TABLE_Z,
                "task.env.objectStartPose": seed_start,
                # Headless batch eval -- no viewer, no viser viz server,
                # no mid-training video captures.
                "task.env.viserViz": False,
                "task.env.capture_video": False,
                **({"task.env.useSDF": True} if collision_method == "sdf" else {}),
                **(extra_overrides or {}),
            },
        )
        zero_vel = torch.zeros(6, device=device, dtype=torch.float32)

        conn.send(("env_ready",))

        current_policy = None

        while True:
            msg = conn.recv()  # blocks until parent sends a command
            tag = msg[0]

            if tag == "quit":
                break

            if tag == "load_policy":
                _, config_path, checkpoint_path = msg
                # Drop the previous policy's GPU buffers before allocating a new one.
                if current_policy is not None:
                    del current_policy
                    torch.cuda.empty_cache()
                    current_policy = None
                try:
                    ckpt = torch.load(str(checkpoint_path), map_location=device)
                    env_state = (ckpt[0].get("env_state", None)
                                 if 0 in ckpt else ckpt.get("env_state", None))
                    env.set_env_state(env_state)
                    current_policy = RlPlayer(
                        OBS_DIM, N_ACT, str(config_path), str(checkpoint_path),
                        device, env.num_envs,
                    )
                    conn.send(("policy_ready",))
                except Exception:
                    conn.send(("error", traceback.format_exc()))
                continue

            if tag == "run":
                _, split, scene_idx, mode = msg
                t0 = time.time()
                try:
                    if current_policy is None:
                        raise RuntimeError("run received before load_policy")
                    scenes = scenes_by_split[split]
                    start_pose, goals_full = _scene_start_and_goals(
                        scenes, scene_idx, part_id
                    )
                    goals = apply_goal_mode(goals_full, mode)

                    # Mutate env trajectory + start pose in place.
                    env.trajectory_states = torch.tensor(
                        goals, device=device, dtype=torch.float32,
                    )
                    env.max_consecutive_successes = len(goals)
                    env.cfg["env"]["fixedGoalStates"] = goals
                    env.cfg["env"]["objectStartPose"] = start_pose
                    env.object_init_state[0, :7] = torch.tensor(
                        start_pose, device=device, dtype=torch.float32,
                    )
                    env.object_init_state[0, 7:13] = zero_vel

                    env.reset_idx(torch.tensor([0], device=device))
                    # After reset_idx the obs_buf holds post-reset obs.
                    obs = env.obs_buf
                    current_policy.reset()

                    step = 0
                    done = False
                    retract_ok = False
                    while not done:
                        action = current_policy.get_normalized_action(
                            obs, deterministic_actions=True,
                        )
                        obs_dict, _, done_tensor, _ = env.step(action)
                        obs = obs_dict["obs"]
                        done = bool(done_tensor[0].item())
                        step += 1
                        if env.extras.get("retract_success_ratio", 0) > 0.5:
                            retract_ok = True

                    goal_pct = (
                        100.0 * int(env.successes[0].item())
                        / max(env.max_consecutive_successes, 1)
                    )
                    wall_s = round(time.time() - t0, 2)
                    conn.send(("ok", float(goal_pct), int(step),
                               bool(retract_ok), wall_s))
                except Exception:
                    conn.send(("error", traceback.format_exc()))
                continue

            conn.send(("error", f"unknown command tag: {tag!r}"))

    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ===================================================================
# Per-part coordinator (runs in the main process, one thread per part)
# ===================================================================

def _drive_part(assembly: str, part_id: str, policies: Dict[str, Tuple[Path, Path]],
                goal_modes: List[str], splits: List[str],
                num_scenes_per_split: Dict[str, int], num_repeats: int,
                scenes_by_split: dict, collision_method: str,
                extra_overrides: dict, per_cmd_timeout: int,
                log_prefix: str) -> dict:
    """Spawn one worker for this part; stream all (policy, mode, split,
    scene, repeat) commands to it; collect replies.

    Returns nested dict:
      part_results[policy_name][mode][split][scene_idx] = [ep_dict, ...]
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    seed_config_path = next(iter(policies.values()))[0]

    proc = ctx.Process(
        target=persistent_eval_worker,
        args=(child_conn, assembly, part_id, collision_method,
              scenes_by_split, extra_overrides, str(seed_config_path)),
        daemon=True,
    )
    proc.start()
    child_conn.close()

    part_results: Dict[str, dict] = {}

    def _recv(timeout):
        if not parent_conn.poll(timeout):
            raise TimeoutError(f"worker {log_prefix} silent for {timeout}s")
        return parent_conn.recv()

    try:
        msg = _recv(300)  # env create can take a while
        if msg[0] != "env_ready":
            raise RuntimeError(f"env_ready expected, got: {msg!r}")
        print(f"[{log_prefix}] env ready")

        total_cmds_per_policy = (
            len(goal_modes) * sum(num_scenes_per_split[s] for s in splits)
            * num_repeats
        )

        for p_idx, (pname, (cfg, ckpt)) in enumerate(policies.items(), 1):
            parent_conn.send(("load_policy", str(cfg), str(ckpt)))
            msg = _recv(120)
            if msg[0] != "policy_ready":
                raise RuntimeError(
                    f"policy_ready expected for {pname}, got: {msg!r}"
                )
            print(f"[{log_prefix}] policy {p_idx}/{len(policies)}: {pname}")
            part_results[pname] = {}

            cmd_i = 0
            for mode in goal_modes:
                part_results[pname].setdefault(mode, {})
                for split in splits:
                    part_results[pname][mode].setdefault(split, {})
                    for scene_idx in range(num_scenes_per_split[split]):
                        part_results[pname][mode][split][scene_idx] = []
                        for rep in range(num_repeats):
                            parent_conn.send(
                                ("run", split, scene_idx, mode)
                            )
                            reply = _recv(per_cmd_timeout)
                            cmd_i += 1
                            if reply[0] == "ok":
                                _, goal_pct, steps, retract_ok, wall = reply
                                part_results[pname][mode][split][scene_idx].append({
                                    "goal_pct": goal_pct,
                                    "steps": steps,
                                    "retract_ok": retract_ok,
                                    "wall_s": wall,
                                    "repeat": rep,
                                    "error": None,
                                })
                                if cmd_i == 1 or cmd_i % 20 == 0:
                                    print(f"[{log_prefix}] {pname} "
                                          f"[{cmd_i}/{total_cmds_per_policy}] "
                                          f"{mode}/{split}/scene{scene_idx}/r{rep} "
                                          f"goal={goal_pct:.0f}% "
                                          f"retract={'OK' if retract_ok else 'FAIL'} "
                                          f"({wall:.1f}s)")
                            else:
                                err = reply[1] if len(reply) > 1 else "unknown"
                                part_results[pname][mode][split][scene_idx].append({
                                    "goal_pct": 0.0, "steps": 0,
                                    "retract_ok": False, "wall_s": 0.0,
                                    "repeat": rep, "error": str(err)[:500],
                                })
                                print(f"[{log_prefix}] {pname} "
                                      f"ERROR {mode}/{split}/scene{scene_idx}/"
                                      f"r{rep}: {str(err).strip().splitlines()[-1][:100]}")
    finally:
        try:
            parent_conn.send(("quit",))
        except Exception:
            pass
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        parent_conn.close()

    return part_results


# ===================================================================
# Summary
# ===================================================================

def compute_summary(results: dict) -> dict:
    """results[policy][mode][split][assembly][part][scene_idx] = list[ep_dict].

    Returns summary[policy][mode][split] with:
      avg / std / min / max of goal_pct across all non-error episodes,
      plus retract_rate, num_episodes, num_errors, and per_part breakdown.
    """
    summary: dict = {}
    for policy, by_mode in results.items():
        summary[policy] = {}
        for mode, by_split in by_mode.items():
            summary[policy][mode] = {}
            for split, by_asm in by_split.items():
                all_goals: List[float] = []
                all_retracts: List[bool] = []
                num_errors = 0
                per_part: Dict[str, Dict[str, float]] = {}
                for asm, by_part in by_asm.items():
                    for part, by_scene in by_part.items():
                        part_goals: List[float] = []
                        part_retracts: List[bool] = []
                        for _scene, eps in by_scene.items():
                            for ep in eps:
                                if ep["error"]:
                                    num_errors += 1
                                    continue
                                all_goals.append(ep["goal_pct"])
                                all_retracts.append(ep["retract_ok"])
                                part_goals.append(ep["goal_pct"])
                                part_retracts.append(ep["retract_ok"])
                        if part_goals:
                            per_part[part] = {
                                "avg_goal_pct": round(float(np.mean(part_goals)), 2),
                                "std_goal_pct": round(float(np.std(part_goals)), 2),
                                "retract_rate": round(
                                    100 * sum(part_retracts) / len(part_retracts), 1
                                ),
                                "n": len(part_goals),
                            }
                n = len(all_goals)
                summary[policy][mode][split] = {
                    "avg_goal_pct": round(float(np.mean(all_goals)), 2) if n else 0.0,
                    "std_goal_pct": round(float(np.std(all_goals)), 2) if n else 0.0,
                    "min_goal_pct": round(float(np.min(all_goals)), 2) if n else 0.0,
                    "max_goal_pct": round(float(np.max(all_goals)), 2) if n else 0.0,
                    "retract_rate": round(100 * sum(all_retracts) / n, 1) if n else 0.0,
                    "num_episodes": n,
                    "num_errors": num_errors,
                    "per_part": per_part,
                }
    return summary


def print_summary_table(summary: dict):
    for policy, by_mode in summary.items():
        print(f"\n==== {policy} ====")
        print(f"  {'mode':<25s} {'split':<6s} "
              f"{'avg':>8s} {'std':>6s} {'min':>6s} {'max':>6s} "
              f"{'retract':>8s} {'n':>4s} {'err':>4s}")
        for mode, by_split in by_mode.items():
            for split, s in by_split.items():
                print(f"  {mode:<25s} {split:<6s} "
                      f"{s['avg_goal_pct']:>7.1f}% "
                      f"{s['std_goal_pct']:>5.1f}  "
                      f"{s['min_goal_pct']:>5.1f}  "
                      f"{s['max_goal_pct']:>5.1f}  "
                      f"{s['retract_rate']:>7.1f}% "
                      f"{s['num_episodes']:>4d} "
                      f"{s['num_errors']:>4d}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Persistent-worker batch eval (subprocess reuse)",
    )
    parser.add_argument("--collision", choices=["vhacd", "coacd", "sdf"],
                        default="coacd")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: "
                             "fabrica/eval_outputs/reuse_<timestamp>.json)")
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory of policy subfolders (each with "
                             "config.yaml + model.pth). Takes precedence "
                             "over --policies when both are given -- "
                             "--policies filters within the scanned dir.")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Policy names (filters --policies-dir).")
    parser.add_argument("--assemblies", nargs="+", default=None,
                        help=f"Default: {ALL_ASSEMBLIES_DEFAULT}")
    parser.add_argument("--parts", nargs="+", default=None,
                        help="Subset of part ids per assembly's assembly_order.")
    parser.add_argument("--goal-modes", nargs="+", choices=GOAL_MODES,
                        default=["dense"])
    parser.add_argument("--split", choices=["train", "val", "both"], default="val")
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--num-scenes-train", type=int, default=None)
    parser.add_argument("--num-scenes-val", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=1,
                        help="Repeats per (scene, part, mode). K>1 is only "
                             "meaningful with reset-noise overrides.")
    parser.add_argument("--parallel", type=int, default=5,
                        help="Simultaneous part workers (default 5).")
    parser.add_argument("--per-cmd-timeout", type=int, default=300,
                        help="Seconds to wait for a single episode reply.")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"), help="Extra config overrides")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]

    # ---- Resolve policies ----
    if args.policies_dir is None:
        print("ERROR: --policies-dir is required (reuse runner does not "
              "consult the static POLICIES registry).")
        sys.exit(1)
    pdir = Path(args.policies_dir)
    if not pdir.is_absolute():
        pdir = REPO_ROOT / pdir
    if not pdir.exists():
        print(f"ERROR: --policies-dir not found: {pdir}")
        sys.exit(1)
    name_filter = set(args.policies) if args.policies else None
    selected_policies: Dict[str, Tuple[Path, Path]] = {}
    for sub in sorted(pdir.iterdir()):
        if name_filter is not None and sub.name not in name_filter:
            continue
        cfg = sub / "config.yaml"
        ckpt = sub / "model.pth"
        if cfg.exists() and ckpt.exists():
            selected_policies[sub.name] = (cfg, ckpt)
    if not selected_policies:
        suffix = f" matching {sorted(name_filter)}" if name_filter else ""
        print(f"ERROR: No policy subfolders in {pdir}{suffix}")
        sys.exit(1)

    # ---- Assemblies + parts + scene counts ----
    assemblies = args.assemblies or ALL_ASSEMBLIES_DEFAULT
    split_num_scenes = {
        "train": args.num_scenes_train if args.num_scenes_train is not None else args.num_scenes,
        "val":   args.num_scenes_val   if args.num_scenes_val   is not None else args.num_scenes,
    }

    # Preload scene dicts per (assembly, split). Also clamp per-split scene
    # counts to the smallest availability across requested assemblies so no
    # worker indexes past its scenes.npz.
    assembly_scenes: Dict[str, Dict[str, dict]] = {}
    effective_num_scenes = dict(split_num_scenes)
    for asm in assemblies:
        per_split = {}
        for split in splits:
            scenes = _load_scenes(asm, split)
            if scenes is None:
                print(f"WARNING: no {SPLIT_FILES[split]} for '{asm}', skipping.")
                continue
            avail = scenes["start_poses"].shape[0]
            if avail < effective_num_scenes[split]:
                print(f"WARNING: '{asm}' {split} has only {avail} scenes, "
                      f"clamping split to {avail}.")
                effective_num_scenes[split] = avail
            per_split[split] = scenes
        if per_split:
            assembly_scenes[asm] = per_split
    split_num_scenes = effective_num_scenes

    if not assembly_scenes:
        print("ERROR: No scenes.npz / scenes_val.npz found for requested "
              "assemblies. Exiting.")
        sys.exit(1)

    # ---- Parts per assembly (from assembly_order, optionally filtered) ----
    parts_by_asm: Dict[str, List[str]] = {}
    for asm, per_split in assembly_scenes.items():
        order = next(iter(per_split.values()))["assembly_order"]
        use_parts = args.parts or order
        parts_by_asm[asm] = [p for p in use_parts if p in order]

    # ---- Warn about repeats vs noise ----
    extra_overrides = parse_overrides(args.override)
    noise_configured = any(
        k in extra_overrides
        and (extra_overrides[k] not in (0, 0.0, False, "0", "False"))
        for k in _NOISE_OVERRIDE_KEYS
    )
    if args.num_repeats > 1 and not noise_configured:
        print("WARNING: --num-repeats > 1 without any reset-noise --override "
              "= bit-identical repeats (deterministic policy + no noise). "
              "Pass e.g. --override task.env.resetPositionNoiseX 0.003 for "
              "real variance.")

    # ---- Build work units: (assembly, part) ----
    units: List[Tuple[str, str]] = []
    for asm, parts in parts_by_asm.items():
        for part in parts:
            units.append((asm, part))

    total_episodes = (
        len(selected_policies) * len(args.goal_modes)
        * sum(split_num_scenes[s] for s in splits)
        * len(units) * args.num_repeats
    )
    print(f"\n{'='*60}")
    print(f"Persistent-worker batch eval")
    print(f"  Policies: {len(selected_policies)} ({', '.join(selected_policies)})")
    print(f"  Assemblies/parts: {units}")
    print(f"  Modes: {args.goal_modes}")
    print(f"  Splits: {splits}  scenes: "
          f"{ {s: split_num_scenes[s] for s in splits} }")
    print(f"  Repeats: {args.num_repeats}")
    print(f"  Parallel part-workers: {args.parallel}")
    print(f"  Total episodes: {total_episodes}")
    print(f"{'='*60}\n")

    # ---- Dispatch ----
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t_start = time.time()
    results: dict = {pname: {mode: {split: {asm: {} for asm in parts_by_asm}
                                    for split in splits}
                             for mode in args.goal_modes}
                     for pname in selected_policies}

    # Each part has its own worker subprocess. We run up to args.parallel
    # parts concurrently via a ThreadPool (NOT ProcessPool) so the work of
    # _drive_part (spawning worker + pipe I/O) happens in this process.
    # The actual IsaacGym work is in the spawned workers.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {}
        for asm, part in units:
            scenes_by_split = assembly_scenes[asm]
            log_prefix = f"{asm}/{part}"
            fut = pool.submit(
                _drive_part, asm, part, selected_policies,
                args.goal_modes, splits, split_num_scenes, args.num_repeats,
                scenes_by_split, args.collision, extra_overrides,
                args.per_cmd_timeout, log_prefix,
            )
            futures[fut] = (asm, part)

        for fut in as_completed(futures):
            asm, part = futures[fut]
            try:
                part_results = fut.result()
            except Exception:
                print(f"[{asm}/{part}] WORKER FAILED:\n{traceback.format_exc()}")
                continue
            # part_results[pname][mode][split][scene_idx] = [ep_dict,...]
            for pname, by_mode in part_results.items():
                for mode, by_split in by_mode.items():
                    for split, by_scene in by_split.items():
                        results[pname][mode][split][asm].setdefault(part, {}).update(by_scene)
            print(f"[{asm}/{part}] done")

    total_time = time.time() - t_start

    summary = compute_summary(results)
    print_summary_table(summary)
    print(f"\nTotal wall time: {total_time/60:.1f} minutes "
          f"({total_episodes} episodes)")

    # ---- Save ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = args.output or str(
        REPO_ROOT / "fabrica" / "eval_outputs"
        / f"reuse_{timestamp}.json"
    )
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "collision_method": args.collision,
            "splits": splits,
            "num_scenes_per_split": split_num_scenes,
            "num_repeats": args.num_repeats,
            "goal_modes": args.goal_modes,
            "assemblies": list(parts_by_asm.keys()),
            "parts_by_assembly": parts_by_asm,
            "num_parallel_workers": args.parallel,
            "total_episodes": total_episodes,
            "total_wall_time_s": round(total_time, 1),
            "policies": {
                name: {"config": str(cfg), "checkpoint": str(ckpt)}
                for name, (cfg, ckpt) in selected_policies.items()
            },
            "overrides": extra_overrides,
        },
        "results": results,
        "summary": summary,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
