"""Batch evaluation of Fabrica policies on multi-init-state scenes.

Runs policies on the first N scenes from either the train (scenes.npz) or
val (scenes_val.npz) split, for each assembly/part, and averages metrics.
Each (policy, split, scene, part) is one headless subprocess.

Policies come from the registry in ``fabrica_eval_all.POLICIES``.

Usage:
    # Default: all registered policies, beam only, val split, 20 scenes
    python fabrica/fabrica_multi_init_eval_all.py

    # Specific policies and 10 scenes on the train split
    python fabrica/fabrica_multi_init_eval_all.py \\
        --policies beam_multi_init_dr beam_multi_init_no_dr \\
        --split train --num-scenes 10

    # Both splits
    python fabrica/fabrica_multi_init_eval_all.py --split both
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

from fabrica.fabrica_eval_all import (
    ASSETS_DIR,
    BASE_OVERRIDES,
    CONTROL_DT,
    N_ACT,
    OBS_DIM,
    POLICIES,
    REPO_ROOT,
    TABLE_Z,
    _create_fabrica_env,
    _sim_reset,
    _table_urdf_rel,
    parse_overrides,
)
from fabrica.fabrica_multi_init_eval import (
    GOAL_MODES,
    _GOAL_MODE_CAMEL,
)

# scenes.npz / scenes_val.npz filenames per split
SPLIT_FILES = {
    "train": "scenes.npz",
    "val": "scenes_val.npz",
}

ALL_ASSEMBLIES_DEFAULT = ["beam"]  # only beam has scenes.npz today


# ===================================================================
# Scene data loading (main process)
# ===================================================================

def _load_scenes(assembly: str, split: str) -> Optional[dict]:
    import numpy as np
    fname = SPLIT_FILES[split]
    path = ASSETS_DIR / assembly / fname
    if not path.exists():
        return None
    data = np.load(str(path))
    order_path = ASSETS_DIR / assembly / "assembly_order.json"
    assembly_order = json.loads(order_path.read_text())["steps"]
    return {
        "start_poses": data["start_poses"],
        "goals": data["goals"],
        "traj_lengths": data["traj_lengths"],
        "assembly_order": assembly_order,
    }


def _scene_start_and_goals(scenes: dict, scene_idx: int, part_id: str
                           ) -> Tuple[List[float], List[List[float]]]:
    order = scenes["assembly_order"]
    p_idx = order.index(part_id)
    start = scenes["start_poses"][scene_idx, p_idx].tolist()
    L = int(scenes["traj_lengths"][scene_idx, p_idx])
    goals = scenes["goals"][scene_idx, p_idx, :L].tolist()
    return start, goals


def build_scene_tasks(assemblies: List[str], split: str, num_scenes: int,
                      part_ids: Optional[List[str]] = None
                      ) -> List[Tuple[str, str, int, str]]:
    """Return list of (assembly, split, scene_idx, part_id)."""
    tasks = []
    for assembly in assemblies:
        scenes = _load_scenes(assembly, split)
        if scenes is None:
            print(f"WARNING: no {SPLIT_FILES[split]} for '{assembly}', skipping.")
            continue
        available = scenes["start_poses"].shape[0]
        N = min(num_scenes, available)
        if N < num_scenes:
            print(f"WARNING: '{assembly}' {split} has only {available} scenes, "
                  f"using all of them (requested {num_scenes}).")
        order = scenes["assembly_order"]
        use_parts = part_ids or order
        for scene_idx in range(N):
            for pid in use_parts:
                if pid not in order:
                    continue
                tasks.append((assembly, split, scene_idx, pid))
    return tasks


# ===================================================================
# Subprocess entry: env creation + one episode
# ===================================================================

def _eval_scene_task(conn, assembly, part_id, config_path, checkpoint_path,
                     collision_method, start_pose, goals, extra_overrides):
    """Run one headless episode for a (scene, part). Sends ("ok", goal_pct,
    steps, retract_ok) or ("error", traceback).
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

        env = _create_fabrica_env(
            config_path=str(config_path), headless=True, device=device,
            overrides={
                **BASE_OVERRIDES,
                "task.env.objectName": object_name,
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": goals,
                "task.env.asset.table": table_urdf,
                "task.env.tableResetZ": TABLE_Z,
                "task.env.objectStartPose": start_pose,
                **({"task.env.useSDF": True} if collision_method == "sdf" else {}),
                **(extra_overrides or {}),
            },
        )

        ckpt = torch.load(str(checkpoint_path), map_location=device)
        env_state = ckpt[0].get("env_state", None) if 0 in ckpt else ckpt.get("env_state", None)
        env.set_env_state(env_state)

        policy = RlPlayer(OBS_DIM, N_ACT, str(config_path), str(checkpoint_path),
                          device, env.num_envs)

        policy.reset()
        obs = _sim_reset(env, device)
        step = 0
        done = False
        retract_ok = False

        while not done:
            action = policy.get_normalized_action(obs, deterministic_actions=True)
            obs_dict, _, done_tensor, _ = env.step(action)
            obs = obs_dict["obs"]
            done = done_tensor[0].item()
            step += 1
            if env.extras.get("retract_success_ratio", 0) > 0.5:
                retract_ok = True

        goal_pct = 100.0 * int(env.successes[0].item()) / env.max_consecutive_successes
        conn.send(("ok", goal_pct, step, retract_ok))

    except Exception:
        conn.send(("error", traceback.format_exc()))

    conn.close()


def run_scene_task(assembly: str, part_id: str, scene_idx: int, split: str,
                   config_path, checkpoint_path, collision_method: str,
                   start_pose, goals, extra_overrides: dict,
                   timeout: int = 300) -> dict:
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(
        target=_eval_scene_task,
        args=(child_conn, assembly, part_id, str(config_path), str(checkpoint_path),
              collision_method, start_pose, goals, extra_overrides),
        daemon=True,
    )
    proc.start()
    child_conn.close()

    t0 = time.time()
    result = {"goal_pct": 0.0, "steps": 0, "retract_ok": False, "error": None}

    if parent_conn.poll(timeout):
        msg = parent_conn.recv()
        if msg[0] == "ok":
            result["goal_pct"] = msg[1]
            result["steps"] = msg[2]
            result["retract_ok"] = msg[3]
        elif msg[0] == "error":
            result["error"] = msg[1]
    else:
        result["error"] = f"Timeout after {timeout}s"

    result["wall_time_s"] = round(time.time() - t0, 1)

    proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    parent_conn.close()
    return result


# ===================================================================
# Orchestration
# ===================================================================

def run_all_evaluations(policies: Dict[str, Tuple[Path, Path]],
                        tasks: List[Tuple[str, str, int, str]],
                        scenes_by_assembly_split: Dict[Tuple[str, str], dict],
                        collision_method: str,
                        extra_overrides: dict,
                        timeout: int) -> dict:
    """Run all policy × (assembly, split, scene, part) evaluations sequentially.

    Returns nested dict:
      results[policy][split][assembly][part_id][scene_idx] = {goal_pct, steps, retract_ok, error}
    """
    total = len(policies) * len(tasks)
    print(f"\n{'='*60}")
    print(f"Fabrica Multi-Init Batch Evaluation")
    print(f"  Policies: {len(policies)} ({', '.join(policies.keys())})")
    print(f"  Tasks: {len(tasks)} (assembly × split × scene × part)")
    print(f"  Total evaluations: {total}")
    print(f"  Collision: {collision_method}")
    print(f"{'='*60}\n")

    results: dict = {}
    completed = 0

    for policy_name, (config_path, checkpoint_path) in policies.items():
        print(f"\n--- Policy: {policy_name} ---")
        results[policy_name] = {}

        for assembly, split, scene_idx, part_id in tasks:
            completed += 1
            tag = f"[{completed}/{total}]"
            print(f"  {tag} {split}/{assembly}/scene{scene_idx}/{part_id} ... ",
                  end="", flush=True)

            scenes = scenes_by_assembly_split[(assembly, split)]
            start, goals = _scene_start_and_goals(scenes, scene_idx, part_id)

            result = run_scene_task(
                assembly, part_id, scene_idx, split,
                config_path, checkpoint_path, collision_method,
                start, goals, extra_overrides, timeout,
            )

            results[policy_name].setdefault(split, {}) \
                .setdefault(assembly, {}) \
                .setdefault(part_id, {})[scene_idx] = result

            if result["error"]:
                err_first_line = result["error"].strip().split("\n")[-1][:80]
                print(f"ERROR: {err_first_line}")
            else:
                retract_str = "OK" if result["retract_ok"] else "FAIL"
                print(f"{result['goal_pct']:.0f}% goals, "
                      f"{result['steps']/60:.1f}s, "
                      f"retract: {retract_str} "
                      f"({result['wall_time_s']:.0f}s wall)")

    return results


def compute_summary(results: dict) -> dict:
    """Aggregate into summary[policy][split] = {
         overall_avg_goal, overall_retract_rate, num_episodes, num_errors,
         per_part: {part_id: {avg_goal, retract_rate, n}},
         per_assembly: {assembly: {avg_goal, retract_rate, n}},
       }
    """
    summary = {}
    for policy_name, by_split in results.items():
        summary[policy_name] = {}
        for split, by_assembly in by_split.items():
            all_goals = []
            all_retracts = []
            num_errors = 0
            per_part_acc: Dict[str, List[Tuple[float, bool]]] = {}
            per_asm_acc: Dict[str, List[Tuple[float, bool]]] = {}

            for assembly, by_part in by_assembly.items():
                for part_id, by_scene in by_part.items():
                    for scene_idx, res in by_scene.items():
                        if res["error"]:
                            num_errors += 1
                            continue
                        all_goals.append(res["goal_pct"])
                        all_retracts.append(res["retract_ok"])
                        per_part_acc.setdefault(part_id, []).append(
                            (res["goal_pct"], res["retract_ok"])
                        )
                        per_asm_acc.setdefault(assembly, []).append(
                            (res["goal_pct"], res["retract_ok"])
                        )

            n = len(all_goals)
            per_part = {
                pid: {
                    "avg_goal_pct": round(sum(g for g, _ in vals) / len(vals), 1),
                    "retract_rate": round(100 * sum(r for _, r in vals) / len(vals), 1),
                    "n": len(vals),
                }
                for pid, vals in per_part_acc.items()
            }
            per_assembly = {
                asm: {
                    "avg_goal_pct": round(sum(g for g, _ in vals) / len(vals), 1),
                    "retract_rate": round(100 * sum(r for _, r in vals) / len(vals), 1),
                    "n": len(vals),
                }
                for asm, vals in per_asm_acc.items()
            }
            summary[policy_name][split] = {
                "avg_goal_pct": round(sum(all_goals) / n, 1) if n else 0.0,
                "retract_rate": round(100 * sum(all_retracts) / n, 1) if n else 0.0,
                "num_episodes": n,
                "num_errors": num_errors,
                "per_part": per_part,
                "per_assembly": per_assembly,
            }
    return summary


def print_summary_table(summary: dict, splits: List[str], policy_names: List[str]):
    print(f"\n{'='*78}")
    print("MULTI-INIT SUMMARY (avg per policy × split)")
    print(f"{'='*78}")
    header = f"{'Policy':<28}"
    for split in splits:
        header += f" | {split+' goal%':>10} | {split+' retr%':>10} | {split+' n':>5}"
    print(header)
    print("-" * len(header))
    for pname in policy_names:
        row = f"{pname:<28}"
        for split in splits:
            s = summary.get(pname, {}).get(split, {})
            g = s.get("avg_goal_pct", 0.0)
            r = s.get("retract_rate", 0.0)
            n = s.get("num_episodes", 0)
            row += f" | {g:>9.1f}% | {r:>9.1f}% | {n:>5}"
        print(row)

    # Per-part breakdown (one table per split)
    for split in splits:
        # collect part_ids seen across policies in this split
        part_ids = sorted({
            pid
            for pname in policy_names
            for pid in summary.get(pname, {}).get(split, {}).get("per_part", {}).keys()
        }, key=lambda x: int(x))
        if not part_ids:
            continue
        print(f"\n{'='*78}")
        print(f"PER-PART BREAKDOWN — split={split} (avg goal% [retract%])")
        print(f"{'='*78}")
        header = f"{'Policy':<28}"
        for pid in part_ids:
            header += f" | {'p'+pid:>13}"
        print(header)
        print("-" * len(header))
        for pname in policy_names:
            row = f"{pname:<28}"
            for pid in part_ids:
                per = summary.get(pname, {}).get(split, {}).get("per_part", {}).get(pid)
                if per:
                    cell = f"{per['avg_goal_pct']:5.1f}% [{per['retract_rate']:4.0f}%]"
                else:
                    cell = "--"
                row += f" | {cell:>13}"
            print(row)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch multi-init-state evaluation of Fabrica policies",
    )
    parser.add_argument("--collision", choices=["vhacd", "coacd", "sdf"],
                        default="coacd")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: fabrica/eval_outputs/"
                             "multi_init_eval_results_<timestamp>.json)")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Subset of registered policies (default: all multi-init policies)")
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory of policy subfolders, each containing "
                             "config.yaml + model.pth. Every subfolder becomes "
                             "one policy. Takes precedence over --policies.")
    parser.add_argument("--goal-modes", nargs="+", choices=GOAL_MODES,
                        default=["dense"],
                        help="One or more goal modes to evaluate. Each selected "
                             "policy is run once per mode; results use "
                             "composite names <policy>__<mode>.")
    parser.add_argument("--assemblies", nargs="+", default=None,
                        help="Subset of assemblies (default: beam)")
    parser.add_argument("--parts", nargs="+", default=None,
                        help="Subset of part ids (default: all parts in assembly_order)")
    parser.add_argument("--split", choices=["train", "val", "both"], default="val",
                        help="Which scenes file(s) to eval on (default: val)")
    parser.add_argument("--num-scenes", type=int, default=20,
                        help="Number of scenes per (assembly, split). "
                             "Clamped to scenes available in the npz.")
    parser.add_argument("--num-scenes-train", type=int, default=None,
                        help="Override --num-scenes for the train split.")
    parser.add_argument("--num-scenes-val", type=int, default=None,
                        help="Override --num-scenes for the val split.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-task timeout in seconds (default: 300)")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"), help="Extra config overrides")
    args = parser.parse_args()

    # Resolve splits
    splits = ["train", "val"] if args.split == "both" else [args.split]

    selected_policies: Dict[str, Tuple[Path, Path]] = {}

    if args.policies_dir is not None:
        pdir = Path(args.policies_dir)
        if not pdir.is_absolute():
            pdir = REPO_ROOT / pdir
        if not pdir.exists():
            print(f"ERROR: --policies-dir not found: {pdir}")
            sys.exit(1)
        name_filter = set(args.policies) if args.policies else None
        for sub in sorted(pdir.iterdir()):
            if name_filter is not None and sub.name not in name_filter:
                continue
            cfg = sub / "config.yaml"
            ckpt = sub / "model.pth"
            if cfg.exists() and ckpt.exists():
                selected_policies[sub.name] = (cfg, ckpt)
        if not selected_policies:
            suffix = f" matching {sorted(name_filter)}" if name_filter else ""
            print(f"ERROR: No policy subfolders in {pdir}{suffix} "
                  f"(each needs config.yaml + model.pth)")
            sys.exit(1)
    else:
        default_policy_subset = [
            "beam_multi_init_dr", "beam_multi_init_no_dr",
            "beam_table_rand_dr", "beam_table_rand_no_dr",
        ]
        selected_names = args.policies or default_policy_subset
        for name in selected_names:
            if name not in POLICIES:
                print(f"WARNING: Unknown policy '{name}', skipping. "
                      f"Available: {list(POLICIES.keys())}")
                continue
            config_path, checkpoint_path = POLICIES[name]
            if not config_path.exists():
                print(f"WARNING: Config not found for '{name}': {config_path}, skipping.")
                continue
            if not checkpoint_path.exists():
                print(f"WARNING: Checkpoint not found for '{name}': {checkpoint_path}, skipping.")
                continue
            selected_policies[name] = (config_path, checkpoint_path)

        if not selected_policies:
            print("ERROR: No valid policies found. Exiting.")
            sys.exit(1)

    assemblies = args.assemblies or ALL_ASSEMBLIES_DEFAULT

    # Per-split scene counts (CLI override → fallback to --num-scenes)
    split_num_scenes = {
        "train": args.num_scenes_train if args.num_scenes_train is not None else args.num_scenes,
        "val": args.num_scenes_val if args.num_scenes_val is not None else args.num_scenes,
    }

    # Build task list, and preload scenes for main-process reuse
    tasks: List[Tuple[str, str, int, str]] = []
    scenes_by_assembly_split: Dict[Tuple[str, str], dict] = {}
    for split in splits:
        for asm in assemblies:
            scenes = _load_scenes(asm, split)
            if scenes is None:
                continue
            scenes_by_assembly_split[(asm, split)] = scenes
        tasks += build_scene_tasks(assemblies, split, split_num_scenes[split], args.parts)

    if not tasks:
        print("ERROR: No scene tasks found. Did you generate scenes.npz / "
              "scenes_val.npz? Exiting.")
        sys.exit(1)

    base_overrides = parse_overrides(args.override)

    # Loop over goal modes. For each mode, run every policy under a
    # composite name "<policy>__<mode>" so the downstream summary /
    # printer treat (policy, mode) pairs as independent rows.
    results: dict = {}
    composite_policies: Dict[str, Tuple[Path, Path]] = {}
    t_start = time.time()
    for mode in args.goal_modes:
        mode_overrides = dict(base_overrides)
        mode_overrides["task.env.goalMode"] = _GOAL_MODE_CAMEL[mode]

        mode_policies = {
            f"{name}__{mode}": paths
            for name, paths in selected_policies.items()
        }
        composite_policies.update(mode_policies)

        print(f"\n########  Goal mode: {mode}  ########")
        mode_results = run_all_evaluations(
            mode_policies, tasks, scenes_by_assembly_split,
            args.collision, mode_overrides, args.timeout,
        )
        results.update(mode_results)

    total_time = time.time() - t_start

    summary = compute_summary(results)

    policy_names = list(composite_policies.keys())
    print_summary_table(summary, splits, policy_names)
    print(f"\nTotal wall time: {total_time/60:.1f} minutes")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = args.output or str(
        REPO_ROOT / "fabrica" / "eval_outputs"
        / f"multi_init_eval_results_{timestamp}.json"
    )

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "collision_method": args.collision,
            "splits": splits,
            "num_scenes_requested": args.num_scenes,
            "num_scenes_per_split": {s: split_num_scenes[s] for s in splits},
            "assemblies": assemblies,
            "parts": args.parts,
            "num_tasks": len(tasks),
            "total_wall_time_s": round(total_time, 1),
            "goal_modes": args.goal_modes,
            "policies": {
                name: {"config": str(cfg), "checkpoint": str(ckpt)}
                for name, (cfg, ckpt) in selected_policies.items()
            },
            "composite_policies": {
                name: {"config": str(cfg), "checkpoint": str(ckpt)}
                for name, (cfg, ckpt) in composite_policies.items()
            },
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
