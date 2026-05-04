"""Batch evaluation of Fabrica policies across all assembly tasks.

Runs 3 policies (pretrained, beam-finetuned, all-assemblies-finetuned) on all 38
insertion tasks headlessly. Each task spawns a fresh subprocess with its own
IsaacGym environment.

Usage:
    python fabrica/fabrica_eval_all.py
    python fabrica/fabrica_eval_all.py --policies pretrained --assemblies beam
    python fabrica/fabrica_eval_all.py --collision sdf --timeout 600
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

# ===================================================================
# Constants (lightweight -- no isaacgym imports)
# ===================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_Z = 0.38
N_ACT = 29
OBS_DIM = 140
CONTROL_DT = 1.0 / 60.0

ALL_ASSEMBLIES = [
    "beam", "car", "cooling_manifold", "duct",
    "gamepad", "plumbers_block", "stool_circular",
]

ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

BASE_OVERRIDES = {
    "task.env.resetPositionNoiseX": 0.0,
    "task.env.resetPositionNoiseY": 0.0,
    "task.env.resetPositionNoiseZ": 0.0,
    "task.env.randomizeObjectRotation": False,
    "task.env.resetDofPosRandomIntervalFingers": 0.0,
    "task.env.resetDofPosRandomIntervalArm": 0.0,
    "task.env.resetDofVelRandomInterval": 0.0,
    "task.env.tableResetZRange": 0.0,
    "task.env.numEnvs": 1,
    "task.env.envSpacing": 0.4,
    "task.env.capture_video": False,
    "task.env.useActionDelay": False,
    "task.env.useObsDelay": False,
    "task.env.useObjectStateDelayNoise": False,
    "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
    "task.env.resetWhenDropped": False,
    "task.env.armMovingAverage": 0.1,
    "task.env.evalSuccessTolerance": 0.01,
    "task.env.forceConsecutiveNearGoalSteps": True,
    "task.env.fixedSizeKeypointReward": True,
    "task.env.useFixedInitObjectPose": True,
    "task.env.startArmHigher": True,
    "task.env.forceScale": 0.0,
    "task.env.torqueScale": 0.0,
    "task.env.linVelImpulseScale": 0.0,
    "task.env.angVelImpulseScale": 0.0,
    "task.env.forceOnlyWhenLifted": True,
    "task.env.torqueOnlyWhenLifted": True,
    "task.env.linVelImpulseOnlyWhenLifted": True,
    "task.env.angVelImpulseOnlyWhenLifted": True,
    "task.env.forceProbRange": [0.0001, 0.0001],
    "task.env.torqueProbRange": [0.0001, 0.0001],
    "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
    "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
}

# ===================================================================
# Policy Registry
# ===================================================================

POLICIES = {
    "pretrained": (
        REPO_ROOT / "pretrained_policy" / "config.yaml",
        REPO_ROOT / "pretrained_policy" / "model.pth",
    ),
    "beam_policy": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_phase4_unified"
        / "all_beam_parts_no_curriculum_retracttol0.005_2026-03-19_19-02-32"
        / "runs" / "00_all_beam_parts_no_curriculum_retracttol0.005_2026-03-19_19-02-32"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_phase4_unified"
        / "all_beam_parts_no_curriculum_retracttol0.005_2026-03-19_19-02-32"
        / "runs" / "00_all_beam_parts_no_curriculum_retracttol0.005_2026-03-19_19-02-32"
        / "nn" / "last_00_all_beam_parts_no_curriculum_retracttol0.005_2026-03-19_19-02-32_ep_243000_rew_13263.113.pth",
    ),
    "all_assemblies_policy": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_phase5_all_assemblies"
        / "all_6_assemblies_v2_2gpu_2026-03-23_21-10-45"
        / "runs" / "00_all_6_assemblies_v2_2gpu_2026-03-23_21-10-45"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_phase5_all_assemblies"
        / "all_6_assemblies_v2_2gpu_2026-03-23_21-10-45"
        / "runs" / "00_all_6_assemblies_v2_2gpu_2026-03-23_21-10-45"
        / "nn" / "last_00_all_6_assemblies_v2_2gpu_2026-03-23_21-10-45_ep_243000_rew_12387.344.pth",
    ),
    "beam_with_dr": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation"
        / "beam_with_dr_2026-04-08_17-22-46"
        / "runs" / "00_beam_with_dr_2026-04-08_17-22-46"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation"
        / "beam_with_dr_2026-04-08_17-22-46"
        / "runs" / "00_beam_with_dr_2026-04-08_17-22-46"
        / "nn" / "last_00_beam_with_dr_2026-04-08_17-22-46_ep_243000_rew_11727.92.pth",
    ),
    "beam_no_dr": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation"
        / "beam_no_dr_2026-04-08_17-22-21"
        / "runs" / "00_beam_no_dr_2026-04-08_17-22-21"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation"
        / "beam_no_dr_2026-04-08_17-22-21"
        / "runs" / "00_beam_no_dr_2026-04-08_17-22-21"
        / "nn" / "last_00_beam_no_dr_2026-04-08_17-22-21_ep_243000_rew_12916.671.pth",
    ),
    "beam_with_dr_no_lift": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation_no_lift_reward"
        / "beam_with_dr_no_lift_2026-04-11_04-05-58"
        / "runs" / "00_beam_with_dr_no_lift_2026-04-11_04-05-58"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation_no_lift_reward"
        / "beam_with_dr_no_lift_2026-04-11_04-05-58"
        / "runs" / "00_beam_with_dr_no_lift_2026-04-11_04-05-58"
        / "last" / "model.pth",
    ),
    "beam_no_dr_no_lift": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation_no_lift_reward"
        / "beam_no_dr_no_lift_2026-04-11_04-05-58"
        / "runs" / "00_beam_no_dr_no_lift_2026-04-11_04-05-58"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_dr_ablation_no_lift_reward"
        / "beam_no_dr_no_lift_2026-04-11_04-05-58"
        / "runs" / "00_beam_no_dr_no_lift_2026-04-11_04-05-58"
        / "last" / "model.pth",
    ),
    "beam_multi_init_dr": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_states"
        / "beam_multi_init_dr_2026-04-12_18-22-09"
        / "runs" / "00_beam_multi_init_dr_2026-04-12_18-22-09"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_states"
        / "beam_multi_init_dr_2026-04-12_18-22-09"
        / "runs" / "00_beam_multi_init_dr_2026-04-12_18-22-09"
        / "last" / "model.pth",
    ),
    "beam_multi_init_no_dr": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_states"
        / "beam_multi_init_no_dr_2026-04-12_18-28-33"
        / "runs" / "00_beam_multi_init_no_dr_2026-04-12_18-28-33"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_states"
        / "beam_multi_init_no_dr_2026-04-12_18-28-33"
        / "runs" / "00_beam_multi_init_no_dr_2026-04-12_18-28-33"
        / "last" / "model.pth",
    ),
    "beam_table_rand_dr": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_table_rand"
        / "beam_multi_init_table_rand_dr_2026-04-13_15-35-49"
        / "runs" / "00_beam_multi_init_table_rand_dr_2026-04-13_15-35-49"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_table_rand"
        / "beam_multi_init_table_rand_dr_2026-04-13_15-35-49"
        / "runs" / "00_beam_multi_init_table_rand_dr_2026-04-13_15-35-49"
        / "last" / "model.pth",
    ),
    "beam_table_rand_no_dr": (
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_table_rand"
        / "beam_multi_init_table_rand_no_dr_2026-04-13_15-35-49"
        / "runs" / "00_beam_multi_init_table_rand_no_dr_2026-04-13_15-35-49"
        / "config.yaml",
        REPO_ROOT / "train_dir" / "FABRICA_TRAINING" / "fabrica_beam_multi_init_table_rand"
        / "beam_multi_init_table_rand_no_dr_2026-04-13_15-35-49"
        / "runs" / "00_beam_multi_init_table_rand_no_dr_2026-04-13_15-35-49"
        / "last" / "model.pth",
    ),
}


# ===================================================================
# Helpers (copied from fabrica_eval.py to avoid viser/trimesh imports)
# ===================================================================

def _load_assembly_order(assembly: str) -> List[str]:
    path = ASSETS_DIR / assembly / "assembly_order.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["steps"]


def _get_available_parts(assembly: str) -> List[str]:
    """Return parts that have both a trajectory and environment URDF."""
    order = _load_assembly_order(assembly)
    available = []
    for pid in order:
        traj = ASSETS_DIR / assembly / "trajectories" / pid / "pick_place.json"
        urdf = ASSETS_DIR / assembly / "environments" / pid / "scene.urdf"
        if traj.exists() and urdf.exists():
            available.append(pid)
    return available


def _table_urdf_rel(assembly: str, active_pid: str, collision_method: str) -> str:
    suffix_map = {"vhacd": "", "sdf": "_sdf", "coacd": "_coacd"}
    suffix = suffix_map[collision_method]
    return f"urdf/fabrica/{assembly}/environments/{active_pid}/scene{suffix}.urdf"


def get_all_tasks() -> List[Tuple[str, str]]:
    """Return list of (assembly, part_id) for all available tasks."""
    tasks = []
    for assembly in ALL_ASSEMBLIES:
        for pid in _get_available_parts(assembly):
            tasks.append((assembly, pid))
    return tasks


# ===================================================================
# Subprocess: env creation + episode
# ===================================================================

def _create_fabrica_env(config_path, headless, device, overrides):
    """Create FabricaEnv regardless of config task name."""
    from deployment.rl_player_utils import read_cfg_omegaconf
    from deployment.isaac.isaac_env import merge_cfg_with_default_config, create_env_from_cfg
    from omegaconf import OmegaConf

    cfg = read_cfg_omegaconf(config_path=config_path, device=device)
    cfg = merge_cfg_with_default_config(cfg)

    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "FabricaEnv"
    cfg.task_name = "FabricaEnv"
    fabrica_defaults = {
        "enableRetract": True,
        "retractDistanceThreshold": 0.1,
        "retractRewardScale": 1.0,
        "retractSuccessBonus": 0.0,
        "multiPart": False,
        "multiInitStates": False,
        "finalGoalOnly": False,
        "preInsertAndFinal": False,
        "objectNames": None,
    }
    for k, v in fabrica_defaults.items():
        OmegaConf.update(cfg, f"task.env.{k}", v, force_add=True)

    return create_env_from_cfg(cfg=cfg, headless=headless, overrides=overrides)


def _sim_reset(env, device):
    import torch
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    return obs["obs"]


def eval_single_task(conn, assembly, part_id, config_path, checkpoint_path,
                     collision_method, extra_overrides):
    """Subprocess entry point: create env, load policy, run 1 episode, send result."""
    try:
        from isaacgym import gymapi  # noqa: F401 isort:skip
        import torch  # noqa: E401
        from deployment.rl_player import RlPlayer
        import fabrica.objects  # noqa: F401

        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj_suffix = {"vhacd": "", "sdf": "_sdf", "coacd": "_coacd"}[collision_method]
        object_name = f"{assembly}_{part_id}{obj_suffix}"
        traj_path = ASSETS_DIR / assembly / "trajectories" / part_id / "pick_place.json"

        with open(traj_path) as f:
            traj = json.load(f)

        table_urdf = _table_urdf_rel(assembly, part_id, collision_method)

        env = _create_fabrica_env(
            config_path=str(config_path), headless=True, device=device,
            overrides={
                **BASE_OVERRIDES,
                "task.env.objectName": object_name,
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": traj["goals"],
                "task.env.asset.table": table_urdf,
                "task.env.tableResetZ": TABLE_Z,
                "task.env.objectStartPose": traj["start_pose"],
                **({"task.env.useSDF": True} if collision_method == "sdf" else {}),
                **(extra_overrides or {}),
            },
        )

        # Load env state from checkpoint (handles both single-GPU and multi-GPU)
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        env_state = ckpt[0].get("env_state", None) if 0 in ckpt else ckpt.get("env_state", None)
        env.set_env_state(env_state)

        policy = RlPlayer(OBS_DIM, N_ACT, str(config_path), str(checkpoint_path),
                          device, env.num_envs)

        # Run one episode
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


# ===================================================================
# Main process: orchestration
# ===================================================================

def run_eval_task(assembly: str, part_id: str, config_path, checkpoint_path,
                  collision_method: str, extra_overrides: dict,
                  timeout: int = 300) -> dict:
    """Spawn subprocess for one task, collect result."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(
        target=eval_single_task,
        args=(child_conn, assembly, part_id, str(config_path), str(checkpoint_path),
              collision_method, extra_overrides),
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


def run_all_evaluations(policies: Dict[str, Tuple[Path, Path]],
                        tasks: List[Tuple[str, str]],
                        collision_method: str,
                        extra_overrides: dict,
                        timeout: int) -> dict:
    """Run all policy x task evaluations sequentially."""
    total = len(policies) * len(tasks)
    print(f"\n{'='*60}")
    print(f"Fabrica Batch Evaluation")
    print(f"  Policies: {len(policies)} ({', '.join(policies.keys())})")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total evaluations: {total}")
    print(f"  Collision: {collision_method}")
    print(f"{'='*60}\n")

    results = {}
    completed = 0

    for policy_name, (config_path, checkpoint_path) in policies.items():
        print(f"\n--- Policy: {policy_name} ---")
        results[policy_name] = {}

        for assembly, part_id in tasks:
            completed += 1
            tag = f"[{completed}/{total}]"
            print(f"  {tag} {assembly}/{part_id} ... ", end="", flush=True)

            result = run_eval_task(
                assembly, part_id, config_path, checkpoint_path,
                collision_method, extra_overrides, timeout,
            )

            if assembly not in results[policy_name]:
                results[policy_name][assembly] = {}
            results[policy_name][assembly][part_id] = result

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


def compute_summary(results: dict, tasks: List[Tuple[str, str]]) -> dict:
    """Compute summary statistics from results."""
    summary = {}
    assemblies_in_tasks = sorted(set(a for a, _ in tasks))

    for policy_name, assembly_results in results.items():
        all_goals = []
        all_retracts = []
        num_full_success = 0
        num_errors = 0
        per_assembly = {}

        for assembly in assemblies_in_tasks:
            part_results = assembly_results.get(assembly, {})
            asm_goals = []
            for pid, res in part_results.items():
                if res["error"]:
                    num_errors += 1
                    continue
                all_goals.append(res["goal_pct"])
                all_retracts.append(res["retract_ok"])
                asm_goals.append(res["goal_pct"])
                if res["goal_pct"] == 100.0 and res["retract_ok"]:
                    num_full_success += 1

            per_assembly[assembly] = {
                "avg_goal_pct": round(sum(asm_goals) / len(asm_goals), 1) if asm_goals else 0.0,
                "num_parts": len(part_results),
            }

        num_valid = len(all_goals)
        summary[policy_name] = {
            "avg_goal_pct": round(sum(all_goals) / num_valid, 1) if num_valid else 0.0,
            "retract_rate": round(100.0 * sum(all_retracts) / num_valid, 1) if num_valid else 0.0,
            "num_full_success": num_full_success,
            "num_tasks": num_valid,
            "num_errors": num_errors,
            "per_assembly": per_assembly,
        }

    return summary


def print_summary_table(results: dict, summary: dict, tasks: List[Tuple[str, str]],
                        policy_names: List[str]):
    """Print formatted summary tables."""
    assemblies_in_tasks = sorted(set(a for a, _ in tasks))

    # Table 1: Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    header = f"{'Policy':<25} | {'Avg Goal%':>9} | {'Retract%':>8} | {'Full Success':>12}"
    print(header)
    print("-" * len(header))
    for pname in policy_names:
        s = summary[pname]
        print(f"{pname:<25} | {s['avg_goal_pct']:>8.1f}% | {s['retract_rate']:>7.1f}% | "
              f"{s['num_full_success']:>5}/{s['num_tasks']:<5}")

    # Table 2: Per-assembly breakdown
    # Short names for columns
    short_names = {
        "beam": "beam", "car": "car", "cooling_manifold": "cool",
        "gamepad": "game", "plumbers_block": "plumb", "stool_circular": "stool",
    }

    print(f"\n{'='*70}")
    print("PER-ASSEMBLY BREAKDOWN (avg goal%)")
    print(f"{'='*70}")

    col_w = 7
    header = f"{'Policy':<25}"
    for asm in assemblies_in_tasks:
        header += f" | {short_names.get(asm, asm[:5]):>{col_w}}"
    print(header)
    print("-" * len(header))

    for pname in policy_names:
        row = f"{pname:<25}"
        for asm in assemblies_in_tasks:
            val = summary[pname]["per_assembly"].get(asm, {}).get("avg_goal_pct", 0.0)
            row += f" | {val:>{col_w}.1f}"
        print(row)

    # Table 3: Per-part detail
    print(f"\n{'='*70}")
    print("PER-PART DETAIL")
    print(f"{'='*70}")

    pol_w = 14
    header = f"{'Assembly':<20} | {'Part':>4}"
    for pname in policy_names:
        # Truncate policy name if needed
        short_pname = pname[:pol_w]
        header += f" | {short_pname:>{pol_w}}"
    print(header)
    print("-" * len(header))

    for assembly in assemblies_in_tasks:
        parts = sorted(results[policy_names[0]].get(assembly, {}).keys(), key=lambda x: int(x))
        for pid in parts:
            row = f"{assembly:<20} | {pid:>4}"
            for pname in policy_names:
                res = results[pname].get(assembly, {}).get(pid, None)
                if res is None:
                    cell = "N/A"
                elif res["error"]:
                    cell = "ERR"
                else:
                    mark = "+" if res["retract_ok"] else " "
                    cell = f"{res['goal_pct']:5.1f}%{mark}"
                row += f" | {cell:>{pol_w}}"
            print(row)

    print(f"\n(+ = retract succeeded)")


# ===================================================================
# CLI
# ===================================================================

def parse_overrides(override_pairs: list) -> dict:
    """Parse --override KEY VALUE pairs, auto-casting types."""
    overrides = {}
    for key, val in override_pairs:
        # Auto-cast
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        overrides[key] = val
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Fabrica policies across all assembly tasks",
    )
    parser.add_argument("--collision", choices=["vhacd", "coacd", "sdf"], default="coacd",
                        help="Collision method (default: coacd)")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: fabrica/eval_results_<timestamp>.json)")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Subset of policies to evaluate (default: all)")
    parser.add_argument("--assemblies", nargs="+", default=None,
                        help="Subset of assemblies to evaluate (default: all)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-task timeout in seconds (default: 300)")
    parser.add_argument("--override", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"), help="Extra config overrides")
    args = parser.parse_args()

    # Validate and filter policies
    selected_policies = {}
    policy_names_to_use = args.policies or list(POLICIES.keys())
    for name in policy_names_to_use:
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

    # Build task list
    all_tasks = get_all_tasks()
    if args.assemblies:
        all_tasks = [(a, p) for a, p in all_tasks if a in args.assemblies]

    if not all_tasks:
        print("ERROR: No tasks found. Exiting.")
        sys.exit(1)

    extra_overrides = parse_overrides(args.override)

    # Run evaluations
    t_start = time.time()
    results = run_all_evaluations(
        selected_policies, all_tasks, args.collision, extra_overrides, args.timeout,
    )
    total_time = time.time() - t_start

    # Compute summary
    policy_names = list(selected_policies.keys())
    summary = compute_summary(results, all_tasks)

    # Print tables
    print_summary_table(results, summary, all_tasks, policy_names)
    print(f"\nTotal wall time: {total_time/60:.1f} minutes")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = args.output or str(REPO_ROOT / "fabrica" / "eval_outputs" / f"eval_results_{timestamp}.json")

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "collision_method": args.collision,
            "num_tasks": len(all_tasks),
            "total_wall_time_s": round(total_time, 1),
            "policies": {
                name: {"config": str(cfg), "checkpoint": str(ckpt)}
                for name, (cfg, ckpt) in selected_policies.items()
            },
            "tasks": [{"assembly": a, "part_id": p} for a, p in all_tasks],
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
