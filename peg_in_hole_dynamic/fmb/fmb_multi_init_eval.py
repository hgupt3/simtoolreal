"""FMB Multi-Init-State Evaluation (interactive viser).

Wraps fabrica/fabrica_multi_init_eval.py with asset paths pointing to
assets/urdf/fmb/ and task class forced to FMBEnv.

Usage:
    python fmb/fmb_multi_init_eval.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth \
        --collision coacd

    python fmb/fmb_multi_init_eval.py \
        --config-path train_dir/.../config.yaml \
        --checkpoint-path train_dir/.../model.pth \
        --collision coacd
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
FMB_ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fmb"

# Patch asset paths before importing anything else.
import fabrica.scene_generation.generate_scenes as gen_mod
gen_mod.ASSETS_DIR = FMB_ASSETS_DIR

import fabrica.benchmark_processing.step3_generate_trajectories as step3_mod
step3_mod.ASSETS_DIR = FMB_ASSETS_DIR

import fabrica.fabrica_eval as eval_mod
eval_mod.ASSETS_DIR = FMB_ASSETS_DIR

# Patch _create_fabrica_env to use FMBEnv instead of FabricaEnv.
def _create_fmb_env(config_path, headless, device, overrides):
    from deployment.rl_player_utils import read_cfg_omegaconf
    from deployment.isaac.isaac_env import merge_cfg_with_default_config, create_env_from_cfg
    from omegaconf import OmegaConf

    cfg = read_cfg_omegaconf(config_path=config_path, device=device)
    cfg = merge_cfg_with_default_config(cfg)

    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "FMBEnv"
    cfg.task_name = "FMBEnv"
    fmb_defaults = {
        "enableRetract": True,
        "retractDistanceThreshold": 0.1,
        "retractRewardScale": 1.0,
        "retractSuccessBonus": 0.0,
        "retractSuccessTolerance": 0.005,
        "assemblyName": "fmb_board_1",
        "scenesFilename": "scenes.npz",
        "goalMode": "dense",
        "forcePartIdx": -1,
        "forceSceneIdx": -1,
        "forceStartIdx": -1,
        "withTableForceSensor": False,
        "tableForceResetThreshold": 100.0,
        "goalXyObsNoise": 0.002,
    }
    for k, v in fmb_defaults.items():
        OmegaConf.update(cfg, f"task.env.{k}", v, force_add=True)

    import fmb.objects  # noqa: F401

    OmegaConf.update(cfg, "task.sim.physx.max_gpu_contact_pairs", 16777216, force_add=True)
    OmegaConf.update(cfg, "task.sim.physx.contact_offset", 0.005, force_add=True)
    OmegaConf.update(cfg, "task.sim.physx.num_position_iterations", 16, force_add=True)
    OmegaConf.update(cfg, "task.sim.physx.num_velocity_iterations", 1, force_add=True)

    return create_env_from_cfg(cfg=cfg, headless=headless, overrides=overrides)

eval_mod._create_fabrica_env = _create_fmb_env

# Patch the multi-init eval module.
import fabrica.fabrica_multi_init_eval as multi_eval_mod
multi_eval_mod.ASSETS_DIR = FMB_ASSETS_DIR

def _discover_fmb_assemblies():
    return sorted(
        d.name for d in FMB_ASSETS_DIR.iterdir()
        if d.is_dir() and (d / "canonical_transforms.json").exists()
    )

multi_eval_mod.ALL_ASSEMBLIES = _discover_fmb_assemblies()

# Inject FMB objects into FABRICA_NAME_TO_OBJECT so the eval can look them up.
import fmb.objects
import fabrica.objects
fabrica.objects.FABRICA_NAME_TO_OBJECT.update(fmb.objects.FMB_NAME_TO_OBJECT)

GOAL_MODES = multi_eval_mod.GOAL_MODES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FMB Multi-Init-State Evaluation (interactive viser)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policies-dir", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--initial-policy", type=str, default=None)
    parser.add_argument("--final-goal-tolerance", type=float, default=None)
    parser.add_argument("--collision", choices=["vhacd", "coacd", "sdf"], default="coacd")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="dense")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--override", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    args = parser.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists():
            return str(path)
        path = REPO_ROOT / p
        if path.exists():
            return str(path)
        raise FileNotFoundError(p)

    extra_overrides = {}
    for key, val in args.override:
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        if val == "True":
            val = True
        elif val == "False":
            val = False
        extra_overrides[key] = val

    policies: Dict[str, Tuple[str, str]] = {}
    if args.policies_dir is not None:
        pdir = Path(_resolve(args.policies_dir))
        for sub in sorted(pdir.iterdir()):
            cfg = sub / "config.yaml"
            ckpt = sub / "model.pth"
            if cfg.exists() and ckpt.exists():
                policies[sub.name] = (str(cfg), str(ckpt))
        if not policies:
            raise SystemExit(f"No policy subfolders in {pdir}")
    if args.config_path and args.checkpoint_path:
        name = Path(args.config_path).parent.name or "policy"
        policies[name] = (_resolve(args.config_path), _resolve(args.checkpoint_path))
    if not policies:
        raise SystemExit("Provide --policies-dir or (--config-path and --checkpoint-path).")

    multi_eval_mod.MultiInitAssemblyDemo(
        policies=policies,
        port=args.port,
        final_goal_tolerance=args.final_goal_tolerance,
        collision_method=args.collision,
        extra_overrides=extra_overrides,
        headless=not args.no_headless,
        goal_mode=args.goal_mode,
        initial_policy=args.initial_policy,
    ).run()
