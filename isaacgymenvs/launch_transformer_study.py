"""Launch one member of the reproducible SimToolReal transformer study."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path


VARIANTS = {
    "rlgames-lstm-sapg": ("rlgames", "SimToolRealStudyRLGamesLSTMSAPG"),
    "legacy-simplerl-lstm-sapg": ("legacy", "SimToolRealStudyLegacyLSTMSAPG"),
    "current-simplerl-lstm-sapg": ("current", "SimToolRealStudyCurrentLSTMSAPG"),
    "current-simplerl-rolling16-sapg": ("current", "SimToolRealStudyRolling16SAPG"),
    "current-simplerl-loco128-sapg": ("current", "SimToolRealStudyLoco128SAPG"),
}


def _legacy_pythonpath(repo_root: Path) -> str:
    temp_root = Path(tempfile.mkdtemp(prefix="simtoolreal_legacy_simplerl_"))
    archive = subprocess.Popen(
        ["git", "archive", "5e831aaa", "simple_rl"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(temp_root)], stdin=archive.stdout, check=True)
    assert archive.stdout is not None
    archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("git archive failed for legacy simple_rl snapshot")
    patch = repo_root / "study" / "legacy_simple_rl_batch_compat.patch"
    subprocess.run(
        ["git", "apply", "--unsafe-paths", str(patch)], cwd=temp_root, check=True
    )
    return str(temp_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--num-envs", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--wandb-project", default="simtoolreal_transformer")
    parser.add_argument("--wandb-entity", default="tylerlum")
    parser.add_argument("--wandb-group", default="2026-08-12-transformer-study")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-wall-time-seconds", type=int, default=604800)
    args = parser.parse_args()

    if args.num_envs % 6:
        parser.error("--num-envs must be divisible by six SAPG policies")
    if args.num_envs * 16 % 98304:
        parser.error("--num-envs * 16 must be divisible by the 98,304 minibatch")

    backend, train_config = VARIANTS[args.variant]
    experiment = args.experiment or f"{args.variant}-seed{args.seed}"
    common = [
        "task=SimToolReal",
        f"train={train_config}",
        f"num_envs={args.num_envs}",
        f"seed={args.seed}",
        f"experiment={experiment}",
        "headless=True",
        "force_render=False",
        "task.env.objectName=allegro_kuka_cuboids",
        "task.env.capture_video=False",
        "task.env.capture_viewer=False",
        "task.env.viserViz=False",
        f"wandb_activate={not args.no_wandb}",
        f"wandb_project={args.wandb_project}",
        f"wandb_entity={args.wandb_entity}",
        f"wandb_group={args.wandb_group}",
        f"wandb_name={experiment}",
        f"wandb_tags=[transformer-study,{args.variant},easy-cuboids,seed-{args.seed}]",
        "++wandb_notes='Six-policy SAPG LF sharing; deterministic Allegro-Kuka cuboids'",
    ]
    if args.checkpoint:
        common.append(f"checkpoint={Path(args.checkpoint).resolve()}")

    environment = os.environ.copy()
    if backend == "rlgames":
        command = ["python", "isaacgymenvs/train.py", *common]
        command.append("train.params.config.expl_coef_block_size=" + str(args.num_envs // 6))
    else:
        command = ["python", "isaacgymenvs/train_simple_rl.py", *common]
        if backend == "current":
            command.append(
                f"train.ppo.max_wall_time_seconds={args.max_wall_time_seconds}"
            )
        else:
            legacy_path = _legacy_pythonpath(Path.cwd())
            previous = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = legacy_path + (os.pathsep + previous if previous else "")

    print("Launching:", shlex.join(command), flush=True)
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
