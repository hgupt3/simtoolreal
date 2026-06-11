"""Baseline run: standard PPO with no advantage reweighting (mode=none)."""

import subprocess
from datetime import datetime
from typing import List

import tyro

from launch_training import LaunchTrainingArgs, launch_training


def main() -> None:
    args: LaunchTrainingArgs = tyro.cli(LaunchTrainingArgs)
    args.custom_experiment_name = f"maxrl_baseline_{args.custom_experiment_name}"
    args.wandb_tags = list(args.wandb_tags) + ["maxrl_baseline"]

    # Override launch_training to inject reweighting config
    original_launch = launch_training.__code__

    # We monkey-patch by calling the training directly with extra args
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{args.custom_experiment_name}_{now}"
    hydra_run_dir = f"./train_dir/{args.wandb_project}/{args.wandb_group}/{experiment_name}"
    wandb_tags_str = "[" + ",".join(args.wandb_tags) + "]"

    cmd_parts = [
        "python", "-m", "isaacgymenvs.train",
        "++task.env.useSparseReward=False",
        "headless=True",
        f"task.env.numEnvs={args.num_envs}",
        # === Training ===
        "train.params.config.minibatch_size=98304",
        "multi_gpu=False",
        "train.params.config.good_reset_boundary=0",
        "task.env.goodResetBoundary=0",
        "train.params.config.use_others_experience=lf",
        "train.params.config.off_policy_ratio=1.0",
        "train.params.config.expl_type=mixed_expl_learn_param",
        "train.params.config.expl_reward_type=entropy",
        f"train.params.config.expl_coef_block_size={args.sapg_block_size}",
        "train.params.config.expl_reward_coef_scale=0.005",
        "train.params.network.space.continuous.fixed_sigma=coef_cond",
        # === MaxRL: BASELINE (no reweighting) ===
        "train.params.config.advantage_reweight_mode=none",
        # === Wandb ===
        f"wandb_project={args.wandb_project}",
        f"wandb_entity={args.wandb_entity}",
        f"wandb_activate={args.wandb_activate}",
        f"wandb_group={args.wandb_group}",
        f"wandb_tags={wandb_tags_str}",
        f"++wandb_notes='{args.wandb_notes}'",
        # === Seed ===
        f"seed={args.seed}",
        # === Experiment ===
        f"experiment=00_{experiment_name}",
        f"hydra.run.dir={hydra_run_dir}",
        "task=SimToolRealLSTMAsymmetric",
        "task.env.objectScaleNoiseMultiplierRange=[0.9,1.1]",
        "task.env.forceConsecutiveNearGoalSteps=True",
        f"task.env.forceScale={args.force_scale}",
        f"task.env.torqueScale={args.torque_scale}",
        f"task.env.objectAngVelPenaltyScale={args.object_ang_vel_penalty_scale}",
    ]

    if args.checkpoint is not None:
        cmd_parts.append(f"checkpoint={args.checkpoint}")

    cmd = " ".join(cmd_parts)
    print(f"Running command:\n{cmd}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
