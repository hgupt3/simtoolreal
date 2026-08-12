"""Training / evaluation entry point using simple_rl (PPO / SAPG / EPO).

Usage
-----
# Train with PPO:
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLPPO

# Train with SAPG (6 blocks):
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLSAPG

# Train with EPO (64 blocks + evolution):
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLEPO

# Evaluate a checkpoint:
python isaacgymenvs/train_simple_rl.py \
    task=SimToolReal train=SimToolRealSimpleRLPPO \
    test=True checkpoint=runs/SimToolRealSimpleRLPPO/nn/best.pth

Mirrors the structure of human2sim2robot/sim_training/run.py.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig) -> None:
    # ── Isaac Gym must be imported before torch ──────────────────────────
    from isaacgymenvs.tasks import isaacgym_task_map  # noqa: F401 (triggers gym import)

    import os
    from pathlib import Path

    import torch
    import wandb
    from omegaconf import OmegaConf

    from simple_rl.agent import Agent, PpoConfig
    from simple_rl.player import InferenceConfig, Player, PlayerConfig
    from simple_rl.utils.dict_to_dataclass import dict_to_dataclass
    from simple_rl.utils.network import NetworkConfig

    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.simple_rl_env_wrapper import SimpleRLEnvWrapper
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    # ── Patch wandb_name early (config.yaml defaults to rl_games path) ────
    # The default is ${train.params.config.name} which doesn't exist in simple_rl
    # configs — resolve it lazily and fall back gracefully.
    OmegaConf.set_struct(cfg, False)
    try:
        _wandb_name = str(cfg.wandb_name)
        # If it still looks like the rl_games default, override it
        if "params" in _wandb_name.lower() or "config" in _wandb_name.lower():
            raise ValueError("looks like rl_games default")
    except Exception:
        # Derive a unique name per algorithm (PPO / SAPG / EPO) so each run
        # gets its own experiment directory and config.yaml.
        from isaacgymenvs.utils.reformat import omegaconf_to_dict as _o2d
        _train_ppo = _o2d(cfg.train).get("ppo", {})
        if _train_ppo.get("epo"):
            _algo = "EPO"
        elif _train_ppo.get("sapg"):
            _algo = "SAPG"
        else:
            _algo = "PPO"
        _wandb_name = f"{cfg.task_name}_SimpleRL_{_algo}"
        cfg.wandb_name = _wandb_name

    # ── Device / seed ─────────────────────────────────────────────────────
    # config.yaml already has sim_device / rl_device / graphics_device_id set.
    # For multi-gpu, override with the local rank; otherwise keep as-is.
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.sim_device = f"cuda:{rank}"
        cfg.rl_device = f"cuda:{rank}"
        cfg.seed = set_seed(cfg.seed + rank, torch_deterministic=cfg.torch_deterministic)
    else:
        # sim_device / rl_device already set from config.yaml (default: "cuda:0")
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    set_np_formatting()

    # ── Experiment directory ───────────────────────────────────────────────
    experiment_name = str(cfg.experiment) if cfg.experiment else _wandb_name
    experiment_dir = Path("runs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False))
    with open(experiment_dir / "config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # ── W&B ───────────────────────────────────────────────────────────────
    train = not cfg.test
    if train and cfg.wandb_activate:
        _display_name = _wandb_name
        _run_id = experiment_name.replace("/", "-")
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=_display_name,
            id=_run_id,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags),
            notes=str(cfg.get("wandb_notes", "")),
            config=omegaconf_to_dict(cfg),
            sync_tensorboard=True,
            resume="allow",
        )

    # ── Build env ─────────────────────────────────────────────────────────
    print("Building environment...")
    raw_env = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=True,
    )
    env = SimpleRLEnvWrapper(raw_env)
    print(f"Environment built.  obs={env.observation_space}  act={env.action_space}")

    # ── Parse train config ─────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    train_params = omegaconf_to_dict(cfg.train)

    network_config = dict_to_dataclass(train_params["network"], NetworkConfig)
    ppo_config = dict_to_dataclass(train_params["ppo"], PpoConfig)
    ppo_config.device = cfg.rl_device

    checkpoint = cfg.checkpoint if cfg.checkpoint else None

    # ── Train or test ──────────────────────────────────────────────────────
    if train:
        agent = Agent(
            experiment_dir=experiment_dir,
            ppo_config=ppo_config,
            network_config=network_config,
            env=env,
        )
        if checkpoint:
            print(f"Restoring from checkpoint: {checkpoint}")
            agent.restore(Path(checkpoint))
        if cfg.sigma:
            agent.override_sigma(float(cfg.sigma))
        agent.train()
    else:
        player_config = dict_to_dataclass(train_params["player"], PlayerConfig)
        inference_config = ppo_config.to_inference_config()

        player = Player(
            inference_config=inference_config,
            player_config=player_config,
            network_config=network_config,
            env=env,
        )
        if checkpoint:
            print(f"Restoring from checkpoint: {checkpoint}")
            player.restore(Path(checkpoint))
        if cfg.sigma:
            player.override_sigma(float(cfg.sigma))
        player.run()


if __name__ == "__main__":
    main()
