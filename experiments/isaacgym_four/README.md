# Native IsaacGym fine-tuning and scratch runs

This is the runnable setup used for the four September 20, 2026 Gym runs.
It contains the worker, four frozen task/PPO recipes, environment versions,
preparation, persistent launch and monitoring commands. It builds on host
commit `fddff7462bfb683becb8aa0d73d496dca6065af4`.

| Run | Seed | Initial actor LR | Exploration reward scale | Initial tolerance |
|---|---:|---:|---:|---:|
| JABS fine-tune | 13 | 2.962962962962963e-5 | 0.002 | 0.01 |
| EigenDExplore fine-tune | 202 | 6.666666666666667e-5 | 0.002 | 0.01 |
| JABS scratch | 13 | 1e-4 | 0.005 | 0.075 |
| EigenDExplore scratch | 13 | 1e-4 | 0.005 | 0.075 |

All runs use 24,576 environments, six SAPG blocks of 4,096, horizon 16,
minibatches of 98,304, two mini-epochs and the expanded Sharpa joint limits.
The released Gym recipe supplies the goal bounds, force scale 2, decay 0.99,
torque 0 and joint-velocity observation noise 0.01. Value bootstrap is off.
Native learning-rate adaptation remains enabled. Scratch tolerance anneals to
0.01. There is no automatic epoch, frame or reward stopping cap.

Fine-tunes initialize exact actor, privileged critic, normalization and learned
exploration tensors from the selected ~61B policies, with fresh optimizers and
Gym episodes. The source learning rates above are retained at initialization.
The two fine-tuning sources have 61,137,616,896 JABS and 60,940,222,464 Eigen
training steps. Both Eigen arms use the egosuite++ hot-additive k=9 basis.
The author's released policy weights are not used as initialization.

## Environment and required inputs

The validated environment is Linux, Python 3.8.20, Isaac Gym Preview 4,
PyTorch 2.4.1+cu121 and NVIDIA L40S. A C++ compiler, Ninja and tmux are needed.
The current training runs use this environment; other GPU architectures have
not been validated for this recipe.

Install the separately downloaded NVIDIA Isaac Gym Preview 4 SDK into a
Python 3.8 environment (see [the Gym installation guide](../../docs/isaacgym_installation.md)).
Use the following commands from the repository root in that environment.
`ISAACGYM_ROOT` points to the extracted SDK directory containing `python/`.
The vendored `rl_games` and this checkout are selected by `run_worker.sh`.

```bash
python -m pip install 'setuptools==75.3.4' 'wheel==0.45.1'
python -m pip install 'torch==2.4.1' 'torchvision==0.19.1' \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r experiments/isaacgym_four/requirements.txt
python -m pip install --no-deps -e "$ISAACGYM_ROOT/python"
PYTORCH3D_NO_EXTENSION=1 python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/pytorch3d.git@89653419d0973396f3eff1a381ba09a07fffc2ed'
python -m wandb login
```

Only PyTorch3D's pure-PyTorch transforms are used by this training path.
The requirements file records versions from the working environment; a complete
installation in a new environment was not repeated for this packaging change.
Do not install the root project's broad dependency list over these pins.

Provide these external artifacts; checkpoints, the SDK and generated run data
are not included in Git:

- A weights directory containing `str_joint_abs_s13.pth` and
  `str_joint_abs_eignoise_espp_hot_s202.pth`, the original selected ~61B
  actor-plus-critic exports. Each must contain `model` and
  `assymetric_vf_nets`. Actor-only deployment exports are not suitable here.
- `sharpa_espp_eignoise_hot_k9_simtoolreal.npz`, the egosuite++ hot-additive
  EigenDExplore exploration basis used by those runs.
- Writable output/cache directories outside the source checkout and an
  authenticated W&B account. Online logging is required for production runs.

## Prepare and launch

Activate the Python 3.8 environment first. Set these shell variables to the
chosen machine-local paths and W&B entity:

```bash
export STR_PYTHON="$(command -v python)"
export STR_RUN_ROOT=/your/run/storage/isaacgym-four
export STR_PREPARED=/your/run/storage/isaacgym-four-configs
export STR_WEIGHTS_DIR=/your/checkpoints/selected-61b-weights
export STR_EIGEN_BASIS=/your/artifacts/sharpa_espp_eignoise_hot_k9_simtoolreal.npz
export STR_CACHE_ROOT=/your/cache/storage/isaacgym-four
export STR_WANDB_ENTITY=your-wandb-entity

"$STR_PYTHON" experiments/isaacgym_four/prepare.py \
  --output-dir "$STR_PREPARED" --run-root "$STR_RUN_ROOT" \
  --weights-dir "$STR_WEIGHTS_DIR" --basis "$STR_EIGEN_BASIS" \
  --wandb-entity "$STR_WANDB_ENTITY" --gpus 0 1 2 3

"$STR_PYTHON" experiments/isaacgym_four/launch.py \
  "$STR_PREPARED/manifest.json" --python "$STR_PYTHON" --dry-run

"$STR_PYTHON" experiments/isaacgym_four/launch.py \
  "$STR_PREPARED/manifest.json" --python "$STR_PYTHON"
```

Preparation is CPU-only, creates fresh W&B IDs and never launches a simulator
or creates a W&B run. It refuses to overwrite prepared configs or existing run
directories. `--wandb-project` and `--wandb-group` can override logging locations.
The GPU arguments are physical indices in the table's run order. The launcher
resolves them to UUIDs, refuses occupied GPUs and starts one persistent tmux
session per run. The dry run only reports planned commands and occupancy.
Each run records its session, PID, effective config and W&B URL locally.

For a small smoke check on a device authorized for the test:

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID \
  experiments/isaacgym_four/run_worker.sh \
  "$STR_PREPARED/configs/eigendexplore_finetune.yaml" \
  --smoke-envs 384 --smoke-epochs 4
```

Smokes use separate output directories and disable W&B. Use
`--smoke-envs 24576 --smoke-epochs 3 --smoke-capture` for the full geometry
check. These commands do run GPU work; preparation and `--dry-run` do not.

## Monitor, stop and resume

```bash
python experiments/isaacgym_four/check_status.py "$STR_PREPARED/manifest.json"
```

The status command reports live workers, progress and first-five-minute
samples measured from actual training start. W&B's `train/episode_reward`
is the native rolling mean task return for the final SAPG block, which has
zero intrinsic-reward coefficient. `train/env_steps` counts this Gym run;
`train/parent_plus_env_steps` also includes the source training. Native
TensorBoard metrics are also synchronized to W&B.

Checkpoints include the complete native training state. The initial checkpoint
is saved after epoch 1, rolling checkpoints every 15 minutes, archives every
four hours, plus native best and graceful-shutdown checkpoints. Each run also
records local 30-second state windows for four worlds at launch, every 500
epochs through 3000, then every 1000. These are state files, not uploaded videos.

To stop one run, send SIGTERM to the PID in that run's `worker.pid`, then check
its exit status and shutdown checkpoint. For a restart, copy its resolved YAML
to a new config, set `campaign.run_dir` and `train.params.config.train_dir` to
a new segment directory, and keep its W&B ID. On the same authorized GPU, use
this command inside a named tmux session:

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID \
  experiments/isaacgym_four/run_worker.sh /your/new-segment-config.yaml \
  --resume /your/previous-run/0_gym_RUN_sSEED/last/model.pth
```

`--resume` uses W&B `resume=must` and restores learner state, counters,
adaptive learning rate and curriculum while starting fresh Gym episodes.
Transient simulator tensors are deliberately not restored: restoring them
caused PhysX contact-capacity errors and non-finite observations in a restart
smoke. The corrected path passed actual PPO updates. Native behavior still
counts the first resumed rollout but skips its PPO update.

## Validation

The active four runs passed native GPU rollouts and PPO updates, exact
actor/critic transfer checks, joint ordering, expanded-limit and nonzero hand
friction checks, initial checkpoint saves and W&B API verification. The
full-size smoke exercised all 24,576 environments and state capture; corrected
resume passed a separate smoke. Packaging adds CPU checks that the portable
configs resolve to those same recipes, plus syntax and launcher dry-run checks.
