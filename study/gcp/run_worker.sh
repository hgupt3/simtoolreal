#!/bin/bash
set -euo pipefail

metadata() {
  curl --fail --silent --show-error \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

REPO_DIR=/home/tylerlum/simtoolreal
RUNTIME_DIR="$REPO_DIR/study/gcp/runtime"
VARIANT="$(metadata study-variant)"
NUM_ENVS="$(metadata study-num-envs)"
SEED="$(metadata study-seed)"
EXPERIMENT="${VARIANT}-seed${SEED}"
mkdir -p "$RUNTIME_DIR"

source /home/tylerlum/miniforge3/etc/profile.d/conda.sh
conda activate isaacgym_env
cd "$REPO_DIR"

export HOME=/home/tylerlum
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export WANDB_MODE=offline

# Resume the newest checkpoint after an unexpected VM/service restart. Normal
# seven-day completion exits successfully and is not restarted by systemd.
checkpoint_args=()
checkpoint="$(find "runs/$EXPERIMENT/nn" -maxdepth 1 -type f -name '*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [[ -n "$checkpoint" ]]; then
  checkpoint_args=(--checkpoint "$checkpoint")
fi

python isaacgymenvs/launch_transformer_study.py \
  --variant "$VARIANT" \
  --num-envs "$NUM_ENVS" \
  --seed "$SEED" \
  --experiment "$EXPERIMENT" \
  --wandb-project simtoolreal_transformer \
  --wandb-group 2026-08-12-transformer-study \
  --max-wall-time-seconds 604800 \
  "${checkpoint_args[@]}" \
  2>&1 | tee -a "$RUNTIME_DIR/$EXPERIMENT.log"

