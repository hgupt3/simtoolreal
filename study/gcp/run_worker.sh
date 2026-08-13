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

# Authenticate non-interactively without putting the credential in Git or VM
# metadata. Refuse to silently run offline: that failure mode makes a healthy
# week-long job invisible in the W&B dashboard.
WANDB_SECRET_PROJECT=gcp-gentoolreal
WANDB_SECRET_NAME=wandb-api-key
if ! WANDB_API_KEY="$(
  gcloud secrets versions access latest \
    --project="$WANDB_SECRET_PROJECT" \
    --secret="$WANDB_SECRET_NAME"
)"; then
  echo "Unable to read $WANDB_SECRET_NAME from GCP Secret Manager." >&2
  echo "Grant roles/secretmanager.secretAccessor to this VM's service account." >&2
  exit 1
fi
if [[ -z "$WANDB_API_KEY" ]]; then
  echo "The W&B API-key secret is empty." >&2
  exit 1
fi
export WANDB_API_KEY
export WANDB_MODE=online

# The source image's legacy 0.12 client only accepts 40-character API keys.
# The private cluster key uses W&B's current 86-character format.
if [[ "$(python -c 'import wandb; print(wandb.__version__)')" != "0.24.2" ]]; then
  python -m pip install --disable-pip-version-check --quiet "wandb==0.24.2"
fi
python - <<'PY'
import os

import wandb

if not wandb.login(key=os.environ["WANDB_API_KEY"], verify=True, relogin=True):
    raise SystemExit("W&B rejected the configured API key")
print(f"W&B online authentication verified with wandb {wandb.__version__}")
PY

# Resume the newest checkpoint after an unexpected VM/service restart. Normal
# seven-day completion exits successfully and is not restarted by systemd.
checkpoint_args=()
checkpoint="$(find "runs/$EXPERIMENT/nn" -maxdepth 1 -type f -name '*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [[ -n "$checkpoint" ]]; then
  checkpoint_args=(--checkpoint "$checkpoint")
fi

set +e
timeout --signal=INT --kill-after=120s 604800 \
python isaacgymenvs/launch_transformer_study.py \
  --variant "$VARIANT" \
  --num-envs "$NUM_ENVS" \
  --seed "$SEED" \
  --experiment "$EXPERIMENT" \
  --wandb-project simtoolreal_transformer \
  --wandb-group 2026-08-12-transformer-study \
  --max-wall-time-seconds 603600 \
  "${checkpoint_args[@]}" \
  2>&1 | tee -a "$RUNTIME_DIR/$EXPERIMENT.log"
run_status=${PIPESTATUS[0]}
set -e

# GNU timeout returns 124 after enforcing the planned study deadline. Treat it
# as successful completion so Restart=on-failure only retries genuine crashes.
if [[ "$run_status" -eq 124 ]]; then
  exit 0
fi
exit "$run_status"
