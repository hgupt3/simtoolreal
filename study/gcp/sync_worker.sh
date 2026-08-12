#!/bin/bash
set -euo pipefail

metadata() {
  curl --fail --silent --show-error \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

REPO_DIR=/home/tylerlum/simtoolreal
VARIANT="$(metadata study-variant)"
SEED="$(metadata study-seed)"
EXPERIMENT="${VARIANT}-seed${SEED}"
BUCKET="$(metadata study-bucket)"
DESTINATION="gs://$BUCKET/$EXPERIMENT"

cd "$REPO_DIR"
mkdir -p study/gcp/runtime
date --iso-8601=seconds > "study/gcp/runtime/$EXPERIMENT.heartbeat"
systemctl is-active simtoolreal-study.service > "study/gcp/runtime/$EXPERIMENT.service" || true
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "study/gcp/runtime/$EXPERIMENT.gpu" || true

gcloud storage rsync --recursive "study/gcp/runtime" "$DESTINATION/runtime"
if [[ -d "runs/$EXPERIMENT" ]]; then
  gcloud storage rsync --recursive "runs/$EXPERIMENT" "$DESTINATION/run"
fi
if [[ -d wandb ]]; then
  gcloud storage rsync --recursive wandb "$DESTINATION/wandb"
fi
