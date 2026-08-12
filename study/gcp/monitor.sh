#!/bin/bash
set -euo pipefail

PROJECT=gcp-gentoolreal
PREFIX=simtoolreal-study-

gcloud compute instances list --project="$PROJECT" \
  --filter="name~'^${PREFIX}run-'" \
  --format='table(name,zone.basename(),status,lastStartTimestamp)'
gcloud storage ls "gs://gcp-gentoolreal-simtoolreal-transformer/**.heartbeat" || true

