#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
export PYTHONPATH="$repo_dir:$repo_dir/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MAX_JOBS=2
if [[ -n "${STR_CACHE_ROOT:-}" ]]; then
    mkdir -p "$STR_CACHE_ROOT/torch_extensions" "$STR_CACHE_ROOT/wandb" "$STR_CACHE_ROOT/tmp"
    export TORCH_EXTENSIONS_DIR="$STR_CACHE_ROOT/torch_extensions"
    export WANDB_CACHE_DIR="$STR_CACHE_ROOT/wandb"
    export TMPDIR="$STR_CACHE_ROOT/tmp"
fi
unset WANDB_SERVICE
exec "${STR_PYTHON:-python}" -B "$script_dir/worker.py" "$@"
