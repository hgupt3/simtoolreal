#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in "$DIR"/*.sub; do
    sbatch "$f"
    echo "Submitted $(basename "$f")"
done
