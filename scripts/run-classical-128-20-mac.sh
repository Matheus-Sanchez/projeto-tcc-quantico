#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" -u -m classic_models.vector_dense20 \
  --features-root "$repo_root/outputs/classic-dense20" \
  --output-root "$repo_root/outputs/classical-128-20-mac" \
  "$@"
