#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/root/.venvs/projeto-tcc-classic/bin/python}"

cd "$repo_root"
exec env PYTHON_BIN="$python_bin" \
  bash scripts/wsl-classic-env.sh \
  "$python_bin" -u -m classic_models.features128 \
  --all \
  --registry configs/datasets.wsl.yaml \
  --fail-fast \
  "$@"
