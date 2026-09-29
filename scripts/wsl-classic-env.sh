#!/usr/bin/env bash
# Expose the CUDA libraries installed by tensorflow[and-cuda] before importing TF.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Uso: PYTHON_BIN=/caminho/python bash scripts/wsl-classic-env.sh <comando> [args...]" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
site_packages="$($python_bin -c 'import site; print(site.getsitepackages()[0])')"
nvidia_root="$site_packages/nvidia"
if [[ -d "$nvidia_root" ]]; then
  cuda_libs="$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -path '*/lib' -print | paste -sd: -)"
  if [[ -n "$cuda_libs" ]]; then
    export LD_LIBRARY_PATH="$cuda_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
