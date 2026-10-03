#!/bin/zsh
# Run the local parallel PQC head with the macOS Python environment.

set -euo pipefail

script_path="${0:A}"
project_root="${script_path:h:h}"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"

cd "$project_root"
exec "$python_bin" -u -m quantum_models.parallel_dense20 "$@"
