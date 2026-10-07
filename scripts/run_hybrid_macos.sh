#!/bin/zsh
# Run the local parallel PQC head with the macOS Python environment.

set -euo pipefail

script_path="${0:A}"
project_root="${script_path:h:h}"
python_bin="${PYTHON_BIN:-$project_root/.venv-quantum/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  print -u2 "Python quântico não encontrado: $python_bin"
  print -u2 "Crie .venv-quantum e instale requirements/quantum-local.txt."
  exit 1
fi

cd "$project_root"
exec "$python_bin" -u -m quantum_models.parallel_dense20 "$@"
