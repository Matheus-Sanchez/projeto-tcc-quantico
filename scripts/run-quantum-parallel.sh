#!/usr/bin/env bash
set -euo pipefail
QUANTUM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${QUANTUM_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1
cd "${QUANTUM_PROJECT_ROOT}"
exec "${PYTHON_BIN:-python}" -m quantum_models.parallel_dense20 "$@"
