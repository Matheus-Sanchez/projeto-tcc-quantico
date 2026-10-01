#!/bin/zsh
# Launch a hybrid run from the active macOS Aqua domain and require Metal.

set -euo pipefail

script_path="${0:A}"
project_root="${script_path:h:h}"

if [[ "${QCNN_METAL_AQUA:-}" != "1" ]]; then
  exec /bin/launchctl asuser "$(id -u)" /usr/bin/env QCNN_METAL_AQUA=1 /bin/zsh "$script_path" "$@"
fi

cd "$project_root"
.venv/bin/python -m quantum_models.metal_preflight
exec .venv/bin/python -u -m quantum_models.smoke --require-tensorflow-gpu "$@"
