#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_version="$(<"$repo_root/.python-version")"
venv_path="$repo_root/.venv"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Este script prepara o ambiente CPU do macOS; sistema detectado: $(uname -s)." >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv não encontrado. Instale-o com: brew install uv" >&2
  exit 2
fi

uv python install "$python_version"

if [[ -x "$venv_path/bin/python" ]]; then
  actual_version="$("$venv_path/bin/python" -c 'import platform; print(platform.python_version())')"
  if [[ "$actual_version" != "$python_version" ]]; then
    echo "O ambiente $venv_path usa Python $actual_version; esperado $python_version." >&2
    echo "Remova esse .venv manualmente e execute o script novamente." >&2
    exit 2
  fi
else
  uv venv --python "$python_version" "$venv_path"
fi

uv pip install --python "$venv_path/bin/python" -r "$repo_root/requirements/classical-128-20-mac.txt"
uv pip check --python "$venv_path/bin/python"
echo "Ambiente pronto em $venv_path (Python $python_version)."
