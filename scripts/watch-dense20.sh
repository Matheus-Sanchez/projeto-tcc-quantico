#!/usr/bin/env bash
set -euo pipefail

root="${1:-outputs/classic-dense20}"
interval="${DENSE20_WATCH_INTERVAL:-2}"

if [[ ! -d "$root" ]]; then
  echo "Diretório de resultados não encontrado: $root" >&2
  exit 2
fi

while true; do
  clear
  printf 'Dense20 clássico — %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%-20s %-11s %-13s %-14s\n' 'dataset' 'status' 'época' 'val_macro_f1'
  printf '%-20s %-11s %-13s %-14s\n' '--------------------' '-----------' '-------------' '--------------'

  while IFS= read -r status_file; do
    run_dir="$(dirname "$status_file")"
    dataset="$(basename "$(dirname "$(dirname "$run_dir")")")"
    history="$run_dir/artifacts/history.csv"

    status="$(python3 - "$status_file" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream).get("status", "unknown"))
PY
)"

    epoch=0
    val_macro_f1='-'
    if [[ -s "$history" ]]; then
      read -r epoch val_macro_f1 < <(python3 - "$history" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    print("0 -")
else:
    value = rows[-1].get("val_macro_f1", "-")
    print(len(rows), value if value else "-")
PY
)
    fi

    filled=$((epoch / 5))
    empty=$((20 - filled))
    bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
    printf '%-20s %-11s %3d/100 [%s] %-14s\n' "$dataset" "$status" "$epoch" "$bar" "$val_macro_f1"
  done < <(find "$root" -path '*/runs/*/status.json' -type f | sort)

  printf '\nGPU: '
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || printf 'nvidia-smi indisponível\n'
  printf '\nAtualização a cada %ss; Ctrl+C encerra apenas este monitor.\n' "$interval"
  sleep "$interval"
done
