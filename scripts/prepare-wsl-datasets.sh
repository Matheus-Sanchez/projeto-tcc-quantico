#!/usr/bin/env bash
# Keep array-backed datasets linked and copy the file-backed GTSRB to native WSL storage.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
source_root="${1:-$project_root/datasets}"
target_root="${2:-${HOME}/datasets-tcc}"
mkdir -p "$target_root"

for name in mnist fashion_mnist kmnist emnist_balanced cifar10 cifar100_coarse svhn fer2013; do
  if [[ ! -e "$target_root/$name" ]]; then
    ln -s "$source_root/$name" "$target_root/$name"
  fi
done

if [[ ! -e "$target_root/gtsrb" ]]; then
  cp -a "$source_root/gtsrb" "$target_root/gtsrb"
fi
touch "$target_root/gtsrb/.native-copy-complete"
echo "Datasets WSL preparados em $target_root"
