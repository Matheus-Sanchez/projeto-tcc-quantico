"""Dataset registry configuration used by the classical pipeline."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the local environment
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_REGISTRY = PROJECT_ROOT / "configs" / "datasets.yaml"
DATASET_ORDER: tuple[str, ...] = (
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_balanced",
    "cifar10",
    "cifar100_coarse",
    "svhn",
    "gtsrb",
    "fer2013",
)


class ConfigurationError(ValueError):
    """Raised when a local dataset registry is malformed."""


@dataclasses.dataclass(frozen=True)
class DatasetEntry:
    """One local dataset and the adapter that knows how to read it."""

    name: str
    adapter: str
    root: Path
    options: dict[str, Any] = dataclasses.field(default_factory=dict)


def _expand_path(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base / path).resolve()


def load_dataset_registry(path: str | Path = DEFAULT_DATASET_REGISTRY) -> dict[str, DatasetEntry]:
    """Load and validate the local dataset registry."""

    if yaml is None:
        raise RuntimeError("PyYAML não está instalado. Execute 'python -m pip install -r requirements.txt'.")

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Registro de datasets não encontrado: {source}")

    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets", payload) if isinstance(payload, Mapping) else None
    if not isinstance(datasets, Mapping):
        raise ConfigurationError("O registro YAML precisa conter o mapa 'datasets'.")

    entries: dict[str, DatasetEntry] = {}
    for raw_name, raw_entry in datasets.items():
        if not isinstance(raw_entry, Mapping):
            raise ConfigurationError(f"Entrada do dataset '{raw_name}' precisa ser um mapa.")
        root = raw_entry.get("root")
        if not root:
            raise ConfigurationError(f"Dataset '{raw_name}' não possui 'root'.")
        name = str(raw_name)
        entries[name] = DatasetEntry(
            name=name,
            adapter=str(raw_entry.get("adapter", name)),
            root=_expand_path(root, source.parent),
            options={key: value for key, value in raw_entry.items() if key not in {"adapter", "root"}},
        )
    return entries


__all__ = ["ConfigurationError", "DATASET_ORDER", "DEFAULT_DATASET_REGISTRY", "DatasetEntry", "load_dataset_registry"]
