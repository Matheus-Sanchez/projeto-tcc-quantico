"""Configuração pequena e resultados reproduzíveis para os experimentos do TCC.

Este módulo não escolhe nem treina um modelo. Ele guarda apenas o que os
braços clássico, PennyLane e Qiskit têm em comum: dados locais, identidade
determinística e gravação segura de artefatos.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DATASET_ORDER: tuple[str, ...] = (
    "mnist", "fashion_mnist", "kmnist", "emnist_balanced", "cifar10",
    "cifar100_coarse", "svhn", "gtsrb", "fer2013",
)
NORMALIZATION_MODES: tuple[str, ...] = ("unit_interval", "zscore")
BALANCE_MODES: tuple[str, ...] = ("all_raw", "undersample", "oversample", "class_weight")
import random
DEFAULT_SEED = random.randint(1, 999999)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_REGISTRY = PROJECT_ROOT / "configs" / "datasets.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tcc-benchmark"


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
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    return path if path.is_absolute() else (base / path).resolve()


def load_dataset_registry(path: str | Path = DEFAULT_DATASET_REGISTRY) -> dict[str, DatasetEntry]:
    """Load the only checked-in YAML configuration: local dataset locations."""

    if yaml is None:
        raise RuntimeError("PyYAML não está instalado. Execute 'python -m pip install -r requirements.txt'.")
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Registro de datasets não encontrado: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    raw_datasets = payload.get("datasets", payload) if isinstance(payload, Mapping) else None
    if not isinstance(raw_datasets, Mapping):
        raise ConfigurationError("O registro YAML precisa conter o mapa 'datasets'.")

    entries: dict[str, DatasetEntry] = {}
    for raw_name, raw_entry in raw_datasets.items():
        if not isinstance(raw_entry, Mapping):
            raise ConfigurationError(f"Entrada do dataset '{raw_name}' precisa ser um mapa.")
        name = str(raw_name)
        root = raw_entry.get("root")
        if not root:
            raise ConfigurationError(f"Dataset '{name}' não possui 'root'.")
        entries[name] = DatasetEntry(
            name=name,
            adapter=str(raw_entry.get("adapter", name)),
            root=_expand_path(root, source.parent),
            options={key: value for key, value in raw_entry.items() if key not in {"adapter", "root"}},
        )
    return entries


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_ready(item) for item in value), key=repr)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_ready(item())
        except (TypeError, ValueError):
            pass
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _json_ready(to_list())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "unknown"


@dataclasses.dataclass(frozen=True)
class Experiment:
    """Portable identity and destination for a single model/dataset execution."""

    dataset: str
    model: str
    seed: int = DEFAULT_SEED
    output_root: Path = DEFAULT_OUTPUT_ROOT
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return "__".join((_slug(self.dataset), _slug(self.model), f"seed-{int(self.seed)}"))

    @property
    def directory(self) -> Path:
        return Path(self.output_root) / self.dataset / self.run_id

    def manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "model": self.model,
            "seed": int(self.seed),
            "settings": dict(self.settings),
            "settings_fingerprint": fingerprint(self.settings),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }

    def save_manifest(self) -> Path:
        return atomic_write_json(self.directory / "manifest.json", self.manifest())


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON atomically, so interrupted experiments never leave partial files."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(_json_ready(payload), handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


__all__ = [
    "BALANCE_MODES", "DATASET_ORDER", "DEFAULT_DATASET_REGISTRY", "DEFAULT_OUTPUT_ROOT", "DEFAULT_SEED",
    "DatasetEntry", "Experiment", "NORMALIZATION_MODES", "atomic_write_json", "canonical_json",
    "fingerprint", "load_dataset_registry",
]
