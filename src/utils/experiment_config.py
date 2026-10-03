"""One protocol and runtime contract for classical and quantum vector heads."""
from __future__ import annotations

import dataclasses
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_ROOT = PROJECT_ROOT / "outputs" / "classic-dense20"
SEED = 42
INPUT_FEATURES = 128
PROJECTION_UNITS = 128
HIDDEN_UNITS = 20
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
TENSORFLOW_VERSION = "2.21.0"
NUMPY_VERSION = "2.0.2"
QUANTUM_VERSIONS = {"pennylane": "0.45.1", "qiskit": "2.2.3", "qiskit-aer": "0.17.2"}
NUM_CLASSES = {
    "mnist": 10, "fashion_mnist": 10, "kmnist": 10, "emnist_balanced": 47,
    "cifar10": 10, "cifar100_coarse": 20, "svhn": 10, "gtsrb": 43, "fer2013": 7,
}


@dataclasses.dataclass(frozen=True)
class VectorHeadSettings:
    seed: int = SEED
    input_features: int = INPUT_FEATURES
    projection_units: int = PROJECTION_UNITS
    projection_activation: str = "relu"
    hidden_units: int = HIDDEN_UNITS
    hidden_activation: str = "tanh"
    precision: str = "float32"
    batch_size: int = BATCH_SIZE
    optimizer: str = "adam"
    learning_rate: float = LEARNING_RATE
    max_epochs: int = MAX_EPOCHS
    checkpoint_metric: str = "val_macro_f1"
    early_stopping: bool = False
    lr_scheduler: bool = False
    telemetry_interval_seconds: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def runtime_versions(*, quantum: bool = False) -> dict[str, Any]:
    expected = {"tensorflow": TENSORFLOW_VERSION, "numpy": NUMPY_VERSION}
    if quantum:
        expected.update(QUANTUM_VERSIONS)
    actual = {}
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
    errors = [f"{name}: esperado {version}, encontrado {actual[name]}"
              for name, version in expected.items() if actual[name] != version]
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 obrigatório; encontrado {sys.version.split()[0]}")
    return {"python": sys.version, "packages": actual, "expected": expected,
            "ok": not errors, "errors": errors}


def require_runtime(*, quantum: bool = False) -> dict[str, Any]:
    report = runtime_versions(quantum=quantum)
    if not report["ok"]:
        raise RuntimeError("Runtime incompatível: " + "; ".join(report["errors"]))
    return report
