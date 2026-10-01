from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data_prep.adapters import DatasetSplit, LoadedDataset
from data_prep.audit import audit_dataset
from data_prep.data import balance_training_data, stratified_split_indices
from metrics.reporting import compute_classification_metrics, write_classification_report
from quantum_models.qcnn import N_FEATURES, N_PARAMETERS, PennyLaneQCNN, normalize_features
from quantum_models.qiskit_qcnn import QiskitQCNN
from quantum_models.metal_preflight import verify_tensorflow_gpu
from quantum_models.smoke import FrozenModelError, SmokeSettings, select_frozen_model
from utils.experiment import Experiment, load_dataset_registry


def test_default_registry_matches_the_nine_local_dataset_folders() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_dataset_registry(root / "configs" / "datasets.yaml")
    assert tuple(registry) == (
        "mnist", "fashion_mnist", "kmnist", "emnist_balanced", "cifar10",
        "cifar100_coarse", "svhn", "gtsrb", "fer2013",
    )
    assert registry["kmnist"].root == root / "datasets" / "kmnist"


def test_audit_metrics_and_all_raw_are_framework_independent(tmp_path: Path) -> None:
    loaded = LoadedDataset(
        "fixture", ["zero", "one"],
        {"train": DatasetSplit(images=np.zeros((2, 3, 3, 1), dtype=np.uint8), labels=np.array([0, 4]))}, tmp_path,
    )
    assert audit_dataset(loaded).to_dict()["invalid_labels"][0]["label"] == 4

    samples = np.arange(12).reshape(6, 2)
    labels = np.array([0, 0, 0, 1, 1, 1])
    raw = balance_training_data(samples, labels, mode="all_raw", seed=42)
    assert raw.is_noop and raw.class_weights is None
    assert np.array_equal(raw.samples, samples) and np.array_equal(raw.labels, labels)

    metrics = compute_classification_metrics([0, 1, 1], [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
    saved = write_classification_report(tmp_path / "result", metrics)
    assert json.loads(saved["metrics"].read_text(encoding="utf-8"))["n_samples"] == 3


def test_split_and_model_selection_are_reproducible(tmp_path: Path) -> None:
    labels = np.repeat(np.arange(3), 10)
    split = stratified_split_indices(labels, seed=42)
    assert len(split.train) + len(split.validation) + len(split.test) == len(labels)
    assert all(len(np.unique(labels[indexes])) == 3 for indexes in (split.train, split.validation, split.test))

    def write_manifest(name: str, *, dtype: str, activation: str, accuracy: float) -> Path:
        run = tmp_path / name / dtype / activation / "runs" / "fixture"
        checkpoint = run / "checkpoints" / "best.keras"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture")
        payload = {
            "status": "completed",
            "run_id": "fixture",
            "config": {
                "dataset": name,
                "class_names": [str(value) for value in range(10)],
                "num_classes": 10,
                "balance_mode": "all_raw",
                "normalization": "unit_interval",
                "native_channels": 1,
                "target_size": 64,
                "protocol": {"dtype_policy": dtype, "hidden_activation": activation, "qat_weight_bits": None},
            },
            "data_metadata": {"image_size": 64, "channels": 1},
            "test_metrics": {"classification": {"accuracy": accuracy}},
        }
        manifest = run / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return checkpoint

    fp32_checkpoint = write_manifest("kmnist", dtype="float32", activation="swish", accuracy=0.98)
    write_manifest("kmnist", dtype="mixed_float16", activation="relu", accuracy=0.99)
    write_manifest("mnist", dtype="float32", activation="relu", accuracy=0.99)
    selected, metadata = select_frozen_model(
        dataset="kmnist", model_root=tmp_path, preferred_dtype="float32", preferred_activation="relu"
    )
    assert selected.checkpoint == fp32_checkpoint.resolve()
    assert metadata["exact_profile_match"] is False
    with pytest.raises(FrozenModelError):
        select_frozen_model(
            dataset="kmnist", model_root=tmp_path, preferred_dtype="float32", preferred_activation="relu", strict_profile=True
        )
    SmokeSettings(samples_per_class=6).validate()


def test_tensorflow_metal_preflight_requires_a_real_gpu_kernel() -> None:
    class Tensor:
        device = "/job:localhost/replica:0/task:0/device:GPU:0"

    class Config:
        soft_placement = True

        @staticmethod
        def list_physical_devices(kind: str):
            assert kind == "GPU"
            return [type("Device", (), {"name": "/physical_device:GPU:0"})()]

        @classmethod
        def get_soft_device_placement(cls):
            return cls.soft_placement

        @classmethod
        def set_soft_device_placement(cls, value: bool):
            cls.soft_placement = value

    class TensorFlow:
        __version__ = "fixture"
        float32 = object()
        config = Config

        @staticmethod
        def ones(shape, dtype):
            assert shape == (16, 16)
            assert dtype is TensorFlow.float32
            return object()

        @staticmethod
        def device(name: str):
            assert name == "/GPU:0"
            return nullcontext()

        @staticmethod
        def matmul(left, right):
            assert left is right
            return Tensor()

    report = verify_tensorflow_gpu(TensorFlow)
    assert report["matmul_device"].endswith("GPU:0")
    assert Config.soft_placement is True

    class NoGpuTensorFlow:
        class config:
            @staticmethod
            def list_physical_devices(kind: str):
                assert kind == "GPU"
                return []

    with pytest.raises(RuntimeError, match="Nenhuma GPU TensorFlow"):
        verify_tensorflow_gpu(NoGpuTensorFlow)


def test_pennylane_head_consumes_only_the_256_frozen_features() -> None:
    model = PennyLaneQCNN(10, seed=7)
    features = torch.ones((1, N_FEATURES), dtype=torch.float64)
    assert torch.allclose(torch.linalg.vector_norm(normalize_features(features), dim=1), torch.ones(1, dtype=torch.float64))
    assert tuple(model(features).shape) == (1, 10)
    assert [name for name, _ in model.named_parameters()] == ["theta", "classifier.weight", "classifier.bias"]
    assert model.theta.numel() == N_PARAMETERS
    with pytest.raises(ValueError):
        model(torch.zeros((1, N_FEATURES), dtype=torch.float64))


def test_qiskit_and_pennylane_have_equivalent_quantum_features() -> None:
    features = torch.arange(1, N_FEATURES + 1, dtype=torch.float64).reshape(1, -1)
    pennylane = PennyLaneQCNN(2, seed=19)
    qiskit = QiskitQCNN(2, seed=19)
    theta = torch.linspace(-0.2, 0.2, N_PARAMETERS, dtype=torch.float64)
    with torch.no_grad():
        pennylane.theta.copy_(theta)
        qiskit.theta.copy_(theta)
    assert torch.allclose(pennylane.quantum_features(features), qiskit.quantum_features(features), atol=1e-10, rtol=1e-8)


def test_experiment_manifest_stays_stable(tmp_path: Path) -> None:
    manifest = Experiment("kmnist", "pennylane", seed=42, output_root=tmp_path, settings={"wires": 8}).save_manifest()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["run_id"] == "kmnist__pennylane__seed-42"
