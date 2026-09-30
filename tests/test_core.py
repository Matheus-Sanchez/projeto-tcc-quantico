from __future__ import annotations

from pathlib import Path

import numpy as np

from data_prep.adapters import DatasetSplit, LoadedDataset
from data_prep.audit import audit_dataset
from data_prep.data import balance_training_data, stratified_split_indices
from data_prep.registry import load_dataset_registry
from metrics.metrics import classification_metrics


def test_default_registry_matches_the_nine_local_dataset_folders() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_dataset_registry(root / "configs" / "datasets.yaml")

    assert tuple(registry) == (
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
    assert registry["kmnist"].root == root / "datasets" / "kmnist"


def test_audit_balancing_split_and_metrics_are_framework_independent(tmp_path: Path) -> None:
    loaded = LoadedDataset(
        "fixture",
        ["zero", "one"],
        {"train": DatasetSplit(images=np.zeros((2, 3, 3, 1), dtype=np.uint8), labels=np.array([0, 4]))},
        tmp_path,
    )
    assert audit_dataset(loaded).to_dict()["invalid_labels"][0]["label"] == 4

    samples = np.arange(12).reshape(6, 2)
    labels = np.array([0, 0, 0, 1, 1, 1])
    balanced = balance_training_data(samples, labels, mode="all_raw", seed=42)
    assert balanced.is_noop and balanced.class_weights is None
    assert np.array_equal(balanced.samples, samples)
    assert np.array_equal(balanced.labels, labels)

    labels = np.repeat(np.arange(3), 10)
    split = stratified_split_indices(labels, seed=42)
    assert len(split.train) + len(split.validation) + len(split.test) == 30
    assert all(len(np.unique(labels[indexes])) == 3 for indexes in (split.train, split.validation, split.test))

    metrics = classification_metrics([0, 1, 1], [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]], num_classes=2)
    assert metrics.accuracy == 2 / 3
    assert metrics.confusion_matrix.tolist() == [[1, 0], [1, 1]]
