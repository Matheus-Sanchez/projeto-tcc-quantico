from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from classic_models.vector_dense20 import (
    INPUT_FEATURES,
    VectorDense20Error,
    build_classical_dense20,
    discover_features128,
    load_feature_bundle,
)


def _write_bundle(root: Path, dataset: str = "mnist", *, width: int = INPUT_FEATURES) -> Path:
    destination = root / dataset / "runs" / "run-1" / "artifacts" / "features128"
    destination.mkdir(parents=True)
    labels = np.arange(20, dtype=np.int64) % 10
    features = np.arange(len(labels) * width, dtype=np.float32).reshape(len(labels), width)
    for split in ("train", "val", "test"):
        np.save(destination / f"{split}_features.npy", features, allow_pickle=False)
        np.save(destination / f"{split}_labels.npy", labels, allow_pickle=False)
    return destination


def test_discovers_and_loads_exact_features128_without_transforming(tmp_path: Path) -> None:
    expected = _write_bundle(tmp_path)

    assert discover_features128("mnist", tmp_path) == expected.resolve()
    bundle = load_feature_bundle("mnist", tmp_path)

    assert bundle.features["train"].shape == (20, 128)
    assert bundle.features["train"].dtype == np.float32
    assert bundle.labels["train"].dtype == np.int64
    np.testing.assert_array_equal(
        bundle.features["train"],
        np.arange(20 * 128, dtype=np.float32).reshape(20, 128),
    )


def test_rejects_wrong_feature_width(tmp_path: Path) -> None:
    _write_bundle(tmp_path, width=20)

    with pytest.raises(VectorDense20Error, match=r"esperado shape \(N,128\)"):
        load_feature_bundle("mnist", tmp_path)


def test_builds_128_relu_20_tanh_logits_head() -> None:
    tf = pytest.importorskip("tensorflow")
    tf.keras.backend.clear_session()
    model = build_classical_dense20(tf, num_classes=10)

    assert model.input_shape == (None, 128)
    assert model.output_shape == (None, 10)
    assert model.get_layer("dense_128_128").units == 128
    assert model.get_layer("dense_128_128").activation.__name__ == "relu"
    assert model.get_layer("dense_128_20").units == 20
    assert model.get_layer("dense_20_tanh").activation.__name__ == "tanh"
    assert model.get_layer("logits").activation.__name__ == "linear"
    assert model.count_params() == 19_092 + 21 * 10
