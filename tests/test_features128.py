from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from classic_models.features128 import (
    FEATURE_WIDTH,
    _existing_export_is_valid,
    build_backbone_encoder128,
    extract_features128,
)
from data_prep.data import build_tf_dataset


def _tensorflow_or_skip():
    return pytest.importorskip("tensorflow")


@pytest.mark.parametrize("spatial", [1, 2])
def test_encoder128_is_exact_gap_output_without_trainable_dense_layers(spatial: int) -> None:
    tf = _tensorflow_or_skip()
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input((8, 8, 1), name="image")
    convolutional = tf.keras.layers.Conv2D(
        FEATURE_WIDTH,
        1,
        use_bias=False,
        kernel_initializer="ones",
        name="source_conv",
    )(inputs)
    convolutional = tf.keras.layers.Resizing(spatial, spatial, name="block5_pool")(convolutional)
    extractor = tf.keras.Model(inputs, convolutional, name="fixture_backbone")
    extractor.trainable = False

    encoder = build_backbone_encoder128(tf, extractor)
    values = encoder(np.ones((3, 8, 8, 1), dtype=np.float32), training=False).numpy()

    assert encoder.output_shape == (None, FEATURE_WIDTH)
    assert not encoder.trainable_variables
    assert values.shape == (3, FEATURE_WIDTH)
    assert values.dtype == np.float32
    assert np.allclose(values, 1.0)
    assert all("dense" not in layer.name.lower() for layer in encoder.layers)


def test_extract_features128_preserves_order_shape_and_dtype() -> None:
    tf = _tensorflow_or_skip()
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input((8, 8, 1), name="image")
    convolutional = tf.keras.layers.Conv2D(
        FEATURE_WIDTH,
        1,
        use_bias=False,
        kernel_initializer="ones",
        name="block5_pool",
    )(inputs)
    extractor = tf.keras.Model(inputs, convolutional)
    extractor.trainable = False
    encoder = build_backbone_encoder128(tf, extractor)

    images = np.stack([np.full((8, 8, 1), value, dtype=np.uint8) for value in (0, 64, 128, 255)])
    labels = np.array([3, 2, 1, 0], dtype=np.int64)
    dataset, _ = build_tf_dataset(
        images,
        labels,
        image_size=8,
        channels=1,
        batch_size=2,
        training=False,
        extra_fraction=0.0,
        shuffle=False,
        deterministic=True,
        output_dtype="float32",
        repeat=False,
    )
    features, exported_labels = extract_features128(
        tf,
        encoder,
        dataset,
        expected_examples=4,
        label="fixture",
    )

    assert features.shape == (4, FEATURE_WIDTH)
    assert features.dtype == np.float32
    assert exported_labels.dtype == np.int64
    assert np.array_equal(exported_labels, labels)
    assert np.all(np.diff(features[:, 0]) > 0)


def test_completed_export_validation_rejects_20_dimensional_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "features128"
    destination.mkdir()
    splits = {}
    for name, count in (("train", 4), ("val", 2), ("test", 2)):
        features = np.zeros((count, FEATURE_WIDTH), dtype=np.float32)
        labels = np.arange(count, dtype=np.int64)
        np.save(destination / f"{name}_features.npy", features, allow_pickle=False)
        np.save(destination / f"{name}_labels.npy", labels, allow_pickle=False)
        splits[name] = {
            "features_shape": list(features.shape),
            "labels_shape": list(labels.shape),
        }
    (destination / "manifest.json").write_text(json.dumps({
        "feature_width": FEATURE_WIDTH,
        "split_fingerprint": "split",
        "splits": splits,
    }), encoding="utf-8")
    (destination / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    assert _existing_export_is_valid(destination, "split")

    np.save(destination / "train_features.npy", np.zeros((4, 20), dtype=np.float32), allow_pickle=False)
    assert not _existing_export_is_valid(destination, "split")
