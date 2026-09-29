from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from classic_models.dense20 import (
    Dense20Error,
    Dense20Settings,
    DatasetMaterials,
    _archive_model_profile,
    _extract_features,
    _training_callbacks,
    _validate_materials,
    build_dense20_model,
    discover_strict_fp32_backbone,
)
from data_prep.data import AugmentationConfig, build_tf_dataset, prepare_run_datasets, stratified_split_indices
from utils.state import RunPaths


def _write_artifact(
    root: Path,
    *,
    dataset: str = "fixture",
    dtype: str = "float32",
    qat: int | None = None,
    status: str = "completed",
) -> Path:
    run = root / "campaign" / "quantization" / dataset / "fp32" / dataset / "runs" / "run"
    checkpoint = run / "checkpoints" / "best.keras"
    checkpoint.parent.mkdir(parents=True)
    config = {
        "config": {
            "layers": [
                {"class_name": "InputLayer", "config": {"name": "image", "batch_shape": [None, 64, 64, 1]}},
                {"class_name": "MaxPooling2D", "config": {"name": "block5_pool"}},
            ]
        }
    }
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("config.json", json.dumps(config))
    labels = np.repeat(np.arange(2), 10)
    split_fp = stratified_split_indices(labels, seed=42).fingerprint()
    payload = {
        "status": status,
        "split_fingerprint": split_fp,
        "config": {
            "dataset": dataset,
            "class_names": ["zero", "one"],
            "num_classes": 2,
            "balance_mode": "all_raw",
            "normalization": "unit_interval",
            "source_fingerprint": "source",
            "protocol": {"dtype_policy": dtype, "hidden_activation": "swish", "qat_weight_bits": qat},
        },
        "data_metadata": {"image_size": 64, "channels": 1},
        "test_metrics": {"classification": {"macro_f1": 0.9, "accuracy": 0.91}},
    }
    (run / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint


def test_strict_fp32_discovery_and_archive_validation(tmp_path: Path) -> None:
    checkpoint = _write_artifact(tmp_path)
    artifact = discover_strict_fp32_backbone("fixture", tmp_path)
    assert artifact.checkpoint == checkpoint.resolve()
    assert artifact.dtype_policy == "float32"
    assert _archive_model_profile(checkpoint)["input_shape"] == [None, 64, 64, 1]

    _write_artifact(tmp_path / "duplicate")
    with pytest.raises(Dense20Error, match="exatamente um"):
        discover_strict_fp32_backbone("fixture", tmp_path)


@pytest.mark.parametrize(
    ("dtype", "qat", "status"),
    [("mixed_float16", None, "completed"), ("float32", 8, "completed"), ("float32", None, "failed")],
)
def test_strict_discovery_rejects_wrong_profiles(
    tmp_path: Path, dtype: str, qat: int | None, status: str
) -> None:
    _write_artifact(tmp_path, dtype=dtype, qat=qat, status=status)
    with pytest.raises(Dense20Error, match="encontrados 0"):
        discover_strict_fp32_backbone("fixture", tmp_path)


def test_split_fingerprint_and_fixed_augmentation_policy(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    artifact = discover_strict_fp32_backbone("fixture", tmp_path)
    labels = np.repeat(np.arange(2), 10)
    materials = DatasetMaterials(
        samples=np.zeros((20, 8, 8, 1), dtype=np.uint8),
        labels=labels,
        channels=1,
        class_names=("zero", "one"),
        source_manifest={},
    )
    assert _validate_materials(artifact, materials) == artifact.split_fingerprint
    broken = DatasetMaterials(
        samples=materials.samples,
        labels=np.roll(labels, 1),
        channels=1,
        class_names=materials.class_names,
        source_manifest={},
    )
    with pytest.raises(Dense20Error, match="split_fingerprint"):
        _validate_materials(artifact, broken)

    settings = Dense20Settings()
    assert settings.extra_fraction == 0.5
    assert settings.augmentation.flip_lr is False


def _tensorflow_or_skip():
    return pytest.importorskip("tensorflow")


@pytest.mark.parametrize("spatial", [1, 2])
def test_dense20_model_freezes_backbone_and_normalizes_spatial_shape(spatial: int) -> None:
    tf = _tensorflow_or_skip()
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input((8, 8, 1), name="source_image")
    x = tf.keras.layers.Conv2D(128, 1, name="source_conv")(inputs)
    x = tf.keras.layers.Resizing(spatial, spatial, name="block5_pool")(x)
    extractor = tf.keras.Model(inputs, x, name="fixture_backbone")
    extractor.trainable = False
    model, encoder = build_dense20_model(tf, extractor, num_classes=7)
    assert model.output_shape == (None, 7)
    assert encoder.output_shape == (None, 20)
    assert sum(np.prod(item.shape) for item in model.trainable_variables) == 2620 + 21 * 7
    assert not extractor.trainable_variables

    before = [np.array(weight.numpy(), copy=True) for weight in extractor.weights]
    model.train_on_batch(np.zeros((2, 8, 8, 1), dtype=np.float32), np.array([0, 1], dtype=np.int32))
    assert all(np.array_equal(old, weight.numpy()) for old, weight in zip(before, extractor.weights, strict=True))


def test_augmentation_fraction_and_fp32_dataset() -> None:
    tf = _tensorflow_or_skip()
    images = np.zeros((10, 8, 8, 1), dtype=np.uint8)
    labels = np.arange(10, dtype=np.int32) % 2
    dataset, info = build_tf_dataset(
        images,
        labels,
        image_size=64,
        channels=1,
        batch_size=128,
        training=True,
        augmentation=AugmentationConfig(flip_lr=False),
        extra_fraction=0.5,
        output_dtype="float32",
        repeat=False,
    )
    batches = list(dataset)
    assert info.source_examples == 10
    assert info.extra_augmented_examples == 5
    assert info.total_examples == 15
    assert sum(len(targets) for _, targets in batches) == 15
    assert batches[0][0].dtype == tf.float32


def test_one_epoch_writes_macro_f1_checkpoints_history_and_features(tmp_path: Path) -> None:
    tf = _tensorflow_or_skip()
    tf.keras.backend.clear_session()
    images = np.zeros((24, 8, 8, 1), dtype=np.uint8)
    images[12:] = 255
    labels = np.repeat(np.arange(2), 12).astype(np.int32)
    prepared = prepare_run_datasets(
        images,
        labels,
        image_size=64,
        channels=1,
        batch_size=8,
        seed=42,
        extra_fraction=0.5,
        augmentation=AugmentationConfig(flip_lr=False),
        output_dtype="float32",
    )
    inputs = tf.keras.layers.Input((64, 64, 1))
    features = tf.keras.layers.Conv2D(128, 1, name="block5_pool")(inputs)
    extractor = tf.keras.Model(inputs, features)
    extractor.trainable = False
    model, encoder = build_dense20_model(tf, extractor, num_classes=2)
    paths = RunPaths.from_root(tmp_path / "run").ensure()
    callbacks = _training_callbacks(tf, prepared, paths, num_classes=2)
    model.fit(
        prepared.train_ds,
        validation_data=prepared.validation_ds,
        epochs=1,
        steps_per_epoch=prepared.metadata["datasets"]["train"]["batches"],
        callbacks=callbacks,
        verbose=0,
    )
    history = (paths.artifacts / "history.csv").read_text(encoding="utf-8")
    assert "val_macro_f1" in history
    assert "train_examples_per_second" in history
    assert (paths.checkpoints / "best.keras").is_file()
    assert (paths.checkpoints / "last.keras").is_file()

    evaluation_ds, _ = build_tf_dataset(
        images[:4], labels[:4], image_size=64, channels=1, batch_size=4,
        training=False, output_dtype="float32", repeat=False,
    )
    latent, exported_labels = _extract_features(encoder, evaluation_ds)
    assert latent.shape == (4, 20) and latent.dtype == np.float32
    assert exported_labels.shape == (4,) and exported_labels.dtype == np.int64
