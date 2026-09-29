"""Train a classical 20-feature head over one strictly frozen CNN backbone.

This runner is intentionally independent from ``quantum_models``.  It reuses
the completed FP32 checkpoints from the controlled classical campaign, keeps
the convolutional extractor in inference mode, and trains only a small Keras
classification head.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_prep.adapters import LoadedDataset, load_image_file, load_local_dataset
from data_prep.data import (
    AugmentationConfig,
    PreparedRunDatasets,
    build_tf_dataset,
    prepare_run_datasets,
    select_samples,
    stratified_split_indices,
)
from logs.telemetry import TelemetrySampler
from metrics.metrics import EvaluationResult, ValidationMacroF1Callback, evaluate_model
from utils.experiment import DATASET_ORDER, DEFAULT_DATASET_REGISTRY, fingerprint, load_dataset_registry
from utils.state import RunPaths, atomic_write_bytes, atomic_write_json, atomic_write_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classic-dense20"
FEATURE_LAYER_NAME = "block5_pool"
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
EXTRA_FRACTION = 0.5
EXPECTED_TENSORFLOW_VERSION = "2.21.0"
EXPECTED_BALANCE_MODE = "all_raw"
EXPECTED_NORMALIZATION = "unit_interval"
EXPECTED_DTYPE = "float32"


class Dense20Error(RuntimeError):
    """The fixed Dense20 protocol cannot be followed safely."""


@dataclasses.dataclass(frozen=True)
class FrozenBackboneArtifact:
    dataset: str
    checkpoint: Path
    manifest: Path
    status: str
    dtype_policy: str
    hidden_activation: str
    balance_mode: str
    normalization: str
    qat_weight_bits: int | None
    image_size: int
    channels: int
    num_classes: int
    class_names: tuple[str, ...]
    split_fingerprint: str
    source_fingerprint: str
    source_macro_f1: float
    source_accuracy: float
    payload: Mapping[str, Any] = dataclasses.field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "checkpoint": str(self.checkpoint),
            "manifest": str(self.manifest),
            "status": self.status,
            "dtype_policy": self.dtype_policy,
            "hidden_activation": self.hidden_activation,
            "balance_mode": self.balance_mode,
            "normalization": self.normalization,
            "qat_weight_bits": self.qat_weight_bits,
            "image_size": self.image_size,
            "channels": self.channels,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "split_fingerprint": self.split_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "source_macro_f1": self.source_macro_f1,
            "source_accuracy": self.source_accuracy,
        }


@dataclasses.dataclass(frozen=True)
class DatasetMaterials:
    samples: Sequence[Any] | np.ndarray
    labels: np.ndarray
    channels: int
    class_names: tuple[str, ...]
    source_manifest: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclasses.dataclass(frozen=True)
class Dense20Settings:
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    max_epochs: int = MAX_EPOCHS
    extra_fraction: float = EXTRA_FRACTION
    preprocess_cache_max_mib: int = 2048
    shuffle_buffer_max_mib: int = 1024

    @property
    def augmentation(self) -> AugmentationConfig:
        return AugmentationConfig(
            flip_lr=False,
            brightness_delta=0.03,
            contrast_lower=0.92,
            contrast_upper=1.08,
            translate_frac=0.05,
            zoom_min=0.96,
            zoom_max=1.04,
            noise_std=0.005,
            cutout_prob=0.25,
            cutout_max_frac=0.08,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "precision": EXPECTED_DTYPE,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "extra_fraction": self.extra_fraction,
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy(from_logits=True)",
            "checkpoint_monitor": "val_macro_f1",
            "checkpoint_mode": "max",
            "early_stopping": False,
            "learning_rate_scheduler": False,
            "normalization": EXPECTED_NORMALIZATION,
            "balance_mode": EXPECTED_BALANCE_MODE,
            "augmentation": self.augmentation.to_dict(),
            "architecture": {
                "cut_layer": FEATURE_LAYER_NAME,
                "pooling": "GlobalAveragePooling2D",
                "dense_features": 20,
                "layer_normalization_epsilon": 1e-3,
                "activation": "relu",
                "output": "logits",
            },
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Dense20Error(f"JSON inválido: {path}") from exc
    if not isinstance(payload, dict):
        raise Dense20Error(f"JSON precisa conter um objeto: {path}")
    return payload


def _parse_artifact(manifest: Path) -> FrozenBackboneArtifact:
    payload = _read_json(manifest)
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    protocol = config.get("protocol") if isinstance(config.get("protocol"), Mapping) else {}
    training = config.get("training") if isinstance(config.get("training"), Mapping) else {}
    metadata = payload.get("data_metadata") if isinstance(payload.get("data_metadata"), Mapping) else {}
    metrics = payload.get("test_metrics") if isinstance(payload.get("test_metrics"), Mapping) else {}
    classification = metrics.get("classification") if isinstance(metrics.get("classification"), Mapping) else {}
    qat = protocol.get("qat_weight_bits", training.get("qat_weight_bits"))
    checkpoint = manifest.parent / "checkpoints" / "best.keras"
    class_names = tuple(str(item) for item in config.get("class_names", ()))
    try:
        return FrozenBackboneArtifact(
            dataset=str(config.get("dataset", "")).strip().lower(),
            checkpoint=checkpoint.resolve(),
            manifest=manifest.resolve(),
            status=str(payload.get("status", "")).strip().lower(),
            dtype_policy=str(protocol.get("dtype_policy", training.get("dtype_policy", ""))).strip().lower(),
            hidden_activation=str(protocol.get("hidden_activation", training.get("hidden_activation", ""))).strip().lower(),
            balance_mode=str(config.get("balance_mode", "")).strip().lower(),
            normalization=str(config.get("normalization", "")).strip().lower(),
            qat_weight_bits=None if qat is None else int(qat),
            image_size=int(metadata.get("image_size", config.get("target_size", 0))),
            channels=int(metadata.get("channels", config.get("native_channels", 0))),
            num_classes=int(config.get("num_classes", len(class_names))),
            class_names=class_names,
            split_fingerprint=str(payload.get("split_fingerprint", metadata.get("split", {}).get("fingerprint", ""))),
            source_fingerprint=str(config.get("source_fingerprint", "")),
            source_macro_f1=float(classification.get("macro_f1")),
            source_accuracy=float(classification.get("accuracy")),
            payload=payload,
        )
    except (TypeError, ValueError) as exc:
        raise Dense20Error(f"Manifesto incompleto: {manifest}") from exc


def discover_strict_fp32_backbone(dataset: str, model_root: str | Path = DEFAULT_MODEL_ROOT) -> FrozenBackboneArtifact:
    """Return the unique completed non-QAT FP32 artifact from the FP32 suite."""

    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise Dense20Error(f"Raiz de modelos não encontrada: {root}")
    candidates: list[FrozenBackboneArtifact] = []
    for manifest in sorted(root.rglob("manifest.json")):
        lowered_parts = {part.lower() for part in manifest.parts}
        if "quantization" not in lowered_parts or "fp32" not in lowered_parts:
            continue
        try:
            artifact = _parse_artifact(manifest)
        except Dense20Error:
            continue
        if (
            artifact.dataset == dataset
            and artifact.status == "completed"
            and artifact.dtype_policy == EXPECTED_DTYPE
            and artifact.balance_mode == EXPECTED_BALANCE_MODE
            and artifact.normalization == EXPECTED_NORMALIZATION
            and artifact.qat_weight_bits is None
            and artifact.checkpoint.is_file()
        ):
            candidates.append(artifact)
    if len(candidates) != 1:
        found = "\n".join(f"  - {item.manifest}" for item in candidates) or "  (nenhum)"
        raise Dense20Error(
            f"Esperado exatamente um checkpoint FP32 estrito para {dataset}; encontrados {len(candidates)}:\n{found}"
        )
    return candidates[0]


def _archive_model_profile(checkpoint: Path) -> dict[str, Any]:
    """Inspect a .keras archive without importing TensorFlow."""

    try:
        with zipfile.ZipFile(checkpoint) as archive:
            config = json.loads(archive.read("config.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise Dense20Error(f"Checkpoint Keras inválido: {checkpoint}") from exc
    layers = config.get("config", {}).get("layers", [])
    names = [str(layer.get("config", {}).get("name", "")) for layer in layers]
    inputs = [layer for layer in layers if layer.get("class_name") == "InputLayer"]
    if FEATURE_LAYER_NAME not in names:
        raise Dense20Error(f"Checkpoint não contém a camada {FEATURE_LAYER_NAME!r}: {checkpoint}")
    if len(inputs) != 1:
        raise Dense20Error(f"Checkpoint deve possuir uma entrada: {checkpoint}")
    return {
        "input_shape": inputs[0].get("config", {}).get("batch_shape"),
        "feature_layer": FEATURE_LAYER_NAME,
        "keras_archive": str(checkpoint),
    }


def _infer_channels(sample: Any) -> int:
    image = load_image_file(sample) if isinstance(sample, (str, Path)) else np.asarray(sample)
    if image.ndim == 2:
        return 1
    if image.ndim == 3 and image.shape[-1] in {1, 3}:
        return int(image.shape[-1])
    raise Dense20Error(f"Não foi possível inferir os canais da amostra com shape={getattr(image, 'shape', None)}")


def _join_source_samples(dataset: LoadedDataset, *, image_size: int) -> DatasetMaterials:
    samples: list[Any] = []
    labels: list[int] = []
    saw_paths = False
    saw_arrays = False
    for image_or_path, label, _ in dataset.iter_samples(load_images=False):
        if isinstance(image_or_path, Path):
            saw_paths = True
            samples.append(str(image_or_path))
        else:
            saw_arrays = True
            samples.append(np.asarray(image_or_path))
        labels.append(int(label))
    if not samples:
        raise Dense20Error(f"Dataset vazio: {dataset.name}")
    if saw_paths and saw_arrays:
        samples = [load_image_file(item) if isinstance(item, (str, Path)) else np.asarray(item) for item in samples]
    values = np.asarray(labels, dtype=np.int64)
    expected = set(range(dataset.num_classes))
    observed = set(int(item) for item in np.unique(values))
    if observed != expected:
        raise Dense20Error(f"Classes inválidas em {dataset.name}: esperado={sorted(expected)}, observado={sorted(observed)}")
    channels = _infer_channels(samples[0])
    source_manifest = dataset.to_manifest_dict()
    source_manifest.update(joined_samples=len(samples), native_channels=channels, target_size=int(image_size))
    homogeneous: Sequence[Any] | np.ndarray
    if saw_arrays and not saw_paths:
        homogeneous = np.asarray(samples)
    else:
        homogeneous = samples
    return DatasetMaterials(
        samples=homogeneous,
        labels=values,
        channels=channels,
        class_names=tuple(dataset.class_names),
        source_manifest=source_manifest,
    )


def _validate_materials(artifact: FrozenBackboneArtifact, materials: DatasetMaterials) -> str:
    if artifact.num_classes != materials.num_classes or artifact.class_names != materials.class_names:
        raise Dense20Error(
            f"Classes locais divergem do checkpoint {artifact.dataset}: "
            f"local={materials.class_names}, checkpoint={artifact.class_names}"
        )
    if artifact.channels != materials.channels:
        raise Dense20Error(
            f"Canais locais divergem do checkpoint {artifact.dataset}: {materials.channels} != {artifact.channels}"
        )
    split = stratified_split_indices(materials.labels, seed=SEED)
    actual = split.fingerprint()
    if actual != artifact.split_fingerprint:
        raise Dense20Error(
            f"split_fingerprint divergente para {artifact.dataset}: atual={actual}, esperado={artifact.split_fingerprint}"
        )
    return actual


def _load_dataset_materials(dataset_name: str, registry_path: str | Path, artifact: FrozenBackboneArtifact) -> DatasetMaterials:
    registry = load_dataset_registry(registry_path)
    try:
        entry = registry[dataset_name]
    except KeyError as exc:
        raise Dense20Error(f"Dataset {dataset_name!r} não existe em {registry_path}") from exc
    dataset = load_local_dataset(entry.adapter, entry.root, **entry.options)
    if dataset.name != dataset_name:
        dataset.metadata.setdefault("registry_name", dataset_name)
    return _join_source_samples(dataset, image_size=artifact.image_size)


def _run_id(dataset: str) -> str:
    return f"{dataset}__dense20_fp32_aug05__seed-{SEED}"


def _config_payload(
    artifact: FrozenBackboneArtifact,
    materials: DatasetMaterials,
    settings: Dense20Settings,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "classic_frozen_backbone_dense20",
        "dataset": artifact.dataset,
        "run_id": _run_id(artifact.dataset),
        "settings": settings.to_dict(),
        "backbone": artifact.to_dict(),
        "source_manifest": materials.source_manifest,
    }


def _initialize_manifest(paths: RunPaths, config: Mapping[str, Any], split_fingerprint: str) -> dict[str, Any]:
    config_hash = fingerprint(config)
    if paths.manifest.exists():
        existing = _read_json(paths.manifest)
        if existing.get("config_fingerprint") != config_hash:
            raise Dense20Error(
                f"Execução existente possui configuração incompatível: {paths.manifest}. Use outro --output-root."
            )
        return existing
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "run_id": config["run_id"],
        "dataset": config["dataset"],
        "status": "pending",
        "split_fingerprint": split_fingerprint,
        "config_fingerprint": config_hash,
        "config": dict(config),
    }
    atomic_write_json(paths.manifest, manifest)
    _write_status(paths, "pending", "Execução planejada")
    return manifest


def _write_status(paths: RunPaths, status: str, detail: str, *, error: str | None = None) -> None:
    payload = {
        "run_id": paths.root.name,
        "status": status,
        "detail": detail,
        "error": error,
        "updated_at": _utc_now(),
    }
    atomic_write_json(paths.status, payload)
    if paths.manifest.exists():
        manifest = _read_json(paths.manifest)
        manifest.update(status=status, status_detail=detail, updated_at=payload["updated_at"])
        if error is not None:
            manifest["error"] = error
        atomic_write_json(paths.manifest, manifest)


def _update_manifest(paths: RunPaths, values: Mapping[str, Any]) -> None:
    manifest = _read_json(paths.manifest)
    manifest.update(values)
    manifest["updated_at"] = _utc_now()
    atomic_write_json(paths.manifest, manifest)


def _require_tensorflow_runtime() -> Any:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise Dense20Error("TensorFlow não está disponível no runtime WSL clássico.") from exc
    if tf.__version__ != EXPECTED_TENSORFLOW_VERSION:
        raise Dense20Error(
            f"TensorFlow {EXPECTED_TENSORFLOW_VERSION} é obrigatório; runtime atual={tf.__version__}."
        )
    tf.keras.mixed_precision.set_global_policy(EXPECTED_DTYPE)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise Dense20Error("Nenhuma GPU TensorFlow foi detectada; a rodada completa exige WSL2 + RTX A2000.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    return tf


def _runtime_preflight(tf: Any, output_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(output_root.parent if not output_root.exists() else output_root)
    if usage.free < 5 * 1024**3:
        raise Dense20Error(f"Espaço livre insuficiente para a campanha: {usage.free / 1024**3:.2f} GiB")
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "timestamp": _utc_now(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "dtype_policy": tf.keras.mixed_precision.global_policy().name,
        "gpus": [str(item) for item in gpus],
        "disk_free_gib": usage.free / 1024**3,
        "ok": True,
    }


def _load_backbone(tf: Any, artifact: FrozenBackboneArtifact) -> tuple[Any, dict[str, Any]]:
    model = tf.keras.models.load_model(artifact.checkpoint, compile=False)
    expected_input = (artifact.image_size, artifact.image_size, artifact.channels)
    if tuple(model.input_shape[1:]) != expected_input:
        raise Dense20Error(f"Entrada do checkpoint {model.input_shape} diverge de {expected_input}")
    try:
        feature_layer = model.get_layer(FEATURE_LAYER_NAME)
    except ValueError as exc:
        raise Dense20Error(f"Camada {FEATURE_LAYER_NAME!r} ausente em {artifact.checkpoint}") from exc
    shape = tuple(feature_layer.output.shape)
    if len(shape) != 4 or int(shape[-1]) != 128:
        raise Dense20Error(f"{FEATURE_LAYER_NAME} deve produzir (*,*,128); recebeu {shape}")
    extractor = tf.keras.Model(model.input, feature_layer.output, name="frozen_backbone")
    extractor.trainable = False
    for layer in extractor.layers:
        layer.trainable = False
    if extractor.trainable_variables:
        raise Dense20Error("O backbone ainda possui variáveis treináveis após o congelamento.")
    return extractor, {"input_shape": list(model.input_shape), "feature_shape": [None if x is None else int(x) for x in shape]}


def build_dense20_model(tf: Any, extractor: Any, *, num_classes: int, learning_rate: float = LEARNING_RATE) -> tuple[Any, Any]:
    """Build the fixed classical head and its 20-dimensional encoder."""

    extractor.trainable = False
    inputs = tf.keras.layers.Input(shape=tuple(extractor.input_shape[1:]), dtype="float32", name="image")
    features = extractor(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="backbone_global_average_pool")(features)
    dense = tf.keras.layers.Dense(20, dtype="float32", name="dense20")(pooled)
    normalized = tf.keras.layers.LayerNormalization(epsilon=1e-3, dtype="float32", name="dense20_layer_norm")(dense)
    latent = tf.keras.layers.Activation("relu", dtype="float32", name="features20")(normalized)
    logits = tf.keras.layers.Dense(num_classes, activation=None, dtype="float32", name="logits")(latent)
    model = tf.keras.Model(inputs, logits, name="frozen_backbone_dense20_classifier")
    encoder = tf.keras.Model(inputs, latent, name="frozen_backbone_encoder20")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )
    expected_trainable = 2620 + 21 * int(num_classes)
    actual_trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_variables))
    if actual_trainable != expected_trainable:
        raise Dense20Error(f"Parâmetros treináveis incorretos: {actual_trainable} != {expected_trainable}")
    return model, encoder


def _weights_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for weight in model.weights:
        digest.update(str(weight.name).encode("utf-8"))
        values = np.asarray(weight.numpy())
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _training_callbacks(tf: Any, prepared: PreparedRunDatasets, paths: RunPaths, *, num_classes: int) -> list[Any]:
    history_path = paths.artifacts / "history.csv"
    metrics = ValidationMacroF1Callback(
        prepared.validation_ds,
        num_classes=num_classes,
        history_path=history_path,
        train_examples_per_epoch=int(prepared.metadata["datasets"]["train"]["total_examples"]),
    )
    return [
        metrics,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths.checkpoints / "best.keras"),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths.checkpoints / "last.keras"),
            save_best_only=False,
            save_weights_only=False,
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(paths.checkpoints / "backup"),
            save_freq="epoch",
            delete_checkpoint=False,
        ),
    ]


def _evaluation_payload(evaluation: EvaluationResult) -> dict[str, Any]:
    return {
        "keras_metrics": evaluation.keras_metrics,
        "classification": evaluation.classification.to_dict(),
    }


def _write_predictions(path: Path, evaluation: EvaluationResult) -> None:
    buffer = io.StringIO(newline="")
    fields = ["sample", "y_true", "y_pred", *[f"probability_{i}" for i in range(evaluation.probabilities.shape[1])]]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    predictions = evaluation.probabilities.argmax(axis=1)
    for index, (truth, predicted, probabilities) in enumerate(
        zip(evaluation.y_true, predictions, evaluation.probabilities, strict=True)
    ):
        row: dict[str, Any] = {"sample": index, "y_true": int(truth), "y_pred": int(predicted)}
        row.update({f"probability_{i}": float(value) for i, value in enumerate(probabilities)})
        writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    atomic_write_bytes(path, buffer.getvalue())


def _extract_features(encoder: Any, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in dataset:
        latent = encoder(images, training=False)
        features.append(np.asarray(latent.numpy(), dtype=np.float32))
        labels.append(np.asarray(targets.numpy(), dtype=np.int64).reshape(-1))
    if not features:
        raise Dense20Error("Dataset vazio durante a exportação de features.")
    joined_features = np.concatenate(features).astype(np.float32, copy=False)
    joined_labels = np.concatenate(labels).astype(np.int64, copy=False)
    if joined_features.ndim != 2 or joined_features.shape[1] != 20:
        raise Dense20Error(f"Features exportadas possuem shape inválido: {joined_features.shape}")
    return joined_features, joined_labels


def _export_features(
    encoder: Any,
    materials: DatasetMaterials,
    prepared: PreparedRunDatasets,
    artifact: FrozenBackboneArtifact,
    paths: RunPaths,
    settings: Dense20Settings,
) -> dict[str, Any]:
    destination = paths.artifacts / "features20"
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for name, indices in (
        ("train", prepared.split.train),
        ("val", prepared.split.validation),
        ("test", prepared.split.test),
    ):
        samples = select_samples(materials.samples, indices)
        labels = materials.labels[indices]
        dataset, info = build_tf_dataset(
            samples,
            labels,
            image_size=artifact.image_size,
            channels=artifact.channels,
            batch_size=settings.batch_size,
            normalization=prepared.normalization,
            training=False,
            extra_fraction=0.0,
            seed=settings.seed,
            shuffle=False,
            deterministic=True,
            output_dtype="float32",
            repeat=False,
        )
        features, exported_labels = _extract_features(encoder, dataset)
        features_path = destination / f"{name}_features.npy"
        labels_path = destination / f"{name}_labels.npy"
        _atomic_save_npy(features_path, features)
        _atomic_save_npy(labels_path, exported_labels)
        result[name] = {
            "features": str(features_path.resolve()),
            "labels": str(labels_path.resolve()),
            "features_shape": list(features.shape),
            "features_dtype": str(features.dtype),
            "labels_shape": list(exported_labels.shape),
            "labels_dtype": str(exported_labels.dtype),
            "examples": info.total_examples,
        }
    atomic_write_json(destination / "manifest.json", result)
    return result


def _epoch_summary(history_path: Path) -> dict[str, Any]:
    if not history_path.exists():
        return {"epochs_completed": 0}
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seconds = [float(row["epoch_seconds"]) for row in rows if row.get("epoch_seconds")]
    throughputs = [float(row["train_examples_per_second"]) for row in rows if row.get("train_examples_per_second")]
    return {
        "epochs_completed": len(rows),
        "total_epoch_seconds": float(sum(seconds)),
        "mean_epoch_seconds": float(np.mean(seconds)) if seconds else None,
        "mean_train_examples_per_second": float(np.mean(throughputs)) if throughputs else None,
        "best_val_macro_f1": max((float(row["val_macro_f1"]) for row in rows if row.get("val_macro_f1")), default=None),
    }


def _validate_all_models(tf: Any, artifacts: Sequence[FrozenBackboneArtifact]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for artifact in artifacts:
        extractor, profile = _load_backbone(tf, artifact)
        validated.append({"dataset": artifact.dataset, **profile})
        del extractor
        tf.keras.backend.clear_session()
        gc.collect()
    return validated


def _prepare_one(
    dataset_name: str,
    *,
    registry: Path,
    model_root: Path,
    output_root: Path,
    settings: Dense20Settings,
) -> tuple[FrozenBackboneArtifact, DatasetMaterials, RunPaths, dict[str, Any]]:
    artifact = discover_strict_fp32_backbone(dataset_name, model_root)
    archive_profile = _archive_model_profile(artifact.checkpoint)
    materials = _load_dataset_materials(dataset_name, registry, artifact)
    split_fingerprint = _validate_materials(artifact, materials)
    paths = RunPaths.from_root(output_root / dataset_name / "runs" / _run_id(dataset_name)).ensure()
    config = _config_payload(artifact, materials, settings)
    manifest = _initialize_manifest(paths, config, split_fingerprint)
    _update_manifest(paths, {"archive_profile": archive_profile, "split_fingerprint": split_fingerprint})
    return artifact, materials, paths, manifest


def _run_one(
    tf: Any,
    artifact: FrozenBackboneArtifact,
    materials: DatasetMaterials,
    paths: RunPaths,
    settings: Dense20Settings,
    *,
    resume: bool,
) -> str:
    existing = _read_json(paths.manifest)
    status = str(existing.get("status", "pending"))
    if status == "completed":
        print(f"[{artifact.dataset}] já concluído; ignorado.")
        return "skipped"
    if status in {"running", "interrupted", "failed"} and not resume:
        raise Dense20Error(f"Execução {status} já existe em {paths.root}; use --resume.")

    telemetry: TelemetrySampler | None = None
    started = time.perf_counter()
    _write_status(paths, "running", "Treinamento Dense20 em andamento")
    try:
        telemetry = TelemetrySampler(
            paths.telemetry,
            interval_seconds=5.0,
            disk_paths={
                "data": Path(materials.source_manifest.get("source_path", artifact.manifest)),
                "output": paths.root,
            },
        ).start()
        telemetry.snapshot(event="data_preparation_start", phase="data")
        prepared = prepare_run_datasets(
            materials.samples,
            materials.labels,
            image_size=artifact.image_size,
            channels=artifact.channels,
            batch_size=settings.batch_size,
            normalization_mode=EXPECTED_NORMALIZATION,
            balance_mode=EXPECTED_BALANCE_MODE,
            seed=settings.seed,
            train_fraction=0.70,
            validation_fraction=0.15,
            test_fraction=0.15,
            extra_fraction=settings.extra_fraction,
            augmentation=settings.augmentation,
            preprocess_cache_max_mib=settings.preprocess_cache_max_mib,
            shuffle_buffer_max_mib=settings.shuffle_buffer_max_mib,
            output_dtype="float32",
        )
        if prepared.split.fingerprint() != artifact.split_fingerprint:
            raise Dense20Error("Fingerprint mudou entre planejamento e preparo dos datasets.")
        _update_manifest(paths, {"data_metadata": prepared.metadata})
        telemetry.snapshot(event="data_preparation_end", phase="data")

        extractor, model_profile = _load_backbone(tf, artifact)
        backbone_before = _weights_fingerprint(extractor)
        model, _ = build_dense20_model(tf, extractor, num_classes=artifact.num_classes, learning_rate=settings.learning_rate)
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        atomic_write_text(paths.logs / "model_summary.txt", "\n".join(summary_lines) + "\n")
        _update_manifest(
            paths,
            {
                "model_profile": model_profile,
                "backbone_weights_before": backbone_before,
                "trainable_parameter_count": int(sum(np.prod(item.shape) for item in model.trainable_variables)),
                "backbone_trainable_variables": len(extractor.trainable_variables),
            },
        )

        callbacks = _training_callbacks(tf, prepared, paths, num_classes=artifact.num_classes)
        telemetry.snapshot(event="fit_start", phase="training")
        fit_started = time.perf_counter()
        model.fit(
            prepared.train_ds,
            validation_data=prepared.validation_ds,
            epochs=settings.max_epochs,
            steps_per_epoch=int(prepared.metadata["datasets"]["train"]["batches"]),
            validation_steps=int(prepared.metadata["datasets"]["validation"]["batches"]),
            class_weight=prepared.class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        fit_seconds = time.perf_counter() - fit_started
        telemetry.snapshot(event="fit_end", phase="training")

        backbone_after = _weights_fingerprint(extractor)
        if backbone_after != backbone_before:
            raise Dense20Error("Pesos do backbone mudaram durante o treino.")
        best_path = paths.checkpoints / "best.keras"
        if not best_path.is_file():
            raise Dense20Error("Checkpoint best.keras não foi produzido.")
        selected = tf.keras.models.load_model(best_path, compile=True)
        encoder = tf.keras.Model(selected.input, selected.get_layer("features20").output, name="frozen_backbone_encoder20")

        telemetry.snapshot(event="evaluation_start", phase="evaluation")
        evaluation_started = time.perf_counter()
        evaluation = evaluate_model(selected, prepared.test_ds, num_classes=artifact.num_classes)
        evaluation_seconds = time.perf_counter() - evaluation_started
        telemetry.snapshot(event="evaluation_end", phase="evaluation")

        selected.save(paths.artifacts / "model.keras")
        encoder.save(paths.artifacts / "encoder20.keras")
        test_payload = _evaluation_payload(evaluation)
        atomic_write_json(paths.artifacts / "test_metrics.json", test_payload)
        _write_predictions(paths.artifacts / "predictions.csv", evaluation)
        _atomic_save_npy(paths.artifacts / "test_logits.npy", evaluation.logits.astype(np.float32))
        feature_manifest = _export_features(encoder, materials, prepared, artifact, paths, settings)
        comparison = {
            "dataset": artifact.dataset,
            "original_macro_f1": artifact.source_macro_f1,
            "dense20_macro_f1": evaluation.classification.macro_f1,
            "delta_macro_f1": artifact.source_macro_f1 - evaluation.classification.macro_f1,
            "original_accuracy": artifact.source_accuracy,
            "dense20_accuracy": evaluation.classification.accuracy,
            "source_checkpoint": str(artifact.checkpoint),
        }
        atomic_write_json(paths.artifacts / "comparison.json", comparison)
        training_summary = {
            **_epoch_summary(paths.artifacts / "history.csv"),
            "fit_seconds_current_attempt": fit_seconds,
            "wall_seconds_current_attempt": time.perf_counter() - started,
            "evaluation_seconds": evaluation_seconds,
            "selected_checkpoint": str(best_path.resolve()),
            "backbone_weights_before": backbone_before,
            "backbone_weights_after": backbone_after,
            "backbone_unchanged": True,
        }
        atomic_write_json(paths.logs / "training_summary.json", training_summary)
        telemetry_summary = telemetry.stop(final_event="run_completed")
        telemetry = None
        _update_manifest(
            paths,
            {
                "training": training_summary,
                "test_metrics": test_payload,
                "comparison": comparison,
                "feature_exports": feature_manifest,
                "telemetry": telemetry_summary,
            },
        )
        _write_status(paths, "completed", "Treino, avaliação e exportação Dense20 concluídos")
        print(
            f"[{artifact.dataset}] concluído: Macro-F1={evaluation.classification.macro_f1:.6f}, "
            f"delta={comparison['delta_macro_f1']:+.6f}"
        )
        return "completed"
    except KeyboardInterrupt:
        if telemetry is not None:
            telemetry.stop(final_event="interrupted")
        _write_status(paths, "interrupted", "Interrompido; backup da última época preservado")
        raise
    except Exception as exc:
        if telemetry is not None:
            telemetry.stop(final_event="failed")
        atomic_write_text(paths.logs / "error.txt", traceback.format_exc())
        _write_status(paths, "failed", "Falha na execução Dense20", error=repr(exc))
        raise
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _write_aggregate(output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for comparison_path in sorted(output_root.glob("*/runs/*/artifacts/comparison.json")):
        payload = _read_json(comparison_path)
        rows.append(payload)
    destination = output_root / "dense20_comparison.csv"
    fields = [
        "dataset", "original_macro_f1", "dense20_macro_f1", "delta_macro_f1",
        "original_accuracy", "dense20_accuracy", "source_checkpoint",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: DATASET_ORDER.index(str(row["dataset"]))))
    atomic_write_text(destination, buffer.getvalue())
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m classic_models.dense20",
        description="Treina a cabeça clássica Dense20 sobre os backbones FP32 congelados.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASET_ORDER)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_DATASET_REGISTRY)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Valida artefatos, dados e fingerprints sem TensorFlow.")
    parser.add_argument("--resume", action="store_true", help="Retoma uma execução interrompida usando BackupAndRestore.")
    parser.add_argument("--fail-fast", action="store_true", help="Interrompe --all na primeira falha.")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_epochs < 1 or args.max_epochs > MAX_EPOCHS:
        print(f"Erro: --max-epochs deve estar entre 1 e {MAX_EPOCHS}.", file=sys.stderr)
        return 2
    selected = list(DATASET_ORDER) if args.all else [args.dataset]
    registry = args.registry.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    settings = Dense20Settings(max_epochs=int(args.max_epochs))
    prepared_runs: list[tuple[FrozenBackboneArtifact, RunPaths]] = []
    failures = 0
    for name in selected:
        try:
            prepared = _prepare_one(
                name,
                registry=registry,
                model_root=model_root,
                output_root=output_root,
                settings=settings,
            )
            artifact, materials, paths, _ = prepared
            prepared_runs.append((artifact, paths))
            print(
                f"[{name}] validado: checkpoint={artifact.checkpoint.name}, exemplos={len(materials.labels)}, "
                f"split={artifact.split_fingerprint[:12]}…, run={paths.root}"
            )
            del materials
            gc.collect()
        except Exception as exc:
            failures += 1
            print(f"[{name}] validação falhou: {exc}", file=sys.stderr)
            if args.fail_fast:
                return 1
    if args.dry_run:
        atomic_write_json(
            output_root / "dry_run.json",
            {
                "timestamp": _utc_now(),
                "datasets": [item[0].dataset for item in prepared_runs],
                "failures": failures,
                "settings": settings.to_dict(),
                "ok": failures == 0 and len(prepared_runs) == len(selected),
            },
        )
        return 1 if failures else 0
    if failures:
        print("Preflight estático falhou; nenhum treinamento será iniciado.", file=sys.stderr)
        return 1

    try:
        tf = _require_tensorflow_runtime()
        runtime = _runtime_preflight(tf, output_root)
        runtime["validated_models"] = _validate_all_models(tf, [item[0] for item in prepared_runs])
        atomic_write_json(output_root / "preflight.json", runtime)
    except Exception as exc:
        atomic_write_json(output_root / "preflight.json", {"timestamp": _utc_now(), "ok": False, "error": repr(exc)})
        print(f"Preflight de runtime falhou: {exc}", file=sys.stderr)
        return 1

    for artifact, paths in prepared_runs:
        try:
            materials = _load_dataset_materials(artifact.dataset, registry, artifact)
            _validate_materials(artifact, materials)
            _run_one(tf, artifact, materials, paths, settings, resume=bool(args.resume))
        except KeyboardInterrupt:
            print("Interrompido pelo usuário; use --resume para continuar.", file=sys.stderr)
            return 130
        except Exception as exc:
            failures += 1
            print(f"[{artifact.dataset}] falhou: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
    aggregate = _write_aggregate(output_root)
    print(f"Consolidado: {aggregate}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
