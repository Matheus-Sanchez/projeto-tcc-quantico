"""Train a QCNN head from the frozen convolution of a saved local CNN.

The convolutional feature extractor is never trained here.  A completed CNN
checkpoint is selected from ``models/`` by dataset and its manifest is checked
before the 256-dimensional ``global_pool_concat`` vector is sent to the QCNN.
Only the circuit parameters and the final dense classifier are optimised.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import secrets
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data_prep.adapters import LoadedDataset, load_local_dataset
from data_prep.data import (
    AugmentationConfig,
    balance_training_data,
    build_tf_dataset,
    select_samples,
    stratified_split_indices,
    unit_interval_stats,
)
from data_prep.data import require_tensorflow
from logs.telemetry import TelemetrySampler
from metrics.reporting import compute_classification_metrics, write_classification_report
from utils.experiment import DATASET_ORDER, DEFAULT_DATASET_REGISTRY, atomic_write_json, load_dataset_registry

from .qcnn import N_FEATURES, PennyLaneQCNN
from .qiskit_qcnn import QiskitQCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_NAME = "kmnist"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / DEFAULT_DATASET_NAME
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "quantum-smoke"
FEATURE_LAYER_NAME = "global_pool_concat"
EXPECTED_BALANCE_MODE = "all_raw"
EXPECTED_NORMALIZATION = "unit_interval"


class FrozenModelError(ValueError):
    """Raised when a saved CNN cannot safely provide frozen QCNN features."""


@dataclasses.dataclass(frozen=True)
class SmokeSettings:
    """Parameters for either a quick smoke run or a complete QCNN experiment."""

    backend: str = "pennylane"
    seed: int = 42
    mode: str = "smoke"
    samples_per_class: int = 12
    head_epochs: int = 1
    head_batch_size: int = 2
    head_train_per_class: int = 1
    head_validation_per_class: int = 1
    learning_rate: float = 0.01
    evaluation_interval: int = 1
    telemetry_interval_seconds: float = 5.0
    preferred_dtype: str | None = "float32"
    preferred_activation: str | None = "relu"
    strict_model_profile: bool = False

    def validate(self) -> None:
        if self.backend not in {"pennylane", "qiskit"}:
            raise ValueError("backend deve ser 'pennylane' ou 'qiskit'.")
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode deve ser 'smoke' ou 'full'.")
        if int(self.evaluation_interval) < 1:
            raise ValueError("evaluation_interval deve ser positivo.")
        if float(self.telemetry_interval_seconds) <= 0:
            raise ValueError("telemetry_interval_seconds deve ser positiva.")
        for name in (
            "samples_per_class",
            "head_epochs",
            "head_batch_size",
            "head_train_per_class",
            "head_validation_per_class",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} deve ser positivo.")
        if self.samples_per_class < 6:
            raise ValueError("samples_per_class deve ser pelo menos 6 para manter treino/validação/teste.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate deve ser positiva.")


@dataclasses.dataclass(frozen=True)
class FrozenModelArtifact:
    """A trained CNN checkpoint and the metadata needed to use it safely."""

    dataset: str
    checkpoint: Path
    manifest: Path
    dtype_policy: str
    hidden_activation: str
    balance_mode: str
    normalization: str
    image_size: int
    channels: int
    num_classes: int
    class_names: tuple[str, ...]
    qat_weight_bits: int | None
    status: str
    run_id: str
    test_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "checkpoint": str(self.checkpoint),
            "manifest": str(self.manifest),
            "dtype_policy": self.dtype_policy,
            "hidden_activation": self.hidden_activation,
            "balance_mode": self.balance_mode,
            "normalization": self.normalization,
            "image_size": self.image_size,
            "channels": self.channels,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "qat_weight_bits": self.qat_weight_bits,
            "status": self.status,
            "run_id": self.run_id,
            "test_accuracy": self.test_accuracy,
        }


def _normalise_preference(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return None if normalized in {"", "any", "none"} else normalized


def _manifest_artifact(path: Path) -> FrozenModelArtifact:
    """Read the subset of the phase-one manifest required by this runner."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FrozenModelError(f"Manifesto inválido: {path}")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise FrozenModelError(f"Manifesto sem seção config: {path}")
    protocol = config.get("protocol") if isinstance(config.get("protocol"), Mapping) else {}
    training = config.get("training") if isinstance(config.get("training"), Mapping) else {}
    data_metadata = payload.get("data_metadata") if isinstance(payload.get("data_metadata"), Mapping) else {}

    dataset = str(config.get("dataset", "")).strip().lower()
    dtype_policy = str(protocol.get("dtype_policy", training.get("dtype_policy", ""))).strip().lower()
    hidden_activation = str(protocol.get("hidden_activation", training.get("hidden_activation", ""))).strip().lower()
    balance_mode = str(config.get("balance_mode", "")).strip().lower()
    normalization = str(config.get("normalization", "")).strip().lower()
    image_size = int(data_metadata.get("image_size", config.get("target_size", 0)) or 0)
    channels = int(data_metadata.get("channels", config.get("native_channels", 0)) or 0)
    class_names = tuple(str(value) for value in config.get("class_names", ()))
    num_classes = int(config.get("num_classes", len(class_names)) or 0)
    qat_value = protocol.get("qat_weight_bits", training.get("qat_weight_bits"))
    qat_weight_bits = None if qat_value is None else int(qat_value)
    accuracy = payload.get("test_metrics", {}).get("classification", {}).get("accuracy")
    test_accuracy = float(accuracy) if accuracy is not None else None

    required = {
        "dataset": dataset,
        "dtype_policy": dtype_policy,
        "hidden_activation": hidden_activation,
        "balance_mode": balance_mode,
        "normalization": normalization,
        "image_size": image_size,
        "channels": channels,
        "num_classes": num_classes,
    }
    missing = [name for name, value in required.items() if not value]
    if missing or len(class_names) != num_classes:
        raise FrozenModelError(f"Manifesto incompleto em {path}: {', '.join(missing or ['class_names'])}.")

    checkpoint = path.parent / "checkpoints" / "best.keras"
    return FrozenModelArtifact(
        dataset=dataset,
        checkpoint=checkpoint.resolve(),
        manifest=path.resolve(),
        dtype_policy=dtype_policy,
        hidden_activation=hidden_activation,
        balance_mode=balance_mode,
        normalization=normalization,
        image_size=image_size,
        channels=channels,
        num_classes=num_classes,
        class_names=class_names,
        qat_weight_bits=qat_weight_bits,
        status=str(payload.get("status", "")).strip().lower(),
        run_id=str(payload.get("run_id", "")).strip(),
        test_accuracy=test_accuracy,
    )


def discover_frozen_models(model_root: str | Path = DEFAULT_MODEL_ROOT) -> list[FrozenModelArtifact]:
    """Return usable phase-one checkpoints found below ``model_root``."""

    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise FrozenModelError(f"Pasta de modelos não encontrada: {root}")
    artifacts: list[FrozenModelArtifact] = []
    for manifest in sorted(root.rglob("manifest.json")):
        try:
            artifact = _manifest_artifact(manifest)
        except (FrozenModelError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if artifact.checkpoint.is_file():
            artifacts.append(artifact)
    return artifacts


def select_frozen_model(
    *,
    dataset: str,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    preferred_dtype: str | None = "float32",
    preferred_activation: str | None = "relu",
    strict_profile: bool = False,
) -> tuple[FrozenModelArtifact, dict[str, Any]]:
    """Pick the best completed phase-one CNN for a requested local dataset.

    Dataset, ``all_raw`` and ``unit_interval`` are mandatory.  Dtype and hidden
    activation are preferences by default, because the phase-one matrix can
    contain them in separate experiments.  ``strict_profile`` changes that
    behavior into an exact-match requirement.
    """

    expected_dataset = str(dataset).strip().lower()
    dtype = _normalise_preference(preferred_dtype)
    activation = _normalise_preference(preferred_activation)
    candidates = [
        artifact
        for artifact in discover_frozen_models(model_root)
        if artifact.dataset == expected_dataset
        and artifact.status == "completed"
        and artifact.balance_mode == EXPECTED_BALANCE_MODE
        and artifact.normalization == EXPECTED_NORMALIZATION
        and artifact.qat_weight_bits is None
    ]
    if not candidates:
        raise FrozenModelError(
            f"Não há checkpoint concluído para dataset={expected_dataset!r}, "
            f"balance_mode={EXPECTED_BALANCE_MODE!r} e normalization={EXPECTED_NORMALIZATION!r} em {Path(model_root)}."
        )

    def matches_preferences(artifact: FrozenModelArtifact) -> bool:
        return (dtype is None or artifact.dtype_policy == dtype) and (
            activation is None or artifact.hidden_activation == activation
        )

    exact = [artifact for artifact in candidates if matches_preferences(artifact)]
    if strict_profile and not exact:
        available = ", ".join(
            sorted({f"{artifact.dtype_policy}/{artifact.hidden_activation}" for artifact in candidates})
        )
        raise FrozenModelError(
            f"Não existe checkpoint exato para {expected_dataset}: dtype={dtype!r}, activation={activation!r}. "
            f"Perfis disponíveis: {available}."
        )
    pool = exact or candidates

    def score(artifact: FrozenModelArtifact) -> tuple[int, int, float, str]:
        return (
            int(dtype is not None and artifact.dtype_policy == dtype),
            int(activation is not None and artifact.hidden_activation == activation),
            artifact.test_accuracy if artifact.test_accuracy is not None else float("-inf"),
            str(artifact.manifest),
        )

    selected = max(pool, key=score)
    return selected, {
        "model_root": str(Path(model_root).expanduser().resolve()),
        "candidate_count": len(candidates),
        "preferred_dtype": dtype,
        "preferred_activation": activation,
        "strict_profile": bool(strict_profile),
        "exact_profile_match": matches_preferences(selected),
    }


def _load_selected_dataset(
    dataset_name: str,
    *,
    dataset_root: str | Path | None,
    registry_path: str | Path,
) -> LoadedDataset:
    if dataset_root is not None:
        return load_local_dataset(dataset_name, Path(dataset_root))
    registry = load_dataset_registry(registry_path)
    try:
        entry = registry[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Dataset {dataset_name!r} não existe no registro {registry_path}.") from exc
    return load_local_dataset(entry.adapter, entry.root, **entry.options)


def _training_samples(dataset: LoadedDataset) -> tuple[Sequence[Any] | np.ndarray, np.ndarray]:
    """Return an undecoded local training split, preserving paths when present."""

    try:
        partition = dataset.splits["train"]
    except KeyError as exc:
        raise ValueError(f"O dataset local {dataset.name!r} não possui split 'train'.") from exc
    if partition.images is not None and partition.labels is not None and not partition.records:
        return np.asarray(partition.images), np.asarray(partition.labels, dtype=np.int64)
    if partition.records and partition.images is None:
        return [record.path for record in partition.records], np.asarray(
            [record.label for record in partition.records], dtype=np.int64
        )
    raise ValueError(f"O split train de {dataset.name!r} mistura arrays e caminhos; esse formato não é suportado pelo smoke.")


def _balanced_local_subset(
    samples: Sequence[Any] | np.ndarray,
    labels: np.ndarray,
    *,
    samples_per_class: int,
    seed: int,
    classes: int,
) -> tuple[Sequence[Any] | np.ndarray, np.ndarray]:
    """Select a deterministic small raw subset; this never balances training."""

    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in range(classes):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < samples_per_class:
            raise ValueError(f"{label}: o dataset não tem {samples_per_class} exemplos para o teste.")
        generator.shuffle(candidates)
        selected.append(candidates[:samples_per_class])
    indices = np.concatenate(selected)
    generator.shuffle(indices)
    return select_samples(samples, indices), labels[indices]


def _split_smoke_data(
    samples: Sequence[Any] | np.ndarray, labels: np.ndarray, *, seed: int
) -> tuple[
    tuple[Sequence[Any] | np.ndarray, np.ndarray],
    tuple[Sequence[Any] | np.ndarray, np.ndarray],
    tuple[Sequence[Any] | np.ndarray, np.ndarray],
]:
    split = stratified_split_indices(labels, seed=seed, train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15)
    return (
        (select_samples(samples, split.train), labels[split.train]),
        (select_samples(samples, split.validation), labels[split.validation]),
        (select_samples(samples, split.test), labels[split.test]),
    )


def _make_dataset(
    samples: Sequence[Any] | np.ndarray,
    labels: np.ndarray,
    *,
    image_size: int,
    channels: int,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    """Use the shared pipeline with the normalization used by the frozen CNN."""

    return build_tf_dataset(
        samples,
        labels,
        image_size=image_size,
        channels=channels,
        batch_size=batch_size,
        normalization=unit_interval_stats(channels=channels, image_size=image_size),
        training=training,
        augmentation=AugmentationConfig(),
        extra_fraction=0.0,
        seed=seed,
        shuffle=training,
        output_dtype="float32",
        repeat=False,
    )[0]


def _validate_artifact_for_dataset(artifact: FrozenModelArtifact, dataset: LoadedDataset) -> None:
    if artifact.dataset != dataset.name:
        raise FrozenModelError(f"Checkpoint é de {artifact.dataset!r}, mas a execução selecionou {dataset.name!r}.")
    if artifact.balance_mode != EXPECTED_BALANCE_MODE or artifact.normalization != EXPECTED_NORMALIZATION:
        raise FrozenModelError("O checkpoint precisa ter sido treinado com all_raw e unit_interval.")
    if artifact.qat_weight_bits is not None:
        raise FrozenModelError("Checkpoints QAT não são aceitos como extrator congelado nesta execução.")
    if artifact.num_classes != dataset.num_classes or artifact.class_names != tuple(dataset.class_names):
        raise FrozenModelError("Classes do checkpoint e do dataset local não correspondem.")


def load_frozen_feature_extractor(artifact: FrozenModelArtifact) -> Any:
    """Load one persisted CNN and expose its frozen 256-D convolution vector."""

    tensorflow = require_tensorflow()
    tensorflow.keras.mixed_precision.set_global_policy("float32")
    model = tensorflow.keras.models.load_model(artifact.checkpoint, compile=False)
    input_shape = tuple(model.input_shape)
    expected_shape = (artifact.image_size, artifact.image_size, artifact.channels)
    if len(input_shape) != 4 or tuple(input_shape[1:]) != expected_shape:
        raise FrozenModelError(
            f"Entrada do checkpoint {input_shape} não corresponde ao manifesto {expected_shape}."
        )
    if artifact.dtype_policy == "float32" and model.dtype_policy.name != "float32":
        raise FrozenModelError(
            f"Checkpoint declarado FP32, mas carregado com política {model.dtype_policy.name!r}."
        )
    try:
        feature_layer = model.get_layer(FEATURE_LAYER_NAME)
    except ValueError as exc:
        raise FrozenModelError(f"Camada de características {FEATURE_LAYER_NAME!r} não existe no checkpoint.") from exc
    output_shape = tuple(feature_layer.output.shape)
    if output_shape[-1] != N_FEATURES:
        raise FrozenModelError(f"O extrator deve produzir {N_FEATURES} características; recebeu {output_shape}.")

    extractor = tensorflow.keras.Model(model.input, feature_layer.output, name="frozen_convolution_features")
    extractor.trainable = False
    for layer in extractor.layers:
        layer.trainable = False
    if extractor.trainable_variables:
        raise RuntimeError("Falha ao congelar a parte convolucional da CNN.")
    return extractor


def _extract_features(extractor: Any, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in dataset:
        values = extractor(images, training=False)
        features.append(np.asarray(values.numpy() if hasattr(values, "numpy") else values, dtype=np.float32))
        labels.append(np.asarray(targets.numpy() if hasattr(targets, "numpy") else targets, dtype=np.int64).reshape(-1))
    if not features:
        raise ValueError("O dataset de características está vazio.")
    matrix = np.concatenate(features)
    target_vector = np.concatenate(labels)
    if matrix.ndim != 2 or matrix.shape[1] != N_FEATURES:
        raise RuntimeError(f"Formato inesperado das características convolucionais: {matrix.shape}.")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) <= 1e-12):
        raise RuntimeError("A CNN congelada produziu características inválidas para amplitude encoding.")
    return matrix, target_vector


def _take_per_class(features: np.ndarray, labels: np.ndarray, *, count: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    for label in range(classes):
        class_indices = np.flatnonzero(labels == label)
        if len(class_indices) < count:
            raise ValueError(f"Split do teste não tem {count} vetores para a classe {label}.")
        indices.append(class_indices[:count])
    joined = np.concatenate(indices)
    return features[joined], labels[joined]


def _head_model(backend: str, *, classes: int, seed: int) -> Any:
    return PennyLaneQCNN(classes, seed=seed) if backend == "pennylane" else QiskitQCNN(classes, seed=seed)


def _iterate_minibatches(features: Any, labels: Any, *, batch_size: int, seed: int) -> Iterable[tuple[Any, Any]]:
    import torch

    order = torch.randperm(len(features), generator=torch.Generator().manual_seed(seed))
    for indices in order.split(batch_size):
        yield features[indices], labels[indices]


def _train_quantum_head(
    backend: str,
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    classes: int,
    settings: SmokeSettings,
) -> tuple[Any, list[float], np.ndarray]:
    import torch
    from torch import nn

    torch.manual_seed(settings.seed)
    model = _head_model(backend, classes=classes, seed=settings.seed)
    # The frozen TensorFlow extractor is deliberately outside this optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.CrossEntropyLoss()
    x_train = torch.as_tensor(train_features, dtype=torch.float64)
    y_train = torch.as_tensor(train_labels, dtype=torch.long)
    x_validation = torch.as_tensor(validation_features, dtype=torch.float64)
    y_validation = torch.as_tensor(validation_labels, dtype=torch.long)
    losses: list[float] = []
    progress_interval = max(1, settings.head_epochs // 10)
    for epoch in range(settings.head_epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_features, batch_labels in _iterate_minibatches(
            x_train, y_train, batch_size=settings.head_batch_size, seed=settings.seed + epoch
        ):
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_labels)
            total_examples += len(batch_labels)
        epoch_loss = total_loss / total_examples
        losses.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % progress_interval == 0 or epoch + 1 == settings.head_epochs:
            print(f"QCNN época {epoch + 1}/{settings.head_epochs}: loss={epoch_loss:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(x_validation).cpu().numpy()
    return model, losses, logits


@dataclasses.dataclass(frozen=True)
class HeadEvaluation:
    loss: float
    classification: dict[str, Any]
    quantum: dict[str, Any]


class _QuantumMeasurementAccumulator:
    """Aggregate physical checks for the probability vector emitted by the QCNN."""

    def __init__(self) -> None:
        self.count = 0
        self._probability_sum = 0.0
        self._max_sum_error = 0.0
        self._entropy_sum = 0.0
        self._effective_state_sum = 0.0
        self._peak_probability_sum = 0.0
        self._minimum_probability = float("inf")

    def add(self, probabilities: Any) -> None:
        values = np.asarray(probabilities.detach().cpu().numpy(), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != N_FEATURES:
            raise RuntimeError(f"Saída quântica inesperada: {values.shape}.")
        sums = values.sum(axis=1)
        clipped = np.clip(values, 1e-15, None)
        entropy = -np.sum(np.where(values > 0.0, values * np.log(clipped), 0.0), axis=1)
        self.count += int(values.shape[0])
        self._probability_sum += float(sums.sum())
        self._max_sum_error = max(self._max_sum_error, float(np.max(np.abs(sums - 1.0))))
        self._entropy_sum += float(entropy.sum())
        self._effective_state_sum += float(np.exp(entropy).sum())
        self._peak_probability_sum += float(values.max(axis=1).sum())
        self._minimum_probability = min(self._minimum_probability, float(values.min()))

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {"n_measurements": 0}
        return {
            "n_measurements": self.count,
            "mean_probability_sum": self._probability_sum / self.count,
            "max_probability_sum_error": self._max_sum_error,
            "mean_entropy_nats": self._entropy_sum / self.count,
            "mean_effective_basis_states": self._effective_state_sum / self.count,
            "mean_peak_probability": self._peak_probability_sum / self.count,
            "minimum_probability": self._minimum_probability,
        }


def _ordered_minibatches(features: Any, labels: Any, *, batch_size: int) -> Iterable[tuple[Any, Any]]:
    for start in range(0, len(features), batch_size):
        yield features[start : start + batch_size], labels[start : start + batch_size]


def _parameter_summary(head: Any, *, theta_before: Any, theta_gradient_norms: list[float]) -> dict[str, Any]:
    theta = head.theta.detach().cpu().numpy().astype(np.float64, copy=False)
    change = head.theta.detach() - theta_before
    return {
        "theta_l2_norm": float(np.linalg.norm(theta)),
        "theta_mean": float(theta.mean()),
        "theta_std": float(theta.std()),
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "theta_update_l2_norm": float(change.detach().norm().cpu()),
        "theta_gradient_l2_norm_mean": float(np.mean(theta_gradient_norms)) if theta_gradient_norms else None,
        "theta_gradient_l2_norm_max": float(np.max(theta_gradient_norms)) if theta_gradient_norms else None,
    }


def _metric_digest(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "n_samples",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    }


def _hardware_digest(sample: Mapping[str, Any]) -> dict[str, Any]:
    system = sample.get("system", {}) if isinstance(sample.get("system"), Mapping) else {}
    process = sample.get("process", {}) if isinstance(sample.get("process"), Mapping) else {}
    gpu = sample.get("gpu", {}) if isinstance(sample.get("gpu"), Mapping) else {}
    devices = gpu.get("devices", []) if isinstance(gpu.get("devices"), Sequence) else []
    return {
        "timestamp": sample.get("timestamp"),
        "system_cpu_percent": system.get("cpu_percent"),
        "system_memory_percent": system.get("memory_percent"),
        "process_cpu_percent": process.get("cpu_percent"),
        "process_rss_mb": process.get("rss_mb"),
        "gpu_utilization_percent": [device.get("utilization_percent") for device in devices if isinstance(device, Mapping)],
        "gpu_memory_used_mb": [device.get("memory_used_mb") for device in devices if isinstance(device, Mapping)],
    }


def _evaluate_quantum_head(
    head: Any,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    classes: int,
    class_names: Sequence[str],
) -> HeadEvaluation:
    """Evaluate loss, AI metrics and quantum probability diagnostics together."""

    import torch
    from torch import nn

    x_values = torch.as_tensor(features, dtype=torch.float64)
    y_values = torch.as_tensor(labels, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    accumulator = _QuantumMeasurementAccumulator()
    total_loss = 0.0
    total_examples = 0
    head.eval()
    with torch.no_grad():
        for batch_features, batch_labels in _ordered_minibatches(x_values, y_values, batch_size=batch_size):
            probabilities, logits = head.forward_with_quantum_features(batch_features)
            loss = criterion(logits, batch_labels)
            batch_size_value = int(len(batch_labels))
            total_loss += float(loss.detach()) * batch_size_value
            total_examples += batch_size_value
            predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False))
            truths.append(batch_labels.cpu().numpy().astype(np.int64, copy=False))
            accumulator.add(probabilities)
    if total_examples == 0:
        raise ValueError("Não é possível avaliar uma cabeça quântica sem exemplos.")
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    return HeadEvaluation(
        loss=total_loss / total_examples,
        classification=compute_classification_metrics(truth, prediction, class_names=class_names),
        quantum=accumulator.summary(),
    )


def _checkpoint_payload(
    *,
    head: Any,
    artifact: FrozenModelArtifact,
    settings: SmokeSettings,
    epoch: int,
    validation: HeadEvaluation | None,
) -> dict[str, Any]:
    return {
        "backend": settings.backend,
        "n_classes": artifact.num_classes,
        "epoch": int(epoch),
        "state_dict": head.state_dict(),
        "settings": dataclasses.asdict(settings),
        "frozen_cnn": artifact.to_dict(),
        "validation": None
        if validation is None
        else {"loss": validation.loss, "classification": validation.classification, "quantum": validation.quantum},
    }


def _print_epoch_progress(epoch: int, total_epochs: int, record: Mapping[str, Any]) -> None:
    train = record["train"]
    validation = record.get("validation")
    timing = record["timing"]
    message = (
        f"Época {epoch}/{total_epochs} - loss: {train['loss']:.6f} - "
        f"accuracy: {train['classification']['accuracy']:.4f} - "
        f"balanced_accuracy: {train['classification']['balanced_accuracy']:.4f} - "
        f"macro_f1: {train['classification']['macro_f1']:.4f}"
    )
    if validation is not None:
        message += (
            f" - val_loss: {validation['loss']:.6f} - "
            f"val_accuracy: {validation['classification']['accuracy']:.4f} - "
            f"val_balanced_accuracy: {validation['classification']['balanced_accuracy']:.4f} - "
            f"val_macro_f1: {validation['classification']['macro_f1']:.4f}"
        )
    else:
        message += " - validação: não medida nesta época"
    message += (
        f" - theta_grad_norm: {record['quantum_parameters']['theta_gradient_l2_norm_mean'] or 0.0:.6f}"
        f" - {timing['epoch_seconds']:.1f}s - {timing['examples_per_second']:.2f} exemplos/s"
    )
    print(message, flush=True)


def _train_complete_quantum_head(
    *,
    artifact: FrozenModelArtifact,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    settings: SmokeSettings,
    destination: Path,
    telemetry: TelemetrySampler,
) -> tuple[Any, list[dict[str, Any]], int, HeadEvaluation, dict[str, Path]]:
    """Train the head on all supplied train features and evaluate validation per epoch."""

    import torch
    from torch import nn

    torch.manual_seed(settings.seed)
    head = _head_model(settings.backend, classes=artifact.num_classes, seed=settings.seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings.learning_rate)
    criterion = nn.CrossEntropyLoss()
    x_train = torch.as_tensor(train_features, dtype=torch.float64)
    y_train = torch.as_tensor(train_labels, dtype=torch.long)
    history: list[dict[str, Any]] = []
    history_path = destination / "epoch-history.json"
    best_path = destination / "qcnn-head-best.pt"
    last_path = destination / "qcnn-head-last.pt"
    best_epoch = 0
    best_score: tuple[float, float] | None = None
    best_validation: HeadEvaluation | None = None

    for epoch_index in range(settings.head_epochs):
        epoch_started = time.perf_counter()
        theta_before = head.theta.detach().clone()
        theta_gradient_norms: list[float] = []
        predictions: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        total_loss = 0.0
        total_examples = 0
        head.train()
        for batch_features, batch_labels in _iterate_minibatches(
            x_train, y_train, batch_size=settings.head_batch_size, seed=settings.seed + epoch_index
        ):
            optimizer.zero_grad(set_to_none=True)
            _probabilities, logits = head.forward_with_quantum_features(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            if head.theta.grad is not None:
                theta_gradient_norms.append(float(head.theta.grad.detach().norm().cpu()))
            optimizer.step()
            batch_count = int(len(batch_labels))
            total_loss += float(loss.detach()) * batch_count
            total_examples += batch_count
            predictions.append(logits.detach().argmax(dim=1).cpu().numpy().astype(np.int64, copy=False))
            truths.append(batch_labels.detach().cpu().numpy().astype(np.int64, copy=False))

        train_classification = compute_classification_metrics(
            np.concatenate(truths), np.concatenate(predictions), class_names=artifact.class_names
        )
        should_evaluate = (epoch_index + 1) % settings.evaluation_interval == 0 or epoch_index + 1 == settings.head_epochs
        validation: HeadEvaluation | None = None
        if should_evaluate:
            validation = _evaluate_quantum_head(
                head,
                features=validation_features,
                labels=validation_labels,
                batch_size=settings.head_batch_size,
                classes=artifact.num_classes,
                class_names=artifact.class_names,
            )
            score = (float(validation.classification["macro_f1"] or 0.0), -validation.loss)
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch_index + 1
                best_validation = validation
                torch.save(
                    _checkpoint_payload(
                        head=head, artifact=artifact, settings=settings, epoch=best_epoch, validation=validation
                    ),
                    best_path,
                )

        epoch_seconds = time.perf_counter() - epoch_started
        hardware_sample = telemetry.snapshot(event="epoch_end", phase=f"epoch_{epoch_index + 1}")
        record = {
            "epoch": epoch_index + 1,
            "train": {
                "loss": total_loss / total_examples,
                "classification": train_classification,
            },
            "validation": None
            if validation is None
            else {
                "loss": validation.loss,
                "classification": validation.classification,
                "quantum_measurements": validation.quantum,
            },
            "quantum_parameters": _parameter_summary(
                head, theta_before=theta_before, theta_gradient_norms=theta_gradient_norms
            ),
            "timing": {
                "epoch_seconds": epoch_seconds,
                "examples_per_second": total_examples / epoch_seconds if epoch_seconds else None,
                "optimizer_updates": int(np.ceil(total_examples / settings.head_batch_size)),
            },
            "hardware": _hardware_digest(hardware_sample),
        }
        history.append(record)
        torch.save(
            _checkpoint_payload(
                head=head, artifact=artifact, settings=settings, epoch=epoch_index + 1, validation=validation
            ),
            last_path,
        )
        atomic_write_json(history_path, {"epochs": history})
        _print_epoch_progress(epoch_index + 1, settings.head_epochs, record)

    if best_validation is None:
        raise RuntimeError("Nenhuma avaliação de validação foi produzida.")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    head.load_state_dict(best_checkpoint["state_dict"])
    return head, history, best_epoch, best_validation, {"history": history_path, "best": best_path, "last": last_path}


def _run_quantum_smoke_legacy(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_root: str | Path | None = None,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    settings: SmokeSettings = SmokeSettings(),
) -> dict[str, Any]:
    """Train a QCNN head using a selected, frozen phase-one CNN checkpoint."""

    settings.validate()
    name = str(dataset_name).strip().lower()
    dataset = _load_selected_dataset(name, dataset_root=dataset_root, registry_path=registry_path)
    if dataset.name != name:
        raise FrozenModelError(f"O adaptador carregou {dataset.name!r}, esperado {name!r}.")
    artifact, selection = select_frozen_model(
        dataset=name,
        model_root=model_root,
        preferred_dtype=settings.preferred_dtype,
        preferred_activation=settings.preferred_activation,
        strict_profile=settings.strict_model_profile,
    )
    _validate_artifact_for_dataset(artifact, dataset)
    extractor = load_frozen_feature_extractor(artifact)

    raw_samples, raw_labels = _training_samples(dataset)
    samples, labels = _balanced_local_subset(
        raw_samples,
        raw_labels,
        samples_per_class=settings.samples_per_class,
        seed=settings.seed,
        classes=dataset.num_classes,
    )
    (train_samples, train_labels), (validation_samples, validation_labels), (test_samples, test_labels) = _split_smoke_data(
        samples, labels, seed=settings.seed
    )
    balanced = balance_training_data(train_samples, train_labels, mode=EXPECTED_BALANCE_MODE, seed=settings.seed)
    if not balanced.is_noop or not np.array_equal(balanced.labels, train_labels):
        raise RuntimeError("A execução quântica deve usar balance_mode=all_raw sem alterar o treino.")

    dataset_args = {
        "image_size": artifact.image_size,
        "channels": artifact.channels,
        "batch_size": settings.head_batch_size,
        "seed": settings.seed,
    }
    train_dataset = _make_dataset(balanced.samples, balanced.labels, training=False, **dataset_args)
    validation_dataset = _make_dataset(validation_samples, validation_labels, training=False, **dataset_args)
    test_dataset = _make_dataset(test_samples, test_labels, training=False, **dataset_args)

    train_features, train_targets = _extract_features(extractor, train_dataset)
    validation_features, validation_targets = _extract_features(extractor, validation_dataset)
    _, test_targets = _extract_features(extractor, test_dataset)
    q_train_features, q_train_labels = _take_per_class(
        train_features, train_targets, count=settings.head_train_per_class, classes=dataset.num_classes
    )
    q_validation_features, q_validation_labels = _take_per_class(
        validation_features, validation_targets, count=settings.head_validation_per_class, classes=dataset.num_classes
    )
    head, losses, validation_logits = _train_quantum_head(
        settings.backend,
        train_features=q_train_features,
        train_labels=q_train_labels,
        validation_features=q_validation_features,
        validation_labels=q_validation_labels,
        classes=dataset.num_classes,
        settings=settings,
    )

    destination = (
        Path(output_root)
        / dataset.name
        / f"{settings.backend}__{artifact.dtype_policy}_{artifact.hidden_activation}_{artifact.balance_mode}__seed-{settings.seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    metrics = compute_classification_metrics(q_validation_labels, validation_logits, class_names=dataset.class_names)
    report_paths = write_classification_report(destination / "head-report", metrics)
    import torch

    head_checkpoint = destination / "qcnn-head.pt"
    torch.save(
        {
            "backend": settings.backend,
            "n_classes": dataset.num_classes,
            "state_dict": head.state_dict(),
            "settings": dataclasses.asdict(settings),
            "frozen_cnn": artifact.to_dict(),
        },
        head_checkpoint,
    )
    result = {
        "status": "passed",
        "scope": "Teste local da CNN treinada e congelada -> QCNN -> Dense; não mede acurácia final.",
        "dataset": dataset.name,
        "frozen_cnn": {
            **artifact.to_dict(),
            "selection": selection,
            "feature_layer": FEATURE_LAYER_NAME,
            "feature_shape": [int(train_features.shape[0]), int(train_features.shape[1])],
            "frozen_trainable_variables": len(extractor.trainable_variables),
        },
        "quantum_head": {
            "backend": settings.backend,
            "n_qubits": 8,
            "input_features": N_FEATURES,
            "quantum_parameters": 51,
            "trainable_parameters": [name for name, _ in head.named_parameters()],
            "train_examples": int(len(q_train_labels)),
            "validation_examples": int(len(q_validation_labels)),
            "losses": losses,
            "checkpoint": str(head_checkpoint.resolve()),
        },
        "metrics": metrics,
        "artifacts": {key: str(path.resolve()) for key, path in report_paths.items()},
        "unused_test_split_examples": int(len(test_targets)),
        "settings": dataclasses.asdict(settings),
    }
    result_path = destination / "smoke-result.json"
    result["artifacts"]["smoke_result"] = str(result_path.resolve())
    atomic_write_json(result_path, result)
    return result


def run_quantum_smoke(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_root: str | Path | None = None,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    settings: SmokeSettings = SmokeSettings(),
) -> dict[str, Any]:
    """Run a frozen-CNN QCNN experiment in ``smoke`` or complete ``full`` mode.

    Full mode preserves every input from the selected dataset's train partition,
    applies the existing 70/15/15 stratified split and reports final test
    metrics.  Smoke mode keeps the historical small per-class subset behaviour.
    """

    settings.validate()
    name = str(dataset_name).strip().lower()
    dataset = _load_selected_dataset(name, dataset_root=dataset_root, registry_path=registry_path)
    if dataset.name != name:
        raise FrozenModelError(f"Checkpoint esperado para {name!r}, mas o adaptador carregou {dataset.name!r}.")
    artifact, selection = select_frozen_model(
        dataset=name,
        model_root=model_root,
        preferred_dtype=settings.preferred_dtype,
        preferred_activation=settings.preferred_activation,
        strict_profile=settings.strict_model_profile,
    )
    _validate_artifact_for_dataset(artifact, dataset)
    folder_name = f"{settings.mode}__{settings.backend}__{artifact.dtype_policy}_{artifact.hidden_activation}_{artifact.balance_mode}__seed-{settings.seed}"
    destination = Path(output_root) / dataset.name / folder_name
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "run-status.json"
    atomic_write_json(
        status_path,
        {
            "status": "running",
            "mode": settings.mode,
            "dataset": dataset.name,
            "seed": settings.seed,
            "started_at": time.time(),
        },
    )
    telemetry = TelemetrySampler(
        destination / "telemetry",
        interval_seconds=settings.telemetry_interval_seconds,
        data_path=dataset.source_path,
    ).start()
    try:
        extractor = load_frozen_feature_extractor(artifact)
        raw_samples, raw_labels = _training_samples(dataset)
        if settings.mode == "full":
            samples, labels = raw_samples, raw_labels
            data_selection = {
                "mode": "all_raw_full_train_partition",
                "source_train_examples": int(len(raw_labels)),
                "per_class_cap": None,
                "balance_transform_applied": False,
            }
        else:
            samples, labels = _balanced_local_subset(
                raw_samples,
                raw_labels,
                samples_per_class=settings.samples_per_class,
                seed=settings.seed,
                classes=dataset.num_classes,
            )
            data_selection = {
                "mode": "smoke_subset",
                "source_train_examples": int(len(raw_labels)),
                "selected_examples": int(len(labels)),
                "samples_per_class": settings.samples_per_class,
                "balance_transform_applied": False,
            }
        (train_samples, train_labels), (validation_samples, validation_labels), (test_samples, test_labels) = _split_smoke_data(
            samples, labels, seed=settings.seed
        )
        balanced = balance_training_data(train_samples, train_labels, mode=EXPECTED_BALANCE_MODE, seed=settings.seed)
        if not balanced.is_noop or not np.array_equal(balanced.labels, train_labels):
            raise RuntimeError("A execução quântica deve preservar all_raw sem alterar o treino.")

        dataset_args = {
            "image_size": artifact.image_size,
            "channels": artifact.channels,
            "batch_size": settings.head_batch_size,
            "seed": settings.seed,
        }
        telemetry.snapshot(event="feature_extraction_start", phase="train")
        print("Extraindo vetores congelados da CNN para o treino...", flush=True)
        train_dataset = _make_dataset(balanced.samples, balanced.labels, training=False, **dataset_args)
        train_features, train_targets = _extract_features(extractor, train_dataset)
        telemetry.snapshot(event="feature_extraction_end", phase="train")
        telemetry.snapshot(event="feature_extraction_start", phase="validation")
        print("Extraindo vetores congelados da CNN para a validação...", flush=True)
        validation_dataset = _make_dataset(validation_samples, validation_labels, training=False, **dataset_args)
        validation_features, validation_targets = _extract_features(extractor, validation_dataset)
        telemetry.snapshot(event="feature_extraction_end", phase="validation")
        telemetry.snapshot(event="feature_extraction_start", phase="test")
        print("Extraindo vetores congelados da CNN para o teste...", flush=True)
        test_dataset = _make_dataset(test_samples, test_labels, training=False, **dataset_args)
        test_features, test_targets = _extract_features(extractor, test_dataset)
        telemetry.snapshot(event="feature_extraction_end", phase="test")

        if settings.mode == "smoke":
            q_train_features, q_train_labels = _take_per_class(
                train_features, train_targets, count=settings.head_train_per_class, classes=dataset.num_classes
            )
            q_validation_features, q_validation_labels = _take_per_class(
                validation_features,
                validation_targets,
                count=settings.head_validation_per_class,
                classes=dataset.num_classes,
            )
        else:
            q_train_features, q_train_labels = train_features, train_targets
            q_validation_features, q_validation_labels = validation_features, validation_targets

        telemetry.snapshot(event="quantum_training_start", phase="training")
        head, history, best_epoch, best_validation, training_paths = _train_complete_quantum_head(
            artifact=artifact,
            train_features=q_train_features,
            train_labels=q_train_labels,
            validation_features=q_validation_features,
            validation_labels=q_validation_labels,
            settings=settings,
            destination=destination,
            telemetry=telemetry,
        )
        telemetry.snapshot(event="quantum_test_start", phase="test")
        test_evaluation = _evaluate_quantum_head(
            head,
            features=test_features,
            labels=test_targets,
            batch_size=settings.head_batch_size,
            classes=dataset.num_classes,
            class_names=dataset.class_names,
        )
        telemetry.snapshot(event="quantum_test_end", phase="test")

        validation_metrics = {
            **best_validation.classification,
            "cross_entropy_loss": best_validation.loss,
            "quantum_measurements": best_validation.quantum,
            "checkpoint_epoch": best_epoch,
        }
        test_metrics = {
            **test_evaluation.classification,
            "cross_entropy_loss": test_evaluation.loss,
            "quantum_measurements": test_evaluation.quantum,
            "checkpoint_epoch": best_epoch,
        }
        validation_report = write_classification_report(destination / "validation-report", validation_metrics)
        test_report = write_classification_report(destination / "test-report", test_metrics)
        telemetry_artifacts = telemetry.stop(final_event="train_end")
        result = {
            "status": "passed",
            "scope": "Experimento completo: CNN congelada -> QCNN -> Dense; métricas de treino/validação por época e teste final.",
            "dataset": dataset.name,
            "data_selection": data_selection,
            "splits": {
                "train_examples": int(len(train_targets)),
                "validation_examples": int(len(validation_targets)),
                "test_examples": int(len(test_targets)),
                "quantum_train_examples": int(len(q_train_labels)),
                "quantum_validation_examples": int(len(q_validation_labels)),
            },
            "frozen_cnn": {
                **artifact.to_dict(),
                "selection": selection,
                "feature_layer": FEATURE_LAYER_NAME,
                "feature_shapes": {
                    "train": list(map(int, train_features.shape)),
                    "validation": list(map(int, validation_features.shape)),
                    "test": list(map(int, test_features.shape)),
                },
                "frozen_trainable_variables": len(extractor.trainable_variables),
            },
            "quantum_head": {
                "backend": settings.backend,
                "configuration": head.quantum_metadata(),
                "trainable_parameters": [name for name, _ in head.named_parameters()],
                "trainable_parameter_count": int(sum(parameter.numel() for parameter in head.parameters())),
                "forward_circuit_evaluations": head.quantum_forward_samples,
                "forward_batches": head.quantum_forward_batches,
                "best_epoch": best_epoch,
                "best_validation": validation_metrics,
                "test": test_metrics,
            },
            "epoch_history": history,
            "settings": dataclasses.asdict(settings),
            "artifacts": {
                **{key: str(path.resolve()) for key, path in training_paths.items()},
                **{f"validation_{key}": str(path.resolve()) for key, path in validation_report.items()},
                **{f"test_{key}": str(path.resolve()) for key, path in test_report.items()},
                "telemetry_hardware": telemetry_artifacts["hardware"],
                "telemetry_samples": telemetry_artifacts["samples"],
                "telemetry_summary": telemetry_artifacts["summary"],
            },
        }
        result_path = destination / "experiment-result.json"
        result["artifacts"]["experiment_result"] = str(result_path.resolve())
        atomic_write_json(result_path, result)
        atomic_write_json(
            status_path,
            {
                "status": "completed",
                "mode": settings.mode,
                "dataset": dataset.name,
                "seed": settings.seed,
                "result": str(result_path.resolve()),
            },
        )
        return result
    except BaseException as exc:
        telemetry_artifacts = telemetry.stop(final_event="run_failed")
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "mode": settings.mode,
                "dataset": dataset.name,
                "seed": settings.seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "telemetry": telemetry_artifacts,
            },
        )
        raise


def run_kmnist_smoke(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    settings: SmokeSettings = SmokeSettings(),
) -> dict[str, Any]:
    """Backward-compatible KMNIST wrapper around :func:`run_quantum_smoke`."""

    return run_quantum_smoke(
        dataset_name="kmnist",
        dataset_root=dataset_root,
        model_root=model_root,
        output_root=output_root,
        settings=settings,
    )


def _resolve_seed(value: int | str) -> int:
    if str(value).strip().lower() == "random":
        return secrets.randbelow(2_147_483_646) + 1
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed deve ser um inteiro não negativo ou 'random'.") from exc
    if seed < 0:
        raise ValueError("seed deve ser um inteiro não negativo ou 'random'.")
    return seed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_ORDER, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Sobrescreve o caminho definido no registro.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_DATASET_REGISTRY)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", choices=("pennylane", "qiskit"), default="pennylane")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--full-experiment", dest="mode", action="store_const", const="full")
    parser.add_argument("--samples-per-class", type=int, default=12, help="Usado somente no modo smoke.")
    parser.add_argument("--epochs", "--head-epochs", dest="head_epochs", type=int, default=1)
    parser.add_argument("--batch-size", "--head-batch-size", dest="head_batch_size", type=int, default=2)
    parser.add_argument("--head-train-per-class", type=int, default=1, help="Usado somente no modo smoke.")
    parser.add_argument("--head-validation-per-class", type=int, default=1, help="Usado somente no modo smoke.")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--evaluation-interval", type=int, default=1, help="Avalia validação a cada N épocas.")
    parser.add_argument("--telemetry-interval-seconds", type=float, default=5.0)
    parser.add_argument("--preferred-dtype", default="float32", help="Preferência de perfil; use 'any' para ignorar.")
    parser.add_argument("--preferred-activation", default="relu", help="Preferência de perfil; use 'any' para ignorar.")
    parser.add_argument("--strict-model-profile", action="store_true", help="Exige perfil exato de dtype e ativação.")
    parser.add_argument("--seed", default="random", help="Inteiro reprodutível ou 'random'.")
    args = parser.parse_args(argv)
    try:
        seed = _resolve_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    settings = SmokeSettings(
        backend=args.backend,
        seed=seed,
        mode=args.mode,
        samples_per_class=args.samples_per_class,
        head_epochs=args.head_epochs,
        head_batch_size=args.head_batch_size,
        head_train_per_class=args.head_train_per_class,
        head_validation_per_class=args.head_validation_per_class,
        learning_rate=args.learning_rate,
        evaluation_interval=args.evaluation_interval,
        telemetry_interval_seconds=args.telemetry_interval_seconds,
        preferred_dtype=args.preferred_dtype,
        preferred_activation=args.preferred_activation,
        strict_model_profile=args.strict_model_profile,
    )
    try:
        result = run_quantum_smoke(
            dataset_name=args.dataset,
            dataset_root=args.dataset_root,
            registry_path=args.registry,
            model_root=args.model_root,
            output_root=args.output_root,
            settings=settings,
        )
    except Exception as exc:
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
