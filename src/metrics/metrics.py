"""Classification metrics and Keras callbacks for classical runs."""

from __future__ import annotations

import csv
import dataclasses
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data_prep.data import require_tensorflow

try:
    import tensorflow as _tf
except Exception:
    _tf = None


def _as_1d_integer(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] > 1:
        array = array.argmax(axis=1)
    if array.ndim != 1:
        raise ValueError(f"{name} deve ser unidimensional; recebido shape={array.shape}.")
    try:
        result = array.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} precisa conter rótulos inteiros.") from exc
    if not np.array_equal(result, array):
        raise ValueError(f"{name} precisa conter rótulos inteiros.")
    return result


@dataclasses.dataclass(frozen=True)
class PerClassMetrics:
    label: int
    support: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_ovr_auc: float | None
    per_class: dict[int, PerClassMetrics]
    confusion_matrix: np.ndarray

    def to_dict(self, *, include_confusion_matrix: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_precision": float(self.macro_precision),
            "macro_recall": float(self.macro_recall),
            "macro_f1": float(self.macro_f1),
            "macro_ovr_auc": float(self.macro_ovr_auc) if self.macro_ovr_auc is not None else None,
            "per_class": {str(label): values.to_dict() for label, values in self.per_class.items()},
        }
        if include_confusion_matrix:
            payload["confusion_matrix"] = self.confusion_matrix.astype(int).tolist()
        return payload


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred_or_probabilities: Sequence[int] | np.ndarray,
    *,
    num_classes: int | None = None,
) -> ClassificationMetrics:
    truth = _as_1d_integer(y_true, name="y_true")
    raw_predictions = np.asarray(y_pred_or_probabilities)
    if raw_predictions.ndim == 2:
        predictions = raw_predictions.argmax(axis=1).astype(np.int64)
        inferred_classes = int(raw_predictions.shape[1])
    else:
        predictions = _as_1d_integer(raw_predictions, name="y_pred")
        inferred_classes = int(max(truth.max(initial=0), predictions.max(initial=0)) + 1)
    if len(truth) != len(predictions):
        raise ValueError("y_true e y_pred possuem tamanhos diferentes.")
    if not len(truth):
        raise ValueError("Não é possível calcular métricas de uma avaliação vazia.")
    classes = int(num_classes) if num_classes is not None else inferred_classes
    if classes < 1:
        raise ValueError("num_classes deve ser positivo.")
    if np.any(truth < 0) or np.any(predictions < 0) or np.any(truth >= classes) or np.any(predictions >= classes):
        raise ValueError("Rótulos/predições estão fora do intervalo [0, num_classes).")

    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    class_values: dict[int, PerClassMetrics] = {}
    f1_values: list[float] = []
    precision_values: list[float] = []
    recalls: list[float] = []
    for label in range(classes):
        true_positive = int(matrix[label, label])
        false_positive = int(matrix[:, label].sum() - true_positive)
        false_negative = int(matrix[label, :].sum() - true_positive)
        support = int(matrix[label, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_values[label] = PerClassMetrics(
            label=label, support=support, precision=float(precision), recall=float(recall), f1=float(f1)
        )
        f1_values.append(float(f1))
        precision_values.append(float(precision))
        recalls.append(float(recall))
    macro_ovr_auc: float | None = None
    if raw_predictions.ndim == 2:
        try:
            from sklearn.metrics import roc_auc_score
            probabilities = np.asarray(raw_predictions, dtype=np.float64)
            macro_ovr_auc = float(roc_auc_score(truth, probabilities, labels=np.arange(classes), multi_class="ovr", average="macro"))
        except (ImportError, ValueError):
            macro_ovr_auc = None
    return ClassificationMetrics(
        accuracy=float(np.trace(matrix) / len(truth)),
        balanced_accuracy=float(np.mean(recalls)),
        macro_precision=float(np.mean(precision_values)),
        macro_recall=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1_values)),
        macro_ovr_auc=macro_ovr_auc,
        per_class=class_values,
        confusion_matrix=matrix,
    )


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    keras_metrics: dict[str, float]
    classification: ClassificationMetrics
    y_true: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray


def _split_batch(batch: Any) -> tuple[Any, Any]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("O dataset de avaliação deve produzir (imagem, rótulo) ou (imagem, rótulo, peso).")
    return batch[0], batch[1]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def collect_model_outputs(model: Any, dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_tensorflow()
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch in dataset:
        images, targets = _split_batch(batch)
        predicted = model(images, training=False)
        target_values = np.asarray(targets.numpy() if hasattr(targets, "numpy") else targets)
        if target_values.ndim == 2 and target_values.shape[-1] > 1:
            target_values = target_values.argmax(axis=1)
        labels.append(np.asarray(target_values, dtype=np.int64).reshape(-1))
        logits.append(np.asarray(predicted.numpy() if hasattr(predicted, "numpy") else predicted, dtype=np.float32))
    if not labels:
        raise ValueError("O dataset de avaliação não contém lotes.")
    joined_logits = np.concatenate(logits)
    return np.concatenate(labels), joined_logits, _softmax(joined_logits)


def collect_predictions(model: Any, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    labels, _logits, probabilities = collect_model_outputs(model, dataset)
    return labels, probabilities


def evaluate_model(model: Any, dataset: Any, *, num_classes: int) -> EvaluationResult:
    require_tensorflow()
    evaluated = model.evaluate(dataset, verbose=0, return_dict=True)
    keras_metrics = {str(key): float(value) for key, value in evaluated.items()}
    y_true, logits, probabilities = collect_model_outputs(model, dataset)
    return EvaluationResult(
        keras_metrics=keras_metrics,
        classification=classification_metrics(y_true, probabilities, num_classes=num_classes),
        y_true=y_true,
        logits=logits,
        probabilities=probabilities,
    )


def _current_learning_rate(model: Any) -> float:
    tensorflow = require_tensorflow()
    optimizer = model.optimizer
    value = optimizer.learning_rate
    if callable(value):
        value = value(optimizer.iterations)
    return float(tensorflow.keras.backend.get_value(value))


def _float_logs(logs: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in logs.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _flatten_per_class(metrics: ClassificationMetrics, *, prefix: str = "val") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for label, item in metrics.per_class.items():
        flattened[f"{prefix}_class_{label}_precision"] = float(item.precision)
        flattened[f"{prefix}_class_{label}_recall"] = float(item.recall)
        flattened[f"{prefix}_class_{label}_f1"] = float(item.f1)
        flattened[f"{prefix}_class_{label}_support"] = float(item.support)
    return flattened


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically persist epoch history so interruptions leave valid CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "epoch", "loss", "accuracy", "val_loss", "val_accuracy",
        "val_balanced_accuracy", "val_macro_f1", "learning_rate",
        "epoch_seconds", "train_examples", "train_examples_per_second",
    ]
    all_keys = {str(key) for row in rows for key in row}
    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames.extend(sorted(all_keys - set(fieldnames)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


if _tf is not None:

    class EpochTimingCallback(_tf.keras.callbacks.Callback):
        def __init__(self, *, train_examples_per_epoch: int | None = None) -> None:
            super().__init__()
            self._started_at: float | None = None
            self.train_examples_per_epoch = int(train_examples_per_epoch) if train_examples_per_epoch is not None else None

        def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            self._started_at = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            if logs is None:
                return
            epoch_seconds = float(time.perf_counter() - self._started_at) if self._started_at is not None else float("nan")
            logs["epoch_seconds"] = epoch_seconds
            logs["learning_rate"] = _current_learning_rate(self.model)
            if self.train_examples_per_epoch is not None:
                logs["train_examples"] = float(self.train_examples_per_epoch)
                logs["train_examples_per_second"] = (
                    float(self.train_examples_per_epoch) / epoch_seconds if epoch_seconds > 0 else float("nan")
                )

    class EpochMetricsCallback(EpochTimingCallback):
        def __init__(
            self,
            validation_data: Any,
            *,
            num_classes: int,
            history_path: str | Path | None = None,
            train_examples_per_epoch: int | None = None,
        ) -> None:
            super().__init__(train_examples_per_epoch=train_examples_per_epoch)
            self.validation_data = validation_data
            self.num_classes = int(num_classes)
            self.history_path = Path(history_path) if history_path is not None else None
            self.rows: list[dict[str, Any]] = []

        def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
            if self.history_path is not None:
                self.rows = _read_history(self.history_path)

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            super().on_epoch_end(epoch, logs)
            if logs is None:
                logs = {}
            y_true, probabilities = collect_predictions(self.model, self.validation_data)
            metrics = classification_metrics(y_true, probabilities, num_classes=self.num_classes)
            logs["val_balanced_accuracy"] = float(metrics.balanced_accuracy)
            logs["val_macro_f1"] = float(metrics.macro_f1)
            row: dict[str, Any] = {
                "epoch": int(epoch) + 1,
                **_float_logs(logs),
                **_flatten_per_class(metrics, prefix="val"),
            }
            # BackupAndRestore may replay the last completed epoch. Replace it
            # instead of duplicating a CSV row.
            self.rows = [item for item in self.rows if int(float(item.get("epoch", -1))) != int(epoch) + 1]
            self.rows.append(row)
            self.rows.sort(key=lambda item: int(float(item["epoch"])))
            if self.history_path is not None:
                _write_history(self.history_path, self.rows)

    class ValidationMacroF1Callback(EpochMetricsCallback):
        pass

else:
    class _TensorFlowRequiredCallback:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_tensorflow()

    EpochTimingCallback = _TensorFlowRequiredCallback
    EpochMetricsCallback = _TensorFlowRequiredCallback
    ValidationMacroF1Callback = _TensorFlowRequiredCallback
