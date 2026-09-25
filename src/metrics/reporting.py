"""Métricas de classificação portáveis para qualquer braço do benchmark."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.experiment import atomic_write_json


def _as_python(value: Any) -> Any:
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    return value


def _labels(values: Any) -> list[Any]:
    """Accept indices, ``(n, 1)`` labels, one-hot labels or score matrices."""

    values = _as_python(values)
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        return [values]
    result: list[Any] = []
    for value in values:
        value = _as_python(value)
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                result.append(value[0])
            elif len(value) == 0:
                result.append(None)
            else:
                result.append(int(np.argmax(np.asarray(value, dtype=float))))
        else:
            result.append(value)
    return result


def _sort_label(value: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except (TypeError, ValueError):
        return (1, str(value))


def compute_classification_metrics(
    y_true: Any, y_pred: Any, *, class_names: Sequence[str] | Mapping[Any, str] | None = None
) -> dict[str, Any]:
    """Calculate accuracy, balanced accuracy, macro-F1 and a confusion matrix.

    The implementation intentionally has no scikit-learn dependency, keeping
    a completed experiment inspectable on the same lightweight environment.
    """

    actual, predicted = _labels(y_true), _labels(y_pred)
    if len(actual) != len(predicted):
        raise ValueError(f"y_true ({len(actual)}) e y_pred ({len(predicted)}) têm tamanhos diferentes")
    label_names: dict[Any, str] = {}
    declared: list[Any] = []
    if isinstance(class_names, Mapping):
        declared.extend(class_names)
        label_names = {key: str(value) for key, value in class_names.items()}
    elif class_names is not None:
        declared.extend(range(len(class_names)))
        label_names = {index: str(name) for index, name in enumerate(class_names)}
    labels = sorted(set(declared + actual + predicted), key=_sort_label)
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(actual, predicted, strict=True):
        matrix[index[truth], index[prediction]] += 1

    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(true_positive, predicted_count, out=np.zeros(len(labels)), where=predicted_count != 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(labels)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels)), where=(precision + recall) != 0)
    names = [label_names.get(label, str(label)) for label in labels]
    per_class = [
        {"label": label, "class_name": names[position], "support": int(support[position]),
         "precision": float(precision[position]), "recall": float(recall[position]), "f1": float(f1[position])}
        for position, label in enumerate(labels)
    ]
    total = int(matrix.sum())
    observed = support > 0
    return {
        "n_samples": total,
        "accuracy": None if total == 0 else float(np.trace(matrix) / total),
        "balanced_accuracy": None if not np.any(observed) else float(np.mean(recall[observed])),
        "macro_precision": None if not len(precision) else float(np.mean(precision)),
        "macro_recall": None if not len(recall) else float(np.mean(recall)),
        "macro_f1": None if not len(f1) else float(np.mean(f1)),
        "labels": labels,
        "class_names": names,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def write_classification_report(destination: str | Path, metrics: Mapping[str, Any]) -> dict[str, Path]:
    """Persist the model-neutral result as JSON plus one easy-to-open CSV."""

    directory = Path(destination)
    json_path = atomic_write_json(directory / "metrics.json", dict(metrics))
    rows = list(metrics.get("per_class", []))
    buffer = io.StringIO(newline="")
    fields = ("label", "class_name", "support", "precision", "recall", "f1")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    csv_path = directory / "per_class.csv"
    csv_path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return {"metrics": json_path, "per_class": csv_path}


__all__ = ["compute_classification_metrics", "write_classification_report"]
