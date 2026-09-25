"""Dataset integrity audits for the local benchmark adapters.

The audit is deliberately independent of TensorFlow.  It can be run before a
GPU environment exists, which is useful for rejecting broken data paths and
corrupt images before a long benchmark matrix begins.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .adapters import DatasetFormatError, LoadedDataset, LocalDatasetNotFoundError, SampleRecord, load_image_file


def _json_value(value: Any) -> Any:
    """Convert NumPy/path values recursively so report dictionaries are JSON safe."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class AuditIssue:
    """A bounded, serializable integrity problem found during audit."""

    severity: str
    category: str
    message: str
    split: str | None = None
    sample_id: str | None = None
    path: str | None = None
    label: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
        }
        if self.split is not None:
            payload["split"] = self.split
        if self.sample_id is not None:
            payload["sample_id"] = self.sample_id
        if self.path is not None:
            payload["path"] = self.path
        if self.label is not None:
            payload["label"] = int(self.label)
        return payload


@dataclass
class AuditReport:
    """Complete serializable audit result.

    ``duplicate_groups`` use a pixel-content SHA-256: equal pixels, dtype and
    shape produce the same hash even when two image files use different image
    encodings.  The report calls these *exact decoded duplicates* to make that
    distinction explicit.
    """

    dataset: str
    source_path: str
    class_names: list[str]
    started_at_utc: str
    completed_at_utc: str | None = None
    requested_samples: int | None = None
    checked_samples: int = 0
    total_samples: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    shape_distribution: dict[str, int] = field(default_factory=dict)
    channel_distribution: dict[str, int] = field(default_factory=dict)
    dtype_distribution: dict[str, int] = field(default_factory=dict)
    missing_files: list[dict[str, Any]] = field(default_factory=list)
    unreadable_images: list[dict[str, Any]] = field(default_factory=list)
    invalid_labels: list[dict[str, Any]] = field(default_factory=list)
    invalid_image_shapes: list[dict[str, Any]] = field(default_factory=list)
    duplicate_group_count: int = 0
    duplicate_sample_count: int = 0
    duplicate_excess_count: int = 0
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)
    error_count_total: int = 0
    warning_count_total: int = 0
    truncated: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return int(self.error_count_total)

    @property
    def warning_count(self) -> int:
        return int(self.warning_count_total)

    @property
    def is_healthy(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "dataset": self.dataset,
                "source_path": self.source_path,
                "class_names": self.class_names,
                "started_at_utc": self.started_at_utc,
                "completed_at_utc": self.completed_at_utc,
                "requested_samples": self.requested_samples,
                "checked_samples": self.checked_samples,
                "total_samples": self.total_samples,
                "split_counts": self.split_counts,
                "label_counts": self.label_counts,
                "shape_distribution": self.shape_distribution,
                "channel_distribution": self.channel_distribution,
                "dtype_distribution": self.dtype_distribution,
                "missing_files": self.missing_files,
                "unreadable_images": self.unreadable_images,
                "invalid_labels": self.invalid_labels,
                "invalid_image_shapes": self.invalid_image_shapes,
                "duplicates": {
                    "definition": "SHA-256 over decoded pixel dtype, shape and bytes",
                    "group_count": self.duplicate_group_count,
                    "sample_count": self.duplicate_sample_count,
                    "excess_sample_count": self.duplicate_excess_count,
                    "groups": self.duplicate_groups,
                },
                "issues": [issue.to_dict() for issue in self.issues],
                "summary": {
                    "healthy": self.is_healthy,
                    "error_count": self.error_count,
                    "warning_count": self.warning_count,
                    "truncated": self.truncated,
                },
                "options": self.options,
            }
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sample_id(split: str, index: int, record: SampleRecord | None) -> str:
    if record is not None:
        return f"{split}:{index}:{record.path}"
    return f"{split}:{index}"


def _shape_key(image: np.ndarray) -> tuple[str | None, str | None]:
    """Return ``(shape-key, channels)`` and leave invalid images detectable."""
    if image.ndim == 2:
        return f"{image.shape[0]}x{image.shape[1]}x1", "1"
    if image.ndim == 3:
        return "x".join(str(value) for value in image.shape), str(image.shape[-1])
    return None, None


def _pixel_hash(image: np.ndarray) -> str:
    value = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii", errors="replace"))
    digest.update(b"\0")
    digest.update(",".join(str(int(dimension)) for dimension in value.shape).encode("ascii", errors="replace"))
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _append_bounded(target: list[dict[str, Any]], value: dict[str, Any], maximum: int) -> None:
    if len(target) < maximum:
        target.append(value)


def _add_issue(report: AuditReport, issue: AuditIssue, *, maximum: int) -> None:
    if issue.severity == "error":
        report.error_count_total += 1
    elif issue.severity == "warning":
        report.warning_count_total += 1
    if len(report.issues) < maximum:
        report.issues.append(issue)


def _iter_dataset_samples(dataset: LoadedDataset) -> Iterable[tuple[str, int, np.ndarray | None, SampleRecord | None]]:
    """Yield native arrays or records without prematurely decoding folder images."""
    for split_name, partition in dataset.splits.items():
        if partition.images is not None and partition.labels is not None:
            for index, (image, label) in enumerate(zip(partition.images, partition.labels, strict=True)):
                yield split_name, index, np.asarray(image), None
        offset = 0 if partition.images is None else len(partition.images)
        for index, record in enumerate(partition.records, start=offset):
            yield split_name, index, None, record


def audit_dataset(
    dataset: LoadedDataset,
    *,
    verify_images: bool = True,
    hash_images: bool = True,
    max_samples: int | None = None,
    max_issue_examples: int = 200,
    max_duplicate_groups: int = 200,
    max_duplicate_members: int = 25,
) -> AuditReport:
    """Inspect local dataset content and return a JSON-serializable report.

    The default processes every sample. ``max_samples`` is a diagnostic shortcut
    for very large datasets and intentionally marks the report as truncated.
    Hashing requires decoding path-backed images; setting both ``verify_images``
    and ``hash_images`` to false provides a labels-only audit.
    """
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    started = _utc_now()
    report = AuditReport(
        dataset=dataset.name,
        source_path=str(dataset.source_path),
        class_names=list(dataset.class_names),
        started_at_utc=started,
        requested_samples=max_samples,
        total_samples=dataset.sample_count(),
        split_counts={name: len(partition) for name, partition in dataset.splits.items()},
        options={
            "verify_images": bool(verify_images),
            "hash_images": bool(hash_images),
            "max_samples": max_samples,
            "hash_definition": "decoded-pixel-content",
        },
    )
    labels_by_split: dict[str, Counter[int]] = defaultdict(Counter)
    shapes: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sequence, (split, index, array_image, record) in enumerate(_iter_dataset_samples(dataset)):
        if max_samples is not None and sequence >= max_samples:
            report.truncated = True
            break
        sample_id = _sample_id(split, index, record)
        label = int(record.label) if record is not None else int(dataset.splits[split].labels[index])  # type: ignore[index]
        report.checked_samples += 1
        labels_by_split[split][label] += 1
        if label < 0 or label >= dataset.num_classes:
            detail = {
                "sample_id": sample_id,
                "split": split,
                "path": str(record.path) if record else None,
                "label": label,
                "expected_range": [0, dataset.num_classes - 1],
            }
            _append_bounded(report.invalid_labels, detail, max_issue_examples)
            _add_issue(
                report,
                AuditIssue(
                    "error", "invalid_label", f"Label {label} is outside 0..{dataset.num_classes - 1}.",
                    split, sample_id, str(record.path) if record else None, label,
                ),
                maximum=max_issue_examples,
            )

        image = array_image
        if record is not None and (verify_images or hash_images):
            try:
                image = load_image_file(record.path)
            except LocalDatasetNotFoundError as exc:
                detail = {"sample_id": sample_id, "split": split, "path": str(record.path), "error": str(exc)}
                _append_bounded(report.missing_files, detail, max_issue_examples)
                _add_issue(report, AuditIssue("error", "missing_file", str(exc), split, sample_id, str(record.path), label), maximum=max_issue_examples)
                continue
            except DatasetFormatError as exc:
                detail = {"sample_id": sample_id, "split": split, "path": str(record.path), "error": str(exc)}
                _append_bounded(report.unreadable_images, detail, max_issue_examples)
                _add_issue(report, AuditIssue("error", "unreadable_image", str(exc), split, sample_id, str(record.path), label), maximum=max_issue_examples)
                continue
            except Exception as exc:  # defensive: audit should not abort on a single decoder failure
                detail = {"sample_id": sample_id, "split": split, "path": str(record.path), "error": repr(exc)}
                _append_bounded(report.unreadable_images, detail, max_issue_examples)
                _add_issue(report, AuditIssue("error", "unreadable_image", repr(exc), split, sample_id, str(record.path), label), maximum=max_issue_examples)
                continue

        if image is None:
            # Paths were intentionally not opened in labels-only mode.
            continue
        image = np.asarray(image)
        shape_key, channel_count = _shape_key(image)
        if shape_key is None:
            detail = {
                "sample_id": sample_id,
                "split": split,
                "path": str(record.path) if record else None,
                "shape": list(image.shape),
            }
            _append_bounded(report.invalid_image_shapes, detail, max_issue_examples)
            _add_issue(
                report,
                AuditIssue("error", "invalid_image_shape", f"Expected rank 2/3 image; got shape {image.shape}.", split, sample_id, str(record.path) if record else None, label),
                maximum=max_issue_examples,
            )
            continue
        shapes[shape_key] += 1
        channels[channel_count or "unknown"] += 1
        dtypes[str(image.dtype)] += 1
        if image.ndim == 3 and image.shape[-1] not in {1, 3}:
            detail = {
                "sample_id": sample_id,
                "split": split,
                "path": str(record.path) if record else None,
                "shape": list(image.shape),
                "channels": int(image.shape[-1]),
            }
            _append_bounded(report.invalid_image_shapes, detail, max_issue_examples)
            _add_issue(
                report,
                AuditIssue("warning", "unusual_channel_count", f"Image has {image.shape[-1]} channels.", split, sample_id, str(record.path) if record else None, label),
                maximum=max_issue_examples,
            )
        if hash_images:
            sample_ref = {
                "sample_id": sample_id,
                "split": split,
                "path": str(record.path) if record else None,
                "label": label,
            }
            hashes[_pixel_hash(image)].append(sample_ref)

    report.label_counts = {
        split: {str(label): int(count) for label, count in sorted(counter.items())}
        for split, counter in sorted(labels_by_split.items())
    }
    report.shape_distribution = {key: int(value) for key, value in sorted(shapes.items())}
    report.channel_distribution = {key: int(value) for key, value in sorted(channels.items(), key=lambda item: item[0])}
    report.dtype_distribution = {key: int(value) for key, value in sorted(dtypes.items())}

    if hash_images:
        all_groups = [(digest, members) for digest, members in hashes.items() if len(members) > 1]
        all_groups.sort(key=lambda item: (-len(item[1]), item[0]))
        report.duplicate_group_count = len(all_groups)
        report.duplicate_sample_count = sum(len(members) for _, members in all_groups)
        report.duplicate_excess_count = sum(len(members) - 1 for _, members in all_groups)
        for digest, members in all_groups[:max_duplicate_groups]:
            label_counts = Counter(member["label"] for member in members)
            report.duplicate_groups.append(
                {
                    "sha256": digest,
                    "count": len(members),
                    "labels": {str(label): int(count) for label, count in sorted(label_counts.items())},
                    "cross_label": len(label_counts) > 1,
                    "members": members[:max_duplicate_members],
                    "members_truncated": len(members) > max_duplicate_members,
                }
            )
        if all_groups:
            _add_issue(
                report,
                AuditIssue(
                    "warning",
                    "exact_decoded_duplicates",
                    f"Found {report.duplicate_group_count} duplicate pixel-content group(s), involving "
                    f"{report.duplicate_sample_count} sample(s).",
                ),
                maximum=max_issue_examples,
            )
        if any(len({member["label"] for member in members}) > 1 for _, members in all_groups):
            _add_issue(
                report,
                AuditIssue(
                    "error",
                    "duplicate_with_conflicting_labels",
                    "At least one exact decoded duplicate appears with different labels.",
                ),
                maximum=max_issue_examples,
            )
    report.completed_at_utc = _utc_now()
    return report


def audit_local_dataset(name: str, path: str | Path, **options: Any) -> AuditReport:
    """Load then audit one local dataset; loader options are forwarded verbatim."""
    from .adapters import load_local_dataset

    audit_keys = {
        "verify_images", "hash_images", "max_samples", "max_issue_examples",
        "max_duplicate_groups", "max_duplicate_members",
    }
    audit_options = {key: value for key, value in options.items() if key in audit_keys}
    loader_options = {key: value for key, value in options.items() if key not in audit_keys}
    return audit_dataset(load_local_dataset(name, path, **loader_options), **audit_options)


def write_audit_report(report: AuditReport | Mapping[str, Any], destination: str | Path) -> Path:
    """Write a stable UTF-8 JSON report and return its resolved path."""
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if isinstance(report, AuditReport) else _json_value(dict(report))
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target.resolve()


__all__ = [
    "AuditIssue",
    "AuditReport",
    "audit_dataset",
    "audit_local_dataset",
    "write_audit_report",
]
