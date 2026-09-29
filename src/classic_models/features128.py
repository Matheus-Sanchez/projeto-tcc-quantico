"""Export the frozen convolutional representation before every dense layer.

The exported vector is exactly::

    image -> frozen backbone -> block5_pool -> GlobalAveragePooling2D -> 128

No Dense20, normalization, activation or classifier output is included.  The
exporter reuses the audited dataset split from the classical campaign and
writes deterministic, non-augmented NumPy arrays for downstream heads.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from classic_models.dense20 import (
    BATCH_SIZE,
    DEFAULT_MODEL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_NORMALIZATION,
    FEATURE_LAYER_NAME,
    SEED,
    DatasetMaterials,
    Dense20Error,
    FrozenBackboneArtifact,
    _atomic_save_npy,
    _load_backbone,
    _load_dataset_materials,
    _require_tensorflow_runtime,
    _run_id,
    _runtime_preflight,
    _validate_all_models,
    _validate_materials,
    _weights_fingerprint,
    discover_strict_fp32_backbone,
)
from data_prep.data import build_tf_dataset, select_samples, stratified_split_indices
from logs.telemetry import TelemetrySampler
from utils.experiment import DATASET_ORDER, DEFAULT_DATASET_REGISTRY
from utils.state import atomic_write_json


FEATURE_WIDTH = 128
EXPORT_SCHEMA_VERSION = 1


class Features128Error(Dense20Error):
    """The pure convolutional feature export cannot be completed safely."""


@dataclasses.dataclass(frozen=True)
class Features128Settings:
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    telemetry_interval_seconds: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "telemetry_interval_seconds": self.telemetry_interval_seconds,
            "normalization": EXPECTED_NORMALIZATION,
            "training": False,
            "augmentation": False,
            "shuffle": False,
            "deterministic": True,
            "output_dtype": "float32",
            "feature_definition": [
                FEATURE_LAYER_NAME,
                "GlobalAveragePooling2D",
            ],
            "feature_width": FEATURE_WIDTH,
        }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Features128Error(f"JSON inválido: {path}") from exc
    if not isinstance(value, dict):
        raise Features128Error(f"JSON precisa conter um objeto: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_fingerprint(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_backbone_encoder128(tf: Any, extractor: Any) -> Any:
    """Build an inference-only 128-D encoder with no dense layers."""

    extractor.trainable = False
    for layer in extractor.layers:
        layer.trainable = False
    inputs = tf.keras.layers.Input(
        shape=tuple(extractor.input_shape[1:]),
        dtype="float32",
        name="image",
    )
    convolutional = extractor(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(
        dtype="float32",
        name="features128",
    )(convolutional)
    encoder = tf.keras.Model(inputs, pooled, name="frozen_backbone_encoder128")
    if tuple(encoder.output_shape) != (None, FEATURE_WIDTH):
        raise Features128Error(
            f"Encoder convolucional deveria produzir (None, {FEATURE_WIDTH}); recebeu {encoder.output_shape}"
        )
    if encoder.trainable_variables:
        raise Features128Error("O encoder128 possui variáveis treináveis; o backbone não está congelado.")
    return encoder


def extract_features128(
    tf: Any,
    encoder: Any,
    dataset: Any,
    *,
    expected_examples: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one deterministic split and validate its 128-D contract."""

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    progress = tf.keras.utils.Progbar(expected_examples, unit_name="imagem")
    seen = 0
    print(f"    {label}: {expected_examples} imagens")
    for images, targets in dataset:
        values = np.asarray(encoder(images, training=False).numpy(), dtype=np.float32)
        target_values = np.asarray(targets.numpy(), dtype=np.int64).reshape(-1)
        if values.ndim != 2 or values.shape[1] != FEATURE_WIDTH:
            raise Features128Error(f"Batch {label} possui shape inválido: {values.shape}")
        if values.shape[0] != target_values.shape[0]:
            raise Features128Error(f"Features e labels desalinhadas em {label}.")
        features.append(values)
        labels.append(target_values)
        seen += int(values.shape[0])
        progress.update(seen)
    if not features:
        raise Features128Error(f"Split vazio durante a exportação: {label}")
    joined_features = np.concatenate(features).astype(np.float32, copy=False)
    joined_labels = np.concatenate(labels).astype(np.int64, copy=False)
    if joined_features.shape != (expected_examples, FEATURE_WIDTH):
        raise Features128Error(
            f"Shape final inválido em {label}: {joined_features.shape} != {(expected_examples, FEATURE_WIDTH)}"
        )
    if joined_labels.shape != (expected_examples,):
        raise Features128Error(
            f"Shape de labels inválido em {label}: {joined_labels.shape} != {(expected_examples,)}"
        )
    return joined_features, joined_labels


def _dense20_run_root(output_root: Path, dataset: str) -> Path:
    root = output_root / dataset / "runs" / _run_id(dataset)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise Features128Error(f"Run Dense20 não encontrada para {dataset}: {root}")
    manifest = _read_json(manifest_path)
    if str(manifest.get("status", "")).lower() != "completed":
        raise Features128Error(f"Run Dense20 de {dataset} não está completed: {manifest.get('status')}")
    if str(manifest.get("dataset", "")).lower() != dataset:
        raise Features128Error(f"Manifest da run Dense20 pertence a outro dataset: {manifest_path}")
    return root


def _existing_export_is_valid(destination: Path, split_fingerprint: str) -> bool:
    manifest_path = destination / "manifest.json"
    status_path = destination / "status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        status = _read_json(status_path)
        if status.get("status") != "completed":
            return False
        if manifest.get("split_fingerprint") != split_fingerprint:
            return False
        if int(manifest.get("feature_width", 0)) != FEATURE_WIDTH:
            return False
        for name in ("train", "val", "test"):
            entry = manifest.get("splits", {}).get(name, {})
            features_path = destination / f"{name}_features.npy"
            labels_path = destination / f"{name}_labels.npy"
            if not features_path.is_file() or not labels_path.is_file():
                return False
            features = np.load(features_path, mmap_mode="r", allow_pickle=False)
            labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
            if tuple(features.shape) != tuple(entry.get("features_shape", ())):
                return False
            if tuple(labels.shape) != tuple(entry.get("labels_shape", ())):
                return False
            if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
                return False
            if features.dtype != np.float32 or labels.dtype != np.int64:
                return False
        return True
    except (OSError, ValueError, TypeError, Features128Error):
        return False


def _split_dataset(
    materials: DatasetMaterials,
    artifact: FrozenBackboneArtifact,
    settings: Features128Settings,
    indices: np.ndarray,
) -> tuple[Any, Any]:
    samples = select_samples(materials.samples, indices)
    labels = materials.labels[indices]
    return build_tf_dataset(
        samples,
        labels,
        image_size=artifact.image_size,
        channels=artifact.channels,
        batch_size=settings.batch_size,
        training=False,
        extra_fraction=0.0,
        seed=settings.seed,
        shuffle=False,
        deterministic=True,
        output_dtype="float32",
        repeat=False,
    )


def _export_one(
    tf: Any,
    artifact: FrozenBackboneArtifact,
    materials: DatasetMaterials,
    run_root: Path,
    settings: Features128Settings,
    *,
    overwrite: bool,
) -> str:
    destination = run_root / "artifacts" / "features128"
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    if not overwrite and _existing_export_is_valid(destination, artifact.split_fingerprint):
        print(f"[{artifact.dataset}] features128 já concluídas e válidas; ignorado.")
        return "skipped"

    started = time.perf_counter()
    atomic_write_json(status_path, {
        "status": "running",
        "dataset": artifact.dataset,
        "started_at": _utc_now(),
    })
    telemetry: TelemetrySampler | None = None
    try:
        telemetry_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        telemetry = TelemetrySampler(
            destination / "telemetry" / telemetry_id,
            interval_seconds=settings.telemetry_interval_seconds,
            disk_paths={
                "data": Path(materials.source_manifest.get("source_path", artifact.manifest)),
                "output": destination,
            },
        ).start()
        telemetry.snapshot(event="encoder_load_start", phase="model")
        extractor, model_profile = _load_backbone(tf, artifact)
        backbone_before = _weights_fingerprint(extractor)
        encoder = build_backbone_encoder128(tf, extractor)
        telemetry.snapshot(event="encoder_load_end", phase="model")

        split = stratified_split_indices(materials.labels, seed=settings.seed)
        if split.fingerprint() != artifact.split_fingerprint:
            raise Features128Error("Fingerprint mudou antes da exportação features128.")
        split_entries = (
            ("train", np.asarray(split.train, dtype=np.int64)),
            ("val", np.asarray(split.validation, dtype=np.int64)),
            ("test", np.asarray(split.test, dtype=np.int64)),
        )
        exported: dict[str, Any] = {}
        print(f"[{artifact.dataset}] extraindo representação convolucional pura de 128 dimensões")
        for name, indices in split_entries:
            telemetry.snapshot(event="split_export_start", phase=name)
            dataset, info = _split_dataset(materials, artifact, settings, indices)
            features, labels = extract_features128(
                tf,
                encoder,
                dataset,
                expected_examples=len(indices),
                label=name,
            )
            features_path = destination / f"{name}_features.npy"
            labels_path = destination / f"{name}_labels.npy"
            _atomic_save_npy(features_path, features)
            _atomic_save_npy(labels_path, labels)
            exported[name] = {
                "examples": int(info.total_examples),
                "indices_sha256": _indices_fingerprint(indices),
                "features": str(features_path.resolve()),
                "features_shape": list(features.shape),
                "features_dtype": str(features.dtype),
                "features_sha256": _sha256_file(features_path),
                "labels": str(labels_path.resolve()),
                "labels_shape": list(labels.shape),
                "labels_dtype": str(labels.dtype),
                "labels_sha256": _sha256_file(labels_path),
            }
            telemetry.snapshot(event="split_export_end", phase=name)
            del dataset, features, labels
            gc.collect()

        backbone_after = _weights_fingerprint(extractor)
        if backbone_after != backbone_before:
            raise Features128Error("Pesos do backbone mudaram durante a extração features128.")
        encoder_path = run_root / "artifacts" / "encoder128.keras"
        encoder.save(encoder_path)
        telemetry_summary = telemetry.stop(final_event="export_completed")
        telemetry = None
        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "status": "completed",
            "dataset": artifact.dataset,
            "created_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "feature_width": FEATURE_WIDTH,
            "feature_semantics": "Saída de block5_pool após GlobalAveragePooling2D; nenhuma camada densa incluída.",
            "source_checkpoint": str(artifact.checkpoint),
            "source_checkpoint_sha256": _sha256_file(artifact.checkpoint),
            "dense20_run": str(run_root.resolve()),
            "encoder128": str(encoder_path.resolve()),
            "encoder128_sha256": _sha256_file(encoder_path),
            "split_fingerprint": artifact.split_fingerprint,
            "class_names": list(artifact.class_names),
            "model_profile": model_profile,
            "settings": settings.to_dict(),
            "backbone_weights_before": backbone_before,
            "backbone_weights_after": backbone_after,
            "backbone_unchanged": True,
            "splits": exported,
            "telemetry": telemetry_summary,
        }
        atomic_write_json(destination / "manifest.json", manifest)
        atomic_write_json(status_path, {
            "status": "completed",
            "dataset": artifact.dataset,
            "completed_at": _utc_now(),
            "elapsed_seconds": manifest["elapsed_seconds"],
        })
        print(f"[{artifact.dataset}] features128 concluídas em {manifest['elapsed_seconds']:.1f}s")
        return "completed"
    except KeyboardInterrupt:
        if telemetry is not None:
            telemetry.stop(final_event="export_interrupted")
        atomic_write_json(status_path, {
            "status": "interrupted",
            "dataset": artifact.dataset,
            "updated_at": _utc_now(),
        })
        raise
    except Exception as exc:
        if telemetry is not None:
            telemetry.stop(final_event="export_failed")
        atomic_write_json(status_path, {
            "status": "failed",
            "dataset": artifact.dataset,
            "updated_at": _utc_now(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _prepare_one(
    dataset: str,
    *,
    registry: Path,
    model_root: Path,
    output_root: Path,
) -> tuple[FrozenBackboneArtifact, DatasetMaterials, Path]:
    artifact = discover_strict_fp32_backbone(dataset, model_root)
    materials = _load_dataset_materials(dataset, registry, artifact)
    _validate_materials(artifact, materials)
    run_root = _dense20_run_root(output_root, dataset)
    run_manifest = _read_json(run_root / "manifest.json")
    if run_manifest.get("split_fingerprint") != artifact.split_fingerprint:
        raise Features128Error(f"Fingerprint da run Dense20 diverge para {dataset}.")
    return artifact, materials, run_root


def _write_consolidated_manifest(output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/runs/*/artifacts/features128/manifest.json")):
        manifest = _read_json(path)
        rows.append({
            "dataset": manifest["dataset"],
            "status": manifest["status"],
            "feature_width": manifest["feature_width"],
            "split_fingerprint": manifest["split_fingerprint"],
            "backbone_unchanged": manifest["backbone_unchanged"],
            "train_examples": manifest["splits"]["train"]["examples"],
            "validation_examples": manifest["splits"]["val"]["examples"],
            "test_examples": manifest["splits"]["test"]["examples"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "manifest": str(path.resolve()),
        })
    rows.sort(key=lambda row: DATASET_ORDER.index(str(row["dataset"])))
    destination = output_root / "features128_manifest.json"
    atomic_write_json(destination, {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "feature_width": FEATURE_WIDTH,
        "datasets": rows,
    })
    return destination


def verify_exports(output_root: Path, datasets: Sequence[str]) -> dict[str, Any]:
    """Open every exported array and verify the reusable feature contract."""

    rows: list[dict[str, Any]] = []
    total_rows = 0
    feature_bytes = 0
    label_bytes = 0
    telemetry_samples = 0
    for dataset in datasets:
        run_root = _dense20_run_root(output_root, dataset)
        dense_manifest = _read_json(run_root / "manifest.json")
        destination = run_root / "artifacts" / "features128"
        if not _existing_export_is_valid(destination, str(dense_manifest.get("split_fingerprint", ""))):
            raise Features128Error(f"Exportação features128 inválida ou incompleta: {dataset}")
        manifest = _read_json(destination / "manifest.json")
        if manifest.get("backbone_weights_before") != manifest.get("backbone_weights_after"):
            raise Features128Error(f"Hash do backbone mudou em {dataset}.")
        if manifest.get("feature_semantics") != (
            "Saída de block5_pool após GlobalAveragePooling2D; nenhuma camada densa incluída."
        ):
            raise Features128Error(f"Semântica de features inesperada em {dataset}.")
        checkpoint = Path(str(manifest["source_checkpoint"]))
        encoder_path = Path(str(manifest["encoder128"]))
        if _sha256_file(checkpoint) != manifest.get("source_checkpoint_sha256"):
            raise Features128Error(f"Hash do checkpoint fonte diverge em {dataset}.")
        if _sha256_file(encoder_path) != manifest.get("encoder128_sha256"):
            raise Features128Error(f"Hash do encoder128 diverge em {dataset}.")
        split_rows: dict[str, int] = {}
        for name in ("train", "val", "test"):
            entry = manifest["splits"][name]
            features_path = destination / f"{name}_features.npy"
            labels_path = destination / f"{name}_labels.npy"
            features = np.load(features_path, mmap_mode="r", allow_pickle=False)
            labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
            previous_labels = np.load(
                run_root / "artifacts" / "features20" / f"{name}_labels.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            if features.dtype != np.float32 or tuple(features.shape) != tuple(entry["features_shape"]):
                raise Features128Error(f"Features {dataset}/{name} não cumprem shape/dtype do manifest.")
            if labels.dtype != np.int64 or tuple(labels.shape) != tuple(entry["labels_shape"]):
                raise Features128Error(f"Labels {dataset}/{name} não cumprem shape/dtype do manifest.")
            if not np.array_equal(labels, previous_labels):
                raise Features128Error(f"Ordem dos labels mudou em {dataset}/{name}.")
            if not bool(np.isfinite(features).all()):
                raise Features128Error(f"Features não finitas em {dataset}/{name}.")
            if _sha256_file(features_path) != entry.get("features_sha256"):
                raise Features128Error(f"Hash das features diverge em {dataset}/{name}.")
            if _sha256_file(labels_path) != entry.get("labels_sha256"):
                raise Features128Error(f"Hash dos labels diverge em {dataset}/{name}.")
            split_rows[name] = int(features.shape[0])
            total_rows += int(features.shape[0])
            feature_bytes += features_path.stat().st_size
            label_bytes += labels_path.stat().st_size
        telemetry = manifest.get("telemetry", {})
        for key in ("hardware", "samples", "summary"):
            if not Path(str(telemetry.get(key, ""))).is_file():
                raise Features128Error(f"Telemetria {key} ausente em {dataset}.")
        telemetry_samples += int(telemetry.get("sample_count", 0))
        rows.append({
            "dataset": dataset,
            "status": "verified",
            "feature_width": FEATURE_WIDTH,
            "split_fingerprint": manifest["split_fingerprint"],
            "backbone_unchanged": True,
            "labels_match_original_splits": True,
            "all_values_finite": True,
            "splits": split_rows,
            "telemetry_samples": int(telemetry.get("sample_count", 0)),
        })
    return {
        "verified_at": _utc_now(),
        "ok": True,
        "dataset_count": len(rows),
        "feature_width": FEATURE_WIDTH,
        "total_rows": total_rows,
        "feature_storage_mib": feature_bytes / 1024**2,
        "label_storage_mib": label_bytes / 1024**2,
        "telemetry_samples": telemetry_samples,
        "datasets": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m classic_models.features128",
        description="Exporta a representação convolucional GAP de 128 dimensões, antes de qualquer Dense.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASET_ORDER)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_DATASET_REGISTRY)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size < 1 or args.telemetry_interval_seconds <= 0:
        print("Erro: batch-size e telemetry-interval-seconds devem ser positivos.", file=sys.stderr)
        return 2
    selected = list(DATASET_ORDER) if args.all else [args.dataset]
    registry = args.registry.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    settings = Features128Settings(
        batch_size=int(args.batch_size),
        telemetry_interval_seconds=float(args.telemetry_interval_seconds),
    )
    if args.verify_only:
        try:
            verification = verify_exports(output_root, selected)
            destination = output_root / "features128_verification.json"
            atomic_write_json(destination, verification)
            print(json.dumps(verification, ensure_ascii=False, indent=2))
            print(f"Verificação salva em: {destination}")
            return 0
        except Exception as exc:
            print(f"Verificação features128 falhou: {exc}", file=sys.stderr)
            return 1
    prepared: list[tuple[FrozenBackboneArtifact, Path]] = []
    failures = 0
    for dataset in selected:
        try:
            artifact, materials, run_root = _prepare_one(
                dataset,
                registry=registry,
                model_root=model_root,
                output_root=output_root,
            )
            prepared.append((artifact, run_root))
            print(
                f"[{dataset}] validado: {len(materials.labels)} imagens, "
                f"split={artifact.split_fingerprint[:12]}…, saída=(*,{FEATURE_WIDTH})"
            )
            del materials
            gc.collect()
        except Exception as exc:
            failures += 1
            print(f"[{dataset}] validação falhou: {exc}", file=sys.stderr)
            if args.fail_fast:
                return 1
    if args.dry_run:
        atomic_write_json(output_root / "features128_dry_run.json", {
            "timestamp": _utc_now(),
            "ok": failures == 0 and len(prepared) == len(selected),
            "datasets": [artifact.dataset for artifact, _ in prepared],
            "failures": failures,
            "settings": settings.to_dict(),
        })
        return 1 if failures else 0
    if failures:
        print("Preflight estático falhou; nenhuma feature será exportada.", file=sys.stderr)
        return 1

    try:
        tf = _require_tensorflow_runtime()
        runtime = _runtime_preflight(tf, output_root)
        runtime["purpose"] = "features128_export"
        runtime["validated_models"] = _validate_all_models(tf, [item[0] for item in prepared])
        atomic_write_json(output_root / "features128_preflight.json", runtime)
    except Exception as exc:
        atomic_write_json(output_root / "features128_preflight.json", {
            "timestamp": _utc_now(),
            "ok": False,
            "error": repr(exc),
        })
        print(f"Preflight de runtime falhou: {exc}", file=sys.stderr)
        return 1

    for artifact, run_root in prepared:
        try:
            materials = _load_dataset_materials(artifact.dataset, registry, artifact)
            _validate_materials(artifact, materials)
            _export_one(
                tf,
                artifact,
                materials,
                run_root,
                settings,
                overwrite=bool(args.overwrite),
            )
        except KeyboardInterrupt:
            print("Exportação interrompida pelo usuário.", file=sys.stderr)
            return 130
        except Exception as exc:
            failures += 1
            print(f"[{artifact.dataset}] exportação falhou: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
    consolidated = _write_consolidated_manifest(output_root)
    print(f"Manifest consolidado: {consolidated}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
