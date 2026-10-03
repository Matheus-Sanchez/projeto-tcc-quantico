"""Train the fixed 128 -> 128 -> 20(tanh) -> C head on saved features128.

This runner never loads images or convolutional models. Its only inputs are
the deterministic ``features128`` NumPy arrays exported immediately after the
frozen CNN and GlobalAveragePooling2D. No normalization, augmentation, PCA, or
feature selection is applied here.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import io
import json
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from logs.telemetry import TelemetrySampler
from metrics.metrics import EvaluationResult, ValidationMacroF1Callback, evaluate_model
from data_prep.registry import DATASET_ORDER
from utils.state import atomic_write_bytes, atomic_write_json, atomic_write_text, config_fingerprint, utc_now


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_ROOT = PROJECT_ROOT / "outputs" / "classic-dense20"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classical-128-20-mac"

from utils.experiment_config import (
    SEED, INPUT_FEATURES, PROJECTION_UNITS, HIDDEN_UNITS, BATCH_SIZE,
    LEARNING_RATE, MAX_EPOCHS, NUM_CLASSES, VectorHeadSettings, require_runtime,
)
from data_prep.feature_vectors import (
    FeatureBundle, FeatureVectorsError, REQUIRED_FILES, discover_features128,
    load_feature_bundle, _read_json, _sha256_file, _atomic_save_npy,
)

# Keep the existing public names and exception catch points.
VectorDense20Error = FeatureVectorsError
VectorDense20Settings = VectorHeadSettings


def configure_reproducibility(tf: Any, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def build_classical_dense20(
    tf: Any,
    *,
    num_classes: int,
    learning_rate: float = LEARNING_RATE,
) -> Any:
    """Build the exact 128 -> 128(ReLU) -> 20(tanh) -> C classical head."""

    inputs = tf.keras.Input(shape=(INPUT_FEATURES,), dtype=tf.float32, name="features128")
    projection = tf.keras.layers.Dense(
        PROJECTION_UNITS,
        activation="relu",
        name="dense_128_128",
    )(inputs)
    hidden_linear = tf.keras.layers.Dense(
        HIDDEN_UNITS,
        activation=None,
        name="dense_128_20",
    )(projection)
    hidden = tf.keras.layers.Activation("tanh", name="dense_20_tanh")(hidden_linear)
    logits = tf.keras.layers.Dense(
        int(num_classes),
        activation=None,
        dtype="float32",
        name="logits",
    )(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="classical_128_20")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )
    expected = 19_092 + 21 * int(num_classes)
    actual = int(sum(np.prod(variable.shape) for variable in model.trainable_variables))
    if actual != expected:
        raise VectorDense20Error(f"Contagem treinável incorreta: {actual} != {expected}.")
    if model.output_shape != (None, int(num_classes)):
        raise VectorDense20Error(f"Shape de saída incorreto: {model.output_shape}.")
    if tf.keras.mixed_precision.global_policy().name != "float32":
        raise VectorDense20Error("A política global precisa permanecer float32.")
    return model


def build_vector_dataset(
    tf: Any,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    training: bool,
    seed: int,
) -> Any:
    x_values = np.asarray(features, dtype=np.float32)
    y_values = np.asarray(labels, dtype=np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((x_values, y_values))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(x_values),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(int(batch_size), drop_remainder=False)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)


def _history_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _callbacks(
    tf: Any,
    validation_ds: Any,
    *,
    num_classes: int,
    run_dir: Path,
    train_examples: int,
) -> list[Any]:
    history_path = run_dir / "history.csv"
    metrics = ValidationMacroF1Callback(
        validation_ds,
        num_classes=num_classes,
        history_path=history_path,
        train_examples_per_epoch=train_examples,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "best.keras"),
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
    )
    previous = [
        float(row["val_macro_f1"])
        for row in _history_rows(history_path)
        if row.get("val_macro_f1")
    ]
    if previous:
        checkpoint.best = max(previous)
    return [
        metrics,
        checkpoint,
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(run_dir / "backup"),
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
    fields = ["sample", "y_true", "y_pred", *[
        f"probability_{index}" for index in range(evaluation.probabilities.shape[1])
    ]]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    predicted = evaluation.probabilities.argmax(axis=1)
    for index, (truth, prediction, probabilities) in enumerate(
        zip(evaluation.y_true, predicted, evaluation.probabilities, strict=True)
    ):
        row: dict[str, Any] = {
            "sample": index,
            "y_true": int(truth),
            "y_pred": int(prediction),
        }
        row.update({f"probability_{label}": float(value) for label, value in enumerate(probabilities)})
        writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def _classification_report(evaluation: EvaluationResult) -> dict[str, Any]:
    metrics = evaluation.classification
    return {
        "accuracy": metrics.accuracy,
        "macro_avg": {
            "precision": metrics.macro_precision,
            "recall": metrics.macro_recall,
            "f1": metrics.macro_f1,
        },
        "classes": {
            str(label): {
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
                "support": item.support,
            }
            for label, item in metrics.per_class.items()
        },
    }


def _timing_summary(history_path: Path, *, fit_seconds: float, evaluation_seconds: float) -> dict[str, Any]:
    rows = _history_rows(history_path)
    epoch_seconds = [float(row["epoch_seconds"]) for row in rows if row.get("epoch_seconds")]
    throughput = [
        float(row["train_examples_per_second"])
        for row in rows if row.get("train_examples_per_second")
    ]
    best_row = max(rows, key=lambda row: float(row.get("val_macro_f1", "-inf")), default=None)
    return {
        "epochs_completed": len(rows),
        "fit_seconds_current_attempt": float(fit_seconds),
        "total_epoch_seconds": float(sum(epoch_seconds)),
        "mean_epoch_seconds": float(np.mean(epoch_seconds)) if epoch_seconds else None,
        "mean_train_examples_per_second": float(np.mean(throughput)) if throughput else None,
        "evaluation_seconds": float(evaluation_seconds),
        "best_epoch": int(float(best_row["epoch"])) if best_row else None,
        "best_val_macro_f1": float(best_row["val_macro_f1"]) if best_row else None,
    }


def _run_directory(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / f"seed-{SEED}"


def _config(bundle: FeatureBundle, settings: VectorDense20Settings) -> dict[str, Any]:
    return {
        "experiment": "classical_128_relu_20_tanh_features128",
        "platform": platform.system().lower(),
        "dataset": bundle.dataset,
        "num_classes": bundle.num_classes,
        **settings.to_dict(),
        "source_features": str(bundle.directory),
        "source_feature_hashes": dict(bundle.hashes),
        "split_fingerprint": bundle.split_fingerprint,
        "transformations": {
            "augmentation": False,
            "normalization": False,
            "scaler": False,
            "pca": False,
            "feature_selection": False,
        },
    }


def _initialize_run(run_dir: Path, bundle: FeatureBundle, settings: VectorDense20Settings) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _config(bundle, settings)
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.json"
    existing_config = _read_json(config_path)
    if existing_config and config_fingerprint(existing_config) != config_fingerprint(config):
        raise VectorDense20Error(
            f"Configuração existente diverge em {run_dir}; use outro --output-root."
        )
    if not existing_config:
        atomic_write_json(config_path, config)
    manifest = _read_json(manifest_path)
    if not manifest:
        atomic_write_json(manifest_path, {
            "status": "pending",
            "dataset": bundle.dataset,
            "created_at": utc_now(),
            "config_fingerprint": config_fingerprint(config),
            "source": bundle.source_manifest,
        })
    atomic_write_json(run_dir / "feature_statistics.json", dict(bundle.statistics))


def _set_status(run_dir: Path, status: str, message: str, **extra: Any) -> None:
    payload = {
        "status": status,
        "dataset": run_dir.parent.name,
        "timestamp": utc_now(),
        "message": message,
        **extra,
    }
    atomic_write_json(run_dir / "status.json", payload)
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(payload)
    atomic_write_json(manifest_path, manifest)


def _run_one(
    tf: Any,
    bundle: FeatureBundle,
    run_dir: Path,
    settings: VectorDense20Settings,
    *,
    resume: bool,
) -> str:
    manifest = _read_json(run_dir / "manifest.json")
    status = str(manifest.get("status", "pending"))
    if status == "completed":
        print(f"[{bundle.dataset}] já concluído; ignorado.")
        return "skipped"
    if status in {"running", "interrupted", "failed"} and not resume:
        raise VectorDense20Error(f"Run {status} já existe em {run_dir}; use --resume.")

    telemetry: TelemetrySampler | None = None
    started = time.perf_counter()
    _set_status(run_dir, "running", "Treinamento da ponta Dense128-ReLU-Dense20-tanh em andamento")
    try:
        telemetry = TelemetrySampler(
            run_dir / "telemetry",
            interval_seconds=settings.telemetry_interval_seconds,
            disk_paths={"features": bundle.directory, "output": run_dir},
        ).start()
        configure_reproducibility(tf, settings.seed)
        telemetry.snapshot(event="dataset_build_start", phase="data")
        train_ds = build_vector_dataset(
            tf,
            bundle.features["train"],
            bundle.labels["train"],
            batch_size=settings.batch_size,
            training=True,
            seed=settings.seed,
        )
        validation_ds = build_vector_dataset(
            tf,
            bundle.features["val"],
            bundle.labels["val"],
            batch_size=settings.batch_size,
            training=False,
            seed=settings.seed,
        )
        telemetry.snapshot(event="dataset_build_end", phase="data")

        model = build_classical_dense20(
            tf,
            num_classes=bundle.num_classes,
            learning_rate=settings.learning_rate,
        )
        summary: list[str] = []
        model.summary(print_fn=summary.append)
        atomic_write_text(run_dir / "model_summary.txt", "\n".join(summary) + "\n")

        callbacks = _callbacks(
            tf,
            validation_ds,
            num_classes=bundle.num_classes,
            run_dir=run_dir,
            train_examples=len(bundle.features["train"]),
        )
        telemetry.snapshot(event="fit_start", phase="training")
        fit_started = time.perf_counter()
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=settings.max_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        fit_seconds = time.perf_counter() - fit_started
        telemetry.snapshot(event="fit_end", phase="training")
        model.save(run_dir / "final.keras")

        best_path = run_dir / "best.keras"
        if not best_path.is_file():
            raise VectorDense20Error("best.keras não foi produzido.")
        selected = tf.keras.models.load_model(best_path, compile=True)

        # The test dataset is instantiated only after checkpoint selection.
        test_ds = build_vector_dataset(
            tf,
            bundle.features["test"],
            bundle.labels["test"],
            batch_size=settings.batch_size,
            training=False,
            seed=settings.seed,
        )
        telemetry.snapshot(event="test_start", phase="test")
        evaluation_started = time.perf_counter()
        evaluation = evaluate_model(selected, test_ds, num_classes=bundle.num_classes)
        evaluation_seconds = time.perf_counter() - evaluation_started
        telemetry.snapshot(event="test_end", phase="test")

        test_payload = _evaluation_payload(evaluation)
        atomic_write_json(run_dir / "test_metrics.json", test_payload)
        atomic_write_json(run_dir / "classification_report.json", _classification_report(evaluation))
        _atomic_save_npy(
            run_dir / "confusion_matrix.npy",
            evaluation.classification.confusion_matrix.astype(np.int64),
        )
        _atomic_save_npy(run_dir / "test_logits.npy", evaluation.logits.astype(np.float32))
        _write_predictions(run_dir / "predictions.csv", evaluation)
        timing = _timing_summary(
            run_dir / "history.csv",
            fit_seconds=fit_seconds,
            evaluation_seconds=evaluation_seconds,
        )
        timing["wall_seconds_current_attempt"] = float(time.perf_counter() - started)
        atomic_write_json(run_dir / "timing.json", timing)
        telemetry_summary = telemetry.stop(final_event="run_completed")
        telemetry = None

        manifest = _read_json(run_dir / "manifest.json")
        manifest.update({
            "status": "completed",
            "completed_at": utc_now(),
            "trainable_parameters": 19_092 + 21 * bundle.num_classes,
            "selected_checkpoint": str(best_path.resolve()),
            "timing": timing,
            "test_metrics": test_payload,
            "telemetry": telemetry_summary,
        })
        atomic_write_json(run_dir / "manifest.json", manifest)
        _set_status(run_dir, "completed", "Treino e avaliação final concluídos")
        print(
            f"[{bundle.dataset}] concluído: "
            f"Macro-F1={evaluation.classification.macro_f1:.6f}"
        )
        return "completed"
    except KeyboardInterrupt:
        if telemetry is not None:
            telemetry.stop(final_event="interrupted")
        _set_status(run_dir, "interrupted", "Interrompido; backup preservado")
        raise
    except Exception as exc:
        if telemetry is not None:
            telemetry.stop(final_event="failed")
        atomic_write_text(run_dir / "error.txt", traceback.format_exc())
        _set_status(run_dir, "failed", "Falha no treino da ponta clássica", error=repr(exc))
        raise
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _write_aggregate(output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/seed-42/test_metrics.json")):
        metrics = _read_json(path)
        classification = metrics.get("classification", {})
        timing = _read_json(path.parent / "timing.json")
        rows.append({
            "dataset": path.parent.parent.name,
            "accuracy": classification.get("accuracy"),
            "balanced_accuracy": classification.get("balanced_accuracy"),
            "macro_precision": classification.get("macro_precision"),
            "macro_recall": classification.get("macro_recall"),
            "macro_f1": classification.get("macro_f1"),
            "macro_auc": classification.get("macro_ovr_auc"),
            "loss": metrics.get("keras_metrics", {}).get("loss"),
            "best_epoch": timing.get("best_epoch"),
            "training_seconds": timing.get("total_epoch_seconds"),
        })
    rows.sort(key=lambda row: DATASET_ORDER.index(str(row["dataset"])))
    destination = output_root / "classical_128_20_results.csv"
    fields = [
        "dataset", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "macro_auc", "loss", "best_epoch", "training_seconds",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(destination, buffer.getvalue())
    return destination


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise VectorDense20Error("TensorFlow não está disponível neste ambiente.") from exc
    require_runtime()
    return tf


def _runtime_preflight(tf: Any, output_root: Path) -> dict[str, Any]:
    configure_reproducibility(tf, SEED)
    usage = shutil.disk_usage(output_root)
    return {
        "timestamp": utc_now(),
        "ok": True,
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "runtime_contract": require_runtime(),
        "dtype_policy": tf.keras.mixed_precision.global_policy().name,
        "physical_devices": [str(device) for device in tf.config.list_physical_devices()],
        "disk_free_gib": usage.free / 1024**3,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m classic_models.vector_dense20",
        description=(
            "Treina a ponta clássica 128 -> 128(ReLU) -> 20(tanh) -> C "
            "sobre features128 salvas."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASET_ORDER)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Valida todos os arrays sem treinar.")
    parser.add_argument("--resume", action="store_true", help="Retoma a partir do BackupAndRestore.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_epochs < 1 or args.max_epochs > MAX_EPOCHS:
        print(f"Erro: --max-epochs deve estar entre 1 e {MAX_EPOCHS}.", file=sys.stderr)
        return 2
    if args.telemetry_interval_seconds <= 0:
        print("Erro: --telemetry-interval-seconds deve ser positivo.", file=sys.stderr)
        return 2

    selected = list(DATASET_ORDER) if args.all else [str(args.dataset)]
    features_root = args.features_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    settings = VectorDense20Settings(
        max_epochs=int(args.max_epochs),
        telemetry_interval_seconds=float(args.telemetry_interval_seconds),
    )
    prepared: list[tuple[FeatureBundle, Path]] = []
    failures = 0
    for dataset in selected:
        try:
            bundle = load_feature_bundle(dataset, features_root)
            run_dir = _run_directory(output_root, dataset)
            _initialize_run(run_dir, bundle, settings)
            prepared.append((bundle, run_dir))
            train_stats = bundle.statistics["train"]
            print(
                f"[{dataset}] validado: shape={train_stats['shape']}, dtype={train_stats['dtype']}, "
                f"min={train_stats['min']:.6f}, max={train_stats['max']:.6f}, "
                f"mean={train_stats['mean']:.6f}, std={train_stats['std']:.6f}"
            )
        except Exception as exc:
            failures += 1
            print(f"[{dataset}] validação falhou: {exc}", file=sys.stderr)
            if args.fail_fast:
                return 1

    if args.dry_run:
        atomic_write_json(output_root / "dry_run.json", {
            "timestamp": utc_now(),
            "ok": failures == 0 and len(prepared) == len(selected),
            "datasets": [bundle.dataset for bundle, _ in prepared],
            "failures": failures,
            "settings": settings.to_dict(),
        })
        return 1 if failures else 0
    if failures:
        print("Preflight dos vetores falhou; nenhum treino será iniciado.", file=sys.stderr)
        return 1

    try:
        tf = _require_tensorflow()
        atomic_write_json(output_root / "preflight.json", _runtime_preflight(tf, output_root))
    except Exception as exc:
        atomic_write_json(output_root / "preflight.json", {
            "timestamp": utc_now(), "ok": False, "error": repr(exc),
        })
        print(f"Preflight do TensorFlow falhou: {exc}", file=sys.stderr)
        return 1

    for bundle, run_dir in prepared:
        try:
            _run_one(tf, bundle, run_dir, settings, resume=bool(args.resume))
        except KeyboardInterrupt:
            print("Interrompido; use --resume para continuar.", file=sys.stderr)
            return 130
        except Exception as exc:
            failures += 1
            print(f"[{bundle.dataset}] falhou: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
    aggregate = _write_aggregate(output_root)
    print(f"Consolidado: {aggregate}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
