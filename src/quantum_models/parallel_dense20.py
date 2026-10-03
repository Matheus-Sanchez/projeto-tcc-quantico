"""Sequential local training matrix for the four-PQC TensorFlow vector head.

    python -m quantum_models.parallel_dense20 --all --backend both --resume

The module-level imports intentionally exclude TensorFlow and simulator SDKs,
so the four spawned workers initialize only their private simulator.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import platform
import time
import traceback
import uuid
import numpy as np
from data_prep.feature_vectors import load_feature_bundle
from data_prep.registry import DATASET_ORDER
from quantum_models.circuit_spec import circuit_metadata
from quantum_models.diagnostics import probe_indices, capture_probes, save_npz
from quantum_models.persistence import CheckpointStore, RunLock, read_json, trim_history, write_csv
from utils.experiment_config import (
    PROJECT_ROOT, DEFAULT_FEATURES_ROOT, VectorHeadSettings, require_runtime, SEED, BATCH_SIZE,
)
from utils.state import atomic_write_json, atomic_write_text, config_fingerprint, utc_now

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT/"outputs"/"quantum-parallel-128-20"


def epoch_order(size, epoch, seed=SEED):
    """Epoch-addressed shuffle: restart does not depend on consumed RNG state."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)])).permutation(size)


def order_hash(order):
    return hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()


def configuration(bundle, backend, *, smoke=False, simulation_batch=256):
    protocol = VectorHeadSettings().to_dict()
    protocol["hidden_activation"] = "parallel_pqc20"
    return {
        **protocol, "schema_version": 1, "experiment": "quantum_128_relu_20_parallel_pqc_features128",
        "dataset": bundle.dataset, "num_classes": bundle.num_classes, "backend": backend,
        "source_feature_hashes": dict(bundle.hashes), "split_fingerprint": bundle.split_fingerprint,
        "circuit": circuit_metadata(), "simulation_batch": int(simulation_batch), "workers": 4,
        "gradient": "parameter_shift_plus_minus_pi_over_2_inputs_and_weights",
        "simulation_precision": "float64/complex128", "mode": "smoke" if smoke else "full",
        "shuffle": "NumPy SeedSequence([seed, epoch_zero_based]); deterministic full permutation",
        "transformations": {"normalization": False, "augmentation": False, "pca": False,
                            "feature_selection": False, "scaler": False},
        "effective_epochs": 1 if smoke else protocol["max_epochs"],
        "effective_train_examples": min(BATCH_SIZE, len(bundle.labels["train"])) if smoke else
                                    len(bundle.labels["train"]),
    }


def classical_reference(bundle, root):
    directory = Path(root)/bundle.dataset/"seed-42"
    config = read_json(directory/"config.json")
    shared = VectorHeadSettings().to_dict()
    keys = ("seed", "batch_size", "optimizer", "learning_rate", "max_epochs", "precision",
            "checkpoint_metric", "early_stopping", "lr_scheduler", "input_features",
            "projection_units", "projection_activation", "hidden_units")
    differences = [key for key in keys if config.get(key) != shared[key]]
    if config.get("source_feature_hashes") != dict(bundle.hashes):
        differences.append("source_feature_hashes")
    if config.get("split_fingerprint") != bundle.split_fingerprint:
        differences.append("split_fingerprint")
    status = read_json(directory/"status.json")
    if status.get("status") != "completed":
        differences.append("status")
    classification = read_json(directory/"test_metrics.json").get("classification", {})
    if not all(key in classification for key in ("accuracy", "macro_f1")):
        differences.append("test_metrics")
    return {
        "directory": str(directory), "eligible": not differences, "differences": differences,
        "classification": classification if not differences else None,
        "timing": read_json(directory/"timing.json") if not differences else None,
        "environment": read_json(directory/"telemetry"/"hardware.json"),
        "preflight": read_json(Path(root)/"preflight.json"),
        "note": "Accuracy uses identical cached data/protocol. Time retains the effective "
                "environment (historical Python 3.10.12 versus current Python 3.12).",
    }


def _dataset(tf, x, y, order=None):
    if order is not None:
        x, y = np.asarray(x[order], np.float32), np.asarray(y[order], np.int64)
    else:
        x, y = np.asarray(x, np.float32), np.asarray(y, np.int64)
    data = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE, drop_remainder=False)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    return data.with_options(options).prefetch(1)


def profile_batch(model, x, y, epochs, train_examples):
    import tensorflow as tf
    layer = model.get_layer("parallel_pqc20")
    # Warm worker startup separately, so throughput estimates exclude import cost.
    started = time.perf_counter()
    model(np.asarray(x), training=False)
    startup = time.perf_counter()-started
    layer.engine.reset_records()
    started = time.perf_counter()
    with tf.GradientTape() as tape:
        predicted = model(np.asarray(x), training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            np.asarray(y), predicted, from_logits=True))
    gradients = tape.gradient(loss, model.trainable_variables)
    elapsed = time.perf_counter()-started
    if any(g is None or not np.isfinite(np.asarray(g)).all() for g in gradients):
        raise RuntimeError("Preflight has missing or nonfinite gradients.")
    evaluations = sum(r["evaluations"] for operation in ("forward", "vjp")
                      for r in layer.engine.records[operation])
    return {
        "batch_examples": len(x), "batch_seconds": elapsed,
        "worker_startup_and_warmup_seconds": startup, "examples_per_second": len(x)/elapsed,
        "logical_circuit_evaluations": evaluations, "logical_evaluations_per_example": evaluations/len(x),
        "logical_evaluations_per_second": evaluations/elapsed,
        "estimated_train_only_seconds": elapsed/len(x)*train_examples*epochs,
        "estimate_excludes": ["validation", "epoch probes", "telemetry and persistence", "post-training noise study"],
        "method": "one real batch forward and backward, no optimizer update",
        "optimizer_iterations": int(model.optimizer.iterations.numpy()),
        "gradient_norms": {getattr(v, "path", v.name): float(np.linalg.norm(g))
                           for v, g in zip(model.trainable_variables, gradients)},
        "branches": layer.engine.records,
    }


def run_job(bundle, backend, args, environment):
    import tensorflow as tf
    from classic_models.vector_dense20 import configure_reproducibility
    from quantum_models.layer import build_parallel_dense20  # registers deserialization
    from quantum_models.callbacks import QuantumBatchCallback, QuantumEpochCommit
    from quantum_models.backends import make_backend
    from quantum_models.noise_study import run_noise_study
    from logs.telemetry import TelemetrySampler
    from metrics.metrics import ValidationMacroF1Callback, evaluate_model

    directory = Path(args.output_root)/bundle.dataset/backend/"seed-42"
    directory.mkdir(parents=True, exist_ok=True)
    config = configuration(bundle, backend, smoke=args.smoke, simulation_batch=args.simulation_batch)
    fingerprint = config_fingerprint(config)
    previous_config = read_json(directory/"config.json")
    if previous_config and config_fingerprint(previous_config) != fingerprint:
        raise RuntimeError(f"Configuração divergente em {directory}; escolha outra saída.")
    atomic_write_json(directory/"config.json", config)
    status_path = directory/"status.json"
    previous_status = read_json(status_path)
    store = CheckpointStore(directory/"checkpoints", fingerprint)
    if args.diagnostics_only and not store.state:
        raise RuntimeError("--diagnostics-only requires a confirmed trained checkpoint.")
    if store.state and not args.resume and not args.diagnostics_only:
        raise RuntimeError(f"Run já possui checkpoint; utilize --resume: {directory}")
    if previous_status.get("status") == "completed" and args.resume and not args.diagnostics_only:
        store.confirmed_path()
        store.publish_best()
        print(f"[{bundle.dataset}/{backend}] complete: skipping", flush=True)
        return read_json(directory/"result.json")
    session = uuid.uuid4().hex
    telemetry = TelemetrySampler(
        directory/"telemetry"/session, interval_seconds=VectorHeadSettings().telemetry_interval_seconds,
        include_children=True,
        disk_paths={"features": bundle.directory, "run": directory})
    model = None
    start_time = time.perf_counter()
    try:
        telemetry.start()
        tf.keras.backend.clear_session()
        configure_reproducibility(tf, SEED)
        last = store.confirmed_path(best=args.diagnostics_only)
        if last:
            model = tf.keras.models.load_model(last)
            if int(model.optimizer.iterations.numpy()) != store.state["optimizer_iterations"] and not args.diagnostics_only:
                raise RuntimeError("Adam iteration count does not match committed checkpoint.")
        else:
            model = build_parallel_dense20(bundle.num_classes, backend=backend, seed=SEED,
                                           simulation_batch=args.simulation_batch)
            variables = {f"weight_{i}": np.asarray(v) for i, v in enumerate(model.weights)}
            save_npz(directory/"initial_weights.npz", **variables)
            atomic_write_json(directory/"initialization.json", {
                "weights_sha256": {name: hashlib.sha256(v.tobytes()).hexdigest()
                                   for name, v in variables.items()},
                "seed": SEED, "dense": "Keras GlorotUniform + zero biases, from scratch",
                "quantum": "independent NumPy default_rng(42) uniform [0,2pi), from scratch",
            })
        summary = io.StringIO()
        model.summary(print_fn=lambda text: summary.write(text+"\n"))
        atomic_write_text(directory/"model_summary.txt", summary.getvalue())
        atomic_write_json(directory/"environment.json", {
            **environment, "tensorflow_devices": [str(d) for d in tf.config.list_physical_devices()],
            "platform": platform.platform(), "simulators": "local CPU, four private persistent workers",
            "circuit_device": "default.qubit" if backend == "pennylane" else "Aer CPU statevector/double",
            "remote_calls": False,
        })
        atomic_write_json(directory/"feature_statistics.json", {
            "statistics": bundle.statistics, "hashes": bundle.hashes,
            "source_manifest": bundle.source_manifest})
        atomic_write_json(directory/"circuit.json", circuit_metadata())
        atomic_write_text(directory/"circuit.txt", make_backend(backend).drawing())
        reference_root = args.classical_root or (Path(args.features_root).parent/"classical-128-20-mac")
        reference = classical_reference(bundle, reference_root)
        atomic_write_json(directory/"classical_reference.json", reference)
        indices = probe_indices(bundle.labels["val"], bundle.num_classes)
        atomic_write_json(directory/"probe_selection.json", {"validation_indices": indices.tolist(),
                                                           "per_class": 2, "seed": SEED})
        initial_epoch = int(store.state.get("epoch", 0))
        if initial_epoch:
            last_order = epoch_order(len(bundle.labels["train"]), initial_epoch-1)
            if args.smoke:
                last_order = last_order[:config["effective_train_examples"]]
            if order_hash(last_order) != store.state["epoch_order_sha256"]:
                raise RuntimeError("Restored epoch data order differs from confirmed state.")
        trim_history(directory/"history.csv", initial_epoch)
        trim_history(directory/"diagnostics"/"batches.csv", initial_epoch)
        atomic_write_json(status_path, {"status": "diagnostics" if args.diagnostics_only else "running",
                                      "session": session, "confirmed_epoch": initial_epoch,
                                      "fingerprint": fingerprint, "started_at": utc_now()})
        if not args.diagnostics_only:
            if not initial_epoch:
                capture_probes(model, bundle, indices, directory/"diagnostics"/"probes", "epoch-00000")
            if not (directory/"profile.json").is_file():
                profile_indices = epoch_order(len(bundle.labels["train"]), 0)[:BATCH_SIZE]
                profile = profile_batch(model, bundle.features["train"][profile_indices],
                                        bundle.labels["train"][profile_indices],
                                        config["effective_epochs"], config["effective_train_examples"])
                atomic_write_json(directory/"profile.json", profile)
                print(f"[{bundle.dataset}/{backend}] batch: {profile['batch_seconds']:.2f}s, "
                      f"{profile['examples_per_second']:.2f} examples/s; "
                      f"estimated train only: {profile['estimated_train_only_seconds']/3600:.2f}h", flush=True)
            val_features = bundle.features["val"][indices] if args.smoke else bundle.features["val"]
            val_labels = bundle.labels["val"][indices] if args.smoke else bundle.labels["val"]
            validation = _dataset(tf, val_features, val_labels)
            for epoch in range(initial_epoch, config["effective_epochs"]):
                order = epoch_order(len(bundle.labels["train"]), epoch)
                if args.smoke:
                    order = order[:config["effective_train_examples"]]
                training = _dataset(tf, bundle.features["train"], bundle.labels["train"], order)
                callbacks = [
                    QuantumBatchCallback(directory/"diagnostics"/"batches"),
                    ValidationMacroF1Callback(validation, num_classes=bundle.num_classes,
                                              history_path=directory/"history.csv",
                                              train_examples_per_epoch=len(order)),
                    QuantumEpochCommit(store, bundle, indices, directory/"diagnostics", order_hash(order)),
                ]
                model.fit(training, validation_data=validation, epochs=epoch+1, initial_epoch=epoch,
                          callbacks=callbacks, shuffle=False, verbose=2)
                atomic_write_json(status_path, {"status": "running", "session": session,
                                              "confirmed_epoch": epoch+1, "fingerprint": fingerprint})
        # Release workers before creating a model loaded from the selected generation.
        model.get_layer("parallel_pqc20").close()
        best_path = store.confirmed_path(best=True)
        model = tf.keras.models.load_model(best_path)
        capture_probes(model, bundle, indices, directory/"diagnostics"/"probes", "best")
        test_indices = probe_indices(bundle.labels["test"], bundle.num_classes) if args.smoke else None
        test = _dataset(tf, bundle.features["test"] if test_indices is None else bundle.features["test"][test_indices],
                        bundle.labels["test"] if test_indices is None else bundle.labels["test"][test_indices])
        evaluation = evaluate_model(model, test, num_classes=bundle.num_classes)
        metrics = evaluation.classification.to_dict()
        save_npz(directory/"test_predictions.npz", y_true=evaluation.y_true, logits=evaluation.logits,
                 probabilities=evaluation.probabilities)
        atomic_write_json(directory/"test_metrics.json", {"keras": evaluation.keras_metrics,
                                                         "classification": metrics,
                                                         "scope": "smoke_subset" if args.smoke else "full_test"})
        # Every condition is exercised in smoke mode, with smaller diagnostic sets.
        study_indices = indices[:4] if args.smoke else indices
        study_root = directory/"diagnostics"/"noise-study"
        study_generation = f"e{store.state['best_epoch']:05d}-{store.state['best_sha256'][:12]}"
        atomic_write_json(study_root/"selection.json", {"generation": study_generation,
                                                       "checkpoint_sha256": store.state["best_sha256"]})
        study = run_noise_study(model, bundle, study_indices, study_root/study_generation,
                                smoke=args.smoke)
        elapsed = time.perf_counter()-start_time
        result = {
            "dataset": bundle.dataset, "backend": backend, "seed": SEED, "status": "completed",
            "mode": config["mode"], "selected_epoch": store.state["best_epoch"],
            "selected_val_macro_f1": store.state["best_val_macro_f1"],
            "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
            "scope": "smoke_subset" if args.smoke else "full_test",
            "trainable_parameters": model.count_params(), "elapsed_this_session_seconds": elapsed,
            "classical_reference_eligible": reference["eligible"] and not args.smoke,
            "classical_test_accuracy": reference["classification"].get("accuracy")
                if reference["eligible"] and not args.smoke else None,
            "classical_test_macro_f1": reference["classification"].get("macro_f1")
                if reference["eligible"] and not args.smoke else None,
            "accuracy_delta_vs_classical": metrics["accuracy"]-reference["classification"]["accuracy"]
                if reference["eligible"] and not args.smoke else None,
            "macro_f1_delta_vs_classical": metrics["macro_f1"]-reference["classification"]["macro_f1"]
                if reference["eligible"] and not args.smoke else None,
            "noise_study_conditions": study["conditions"], "run": str(directory),
        }
        atomic_write_json(directory/"result.json", result)
        atomic_write_json(status_path, {**result, "status": "completed", "session": session,
                                      "fingerprint": fingerprint, "completed_at": utc_now()})
        return result
    except BaseException as exc:
        atomic_write_json(status_path, {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                      "session": session, "confirmed_epoch": store.state.get("epoch", 0),
                                      "fingerprint": fingerprint, "error": str(exc),
                                      "traceback": traceback.format_exc(), "updated_at": utc_now()})
        raise
    finally:
        if model is not None:
            model.get_layer("parallel_pqc20").close()
        telemetry.stop(final_event="session_finished")
        tf.keras.backend.clear_session()
        gc.collect()


def parser():
    command = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    datasets = command.add_mutually_exclusive_group(required=True)
    datasets.add_argument("--all", action="store_true")
    datasets.add_argument("--dataset", choices=DATASET_ORDER)
    command.add_argument("--backend", choices=("pennylane", "qiskit", "both"), default="both")
    command.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    command.add_argument("--output-root", type=Path)
    command.add_argument("--classical-root", type=Path)
    command.add_argument("--simulation-batch", type=int, default=256)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--smoke", action="store_true", help="one real optimizer batch, one epoch; reduced test/noise subsets")
    command.add_argument("--diagnostics-only", action="store_true", help="inspect the confirmed best checkpoint; no training")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT/("smoke" if args.smoke else "runs")
    if args.simulation_batch < 1:
        raise ValueError("--simulation-batch must be positive.")
    # Set before importing NumPy-dependent child SDKs or TensorFlow.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    names = DATASET_ORDER if args.all else (args.dataset,)
    backends = ("pennylane", "qiskit") if args.backend == "both" else (args.backend,)
    with RunLock(args.output_root):
        environment = require_runtime(quantum=True)
        environment["installed_packages"] = {d.metadata["Name"]: d.version
                                              for d in importlib.metadata.distributions()}
        atomic_write_json(args.output_root/"preflight.json", {**environment, "created_at": utc_now()})
        # Validate ALL datasets before the first job mutates a model or run.
        bundles = {}
        for name in names:
            bundles[name] = load_feature_bundle(name, args.features_root)
            print(f"validated {name}: {len(bundles[name].labels['train'])} train / "
                  f"{len(bundles[name].labels['val'])} val / {len(bundles[name].labels['test'])} test", flush=True)
        jobs = [{"dataset": name, "backend": backend, "seed": SEED,
                 "run": str(args.output_root/name/backend/"seed-42"),
                 "configuration": configuration(bundles[name], backend, smoke=args.smoke,
                                                simulation_batch=args.simulation_batch)}
                for name in names for backend in backends]
        atomic_write_json(args.output_root/"matrix.json", {
            "jobs": jobs, "job_count": len(jobs), "concurrent_jobs": 1,
            "order": "dataset registry order; PennyLane then Qiskit",
            "dry_run": args.dry_run, "mode": "smoke" if args.smoke else "full",
        })
        if args.dry_run:
            print(f"dry-run: {len(jobs)} sequential local jobs configured in {args.output_root}", flush=True)
            return 0
        results = []
        for job in jobs:
            print(f"starting {job['dataset']}/{job['backend']}", flush=True)
            results.append(run_job(bundles[job["dataset"]], job["backend"], args, environment))
            write_csv(args.output_root/"results.csv", results)
        atomic_write_json(args.output_root/"matrix_status.json", {
            "status": "completed", "jobs": len(results), "mode": "smoke" if args.smoke else "full",
            "completed_at": utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
