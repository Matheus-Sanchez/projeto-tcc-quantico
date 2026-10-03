"""Batch diagnostics and transactional epoch callbacks."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from quantum_models.diagnostics import capture_probes
from quantum_models.persistence import read_json, write_csv
from utils.state import atomic_write_json


def statistics(values):
    values = np.asarray(values, np.float64)
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "norm": float(np.linalg.norm(values)), "variance": float(np.var(values))}


class QuantumBatchCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.epoch = 0
        self.phase = "train"

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch+1

    def _begin(self, phase):
        self.phase = phase
        self.started = time.perf_counter()
        self.layer = self.model.get_layer("parallel_pqc20")
        self.before = np.asarray(self.layer.theta).copy()
        self.layer.engine.reset_records()

    def on_train_batch_begin(self, batch, logs=None):
        self._begin("train")

    def on_test_batch_begin(self, batch, logs=None):
        self._begin("validation")

    def _end(self, batch, logs):
        elapsed = time.perf_counter()-self.started
        records = self.layer.engine.records
        update = np.asarray(self.layer.theta)-self.before
        rows = []
        for circuit in range(4):
            forward = [r["circuits"][circuit] for r in records["forward"]]
            backward = [r["circuits"][circuit] for r in records["vjp"]]
            rows.append({
                "id": f"epoch-{self.epoch:05d}/{self.phase}/batch-{batch:05d}/circuit-{circuit}",
                "epoch": self.epoch, "phase": self.phase, "batch": int(batch), "circuit": circuit,
                "inputs_radians": statistics(self.layer.last_inputs[:, circuit*5:(circuit+1)*5]),
                "outputs_y": statistics(self.layer.last_outputs[:, circuit*5:(circuit+1)*5]),
                "parameters": statistics(np.asarray(self.layer.theta)[circuit]),
                "parameter_update": statistics(update[circuit]),
                "forward_seconds": sum(r["seconds"] for r in forward),
                "backward_seconds": sum(r["seconds"] for r in backward),
                "evaluations": sum(r["evaluations"] for r in forward+backward),
                "gradients": backward,
            })
        count = sum(row["evaluations"] for row in rows)
        payload = {"epoch": self.epoch, "batch": int(batch), "phase": self.phase,
                   "batch_seconds": elapsed, "logical_circuit_evaluations": count,
                   "logical_evaluations_per_second": count/elapsed,
                   "examples_per_second": len(self.layer.last_inputs)/elapsed,
                   "keras": logs or {}, "circuits": rows}
        # The same epoch/batch ID is replaced when an uncommitted epoch replays.
        atomic_write_json(self.output_dir/f"epoch-{self.epoch:05d}"/self.phase/
                          f"batch-{batch:05d}.json", payload)

    def on_train_batch_end(self, batch, logs=None):
        self._end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self._end(batch, logs)


class QuantumEpochCommit(tf.keras.callbacks.Callback):
    """Run after shared ValidationMacroF1Callback, commit all epoch artifacts."""
    def __init__(self, store, bundle, indices, output_dir, order_hash):
        super().__init__()
        self.store, self.bundle, self.indices = store, bundle, indices
        self.output_dir, self.order_hash = Path(output_dir), order_hash

    def on_epoch_end(self, epoch, logs=None):
        capture_probes(self.model, self.bundle, self.indices, self.output_dir/"probes",
                       f"epoch-{epoch+1:05d}")
        rows = []
        for path in sorted((self.output_dir/"batches").glob("epoch-*/**/batch-*.json")):
            data = read_json(path)
            if data["epoch"] <= epoch+1:
                for branch in data["circuits"]:
                    rows.append({
                        "id": branch["id"], "epoch": branch["epoch"], "phase": branch["phase"],
                        "batch": branch["batch"], "circuit": branch["circuit"],
                        "batch_seconds": data["batch_seconds"],
                        **{key: branch[key] for key in ("forward_seconds", "backward_seconds", "evaluations")},
                        **{f"input_{key}": value for key, value in branch["inputs_radians"].items()},
                        **{f"output_{key}": value for key, value in branch["outputs_y"].items()},
                        **{f"update_{key}": value for key, value in branch["parameter_update"].items()},
                        **{key: value for gradient in branch["gradients"] for key, value in gradient.items()
                           if key not in ("seconds", "evaluations")},
                    })
        write_csv(self.output_dir/"batches.csv", rows)
        self.store.commit(self.model, epoch+1, float(logs["val_macro_f1"]), self.order_hash)
