"""Atomic complete-Keras checkpoints and an epoch commit pointer."""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import shutil
import uuid
from data_prep.feature_vectors import _sha256_file
from utils.state import atomic_write_json, atomic_write_text


def write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def trim_history(path, confirmed_epoch):
    path = Path(path)
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = [row for row in reader if int(float(row["epoch"])) <= confirmed_epoch]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(path, output.getvalue())


class CheckpointStore:
    def __init__(self, directory, fingerprint):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.pointer = self.directory/"state.json"
        self.state = read_json(self.pointer)
        if self.state and self.state["fingerprint"] != fingerprint:
            raise RuntimeError("Run configuration or feature hashes changed; use a new output directory.")

    def confirmed_path(self, *, best=False):
        key = "best_checkpoint" if best else "checkpoint"
        name = self.state.get(key)
        if not name:
            return None
        path = self.directory/name
        expected = self.state["best_sha256" if best else "sha256"]
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"Checkpoint has no valid confirmed hash: {path}")
        return path

    def commit(self, model, epoch, macro_f1, order_hash):
        destination = self.directory/f"epoch-{epoch:05d}.keras"
        temporary = self.directory/f".pending-{uuid.uuid4().hex}.keras"
        try:
            model.save(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        digest = _sha256_file(destination)
        is_best = not self.state or macro_f1 > self.state["best_val_macro_f1"]
        state = {
            **self.state, "fingerprint": self.fingerprint, "epoch": int(epoch),
            "checkpoint": destination.name, "sha256": digest,
            "optimizer_iterations": int(model.optimizer.iterations.numpy()),
            "epoch_order_sha256": order_hash, "resume_granularity": "last confirmed full epoch",
        }
        if is_best:
            state.update(best_checkpoint=destination.name, best_sha256=digest,
                         best_epoch=int(epoch), best_val_macro_f1=float(macro_f1))
        # The pointer is the sole transaction commit; partial epochs are replayed.
        atomic_write_json(self.pointer, state)
        self.state = state
        self.publish_best()
        # Keep the best, current and previous confirmed generation.
        preserve = {state["checkpoint"], state["best_checkpoint"],
                    f"epoch-{max(0, epoch-1):05d}.keras"}
        for path in self.directory.glob("epoch-*.keras"):
            if path.name not in preserve:
                path.unlink()
        return state

    def publish_best(self):
        source = self.confirmed_path(best=True)
        if source is not None:
            temporary = self.directory/".best.pending.keras"
            shutil.copyfile(source, temporary)
            os.replace(temporary, self.directory/"best.keras")


class RunLock:
    """One matrix per output root. Recover only a dead owner's stale lock."""
    def __init__(self, root):
        self.path = Path(root)/".matrix.lock"
        self.owner = None

    def __enter__(self):
        import psutil
        self.path.parent.mkdir(parents=True, exist_ok=True)
        process = psutil.Process()
        self.owner = {"pid": process.pid, "create_time": process.create_time()}
        for _ in range(2):
            try:
                with self.path.open("x", encoding="utf-8") as stream:
                    json.dump(self.owner, stream)
                return self
            except FileExistsError:
                previous = read_json(self.path)
                try:
                    active = psutil.Process(previous["pid"])
                    if abs(active.create_time()-previous["create_time"]) < 0.001:
                        raise RuntimeError(f"A matrix is already running (PID {active.pid}).")
                except (psutil.NoSuchProcess, KeyError):
                    pass
                self.path.unlink()
        raise RuntimeError(f"Could not acquire run lock: {self.path}")

    def __exit__(self, *_):
        if read_json(self.path) == self.owner:
            self.path.unlink(missing_ok=True)
