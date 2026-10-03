"""Replay an incomplete epoch, then restore a confirmed one without duplication."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from data_prep.feature_vectors import load_feature_bundle, FeatureVectorsError
from quantum_models.parallel_dense20 import run_job, epoch_order, order_hash
from quantum_models.persistence import read_json
from utils.experiment_config import require_runtime


def write_features(root):
    destination = Path(root)/"mnist"
    destination.mkdir(parents=True)
    rng = np.random.default_rng(43)
    for split in ("train", "val", "test"):
        np.save(destination/f"{split}_features.npy", rng.normal(size=(20, 128)).astype(np.float32))
        np.save(destination/f"{split}_labels.npy", np.tile(np.arange(10), 2))
    return destination


def test_loader_rejects_false_declared_hash(tmp_path):
    destination = write_features(tmp_path)
    (destination/"manifest.json").write_text(json.dumps({
        "dataset": "mnist", "splits": {"train": {"features_sha256": "wrong"}}}), encoding="utf-8")
    with pytest.raises(FeatureVectorsError, match="Hash declarado"):
        load_feature_bundle("mnist", tmp_path)


def test_replay_and_confirmed_resume_keep_one_history_and_batch_record(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("pennylane")
    from quantum_models.callbacks import QuantumBatchCallback, QuantumEpochCommit
    from quantum_models.layer import ParallelPQC20
    write_features(tmp_path/"features")
    bundle = load_feature_bundle("mnist", tmp_path/"features")
    args = argparse.Namespace(output_root=tmp_path/"runs", features_root=tmp_path/"features",
                              classical_root=None, simulation_batch=16, smoke=True,
                              resume=True, diagnostics_only=False)
    environment = require_runtime(quantum=True)
    directory = args.output_root/"mnist"/"pennylane"/"seed-42"

    original_batch = QuantumBatchCallback.on_train_batch_end
    def interrupt_batch(self, batch, logs=None):
        original_batch(self, batch, logs)
        raise RuntimeError("simulated interruption before epoch commit")
    with monkeypatch.context() as patch:
        patch.setattr(QuantumBatchCallback, "on_train_batch_end", interrupt_batch)
        with pytest.raises(RuntimeError, match="before epoch commit"):
            run_job(bundle, "pennylane", args, environment)
    assert not (directory/"checkpoints"/"state.json").exists()
    batch_path = directory/"diagnostics"/"batches"/"epoch-00001"/"train"/"batch-00000.json"
    first_batch = read_json(batch_path)

    original_epoch = QuantumEpochCommit.on_epoch_end
    def interrupt_epoch(self, epoch, logs=None):
        original_epoch(self, epoch, logs)
        raise RuntimeError("simulated interruption after confirmed epoch")
    with monkeypatch.context() as patch:
        patch.setattr(QuantumEpochCommit, "on_epoch_end", interrupt_epoch)
        with pytest.raises(RuntimeError, match="after confirmed epoch"):
            run_job(bundle, "pennylane", args, environment)
    state = read_json(directory/"checkpoints"/"state.json")
    assert state["epoch"] == 1
    assert state["optimizer_iterations"] == 1
    assert state["epoch_order_sha256"] == order_hash(epoch_order(20, 0))
    np.testing.assert_allclose(first_batch["keras"]["loss"], read_json(batch_path)["keras"]["loss"], atol=1e-7)
    committed_hash = state["sha256"]

    result = run_job(bundle, "pennylane", args, environment)
    assert result["status"] == "completed"
    assert read_json(directory/"checkpoints"/"state.json")["sha256"] == committed_hash
    with (directory/"history.csv").open(encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 1 and int(history[0]["epoch"]) == 1
    with (directory/"diagnostics"/"batches.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert len({r["id"] for r in rows}) == 8
    study = directory/"diagnostics"/"noise-study"
    selection = read_json(study/"selection.json")
    conditions = list((study/selection["generation"]/"conditions").glob("*.json"))
    assert len(conditions) == 64
    # A completed resume must skip training, study, and checkpoint mutation.
    repeated = run_job(bundle, "pennylane", args, environment)
    assert repeated == result
    assert read_json(directory/"checkpoints"/"state.json")["sha256"] == committed_hash
