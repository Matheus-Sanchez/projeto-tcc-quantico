"""Audit the 18 real-feature smoke artifacts; no TensorFlow/simulator imports."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from data_prep.registry import DATASET_ORDER
from utils.experiment_config import NUM_CLASSES
from utils.state import atomic_write_json
from quantum_models.persistence import read_json


def verify(root):
    root = Path(root)
    matrix = read_json(root/"matrix.json")
    expected_jobs = [(name, backend) for name in DATASET_ORDER for backend in ("pennylane", "qiskit")]
    assert [(j["dataset"], j["backend"]) for j in matrix["jobs"]] == expected_jobs
    assert matrix["concurrent_jobs"] == 1
    report = {"job_count": 18, "noise_conditions": 0, "datasets": [],
              "scope": "one real optimizer batch per job, not the 100-epoch matrix"}
    for name in DATASET_ORDER:
        paths = [root/name/backend/"seed-42" for backend in ("pennylane", "qiskit")]
        assert read_json(paths[0]/"initialization.json") == read_json(paths[1]/"initialization.json")
        for path in paths:
            result = read_json(path/"result.json")
            assert result["status"] == "completed" and result["mode"] == "smoke"
            assert result["trainable_parameters"] == 19_272+21*NUM_CLASSES[name]
            assert result["scope"] == "smoke_subset"
            assert not result["classical_reference_eligible"]
            state = read_json(path/"checkpoints"/"state.json")
            assert state["epoch"] == state["optimizer_iterations"] == 1
            with (path/"history.csv").open(encoding="utf-8") as stream:
                assert len(list(csv.DictReader(stream))) == 1
            batch = read_json(path/"diagnostics"/"batches"/"epoch-00001"/"train"/"batch-00000.json")
            assert batch["logical_circuit_evaluations"] == 128*404
            for circuit in batch["circuits"]:
                assert circuit["parameter_update"]["norm"] > 0
                assert circuit["gradients"][0]["input_gradient_norm"] > 0
                assert circuit["gradients"][0]["weight_gradient_norm"] > 0
            assert read_json(path/"profile.json")["logical_evaluations_per_example"] == 404
            for tag in ("epoch-00000", "epoch-00001", "best"):
                with np.load(path/"diagnostics"/"probes"/f"{tag}.npz", allow_pickle=False) as data:
                    assert data["statevectors"].shape == (NUM_CLASSES[name]*2, 4, 4, 32)
                    np.testing.assert_allclose(data["probabilities_z"].sum(-1), 1, atol=1e-11)
                    assert np.bincount(data["labels"], minlength=NUM_CLASSES[name]).tolist() == [2]*NUM_CLASSES[name]
                    assert data["two_qubit_pauli_correlations_xyz"].shape[-3:] == (10, 3, 3)
            study = path/"diagnostics"/"noise-study"
            selection = read_json(study/"selection.json")
            generation = study/selection["generation"]
            summary = read_json(generation/"summary.json")
            assert summary["conditions"] == 64 and summary["status"] == "complete"
            report["noise_conditions"] += 64
            conditions = list((generation/"conditions").glob("*.json"))
            assert len(conditions) == 64
            for condition_file in conditions:
                condition = read_json(condition_file)
                if condition["shots"]:
                    with np.load(generation/"counts"/f"{condition['id']}.npz", allow_pickle=False) as data:
                        np.testing.assert_array_equal(data["counts_xyz"].sum(-1), condition["shots"])
                        np.testing.assert_array_equal(data["probe_counts_xyz"].sum(-1), condition["shots"])
                if condition["noise_scale"] == 0 and not condition["shots"]:
                    assert abs(condition["fidelity_mean"]-1) < 1e-9
                    assert condition["trace_distance_max"] < 1e-9
                    assert condition["output_rmse"] < 1e-9
            process_samples = []
            for sample_file in (path/"telemetry").glob("*/samples.jsonl"):
                process_samples.extend(json.loads(line)["process"] for line in sample_file.read_text(
                    encoding="utf-8").splitlines() if line.strip())
            assert max(sample.get("children_count", 0) for sample in process_samples) == 4
        with np.load(paths[0]/"test_predictions.npz") as p, np.load(paths[1]/"test_predictions.npz") as q:
            np.testing.assert_array_equal(p["y_true"], q["y_true"])
            np.testing.assert_allclose(p["logits"], q["logits"], atol=1e-6, rtol=0)
            max_logit_error = float(np.max(np.abs(p["logits"]-q["logits"])))
        with np.load(paths[0]/"diagnostics"/"probes"/"best.npz") as p, np.load(
                paths[1]/"diagnostics"/"probes"/"best.npz") as q:
            a, b = p["statevectors"], q["statevectors"]
            phases = np.einsum("...i,...i->...", a.conj(), b)
            max_state_error = float(np.max(np.abs(a-b*np.exp(-1j*np.angle(phases))[..., None])))
            assert max_state_error < 1e-9
        report["datasets"].append({"dataset": name, "max_fp32_logit_difference": max_logit_error,
                                   "max_state_difference_up_to_global_phase": max_state_error,
                                   "identical_initial_weights": True, "private_workers_per_job": 4})
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.root)
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
