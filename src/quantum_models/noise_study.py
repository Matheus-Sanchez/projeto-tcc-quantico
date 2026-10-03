"""Separate post-selection exact / finite-shot local noise study."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from quantum_models.circuit_spec import SIGNS
from quantum_models.diagnostics import (
    capture_probes, basis_probabilities, shot_measurements, density, entropy, purity,
    fidelity_and_trace_distance, output_errors, save_npz,
)
from quantum_models.persistence import read_json, write_csv
from utils.state import atomic_write_json

SHOTS = (None, 256, 1024, 4096)
NOISE_SCALES = (0.0, 0.5, 1.0, 2.0)
SAMPLING_SEEDS = (42, 43, 44, 45, 46)


def _probabilities(logits):
    logits = np.asarray(logits, np.float64)
    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
    return exp/exp.sum(axis=1, keepdims=True)


def _condition(scale, shots, seed):
    # Short stable IDs also keep atomic temporary filenames below Win32's
    # path limit in deeply nested test/output directories.
    return f"n{scale:g}-s{shots or 'exact'}-r{seed if seed is not None else 'none'}"


def run_noise_study(model, bundle, indices, output_dir, *, smoke=False):
    import tensorflow as tf
    from metrics.metrics import classification_metrics
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer = model.get_layer("parallel_pqc20")
    encoder = tf.keras.Model(model.input, model.get_layer("dense_128_20").output)
    # Smoke verifies every noise / shots condition on a small real test subset.
    test_indices = (np.arange(min(8, len(bundle.labels["test"]))) if smoke else
                    np.arange(len(bundle.labels["test"])))
    angles = np.asarray(encoder(np.asarray(bundle.features["test"][test_indices]),
                                training=False), np.float64)
    theta = np.asarray(layer.theta, np.float64)
    classifier = model.get_layer("logits")
    def classify(readings):
        # Preserve the Keras FP32 interface and classifier in every study condition.
        return _probabilities(np.asarray(classifier(np.asarray(readings, np.float32), training=False)))
    ideal_readings = layer.engine.forward(angles, theta)
    ideal_probabilities = classify(ideal_readings)
    labels = bundle.labels["test"][test_indices]
    baseline = classification_metrics(labels, ideal_probabilities.argmax(axis=1),
                                      num_classes=bundle.num_classes).to_dict()
    save_npz(output_dir/"ideal-test.npz", indices=test_indices, labels=labels,
             angles=angles, readings_y=ideal_readings, probabilities=ideal_probabilities)
    rows = []
    for scale in NOISE_SCALES:
        cache = output_dir/f"noise-{scale:g}-test-probabilities.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                if not np.array_equal(saved["indices"], test_indices) or not np.array_equal(saved["theta"], theta):
                    raise RuntimeError("Noise cache belongs to different test data or quantum weights.")
                basis = saved["basis_probabilities"]
                state_metrics = read_json(output_dir/f"noise-{scale:g}-states.json")
        else:
            chunks = []
            fidelities, distances, purities, entropies = [], [], [], []
            # Avoid storing a density matrix for every sample in a large test set.
            for start in range(0, len(angles), 128):
                current = angles[start:start+128]
                ideal_states = layer.engine.snapshots(current, theta)[:, :, -1]
                candidate = (ideal_states if not scale else
                             layer.engine.snapshots(current, theta, scale=scale)[:, :, -1])
                chunks.append(np.stack([basis_probabilities(candidate, b) for b in ("X", "Y", "Z")],
                                       axis=2))
                fidelity, distance = fidelity_and_trace_distance(ideal_states, candidate)
                fidelities.append(fidelity)
                distances.append(distance)
                rho = density(candidate)
                purities.append(purity(rho))
                entropies.append(entropy(rho))
            basis = np.concatenate(chunks)
            state_metrics = {
                "fidelity_mean": float(np.mean(np.concatenate(fidelities))),
                "fidelity_min": float(np.min(np.concatenate(fidelities))),
                "trace_distance_mean": float(np.mean(np.concatenate(distances))),
                "trace_distance_max": float(np.max(np.concatenate(distances))),
                "purity_global_mean": float(np.mean(np.concatenate(purities))),
                "entropy_global_bits_mean": float(np.mean(np.concatenate(entropies))),
            }
            save_npz(cache, indices=test_indices, theta=theta, basis_probabilities=basis)
            atomic_write_json(output_dir/f"noise-{scale:g}-states.json", state_metrics)
        probe_states = capture_probes(model, bundle, indices, output_dir/"probes",
                                      f"noise-{scale:g}", scale=scale)
        probe_basis = np.stack([basis_probabilities(probe_states[:, :, -1], b)
                                for b in ("X", "Y", "Z")], axis=2)
        exact = (basis[:, :, 1] @ SIGNS).reshape(len(angles), 20)
        for shots in SHOTS:
            for seed in (SAMPLING_SEEDS if shots else (None,)):
                condition = _condition(scale, shots, seed)
                existing = read_json(output_dir/"conditions"/f"{condition}.json")
                if existing:
                    rows.append(existing)
                    continue
                if shots:
                    # Separate independently seeded X/Y/Z settings. The same
                    # canonical sampler is used after each SDK's local simulation.
                    rng = np.random.default_rng(np.random.SeedSequence([seed, int(scale*2), shots, 0]))
                    observed, counts, se = shot_measurements(basis, shots, rng)
                    readings = observed[:, :, 1].reshape(len(angles), 20)
                    probe_rng = np.random.default_rng(np.random.SeedSequence([seed, int(scale*2), shots, 1]))
                    probe_observed, probe_counts, probe_se = shot_measurements(probe_basis, shots, probe_rng)
                    save_npz(output_dir/"counts"/f"{condition}.npz",
                             test_indices=test_indices, counts_xyz=counts, standard_error_xyz=se,
                             probe_indices=indices, probe_counts_xyz=probe_counts,
                             probe_standard_error_xyz=probe_se, probe_readings_xyz=probe_observed)
                    se_mean = float(se[:, :, 1].mean())
                else:
                    readings, se_mean = (ideal_readings if not scale else exact), 0.0
                probabilities = classify(readings)
                classification = classification_metrics(labels, probabilities.argmax(axis=1),
                                                       num_classes=bundle.num_classes).to_dict()
                save_npz(output_dir/"predictions"/f"{condition}.npz", indices=test_indices,
                         labels=labels, readings_y=readings, probabilities=probabilities)
                row = {
                    "id": condition, "noise_scale": scale, "shots": shots, "sampling_seed": seed,
                    "test_examples": len(labels), "scope": "smoke_subset" if smoke else "full_test",
                    "accuracy": classification["accuracy"], "macro_f1": classification["macro_f1"],
                    "accuracy_degradation": baseline["accuracy"]-classification["accuracy"],
                    "macro_f1_degradation": baseline["macro_f1"]-classification["macro_f1"],
                    "shot_standard_error_y_mean": se_mean, **state_metrics,
                    **output_errors(ideal_readings, readings),
                }
                atomic_write_json(output_dir/"conditions"/f"{condition}.json", row)
                rows.append(row)
        write_csv(output_dir/"conditions.csv", rows)
    summary = {
        "status": "complete", "scope": "smoke_subset" if smoke else "full_test",
        "baseline": baseline, "conditions": len(rows), "noise_scales": NOISE_SCALES,
        "shots": SHOTS, "sampling_seeds": SAMPLING_SEEDS,
        "sampling": "independent multinomial X/Y/Z measurements from each local SDK's exact density matrix",
        "bit_order": "q0 leftmost, canonical bitstrings 00000..11111",
        "fidelity": "squared Uhlmann fidelity between ideal and noisy pre-measurement states",
        "trace_distance": "0.5 * trace norm of ideal_rho - noisy_rho before sampling",
        "shot_effect": "finite shots affect readings and classification, not pre-measurement states",
        "checkpoint_selection": "validation val_macro_f1 before any test study",
    }
    atomic_write_json(output_dir/"summary.json", summary)
    return summary
