"""State, subsystem, measurement, and noise diagnostics in canonical wire order."""
from __future__ import annotations

import io
import itertools
from pathlib import Path
import numpy as np
from quantum_models.circuit_spec import PAULIS, SIGNS, STAGES
from utils.state import atomic_write_bytes, atomic_write_json

PAIRS = tuple(itertools.combinations(range(5), 2))


def save_npz(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_write_bytes(Path(path), stream.getvalue())


def density(states):
    values = np.asarray(states, np.complex128)
    if values.shape[-2:] == (32, 32):
        return values
    if values.shape[-1] != 32:
        raise ValueError("Expected five-qubit states.")
    return values[..., :, None] * values[..., None, :].conj()


def reduced_density(rho, keep):
    rho = np.asarray(rho, np.complex128)
    keep = tuple(keep)
    if not keep or len(set(keep)) != len(keep) or not set(keep) <= set(range(5)):
        raise ValueError("Invalid subsystem.")
    rest = tuple(w for w in range(5) if w not in keep)
    prefix = rho.ndim - 2
    permutation = (*range(prefix), *(prefix+w for w in (*keep, *rest)),
                   *(prefix+5+w for w in (*keep, *rest)))
    values = rho.reshape(*rho.shape[:-2], *((2,)*10)).transpose(permutation)
    values = values.reshape(*rho.shape[:-2], 2**len(keep), 2**len(rest),
                            2**len(keep), 2**len(rest))
    return np.trace(values, axis1=-3, axis2=-1)


def entropy(rho):
    eigenvalues = np.clip(np.linalg.eigvalsh((rho+rho.swapaxes(-1, -2).conj())/2).real, 0, 1)
    return -np.sum(eigenvalues * np.log2(np.maximum(eigenvalues, 1e-300)), axis=-1)


def purity(rho):
    return np.einsum("...ij,...ji->...", rho, rho).real


def observable(wires, bases):
    operator = np.array([[1]], dtype=complex)
    for w in range(5):
        operator = np.kron(operator, PAULIS[bases[wires.index(w)]] if w in wires else np.eye(2))
    return operator


def state_diagnostics(states):
    rho = density(states)
    one = np.stack([reduced_density(rho, (w,)) for w in range(5)], axis=-3)
    two = np.stack([reduced_density(rho, pair) for pair in PAIRS], axis=-3)
    bloch = np.stack([np.einsum("...ij,ji->...", one, PAULIS[b]).real
                      for b in ("X", "Y", "Z")], axis=-1)
    correlations = np.stack([np.einsum("...ij,ji->...", rho,
                                     observable(pair, ("Y", "Y"))).real for pair in PAIRS], axis=-1)
    covariance = correlations - np.stack([bloch[..., a, 1]*bloch[..., b, 1]
                                         for a, b in PAIRS], axis=-1)
    pair_xyz = np.stack([
        np.stack([np.einsum("...ij,ji->...", two, np.kron(PAULIS[a], PAULIS[b])).real
                  for b in ("X", "Y", "Z")], axis=-1) for a in ("X", "Y", "Z")], axis=-2)
    pair_means = np.stack([bloch[..., a, :, None]*bloch[..., b, None, :]
                           for a, b in PAIRS], axis=-3)
    probabilities = np.clip(np.diagonal(rho, axis1=-2, axis2=-1).real, 0, 1)
    return {
        "probabilities_z": probabilities, "bloch_xyz": bloch,
        "pauli_y_variance": np.maximum(0, 1-bloch[..., 1]**2),
        "yy_correlations": correlations, "yy_covariance": covariance,
        "two_qubit_pauli_correlations_xyz": pair_xyz,
        "two_qubit_pauli_covariance_xyz": pair_xyz-pair_means,
        "reduced_one_qubit": one, "reduced_two_qubit": two,
        "purity_global": purity(rho), "entropy_global_bits": entropy(rho),
        "purity_one_qubit": purity(one), "entropy_one_qubit_bits": entropy(one),
        "purity_two_qubit": purity(two), "entropy_two_qubit_bits": entropy(two),
        "measurement_entropy_z_bits": -np.sum(probabilities *
                                              np.log2(np.maximum(probabilities, 1e-300)), axis=-1),
    }


def basis_probabilities(states, basis):
    if basis not in ("X", "Y", "Z"):
        raise ValueError(basis)
    h = np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2)
    rotation = {"X": h, "Y": h @ np.diag([1, -1j]), "Z": np.eye(2)}[basis]
    unitary = rotation
    for _ in range(4):
        unitary = np.kron(unitary, rotation)
    rho = density(states)
    transformed = unitary @ rho @ unitary.conj().T
    probabilities = np.maximum(np.diagonal(transformed, axis1=-2, axis2=-1).real, 0)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def shot_measurements(probabilities, shots, rng):
    probabilities = np.asarray(probabilities, np.float64)
    counts = np.array([rng.multinomial(int(shots), p/p.sum())
                       for p in probabilities.reshape(-1, 32)], dtype=np.int64).reshape(
                           *probabilities.shape[:-1], 32)
    readings = counts @ SIGNS / shots
    exact = probabilities @ SIGNS
    standard_error = np.sqrt(np.maximum(0, 1-exact**2)/shots)
    return readings, counts, standard_error


def fidelity_and_trace_distance(reference, candidate):
    a, b = density(reference), density(candidate)
    a, b = np.broadcast_arrays(a, b)
    eig, vectors = np.linalg.eigh((a+a.swapaxes(-1, -2).conj())/2)
    root = (vectors * np.sqrt(np.maximum(eig, 0))[..., None, :]) @ vectors.swapaxes(-1, -2).conj()
    product = root @ b @ root
    fidelity = np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(
        (product+product.swapaxes(-1, -2).conj())/2), 0)), axis=-1)**2
    difference = a-b
    trace_distance = 0.5*np.sum(np.abs(np.linalg.eigvalsh(
        (difference+difference.swapaxes(-1, -2).conj())/2)), axis=-1)
    return np.clip(fidelity.real, 0, 1), trace_distance.real


def output_errors(reference, candidate):
    reference, candidate = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    error = candidate-reference
    signal_power, error_power = float(np.mean(reference**2)), float(np.mean(error**2))
    snr = 10*np.log10(signal_power/error_power) if error_power and signal_power else None
    return {
        "output_mae": float(np.mean(np.abs(error))),
        "output_rmse": float(np.sqrt(error_power)),
        "relative_l2_error": float(np.linalg.norm(error)/max(np.linalg.norm(reference), 1e-300)),
        "snr_db_auxiliary": snr,
        "snr_status": "finite" if snr is not None else
                      ("zero_error_infinite" if not error_power else "zero_signal"),
    }


def probe_indices(labels, num_classes, *, seed=42):
    """Two fixed validation samples per class; never use test for selection."""
    rng = np.random.default_rng(seed)
    result = []
    for label in range(num_classes):
        choices = np.flatnonzero(np.asarray(labels) == label)
        if len(choices) < 2:
            raise ValueError(f"Validation class {label} needs at least two probes.")
        result.extend(sorted(rng.choice(choices, 2, replace=False).tolist()))
    return np.asarray(result, np.int64)


def capture_probes(model, bundle, indices, output_dir, tag, *, scale=0):
    import tensorflow as tf
    layer = model.get_layer("parallel_pqc20")
    encoder = tf.keras.Model(model.input, model.get_layer("dense_128_20").output)
    angles = np.asarray(encoder(np.asarray(bundle.features["val"][indices]), training=False), np.float64)
    states = layer.engine.snapshots(angles, np.asarray(layer.theta), scale=scale)
    diagnostics = state_diagnostics(states)
    output_dir = Path(output_dir)
    save_npz(output_dir/f"{tag}.npz", indices=indices, labels=bundle.labels["val"][indices],
             angles=angles, quantum_weights=np.asarray(layer.theta),
             **({"statevectors": states} if scale == 0 else {"density_matrices": states}),
             **diagnostics)
    payload = {
        "id": tag, "split": "validation", "indices": indices.tolist(),
        "labels": bundle.labels["val"][indices].tolist(), "noise_scale": scale,
        "stage_order": list(STAGES), "branch_order": [0, 1, 2, 3],
        "bitstrings": [f"{i:05b}" for i in range(32)], "two_qubit_pairs": list(PAIRS),
        "probability_sum_max_error": float(np.max(np.abs(
            diagnostics["probabilities_z"].sum(axis=-1)-1))),
        "entropy_definitions": {
            "global": "von Neumann S(rho) = -Tr rho log2 rho, bits; zero for an ideal pure state",
            "subsystem": "von Neumann S(Tr_other rho), bits; entanglement for ideal pure states",
            "measurement": "Shannon entropy of Z measurement probabilities, bits",
            "snr": "auxiliary 10 log10(mean(ideal_Y^2)/mean((observed_Y-ideal_Y)^2)), dB",
        },
        "arrays": f"{tag}.npz",
    }
    atomic_write_json(output_dir/f"{tag}.json", payload)
    return states
