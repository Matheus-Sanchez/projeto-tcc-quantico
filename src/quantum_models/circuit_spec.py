"""Library-independent definition and canonical wire convention.

Wire 0 is the leftmost bit / most significant axis of a statevector. Each
repetition uses the directed circular CNOT chain in the supplied reference:
control w, target (w + range) % 5, in ascending control order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

N_QUBITS = 5
N_CIRCUITS = 4
N_LAYERS = 3
RANGES = (1, 2, 3)
WEIGHT_SHAPE = (N_CIRCUITS, N_LAYERS, N_QUBITS, 3)
BRANCH_SHAPE = WEIGHT_SHAPE[1:]
GROUPS = tuple((5 * i, 5 * (i + 1)) for i in range(4))
STAGES = ("embedding", "layer_1", "layer_2", "layer_3")
SHIFT = np.pi / 2
PAULIS = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}
BITS = ((np.arange(32)[:, None] >> np.arange(4, -1, -1)) & 1)
SIGNS = (1 - 2 * BITS).astype(np.float64)


@dataclass(frozen=True)
class NoiseProfile:
    one_qubit_depolarizing: float = 0.002
    cnot_per_qubit_depolarizing: float = 0.01
    amplitude_damping: float = 0.001
    phase_damping: float = 0.001

    def to_dict(self):
        return asdict(self)


def noise_channels(scale: float, *, cnot: bool = False) -> tuple[tuple[np.ndarray, ...], ...]:
    """Identical Kraus channels in both SDKs; p is Pauli-error probability."""
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Noise scale must be finite and nonnegative.")
    if scale == 0:
        return ()
    profile = NoiseProfile()
    p = scale * (profile.cnot_per_qubit_depolarizing if cnot
                 else profile.one_qubit_depolarizing)
    a = scale * profile.amplitude_damping
    d = scale * profile.phase_damping
    if max(p, a, d) > 1:
        raise ValueError("Scaled noise probabilities exceed one.")
    depol = (np.sqrt(1-p) * np.eye(2, dtype=complex),
             *(np.sqrt(p/3) * PAULIS[b] for b in ("X", "Y", "Z")))
    amplitude = (np.diag([1, np.sqrt(1-a)]).astype(complex),
                 np.array([[0, np.sqrt(a)], [0, 0]], dtype=complex))
    phase = (np.diag([1, np.sqrt(1-d)]).astype(complex),
             np.diag([0, np.sqrt(d)]).astype(complex))
    return depol, amplitude, phase


def gates() -> list[dict]:
    result = [{"gate": "RX", "wire": w, "input": w} for w in range(5)]
    for layer, distance in enumerate(RANGES):
        for wire in range(5):
            for rotation, name in enumerate(("RZ", "RY", "RZ")):
                result.append({"gate": name, "wire": wire,
                               "weight": [layer, wire, rotation]})
        result.extend({"gate": "CX", "control": w, "target": (w+distance) % 5,
                       "layer": layer} for w in range(5))
    return result


def circuit_metadata() -> dict:
    schedule = gates()
    depth = [0] * 5
    for gate in schedule:
        wires = [gate["wire"]] if "wire" in gate else [gate["control"], gate["target"]]
        step = max(depth[w] for w in wires) + 1
        for wire in wires:
            depth[wire] = step
    return {
        "qubits_per_branch": 5, "branches": 4, "repetitions": 3,
        "ranges": list(RANGES), "groups": list(GROUPS), "weight_shape": list(WEIGHT_SHAPE),
        "parameters_per_branch": 45, "parameters": 180,
        "elementary_gates_per_branch": len(schedule), "cnot_per_branch": 15,
        "depth_per_branch": max(depth), "observable": "PauliY on wires 0..4",
        "wire_order": "q0 is the leftmost bit and most significant state axis",
        "embedding": "RX(raw Dense20 output in radians), once",
        "noise_insertion": "after RX, after each complete RZ-RY-RZ triplet, "
                           "and after each CNOT on control then target",
        "noise_profile": NoiseProfile().to_dict(),
        "gates": schedule,
        "logical_evaluations_per_training_example": 4 * (1 + 2 * (45 + 5)),
    }


def initial_weights(seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 2*np.pi, WEIGHT_SHAPE).astype(np.float32)


def bit_reversal(values: np.ndarray) -> np.ndarray:
    """Aer little-endian amplitudes -> canonical wire-0-first amplitudes."""
    indices = BITS[:, ::-1] @ (1 << np.arange(4, -1, -1))
    values = np.asarray(values)
    if values.shape[-2:] == (32, 32):
        return values[..., indices, :][..., :, indices]
    return values[..., indices]
