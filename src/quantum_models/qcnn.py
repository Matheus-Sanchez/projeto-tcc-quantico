"""Shared QCNN topology and the PennyLane implementation.

The ``U_SU4``/``Pooling_ansatz1`` blocks and connectivity were adapted from
takh04/QCNN, commit 9091189e198cb9d50c5e938b224a17bcfb2c14f9, Apache-2.0.
See ``third_party/takh04-qcnn-NOTICE.md``. The input is always a detached
256-dimensional vector from a frozen CNN; the output is multiclass logits.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

N_QUBITS, N_FEATURES, N_PARAMETERS = 8, 256, 51


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Validate and L2-normalize a detached CPU feature batch."""

    if not isinstance(features, torch.Tensor):
        raise TypeError("Converta as características para torch.Tensor.")
    if features.requires_grad:
        raise ValueError("A CNN deve estar congelada: use features.detach().")
    if features.device.type != "cpu":
        raise ValueError("A simulação QCNN usa CPU; use features.cpu().")
    if features.ndim != 2 or features.shape[1] != N_FEATURES or features.shape[0] == 0:
        raise ValueError(f"A entrada deve ter forma (batch, {N_FEATURES}), com batch > 0.")
    if features.is_complex():
        raise ValueError("A QCNN recebe características reais.")
    values = features.to(dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("A entrada contém NaN ou infinito.")
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or torch.any(norms <= 1e-12):
        raise ValueError("Amplitude encoding exige vetores finitos e não nulos.")
    return values / norms


def apply_qcnn(theta: Any, su4: Any, pool: Any) -> None:
    """Schedule the shared-parameter QCNN topology through backend callbacks."""

    for first, second in [(0, 7), (0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6)]:
        su4(theta[:15], first, second)
    for control, target in [(1, 0), (3, 2), (5, 4), (7, 6)]:
        pool(theta[45:47], control, target)
    for first, second in [(0, 6), (0, 2), (4, 6), (2, 4)]:
        su4(theta[15:30], first, second)
    for control, target in [(2, 0), (6, 4)]:
        pool(theta[47:49], control, target)
    su4(theta[30:45], 0, 4)
    pool(theta[49:51], 0, 4)


class HybridHead(nn.Module):
    """The only trainable stage: 51 quantum parameters plus ``Linear(256,C)``."""

    def __init__(self, n_classes: int, seed: int = None) -> None:
        super().__init__()
        import random
        self.seed = seed if seed is not None else random.randint(1, 999999)
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("n_classes deve ser um inteiro >= 2.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.theta = nn.Parameter(0.1 * torch.randn(N_PARAMETERS, dtype=torch.float64))
            self.classifier = nn.Linear(N_FEATURES, n_classes, dtype=torch.float64)
        self._quantum_forward_samples = 0
        self._quantum_forward_batches = 0

    def quantum_features(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _record_quantum_forward(self, batch_size: int) -> None:
        self._quantum_forward_samples += int(batch_size)
        self._quantum_forward_batches += 1

    @property
    def quantum_forward_samples(self) -> int:
        """Logical QNode evaluations requested in the forward direction."""

        return int(self._quantum_forward_samples)

    @property
    def quantum_forward_batches(self) -> int:
        return int(self._quantum_forward_batches)

    def quantum_metadata(self) -> dict[str, Any]:
        """Describe the common circuit independently of a framework backend."""

        return {
            "n_qubits": N_QUBITS,
            "state_dimension": N_FEATURES,
            "measurement": "probabilities on all 8 qubits",
            "encoding": "L2-normalized amplitude encoding",
            "quantum_parameters": N_PARAMETERS,
            "parameter_groups": {"su4": 45, "pooling": 6},
        }

    def forward_with_quantum_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probabilities and logits without evaluating the circuit twice."""

        probabilities = self.quantum_features(features)
        return probabilities, self.classifier(probabilities)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _probabilities, logits = self.forward_with_quantum_features(features)
        return logits


class PennyLaneQCNN(HybridHead):
    """Exact PennyLane simulator with native PyTorch backpropagation."""

    def __init__(self, n_classes: int, seed: int = None) -> None:
        super().__init__(n_classes, seed)
        try:
            import pennylane as qml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("PennyLane não está instalado.") from exc

        def su4(parameters: torch.Tensor, first: int, second: int) -> None:
            qml.U3(parameters[0], parameters[1], parameters[2], wires=first)
            qml.U3(parameters[3], parameters[4], parameters[5], wires=second)
            qml.CNOT(wires=[first, second])
            qml.RY(parameters[6], wires=first)
            qml.RZ(parameters[7], wires=second)
            qml.CNOT(wires=[second, first])
            qml.RY(parameters[8], wires=first)
            qml.CNOT(wires=[first, second])
            qml.U3(parameters[9], parameters[10], parameters[11], wires=first)
            qml.U3(parameters[12], parameters[13], parameters[14], wires=second)

        def pool(parameters: torch.Tensor, control: int, target: int) -> None:
            qml.CRZ(parameters[0], wires=[control, target])
            qml.PauliX(wires=control)
            qml.CRX(parameters[1], wires=[control, target])

        device = qml.device("default.qubit", wires=N_QUBITS, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, theta: torch.Tensor) -> Any:
            qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=False)
            apply_qcnn(theta, su4, pool)
            return qml.probs(wires=range(N_QUBITS))

        self.circuit = circuit
        self._pennylane_device = "default.qubit"

    def quantum_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = normalize_features(features)
        self._record_quantum_forward(int(normalized.shape[0]))
        return torch.stack([self.circuit(row, self.theta) for row in normalized])

    def quantum_metadata(self) -> dict[str, Any]:
        return {
            **super().quantum_metadata(),
            "framework": "pennylane",
            "device": self._pennylane_device,
            "shots": None,
            "differentiation": "backprop",
        }


__all__ = [
    "HybridHead", "N_FEATURES", "N_PARAMETERS", "N_QUBITS", "PennyLaneQCNN", "apply_qcnn", "normalize_features",
]
