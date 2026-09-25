"""Qiskit implementation equivalent to :mod:`quantum_models.qcnn`.

Adapted source: takh04/QCNN (Apache-2.0); see the third-party notice. The
autograd bridge supports first-order gradients for the frozen-CNN setting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .qcnn import HybridHead, N_PARAMETERS, N_QUBITS, apply_qcnn, normalize_features


def _imports() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterVector
        from qiskit.circuit.library import StatePreparation
        from qiskit.quantum_info import Operator, Statevector
        from qiskit_algorithms.gradients import ReverseEstimatorGradient
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("Qiskit e qiskit-algorithms são necessários para este backend.") from exc
    return QuantumCircuit, transpile, ParameterVector, StatePreparation, Operator, Statevector, ReverseEstimatorGradient


def build_ansatz() -> tuple[Any, Any]:
    QuantumCircuit, transpile, ParameterVector, _, _, _, _ = _imports()
    theta = ParameterVector("theta", N_PARAMETERS)
    circuit = QuantumCircuit(N_QUBITS)

    def wire(logical: int) -> int:
        # Align PennyLane |q0...q7> to Qiskit's |q7...q0> ordering.
        return N_QUBITS - 1 - logical

    def su4(parameters: Any, logical_first: int, logical_second: int) -> None:
        first, second = wire(logical_first), wire(logical_second)
        circuit.u(parameters[0], parameters[1], parameters[2], first)
        circuit.u(parameters[3], parameters[4], parameters[5], second)
        circuit.cx(first, second)
        circuit.ry(parameters[6], first)
        circuit.rz(parameters[7], second)
        circuit.cx(second, first)
        circuit.ry(parameters[8], first)
        circuit.cx(first, second)
        circuit.u(parameters[9], parameters[10], parameters[11], first)
        circuit.u(parameters[12], parameters[13], parameters[14], second)

    def pool(parameters: Any, logical_control: int, logical_target: int) -> None:
        control, target = wire(logical_control), wire(logical_target)
        circuit.crz(parameters[0], control, target)
        circuit.x(control)
        circuit.crx(parameters[1], control, target)

    apply_qcnn(theta, su4, pool)
    circuit = transpile(circuit, basis_gates=["rx", "ry", "rz", "cx", "x"], optimization_level=0)
    # This parameterized global phase from U3 does not change probabilities.
    circuit.global_phase = 0
    return circuit, theta


def with_state_preparation(vector: np.ndarray, ansatz: Any) -> Any:
    QuantumCircuit, _, _, StatePreparation, _, _, _ = _imports()
    circuit = QuantumCircuit(N_QUBITS)
    circuit.append(StatePreparation(np.asarray(vector, dtype=np.complex128)), range(N_QUBITS))
    circuit.compose(ansatz, inplace=True)
    return circuit


class _QiskitProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, theta: torch.Tensor, ansatz: Any, parameters: Any) -> torch.Tensor:
        _, _, _, _, _, Statevector, _ = _imports()
        ctx.values = theta.detach().cpu().numpy().copy()
        ctx.rows = features.detach().cpu().numpy().copy()
        ctx.ansatz, ctx.parameters = ansatz, parameters
        bound = ansatz.assign_parameters(dict(zip(parameters, ctx.values)))
        probabilities = np.stack([Statevector(row).evolve(bound).probabilities() for row in ctx.rows])
        return torch.from_numpy(probabilities).to(dtype=theta.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None, None]:
        _, _, _, _, Operator, _, ReverseEstimatorGradient = _imports()
        gradient = ReverseEstimatorGradient()
        circuits = [with_state_preparation(row, ctx.ansatz) for row in ctx.rows]
        observables = [Operator(np.diag(row)) for row in grad_output.detach().cpu().numpy()]
        result = gradient.run(
            circuits=circuits,
            observables=observables,
            parameter_values=[ctx.values] * len(circuits),
            parameters=[list(ctx.parameters)] * len(circuits),
        ).result()
        grad = np.sum(np.asarray(result.gradients), axis=0)
        return None, torch.from_numpy(grad).to(dtype=grad_output.dtype), None, None


class QiskitQCNN(HybridHead):
    def __init__(self, n_classes: int, seed: int = None) -> None:
        super().__init__(n_classes, seed)
        self.ansatz, self.parameters_qiskit = build_ansatz()

    def quantum_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = normalize_features(features)
        self._record_quantum_forward(int(normalized.shape[0]))
        return _QiskitProbabilities.apply(normalized, self.theta, self.ansatz, self.parameters_qiskit)

    def quantum_metadata(self) -> dict[str, Any]:
        return {
            **super().quantum_metadata(),
            "framework": "qiskit",
            "device": "Statevector",
            "shots": None,
            "differentiation": "ReverseEstimatorGradient",
        }


__all__ = ["QiskitQCNN", "build_ansatz", "with_state_preparation"]
