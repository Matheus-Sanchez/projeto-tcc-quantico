"""Local simulator adapters. No provider, credentials, or remote API."""
from __future__ import annotations

import numpy as np
from quantum_models.circuit_spec import RANGES, BRANCH_SHAPE, bit_reversal, noise_channels


class PennyLaneBackend:
    name = "pennylane"

    def __init__(self):
        import pennylane as qml
        self.qml = qml
        self.device = qml.device("default.qubit", wires=5, shots=None)

        @qml.qnode(self.device, interface=None, diff_method=None)
        def forward(x, weights):
            self._apply(x, weights)
            return tuple(qml.expval(qml.PauliY(w)) for w in range(5))
        self.qnode = forward
        self._probes = {}

    def _apply(self, x, weights, *, scale=0.0, stage=3):
        qml = self.qml
        single = noise_channels(scale)
        double = noise_channels(scale, cnot=True)

        def channels(wire, matrices):
            for operators in matrices:
                qml.QubitChannel(list(operators), wires=wire)

        for wire in range(5):
            qml.RX(x[..., wire], wires=wire)
            channels(wire, single)
        for layer in range(stage):
            for wire in range(5):
                qml.RZ(weights[..., layer, wire, 0], wires=wire)
                qml.RY(weights[..., layer, wire, 1], wires=wire)
                qml.RZ(weights[..., layer, wire, 2], wires=wire)
                channels(wire, single)
            for wire in range(5):
                target = (wire + RANGES[layer]) % 5
                qml.CNOT(wires=[wire, target])
                channels(wire, double)
                channels(target, double)

    def evaluate(self, inputs, weights):
        inputs = np.asarray(inputs, dtype=np.float64).reshape(-1, 5)
        weights = np.asarray(weights, dtype=np.float64)
        values = np.asarray(self.qnode(inputs, weights), dtype=np.float64)
        return np.moveaxis(values, 0, -1).reshape(len(inputs), 5)

    def snapshots(self, inputs, weights, *, scale=0.0):
        """Four snapshots, [example, stage, state] or density matrices."""
        qml = self.qml
        key = float(scale)
        if key not in self._probes:
            nodes = []
            for stage in range(4):
                device = qml.device("default.mixed" if scale else "default.qubit",
                                    wires=5, shots=None)

                def circuit(x, theta, stage=stage):
                    self._apply(x, theta, scale=scale, stage=stage)
                    return qml.density_matrix(wires=range(5)) if scale else qml.state()

                nodes.append(qml.QNode(circuit, device, interface=None, diff_method=None))
            self._probes[key] = nodes
        # default.mixed does not broadcast arbitrary Kraus channels reliably.
        return np.array([[node(np.asarray(x, np.float64), np.asarray(weights, np.float64))
                          for node in self._probes[key]] for x in inputs], dtype=np.complex128)

    def drawing(self):
        return self.qml.draw(self.qnode)(np.zeros(5), np.zeros(BRANCH_SHAPE))


class QiskitBackend:
    name = "qiskit"

    def __init__(self):
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import Pauli
        from qiskit_aer import AerSimulator
        self.simulator = AerSimulator(method="statevector", device="CPU", precision="double",
                                      max_parallel_threads=1, max_parallel_experiments=1)
        self.inputs = ParameterVector("x", 5)
        self.weights = ParameterVector("theta", 45)
        qc = QuantumCircuit(5)
        self._apply(qc, self.inputs, np.array(self.weights, dtype=object).reshape(BRANCH_SHAPE))
        self.diagram = str(qc.draw(output="text"))
        for wire in range(5):
            qc.save_expectation_value(Pauli("Y"), [wire], label=f"y{wire}")
        self.circuit = transpile(qc, self.simulator, optimization_level=0)
        self._probe_circuits = {}

    @staticmethod
    def _apply(qc, x, weights, *, scale=0.0, snapshots=False):
        from qiskit.quantum_info import Kraus

        def channels(wire, cnot=False):
            for matrices in noise_channels(scale, cnot=cnot):
                qc.append(Kraus(list(matrices)).to_instruction(), [wire])

        def save(label):
            if scale:
                qc.save_density_matrix(label=label)
            else:
                qc.save_statevector(label=label)

        for wire in range(5):
            qc.rx(x[wire], wire)
            channels(wire)
        if snapshots:
            save("stage0")
        for layer, distance in enumerate(RANGES):
            for wire in range(5):
                qc.rz(weights[layer, wire, 0], wire)
                qc.ry(weights[layer, wire, 1], wire)
                qc.rz(weights[layer, wire, 2], wire)
                channels(wire)
            for wire in range(5):
                target = (wire + distance) % 5
                qc.cx(wire, target)
                channels(wire, True)
                channels(target, True)
            if snapshots:
                save(f"stage{layer+1}")

    def _bindings(self, inputs, weights):
        inputs = np.asarray(inputs, np.float64).reshape(-1, 5)
        weights = np.broadcast_to(np.asarray(weights, np.float64), (len(inputs), *BRANCH_SHAPE))
        return {**{p: inputs[:, w].tolist() for w, p in enumerate(self.inputs)},
                **{p: weights.reshape(len(inputs), 45)[:, w].tolist()
                   for w, p in enumerate(self.weights)}}

    def evaluate(self, inputs, weights):
        result = self.simulator.run(self.circuit,
                                    parameter_binds=[self._bindings(inputs, weights)],
                                    shots=None).result()
        if not result.success:
            raise RuntimeError(f"Aer local simulation failed: {result.status}")
        return np.array([[result.data(i)[f"y{wire}"] for wire in range(5)]
                         for i in range(len(inputs))], dtype=np.float64)

    def snapshots(self, inputs, weights, *, scale=0.0):
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        if float(scale) not in self._probe_circuits:
            simulator = AerSimulator(method="density_matrix" if scale else "statevector",
                                     device="CPU", precision="double",
                                     max_parallel_threads=1, max_parallel_experiments=1)
            qc = QuantumCircuit(5)
            self._apply(qc, self.inputs,
                        np.array(self.weights, dtype=object).reshape(BRANCH_SHAPE),
                        scale=scale, snapshots=True)
            self._probe_circuits[float(scale)] = (
                simulator, transpile(qc, simulator, optimization_level=0))
        simulator, circuit = self._probe_circuits[float(scale)]
        result = simulator.run(circuit, parameter_binds=[self._bindings(inputs, weights)],
                               shots=None).result()
        if not result.success:
            raise RuntimeError(f"Aer local probe failed: {result.status}")
        states = np.array([[np.asarray(result.data(i)[f"stage{stage}"])
                            for stage in range(4)] for i in range(len(inputs))],
                          dtype=np.complex128)
        return bit_reversal(states)

    def drawing(self):
        return self.diagram


def make_backend(name):
    if name == "pennylane":
        return PennyLaneBackend()
    if name == "qiskit":
        return QiskitBackend()
    raise ValueError(f"Unknown local backend: {name}")
