"""Independent numerical checks for the circuit contract and local SDKs."""
import numpy as np
import pytest
from quantum_models.circuit_spec import gates, circuit_metadata, initial_weights, bit_reversal
from quantum_models.diagnostics import (
    basis_probabilities, density, state_diagnostics, fidelity_and_trace_distance,
    reduced_density, probe_indices, shot_measurements,
)


@pytest.fixture(scope="module", params=["pennylane", "qiskit"])
def backend(request):
    pytest.importorskip(request.param)
    from quantum_models.backends import make_backend
    return make_backend(request.param)


def test_specification():
    spec = circuit_metadata()
    assert spec["groups"] == [(0, 5), (5, 10), (10, 15), (15, 20)]
    assert spec["parameters"] == 180
    assert spec["elementary_gates_per_branch"] == 65
    assert spec["logical_evaluations_per_training_example"] == 404
    schedule = gates()
    assert [g["gate"] for g in schedule[:8]] == ["RX"]*5 + ["RZ", "RY", "RZ"]
    cnot = [g for g in schedule if g["gate"] == "CX"]
    for layer, distance in enumerate((1, 2, 3)):
        assert [(g["control"], g["target"]) for g in cnot[layer*5:(layer+1)*5]] == [
            (w, (w+distance) % 5) for w in range(5)]
    assert initial_weights().shape == (4, 3, 5, 3)
    assert np.all((initial_weights() >= 0) & (initial_weights() < 2*np.pi))


def test_embedding_y_sign_and_wire_order(backend):
    x = np.array([[0.1, -0.3, 0.8, 1.2, -0.9]])
    states = backend.snapshots(x, np.zeros((3, 5, 3)))
    embedding = states[:, 0]
    # RX(x)|0> has <Y> = -sin(x), an independent analytic sign test.
    bloch = state_diagnostics(embedding)["bloch_xyz"]
    np.testing.assert_allclose(bloch[..., 1], -np.sin(x), atol=1e-12)
    np.testing.assert_allclose(bloch[..., 2], np.cos(x), atol=1e-12)
    from quantum_models.circuit_spec import SIGNS
    np.testing.assert_allclose(basis_probabilities(embedding, "Y") @ SIGNS, -np.sin(x), atol=1e-12)


def test_backends_identical_states_outputs_and_kraus_noise():
    pytest.importorskip("pennylane")
    pytest.importorskip("qiskit_aer")
    from quantum_models.backends import make_backend
    p, q = make_backend("pennylane"), make_backend("qiskit")
    x = np.random.default_rng(76).normal(size=(3, 5))
    weights = np.asarray(initial_weights()[0], np.float64)
    a, b = p.evaluate(x, weights), q.evaluate(x, weights)
    np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)
    np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), atol=1e-6, rtol=0)
    for scale in (0, 0.5, 1, 2):
        a, b = p.snapshots(x[:1], weights, scale=scale), q.snapshots(x[:1], weights, scale=scale)
        np.testing.assert_allclose(density(a), density(b), atol=1e-9, rtol=0)
        if not scale:
            phases = np.einsum("...i,...i->...", a.conj(), b)
            np.testing.assert_allclose(a, b*np.exp(-1j*np.angle(phases))[..., None], atol=1e-9, rtol=0)


def test_state_and_subsystem_properties(backend):
    x = np.random.default_rng(1).normal(size=(1, 5))
    for scale in (0, 1):
        state = backend.snapshots(x, initial_weights()[0], scale=scale)
        data = state_diagnostics(state)
        np.testing.assert_allclose(data["probabilities_z"].sum(-1), 1, atol=1e-12)
        for name in ("reduced_one_qubit", "reduced_two_qubit"):
            rho = data[name]
            np.testing.assert_allclose(np.trace(rho, axis1=-2, axis2=-1), 1, atol=1e-12)
            np.testing.assert_allclose(rho, rho.swapaxes(-1, -2).conj(), atol=1e-12)
            assert np.linalg.eigvalsh(rho).min() > -1e-12
        assert np.min(data["entropy_global_bits"]) > -1e-12
        assert np.max(data["purity_global"]) <= 1+1e-12
        if scale == 0:
            np.testing.assert_allclose(data["entropy_global_bits"], 0, atol=1e-11)
            fidelity, distance = fidelity_and_trace_distance(state, state)
            np.testing.assert_allclose(fidelity, 1, atol=1e-9)
            np.testing.assert_allclose(distance, 0, atol=1e-12)


def test_bit_order_bell_subsystem_and_shots():
    state = np.zeros(32, complex)
    state[0] = state[24] = 1/np.sqrt(2)
    np.testing.assert_allclose(reduced_density(density(state), (0,)), np.eye(2)/2, atol=1e-12)
    np.testing.assert_array_equal(bit_reversal(bit_reversal(state)), state)
    readings, counts, se = shot_measurements(basis_probabilities(state, "Z"), 4096, np.random.default_rng(42))
    assert counts.sum() == 4096
    np.testing.assert_array_equal(readings[2:], np.ones(3))
    np.testing.assert_allclose(se[2:], np.zeros(3), atol=1e-9)
    assert np.max(np.abs(readings[:2])) < 0.06
    assert len(probe_indices(np.tile(np.arange(3), 5), 3)) == 6
