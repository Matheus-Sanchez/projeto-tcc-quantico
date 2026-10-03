"""Gradient, branch independence, optimizer, and cross-process restore checks."""
import os
import subprocess
import sys
import numpy as np
import pytest
from quantum_models.circuit_spec import initial_weights
from quantum_models.engine import ParallelEngine
from quantum_models.parallel_dense20 import epoch_order, order_hash
from quantum_models.persistence import CheckpointStore, trim_history


@pytest.fixture(scope="module", params=["pennylane", "qiskit"])
def engine(request):
    pytest.importorskip(request.param)
    instance = ParallelEngine(request.param, chunk_size=16)
    yield instance
    instance.close()


def test_groups_independence_and_parameter_shift_vjp(engine):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 20))
    theta = initial_weights().astype(np.float64)
    upstream = rng.normal(size=(2, 20))
    y = engine.forward(x, theta)
    changed = x.copy()
    changed[:, :5] += 0.2
    other = engine.forward(changed, theta)
    assert y.shape == (2, 20)
    np.testing.assert_allclose(y[:, 5:], other[:, 5:], atol=1e-12, rtol=0)
    dx, dt = engine.vjp(x, theta, upstream)
    epsilon = 1e-6
    for row, column in ((0, 0), (1, 6), (0, 12), (1, 19)):
        plus, minus = x.copy(), x.copy()
        plus[row, column] += epsilon
        minus[row, column] -= epsilon
        finite = np.sum((engine.forward(plus, theta)-engine.forward(minus, theta))*upstream)/(2*epsilon)
        np.testing.assert_allclose(dx[row, column], finite, atol=2e-7, rtol=0)
    for index in ((0, 0, 0, 0), (1, 1, 3, 1), (2, 2, 4, 2), (3, 1, 2, 0)):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite = np.sum((engine.forward(x, plus)-engine.forward(x, minus))*upstream)/(2*epsilon)
        np.testing.assert_allclose(dt[index], finite, atol=2e-7, rtol=0)
    assert dt.shape == (4, 3, 5, 3)


@pytest.mark.parametrize("backend", ["pennylane", "qiskit"])
def test_keras_all_layers_update_and_initialization_matches_classical(backend, tmp_path):
    tf = pytest.importorskip("tensorflow")
    from classic_models.vector_dense20 import build_classical_dense20, configure_reproducibility
    from quantum_models.layer import build_parallel_dense20
    configure_reproducibility(tf)
    classic = build_classical_dense20(tf, num_classes=7)
    expected = {name: classic.get_layer(name).get_weights()
                for name in ("dense_128_128", "dense_128_20", "logits")}
    tf.keras.backend.clear_session()
    configure_reproducibility(tf)
    model = build_parallel_dense20(7, backend=backend, simulation_batch=16)
    layer = model.get_layer("parallel_pqc20")
    try:
        assert model.count_params() == 19_419
        for name in expected:
            for a, b in zip(expected[name], model.get_layer(name).get_weights()):
                np.testing.assert_array_equal(a, b)
        rng = np.random.default_rng(20)
        x, y = rng.normal(size=(2, 128)).astype(np.float32), np.array([0, 4])
        before = [v.numpy().copy() for v in model.weights]
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None and np.isfinite(g).all() for g in gradients)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for name in ("dense_128_128", "dense_128_20", "logits"):
            variable = model.get_layer(name).kernel
            index = next(i for i, v in enumerate(model.weights) if v is variable)
            assert not np.array_equal(before[index], variable.numpy())
        theta_index = next(i for i, v in enumerate(model.weights) if v is layer.theta)
        assert all(not np.array_equal(before[theta_index][i], np.asarray(layer.theta)[i]) for i in range(4))
        reference = np.asarray(model(x))
        store = CheckpointStore(tmp_path/"checkpoints", "fixed")
        state = store.commit(model, 1, 0.3, order_hash(epoch_order(17, 0)))
        np.save(tmp_path/"inputs.npy", x)
        checkpoint = store.confirmed_path()
        layer.close()
        code = (
            "import numpy as np; import tensorflow as tf; import quantum_models.layer; "
            "from pathlib import Path; import sys; "
            "root=Path(sys.argv[1]); model=tf.keras.models.load_model(sys.argv[2]); "
            "np.save(root/'restored.npy',np.asarray(model(np.load(root/'inputs.npy')))); "
            "np.savez(root/'adam.npz',**{f'v{i}':np.asarray(v) for i,v in enumerate(model.optimizer.variables)}); "
            "model.get_layer('parallel_pqc20').close()"
        )
        subprocess.run([sys.executable, "-c", code, str(tmp_path), str(checkpoint)], check=True,
                       env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0"}, timeout=180)
        np.testing.assert_allclose(np.load(tmp_path/"restored.npy"), reference, atol=1e-6, rtol=0)
        with np.load(tmp_path/"adam.npz") as restored:
            for i, variable in enumerate(model.optimizer.variables):
                np.testing.assert_array_equal(restored[f"v{i}"], np.asarray(variable))
        assert state["epoch_order_sha256"] == order_hash(epoch_order(17, 0))
        assert not np.array_equal(epoch_order(17, 0), epoch_order(17, 1))
    finally:
        layer.close()
        tf.keras.backend.clear_session()


def test_uncommitted_history_and_checkpoint_are_ignored(tmp_path):
    path = tmp_path/"history.csv"
    path.write_text("epoch,loss\n1,1.0\n2,0.8\n3,0.7\n", encoding="utf-8")
    trim_history(path, 1)
    assert path.read_text(encoding="utf-8").splitlines() == ["epoch,loss", "1,1.0"]
    store = CheckpointStore(tmp_path/"checkpoints", "one")
    (store.directory/"epoch-00001.keras").write_bytes(b"partial unconfirmed checkpoint")
    assert store.confirmed_path() is None
    np.testing.assert_array_equal(epoch_order(19, 7), epoch_order(19, 7))
