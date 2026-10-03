"""Serializable FP32 Keras layer with explicit input and weight gradients."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from quantum_models.circuit_spec import WEIGHT_SHAPE, initial_weights
from quantum_models.engine import ParallelEngine


@tf.keras.utils.register_keras_serializable(package="TCC")
class ParallelPQC20(tf.keras.layers.Layer):
    def __init__(self, backend="pennylane", seed=42, simulation_batch=256, workers=4, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        if backend not in ("pennylane", "qiskit"):
            raise ValueError(backend)
        self.backend, self.seed = backend, int(seed)
        self.simulation_batch, self.workers = int(simulation_batch), int(workers)
        self.input_spec = tf.keras.layers.InputSpec(ndim=2, axes={-1: 20})
        self.engine = ParallelEngine(backend, self.simulation_batch, self.workers)
        self.last_inputs = None
        self.last_outputs = None

    def build(self, input_shape):
        self.theta = self.add_weight(
            name="quantum_weights", shape=WEIGHT_SHAPE, dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(initial_weights(self.seed)), trainable=True)
        super().build(input_shape)

    def _forward_numpy(self, x, theta):
        output = self.engine.forward(x, theta).astype(np.float32)
        self.last_inputs = np.asarray(x).copy()
        self.last_outputs = output.copy()
        return output

    def _backward_numpy(self, x, theta, upstream):
        dx, dw = self.engine.vjp(x, theta, upstream)
        return dx.astype(np.float32), dw.astype(np.float32)

    def call(self, inputs):
        @tf.custom_gradient
        def simulate(x, theta):
            output = tf.numpy_function(self._forward_numpy, [x, theta], tf.float32)
            output.set_shape(x.shape)

            def gradient(upstream):
                dx, dw = tf.numpy_function(self._backward_numpy, [x, theta, upstream],
                                          [tf.float32, tf.float32])
                dx.set_shape(x.shape)
                dw.set_shape(WEIGHT_SHAPE)
                return dx, dw

            return output, gradient

        return simulate(tf.cast(inputs, tf.float32), tf.convert_to_tensor(self.theta))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "backend": self.backend, "seed": self.seed,
                "simulation_batch": self.simulation_batch, "workers": self.workers}

    def close(self):
        self.engine.close()


def build_parallel_dense20(num_classes, *, backend="pennylane", seed=42,
                           simulation_batch=256, learning_rate=None):
    from utils.experiment_config import INPUT_FEATURES, PROJECTION_UNITS, HIDDEN_UNITS, LEARNING_RATE
    # This is the same Keras initializer sequence as the classical vector head.
    # Quantum initialization uses a separate NumPy generator and consumes no TF RNG.
    inputs = tf.keras.Input((INPUT_FEATURES,), dtype=tf.float32, name="features128")
    projection = tf.keras.layers.Dense(PROJECTION_UNITS, activation="relu",
                                       name="dense_128_128")(inputs)
    angles = tf.keras.layers.Dense(HIDDEN_UNITS, name="dense_128_20")(projection)
    readings = ParallelPQC20(backend=backend, seed=seed, simulation_batch=simulation_batch,
                            name="parallel_pqc20")(angles)
    logits = tf.keras.layers.Dense(int(num_classes), dtype="float32", name="logits")(readings)
    model = tf.keras.Model(inputs, logits, name=f"quantum_128_20_{backend}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE if learning_rate is None else learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")], jit_compile=False)
    expected = 19_272 + 21 * int(num_classes)
    if model.count_params() != expected:
        raise RuntimeError(f"Unexpected parameter count: {model.count_params()} != {expected}")
    return model
