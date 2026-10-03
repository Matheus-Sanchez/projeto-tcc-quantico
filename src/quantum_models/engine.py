"""Four persistent private workers and bounded parameter-shift batches."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import time
import threading
import numpy as np
from quantum_models.circuit_spec import WEIGHT_SHAPE, BRANCH_SHAPE, SHIFT

_BACKEND = None


def _initialize(name):
    global _BACKEND
    # Each process owns one SDK device; avoid BLAS/OpenMP oversubscription.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    from quantum_models.backends import make_backend
    _BACKEND = make_backend(name)


def _evaluate(backend, inputs, weights, chunk_size):
    return np.concatenate([backend.evaluate(inputs[i:i+chunk_size],
                                           weights if weights.ndim == 3 else weights[i:i+chunk_size])
                           for i in range(0, len(inputs), chunk_size)], axis=0)


def _branch_task(operation, inputs, weights, upstream, chunk_size, scale):
    start = time.perf_counter()
    inputs, weights = np.asarray(inputs, np.float64), np.asarray(weights, np.float64)
    if operation == "forward":
        output = _evaluate(_BACKEND, inputs, weights, chunk_size)
        return output, {"seconds": time.perf_counter()-start, "evaluations": len(inputs)}
    if operation == "snapshots":
        output = np.concatenate([_BACKEND.snapshots(inputs[i:i+chunk_size], weights, scale=scale)
                                 for i in range(0, len(inputs), chunk_size)])
        return output, {"seconds": time.perf_counter()-start, "evaluations": len(inputs)*4}
    if operation != "vjp":
        raise ValueError(operation)
    # Build only bounded [simulation_batch, 5] and [simulation_batch,3,5,3]
    # arrays, instead of materializing B*100 states or a dense Jacobian.
    dx = np.empty_like(inputs)
    dw = np.zeros(BRANCH_SHAPE, np.float64)
    derivative_stats = []
    for parameter in range(50):
        plus_x = inputs.copy()
        minus_x = inputs.copy()
        plus_w = np.broadcast_to(weights, (len(inputs), *BRANCH_SHAPE)).copy()
        minus_w = plus_w.copy()
        if parameter < 5:
            plus_x[:, parameter] += SHIFT
            minus_x[:, parameter] -= SHIFT
        else:
            plus_w.reshape(len(inputs), 45)[:, parameter-5] += SHIFT
            minus_w.reshape(len(inputs), 45)[:, parameter-5] -= SHIFT
        jac = (_evaluate(_BACKEND, plus_x, plus_w, chunk_size) -
               _evaluate(_BACKEND, minus_x, minus_w, chunk_size)) * 0.5
        contracted = np.einsum("bi,bi->b", jac, upstream)
        if parameter < 5:
            dx[:, parameter] = contracted
        else:
            dw.flat[parameter-5] = np.sum(contracted, dtype=np.float64)
        derivative_stats.append(float(np.var(contracted)))
    return (dx, dw), {"seconds": time.perf_counter()-start, "evaluations": len(inputs)*100,
                     "input_gradient_norm": float(np.linalg.norm(dx)),
                     "input_gradient_variance": float(np.var(dx)),
                     "weight_gradient_norm": float(np.linalg.norm(dw)),
                     "weight_gradient_variance": float(np.var(dw)),
                     "per_example_gradient_variance_mean": float(np.mean(derivative_stats))}


class ParallelEngine:
    """Exactly one process per branch, created lazily and reused until close."""

    def __init__(self, backend="pennylane", chunk_size=256, workers=4):
        if backend not in ("pennylane", "qiskit"):
            raise ValueError(backend)
        if workers != 4:
            raise ValueError("ParallelPQC20 requires four private local workers.")
        if int(chunk_size) < 1:
            raise ValueError("simulation batch must be positive")
        self.backend, self.chunk_size = backend, int(chunk_size)
        self.pools = []
        self.records = {"forward": [], "vjp": [], "snapshots": []}
        self.lock = threading.RLock()

    def _start(self):
        if not self.pools:
            context = multiprocessing.get_context("spawn")
            self.pools = [ProcessPoolExecutor(max_workers=1, mp_context=context,
                                             initializer=_initialize, initargs=(self.backend,))
                          for _ in range(4)]

    def reset_records(self):
        with self.lock:
            self.records = {"forward": [], "vjp": [], "snapshots": []}

    def _run(self, operation, inputs, weights, upstream=None, scale=0):
        inputs, weights = np.asarray(inputs, np.float64), np.asarray(weights, np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != 20 or weights.shape != WEIGHT_SHAPE:
            raise ValueError(f"Invalid PQC input/weight shape: {inputs.shape}, {weights.shape}")
        if not len(inputs) or not np.isfinite(inputs).all() or not np.isfinite(weights).all():
            raise ValueError("PQC inputs and weights must be finite and nonempty.")
        with self.lock:
            self._start()
            start = time.perf_counter()
            futures = [pool.submit(_branch_task, operation, inputs[:, i*5:(i+1)*5],
                                   weights[i], None if upstream is None else upstream[:, i*5:(i+1)*5],
                                   self.chunk_size, float(scale)) for i, pool in enumerate(self.pools)]
            results = [future.result() for future in futures]
            self.records[operation].append({
                "wall_seconds": time.perf_counter()-start,
                "examples": len(inputs), "circuits": [r[1] for r in results],
                "evaluations": sum(r[1]["evaluations"] for r in results),
            })
        if operation == "vjp":
            return (np.concatenate([r[0][0] for r in results], axis=1),
                    np.stack([r[0][1] for r in results]))
        if operation == "snapshots":
            return np.stack([r[0] for r in results], axis=1)
        return np.concatenate([r[0] for r in results], axis=1)

    def forward(self, inputs, weights):
        return self._run("forward", inputs, weights)

    def vjp(self, inputs, weights, upstream):
        upstream = np.asarray(upstream, np.float64)
        if upstream.shape != np.shape(inputs) or not np.isfinite(upstream).all():
            raise ValueError("Invalid upstream PQC gradient.")
        return self._run("vjp", inputs, weights, upstream)

    def snapshots(self, inputs, weights, scale=0):
        return self._run("snapshots", inputs, weights, scale=scale)

    def close(self):
        with self.lock:
            for pool in self.pools:
                pool.shutdown(wait=True, cancel_futures=True)
            self.pools.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
