"""Quantization helpers shared by the reproducible KMNIST experiment."""

from __future__ import annotations

import csv
import dataclasses
import html
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from utils.state import atomic_write_bytes, atomic_write_json, atomic_write_text


@dataclasses.dataclass(frozen=True)
class QuantizationVariant:
    key: str
    label: str
    dtype_policy: str
    qat_weight_bits: int | None = None
    post_training: bool = False
    emulated: bool = False

    @property
    def trains(self) -> bool:
        return not self.post_training


VARIANTS: tuple[QuantizationVariant, ...] = (
    QuantizationVariant("fp32", "CNN FP32", "float32"),
    QuantizationVariant("fp16", "CNN FP16", "mixed_float16"),
    QuantizationVariant("int8_qat", "CNN INT8 QAT", "float32", qat_weight_bits=8),
    QuantizationVariant("int4_qat", "CNN INT4 QAT (emulado)", "float32", qat_weight_bits=4, emulated=True),
    QuantizationVariant("int8_ptq", "CNN INT8 PTQ", "float32", post_training=True),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}
TRAINING_VARIANTS = tuple(variant for variant in VARIANTS if variant.trains)


def variant_for(value: str) -> QuantizationVariant:
    try:
        return VARIANT_BY_KEY[str(value)]
    except KeyError as exc:
        raise ValueError(f"Variante desconhecida: {value!r}.") from exc


try:
    import tensorflow as _tf
except Exception:
    _tf = None


if _tf is not None:

    @_tf.keras.utils.register_keras_serializable(package="tcc_benchmark")
    class FakeQuantize(_tf.keras.layers.Layer):
        def __init__(self, *, num_bits: int = 8, momentum: float = 0.99, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.num_bits = int(num_bits)
            self.momentum = float(momentum)

        def build(self, input_shape: Any) -> None:
            self.minimum = self.add_weight(
                name="minimum", shape=(), dtype="float32", trainable=False, initializer=_tf.keras.initializers.Constant(-6.0)
            )
            self.maximum = self.add_weight(
                name="maximum", shape=(), dtype="float32", trainable=False, initializer=_tf.keras.initializers.Constant(6.0)
            )
            super().build(input_shape)

        def call(self, inputs: Any, training: bool | None = None) -> Any:
            values = _tf.cast(inputs, _tf.float32)
            if training is True:
                observed_minimum = _tf.minimum(_tf.reduce_min(values), _tf.constant(-1e-6, dtype=_tf.float32))
                observed_maximum = _tf.maximum(_tf.reduce_max(values), _tf.constant(1e-6, dtype=_tf.float32))
                self.minimum.assign(self.momentum * self.minimum + (1.0 - self.momentum) * observed_minimum)
                self.maximum.assign(self.momentum * self.maximum + (1.0 - self.momentum) * observed_maximum)
            return _tf.quantization.fake_quant_with_min_max_vars(
                values, min=self.minimum, max=self.maximum, num_bits=self.num_bits, narrow_range=False
            )

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "num_bits": self.num_bits, "momentum": self.momentum}


    def _fake_quantize_weight(values: Any, *, num_bits: int) -> Any:
        values = _tf.cast(values, _tf.float32)
        bound = _tf.maximum(_tf.reduce_max(_tf.abs(values)), _tf.constant(1e-6, dtype=_tf.float32))
        return _tf.quantization.fake_quant_with_min_max_vars(
            values, min=-bound, max=bound, num_bits=int(num_bits), narrow_range=True
        )


    @_tf.keras.utils.register_keras_serializable(package="tcc_benchmark")
    class QATSeparableConv2D(_tf.keras.layers.Layer):
        def __init__(self, filters: int, kernel_size: Any, *, strides: Any = 1, padding: str = "same", use_bias: bool = False, depthwise_initializer: Any = "he_normal", pointwise_initializer: Any = "he_normal", weight_bits: int = 8, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.filters = int(filters)
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.strides = strides if isinstance(strides, tuple) else (strides, strides)
            self.padding = str(padding).upper()
            self.use_bias = bool(use_bias)
            self.depthwise_initializer = _tf.keras.initializers.get(depthwise_initializer)
            self.pointwise_initializer = _tf.keras.initializers.get(pointwise_initializer)
            self.weight_bits = int(weight_bits)

        def build(self, input_shape: Any) -> None:
            channels = int(input_shape[-1])
            self.depthwise_kernel = self.add_weight(name="depthwise_kernel", shape=(*self.kernel_size, channels, 1), initializer=self.depthwise_initializer, trainable=True)
            self.pointwise_kernel = self.add_weight(name="pointwise_kernel", shape=(1, 1, channels, self.filters), initializer=self.pointwise_initializer, trainable=True)
            self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True) if self.use_bias else None
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            values = _tf.cast(inputs, _tf.float32)
            depthwise = _tf.nn.depthwise_conv2d(values, _fake_quantize_weight(self.depthwise_kernel, num_bits=self.weight_bits), strides=(1, *self.strides, 1), padding=self.padding)
            output = _tf.nn.conv2d(depthwise, _fake_quantize_weight(self.pointwise_kernel, num_bits=self.weight_bits), strides=(1, 1, 1, 1), padding="SAME")
            if self.bias is not None:
                output = _tf.nn.bias_add(output, self.bias)
            return output

        def compute_output_shape(self, input_shape: Any) -> Any:
            height, width = input_shape[1], input_shape[2]
            if self.padding == "SAME":
                height = math.ceil(height / self.strides[0]) if height is not None else None
                width = math.ceil(width / self.strides[1]) if width is not None else None
            else:
                height = math.floor((height - self.kernel_size[0]) / self.strides[0]) + 1 if height is not None else None
                width = math.floor((width - self.kernel_size[1]) / self.strides[1]) + 1 if width is not None else None
            return (input_shape[0], height, width, self.filters)

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides, "padding": self.padding.lower(), "use_bias": self.use_bias, "depthwise_initializer": _tf.keras.initializers.serialize(self.depthwise_initializer), "pointwise_initializer": _tf.keras.initializers.serialize(self.pointwise_initializer), "weight_bits": self.weight_bits}


    @_tf.keras.utils.register_keras_serializable(package="tcc_benchmark")
    class QATDense(_tf.keras.layers.Layer):
        def __init__(self, units: int, *, activation: str | None = None, use_bias: bool = True, kernel_initializer: Any = "glorot_uniform", weight_bits: int = 8, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.units = int(units)
            self.activation = _tf.keras.activations.get(activation)
            self.use_bias = bool(use_bias)
            self.kernel_initializer = _tf.keras.initializers.get(kernel_initializer)
            self.weight_bits = int(weight_bits)

        def build(self, input_shape: Any) -> None:
            features = int(input_shape[-1])
            self.kernel = self.add_weight(name="kernel", shape=(features, self.units), initializer=self.kernel_initializer, trainable=True)
            self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True) if self.use_bias else None
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            output = _tf.linalg.matmul(_tf.cast(inputs, _tf.float32), _fake_quantize_weight(self.kernel, num_bits=self.weight_bits))
            if self.bias is not None:
                output = _tf.nn.bias_add(output, self.bias)
            return self.activation(output) if self.activation is not None else output

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "units": self.units, "activation": _tf.keras.activations.serialize(self.activation), "use_bias": self.use_bias, "kernel_initializer": _tf.keras.initializers.serialize(self.kernel_initializer), "weight_bits": self.weight_bits}

else:

    class _TensorFlowRequiredLayer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("TensorFlow é necessário para construir camadas QAT.")

    FakeQuantize = _TensorFlowRequiredLayer
    QATSeparableConv2D = _TensorFlowRequiredLayer
    QATDense = _TensorFlowRequiredLayer

