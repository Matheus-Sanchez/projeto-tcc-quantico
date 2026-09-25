"""Scratch-only reconstruction of the legacy separable-convolution CNN."""

from __future__ import annotations

from typing import Any

from data_prep.data import require_tensorflow
from classic_models.quantization import FakeQuantize, QATDense, QATSeparableConv2D

LEGACY_FILTERS: tuple[int, ...] = (32, 48, 64, 96, 128)
TRAINING_DTYPE_POLICY = "mixed_float16"

def set_dtype_policy(dtype_policy: str) -> None:
    """Enable float16 compute while retaining float32 variables and outputs."""
    tensorflow = require_tensorflow()
    current = tensorflow.keras.mixed_precision.global_policy()
    if current.name != str(dtype_policy):
        tensorflow.keras.mixed_precision.set_global_policy(str(dtype_policy))

def set_mixed_float16_policy() -> None:
    set_dtype_policy(TRAINING_DTYPE_POLICY)

def set_training_seed(seed: int, *, enable_op_determinism: bool = True) -> None:
    tensorflow = require_tensorflow()
    tensorflow.keras.utils.set_random_seed(int(seed))
    if enable_op_determinism:
        try:
            tensorflow.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass

def build_legacy_cnn(
    *,
    image_size: int,
    channels: int,
    num_classes: int,
    name: str = "legacy_scratch_cnn",
    dtype_policy: str = "mixed_float16",
    hidden_activation: str = "swish",
    qat_weight_bits: int | None = None,
) -> Any:
    tensorflow = require_tensorflow()
    image_size = int(image_size)
    channels = int(channels)
    num_classes = int(num_classes)
    if image_size < 64:
        raise ValueError("A CNN legada exige image_size >= 64 por causa dos cinco blocos de pooling.")
    if channels not in {1, 3}:
        raise ValueError("channels deve ser 1 (cinza) ou 3 (RGB).")
    if num_classes < 2:
        raise ValueError("num_classes deve ser pelo menos 2 para classificação.")
    set_dtype_policy(dtype_policy)
    if qat_weight_bits is not None and int(qat_weight_bits) not in {4, 8}:
        raise ValueError("qat_weight_bits deve ser 4, 8 ou nulo.")
    if str(hidden_activation) not in {"swish", "relu", "sigmoid", "softmax"}:
        raise ValueError("hidden_activation deve ser swish, relu, sigmoid ou softmax.")
    layers = tensorflow.keras.layers
    if not hasattr(layers, "GroupNormalization"):
        raise RuntimeError(
            "Esta versão do TensorFlow/Keras não oferece GroupNormalization; "
            "use o perfil TensorFlow 2.21 do projeto."
        )

    inputs = layers.Input(shape=(image_size, image_size, channels), dtype="float32", name="image")
    qat_bits = int(qat_weight_bits) if qat_weight_bits is not None else None
    x = FakeQuantize(num_bits=8, name="input_fake_quant")(inputs) if qat_bits else inputs
    for index, filters in enumerate(LEGACY_FILTERS, start=1):
        convolution = QATSeparableConv2D if qat_bits else layers.SeparableConv2D
        quantized_arguments = {"weight_bits": qat_bits} if qat_bits else {}
        x = convolution(
            filters,
            5,
            strides=2 if index == 1 else 1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name=f"block{index}_sepconv5",
            **quantized_arguments,
        )(x)
        x = layers.GroupNormalization(groups=16, axis=-1, name=f"block{index}_gn1")(x)
        x = layers.Activation(hidden_activation, name=f"block{index}_activation1")(x)
        x = FakeQuantize(num_bits=8, name=f"block{index}_fake_quant1")(x) if qat_bits else x
        x = convolution(
            filters,
            3,
            strides=1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name=f"block{index}_sepconv3",
            **quantized_arguments,
        )(x)
        x = layers.GroupNormalization(groups=16, axis=-1, name=f"block{index}_gn2")(x)
        x = layers.Activation(hidden_activation, name=f"block{index}_activation2")(x)
        x = FakeQuantize(num_bits=8, name=f"block{index}_fake_quant2")(x) if qat_bits else x
        x = layers.MaxPooling2D(2, name=f"block{index}_pool")(x)

    gap = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    gmp = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Concatenate(name="global_pool_concat")((gap, gmp))
    x = layers.Dropout(0.4, name="dropout1")(x)
    dense = QATDense if qat_bits else layers.Dense
    dense_arguments = {"weight_bits": qat_bits} if qat_bits else {}
    x = dense(256, activation=hidden_activation, name="dense1", **dense_arguments)(x)
    x = FakeQuantize(num_bits=8, name="dense1_fake_quant")(x) if qat_bits else x
    x = layers.Dropout(0.4, name="dropout2")(x)
    x = dense(256, activation=hidden_activation, name="dense2", **dense_arguments)(x)
    x = FakeQuantize(num_bits=8, name="dense2_fake_quant")(x) if qat_bits else x
    x = layers.Dropout(0.4, name="dropout3")(x)
    logits = dense(num_classes, activation=None, dtype="float32", name="logits", **dense_arguments)(x)
    logits = FakeQuantize(num_bits=8, name="logits_fake_quant")(logits) if qat_bits else logits
    return tensorflow.keras.Model(inputs=inputs, outputs=logits, name=name)

def compile_legacy_cnn(
    model: Any,
    *,
    learning_rate: float = 0.001,
    dtype_policy: str = "mixed_float16",
) -> Any:
    tensorflow = require_tensorflow()
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate deve ser positivo.")
    set_dtype_policy(dtype_policy)
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tensorflow.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )
    return model

def build_and_compile_legacy_cnn(
    *,
    image_size: int,
    channels: int,
    num_classes: int,
    learning_rate: float = 0.001,
    dtype_policy: str = "mixed_float16",
    hidden_activation: str = "swish",
    qat_weight_bits: int | None = None,
    seed: int | None = None,
) -> Any:
    if seed is not None:
        set_training_seed(int(seed), enable_op_determinism=qat_weight_bits is None)
    model = build_legacy_cnn(
        image_size=image_size,
        channels=channels,
        num_classes=num_classes,
        dtype_policy=dtype_policy,
        hidden_activation=hidden_activation,
        qat_weight_bits=qat_weight_bits,
    )
    return compile_legacy_cnn(model, learning_rate=learning_rate, dtype_policy=dtype_policy)

build_model = build_legacy_cnn
compile_model = compile_legacy_cnn
