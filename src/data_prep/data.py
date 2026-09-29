"""Deterministic preparation of image-classification data.

The module deliberately does not know about a particular on-disk dataset
format.  Adapters hand it either a sequence of image paths or an in-memory
``numpy.ndarray`` of images together with integer labels.  This keeps the
experiment logic identical for all ten datasets.

TensorFlow is optional at import time so commands such as ``audit`` can run on
machines that only have the lightweight dependency profile installed.  A clear
error is raised only when a TensorFlow-backed operation is requested.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Keep inspection/audit commands usable without the TF profile.
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - missing DLLs can raise OSError on Windows
    tf = None  # type: ignore[assignment]
    _TENSORFLOW_IMPORT_ERROR: BaseException | None = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


NORMALIZATION_ALIASES: Mapping[str, str] = {
    "unit_interval": "unit_interval",
    "zero_one": "unit_interval",
    "0_1": "unit_interval",
    "[0,1]": "unit_interval",
    "zscore": "zscore",
    "z_score": "zscore",
    "z-score": "zscore",
}
BALANCE_ALIASES: Mapping[str, str] = {
    "all_raw": "all_raw",
    "all": "all_raw",
    "raw": "all_raw",
    "none": "all_raw",
    "undersample": "undersample",
    "undersampling": "undersample",
    "oversample": "oversample",
    "oversampling": "oversample",
    "class_weight": "class_weight",
    "class_weights": "class_weight",
}


class TensorFlowUnavailableError(RuntimeError):
    """Raised when a training-only operation is invoked without TensorFlow."""


def require_tensorflow() -> Any:
    """Return TensorFlow or raise an actionable error.

    Importing this module must remain safe on the Windows audit-only profile;
    callers that build a model or ``tf.data`` pipeline get this explicit error
    instead of an obscure ``NoneType`` exception.
    """

    if tf is None:
        raise TensorFlowUnavailableError(
            "TensorFlow não está instalado. Instale requirements/windows-smoke.txt "
            "ou requirements/wsl-gpu.txt antes de executar treino, smoke ou "
            "normalização baseada em imagens."
        ) from _TENSORFLOW_IMPORT_ERROR
    return tf


def canonical_normalization_mode(mode: str) -> str:
    """Normalize public normalization spellings to the suite's canonical names."""

    key = str(mode).strip().lower().replace(" ", "")
    try:
        return NORMALIZATION_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(set(NORMALIZATION_ALIASES.values())))
        raise ValueError(f"Normalização inválida: {mode!r}. Use: {valid}.") from exc


def canonical_balance_mode(mode: str) -> str:
    """Normalize public balancing spellings to the four experimental modes."""

    key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return BALANCE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(("all_raw", "undersample", "oversample", "class_weight"))
        raise ValueError(f"Estratégia de balanceamento inválida: {mode!r}. Use: {valid}.") from exc


def _as_label_array(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError(f"labels deve ser unidimensional; recebido shape={values.shape}.")
    if not len(values):
        raise ValueError("Não é possível preparar um dataset vazio.")
    if not np.issubdtype(values.dtype, np.integer):
        # Numeric labels encoded as 0.0 are accepted, but no lossy conversion.
        try:
            converted = values.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Os rótulos devem ser inteiros para classificação esparsa.") from exc
        if not np.array_equal(converted, values):
            raise ValueError("Os rótulos devem ser inteiros para classificação esparsa.")
        values = converted
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError("Os rótulos devem ser inteiros não negativos.")
    return values


def class_counts(labels: Sequence[int] | np.ndarray) -> dict[int, int]:
    """Return sorted class frequencies as JSON-friendly Python integers."""

    values = _as_label_array(labels)
    classes, counts = np.unique(values, return_counts=True)
    return {int(label): int(count) for label, count in zip(classes, counts, strict=True)}


@dataclasses.dataclass(frozen=True)
class SplitIndices:
    """Disjoint deterministic indices for the train/validation/test split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.train) + len(self.validation) + len(self.test))

    def validate(self, total_size: int | None = None) -> None:
        """Assert that the three sets are disjoint and cover the source rows."""

        merged = np.concatenate((self.train, self.validation, self.test))
        if len(np.unique(merged)) != len(merged):
            raise ValueError("O split contém índices repetidos entre as partições.")
        if total_size is not None:
            expected = np.arange(int(total_size), dtype=np.int64)
            if len(merged) != len(expected) or not np.array_equal(np.sort(merged), expected):
                raise ValueError("O split não cobre exatamente todos os exemplos de origem.")

    def fingerprint(self) -> str:
        """Stable hash used to verify a resumed run is using the same split."""

        digest = hashlib.sha256()
        for name, indices in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            digest.update(name.encode("ascii"))
            values = np.asarray(indices, dtype="<i8")
            digest.update(values.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        """Salva as partições do split em um arquivo NPZ."""
        np.savez_compressed(
            path,
            train=self.train,
            validation=self.validation,
            test=self.test,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SplitIndices":
        """Carrega as partições do split de um arquivo NPZ."""
        with np.load(path) as data:
            return cls(
                train=data["train"],
                validation=data["validation"],
                test=data["test"],
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_size": int(len(self.train)),
            "validation_size": int(len(self.validation)),
            "test_size": int(len(self.test)),
            "fingerprint": self.fingerprint(),
        }


def _validate_fractions(train_fraction: float, validation_fraction: float, test_fraction: float) -> np.ndarray:
    fractions = np.asarray((train_fraction, validation_fraction, test_fraction), dtype=np.float64)
    if np.any(fractions < 0.0) or not np.isclose(float(fractions.sum()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("As frações train/validation/test devem ser não negativas e somar 1.0.")
    if np.count_nonzero(fractions) < 2:
        raise ValueError("O benchmark exige pelo menos duas partições não vazias.")
    return fractions


def _allocate_class_counts(count: int, fractions: np.ndarray) -> np.ndarray:
    """Allocate one class across split fractions while preserving stratification.

    Every non-empty split receives at least one example when mathematically
    possible.  The remaining examples are handed to the most underrepresented
    split relative to the requested fraction.  This makes very small fixtures
    behave predictably while large datasets converge to 70/15/15.
    """

    active = fractions > 0
    minimum = active.astype(np.int64)
    required = int(minimum.sum())
    if count < required:
        raise ValueError(
            f"A classe tem apenas {count} exemplos, insuficientes para preservar "
            f"estratificação nas {required} partições não vazias."
        )

    allocation = minimum.copy()
    target = fractions * count
    remaining = int(count - allocation.sum())
    # Allocate one at a time.  Stable argmax makes ties reproducible and avoids
    # implementation/version differences from a random quota allocation.
    for _ in range(remaining):
        deficits = target - allocation
        deficits[~active] = -np.inf
        allocation[int(np.argmax(deficits))] += 1
    return allocation


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitIndices:
    """Create a deterministic, disjoint 70/15/15-style stratified split.

    The function does not use the official dataset train/test partition: all
    supplied rows participate in a fresh split, exactly as required by the
    benchmark protocol.  It raises rather than silently losing stratification
    when a class is too small to appear in every requested partition.
    """

    values = _as_label_array(labels)
    fractions = _validate_fractions(train_fraction, validation_fraction, test_fraction)
    rng = np.random.default_rng(int(seed))
    parts: list[list[np.ndarray]] = [[], [], []]

    for label in np.unique(values):
        class_indices = np.flatnonzero(values == label).astype(np.int64, copy=False)
        class_indices = rng.permutation(class_indices)
        allocation = _allocate_class_counts(len(class_indices), fractions)
        start = 0
        for split_index, amount in enumerate(allocation):
            stop = start + int(amount)
            parts[split_index].append(class_indices[start:stop])
            start = stop

    result: list[np.ndarray] = []
    for chunks in parts:
        merged = np.concatenate(chunks).astype(np.int64, copy=False) if chunks else np.empty(0, dtype=np.int64)
        # Mix classes within each partition while preserving the fixed seed.
        result.append(rng.permutation(merged).astype(np.int64, copy=False))
    split = SplitIndices(train=result[0], validation=result[1], test=result[2])
    split.validate(total_size=len(values))
    return split


def select_samples(samples: Sequence[Any] | np.ndarray, indices: Sequence[int] | np.ndarray) -> Sequence[Any] | np.ndarray:
    """Select samples without coercing image tensors into object arrays."""

    idx = np.asarray(indices, dtype=np.int64)
    if isinstance(samples, np.ndarray):
        return samples[idx]
    return [samples[int(position)] for position in idx]


@dataclasses.dataclass(frozen=True)
class NormalizationStats:
    """Train-only per-channel normalization parameters after resize/padding."""

    mode: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    channels: int
    image_size: int
    pixel_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mean": list(self.mean),
            "std": list(self.std),
            "channels": int(self.channels),
            "image_size": int(self.image_size),
            "pixel_count": int(self.pixel_count),
        }


def unit_interval_stats(*, channels: int, image_size: int) -> NormalizationStats:
    """Return identity stats for the [0, 1] branch of the experiment matrix."""

    _validate_image_shape_arguments(image_size=image_size, channels=channels)
    return NormalizationStats(
        mode="unit_interval",
        mean=tuple(0.0 for _ in range(channels)),
        std=tuple(1.0 for _ in range(channels)),
        channels=int(channels),
        image_size=int(image_size),
        pixel_count=0,
    )


def _validate_image_shape_arguments(*, image_size: int, channels: int) -> None:
    if int(image_size) < 1:
        raise ValueError("image_size deve ser positivo.")
    if int(channels) not in {1, 3}:
        raise ValueError("O benchmark aceita imagens nativas em 1 (cinza) ou 3 (RGB) canais.")


def _sample_count(samples: Sequence[Any] | np.ndarray) -> int:
    try:
        count = len(samples)
    except TypeError as exc:
        raise ValueError("samples precisa ser uma sequência de imagens ou caminhos.") from exc
    if not count:
        raise ValueError("Não é possível preparar um dataset vazio.")
    return int(count)


def _samples_are_paths(samples: Sequence[Any] | np.ndarray) -> bool:
    if isinstance(samples, np.ndarray) and samples.ndim != 1:
        return False
    if _sample_count(samples) == 0:  # kept for type narrowing; _sample_count raises first
        return False
    first = samples[0]  # type: ignore[index]
    return isinstance(first, (str, Path))


def _source_channels_for_arrays(samples: Sequence[Any] | np.ndarray) -> int:
    if isinstance(samples, np.ndarray):
        shape = samples.shape
        # A dense stack includes its leading N dimension.
        if len(shape) == 3:
            return 1
        if len(shape) == 4 and shape[-1] in {1, 3}:
            return int(shape[-1])
    else:
        # Folder-backed adapters legitimately yield a list of differently sized
        # arrays.  Infer channels from one image instead of coercing that list
        # into a ragged NumPy array.
        shape = np.asarray(samples[0]).shape  # type: ignore[index]
        if len(shape) == 2:
            return 1
        if len(shape) == 3 and shape[-1] in {1, 3}:
            return int(shape[-1])
    raise ValueError(
        "Imagens em memória devem ter shape [N,H,W] ou [N,H,W,C], com C igual a 1 ou 3; "
        f"recebido {shape}."
    )


def _make_source_dataset(
    samples: Sequence[Any] | np.ndarray,
    labels: np.ndarray,
) -> tuple[Any, bool, int | None]:
    """Build the raw tf.data source and report whether its samples are paths."""

    tensorflow = require_tensorflow()
    paths = _samples_are_paths(samples)
    if paths:
        source = np.asarray([str(item) for item in samples], dtype=np.str_)
        return tensorflow.data.Dataset.from_tensor_slices((source, labels.astype(np.int32))), True, None

    source_channels = _source_channels_for_arrays(samples)
    if isinstance(samples, np.ndarray):
        return tensorflow.data.Dataset.from_tensor_slices((samples, labels.astype(np.int32))), False, source_channels

    # Use a generator for a list of images.  Unlike np.asarray/np.stack, this
    # supports naturally varying resolutions (the resize-with-pad step happens
    # immediately afterwards) while retaining a typed, rank-stable dataset.
    first = np.asarray(samples[0])  # type: ignore[index]
    source_dtype = tensorflow.as_dtype(first.dtype)

    def as_rank_three(value: Any) -> np.ndarray:
        image = np.asarray(value, dtype=first.dtype)
        if source_channels == 1 and image.ndim == 2:
            image = image[..., np.newaxis]
        if image.ndim != 3 or image.shape[-1] != source_channels:
            raise ValueError(
                "Todas as imagens em memória precisam ter H×W×C compatível; "
                f"esperado C={source_channels}, recebido {image.shape}."
            )
        return image

    materialized = list(samples)

    def generator() -> Iterable[tuple[np.ndarray, np.int32]]:
        for image, label in zip(materialized, labels, strict=True):
            yield as_rank_three(image), np.int32(label)

    source = tensorflow.data.Dataset.from_generator(
        generator,
        output_signature=(
            tensorflow.TensorSpec(shape=(None, None, source_channels), dtype=source_dtype),
            tensorflow.TensorSpec(shape=(), dtype=tensorflow.int32),
        ),
    )
    return source, False, source_channels


def _decode_resize_image(
    sample: Any,
    *,
    sample_is_path: bool,
    source_channels: int | None,
    image_size: int,
    channels: int,
) -> Any:
    """Decode/convert one image to native channels and aspect-ratio-safe size."""

    tensorflow = require_tensorflow()
    if sample_is_path:
        # TensorFlow's DecodeImage deliberately does not support Netpbm/PPM.
        # GTSRB's official archive, however, consists of .ppm files.  Keep the
        # native TensorFlow path for the usual formats and dispatch only PPM to
        # Pillow through numpy_function.  This is part of decoding, not a
        # dataset transformation, so it preserves the fixed benchmark protocol.
        def decode_standard() -> Any:
            return tensorflow.io.decode_image(
                tensorflow.io.read_file(sample), channels=channels, expand_animations=False
            )

        def decode_ppm() -> Any:
            decoded = tensorflow.numpy_function(
                lambda raw_path: _decode_ppm_with_pillow(raw_path, channels=channels),
                [sample],
                tensorflow.uint8,
            )
            decoded.set_shape((None, None, channels))
            return decoded

        is_ppm = tensorflow.strings.regex_full_match(tensorflow.strings.lower(sample), r".*\.ppm")
        image = tensorflow.cond(is_ppm, decode_ppm, decode_standard)
        image.set_shape((None, None, channels))
    else:
        image = sample
        if source_channels == 1 and image.shape.rank == 2:
            image = image[..., tensorflow.newaxis]
        if source_channels == 1 and channels == 3:
            image = tensorflow.image.grayscale_to_rgb(image)
        elif source_channels == 3 and channels == 1:
            image = tensorflow.image.rgb_to_grayscale(image)
        elif source_channels != channels:
            raise ValueError(f"Não é possível converter {source_channels} canais para {channels} canais.")
        image.set_shape((None, None, channels))

    # Integer source formats become [0, 1]; adapters should expose float images
    # already in that interval, matching TensorFlow's convert_image_dtype rules.
    image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)
    image = tensorflow.image.resize_with_pad(
        image,
        target_height=int(image_size),
        target_width=int(image_size),
        method="bilinear",
        antialias=True,
    )
    image.set_shape((int(image_size), int(image_size), int(channels)))
    return image


def _decode_ppm_with_pillow(raw_path: Any, *, channels: int) -> np.ndarray:
    """Decode one official GTSRB PPM image for a ``tf.numpy_function`` call."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency checked in GPU profile
        raise RuntimeError("Decodificar imagens PPM exige Pillow.") from exc

    value = raw_path.item() if isinstance(raw_path, np.ndarray) and raw_path.ndim == 0 else raw_path
    if isinstance(value, bytes):
        path = value.decode("utf-8")
    else:
        path = str(value)
    with Image.open(path) as opened:
        converted = opened.convert("L" if channels == 1 else "RGB")
        array = np.asarray(converted, dtype=np.uint8)
    if channels == 1:
        array = array[..., np.newaxis]
    return array


def compute_normalization_stats(
    train_samples: Sequence[Any] | np.ndarray,
    *,
    image_size: int,
    channels: int,
    mode: str,
    batch_size: int,
    epsilon: float = 1e-6,
) -> NormalizationStats:
    """Compute normalization parameters using *only* raw training examples.

    Images are decoded, converted to [0,1], resized with proportional padding,
    and then aggregated.  No balance sampling or augmentation occurs here.  The
    returned values therefore can safely be applied to train, validation, and
    test for one seed.
    """

    mode = canonical_normalization_mode(mode)
    _validate_image_shape_arguments(image_size=image_size, channels=channels)
    if int(batch_size) < 1:
        raise ValueError("batch_size deve ser positivo.")
    if epsilon <= 0:
        raise ValueError("epsilon deve ser positivo.")
    sample_count = _sample_count(train_samples)
    if mode == "unit_interval":
        return unit_interval_stats(channels=channels, image_size=image_size)

    tensorflow = require_tensorflow()
    placeholder_labels = np.zeros(sample_count, dtype=np.int32)
    source, sample_is_path, source_channels = _make_source_dataset(train_samples, placeholder_labels)

    def _decode(sample: Any, label: Any) -> tuple[Any, Any]:
        return (
            _decode_resize_image(
                sample,
                sample_is_path=sample_is_path,
                source_channels=source_channels,
                image_size=image_size,
                channels=channels,
            ),
            label,
        )

    options = tensorflow.data.Options()
    options.experimental_deterministic = True
    dataset = source.with_options(options).map(_decode, num_parallel_calls=tensorflow.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(int(batch_size))

    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    pixels = 0
    for batch, _ in dataset:
        values = np.asarray(batch.numpy(), dtype=np.float64)
        total += values.sum(axis=(0, 1, 2))
        total_sq += np.square(values).sum(axis=(0, 1, 2))
        pixels += int(values.shape[0] * values.shape[1] * values.shape[2])
    if pixels == 0:
        raise ValueError("Não foi possível calcular estatísticas de um treino vazio.")
    mean = total / pixels
    variance = np.maximum(total_sq / pixels - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), float(epsilon))
    return NormalizationStats(
        mode="zscore",
        mean=tuple(float(item) for item in mean),
        std=tuple(float(item) for item in std),
        channels=int(channels),
        image_size=int(image_size),
        pixel_count=int(pixels),
    )


def normalize_image(image: Any, stats: NormalizationStats) -> Any:
    """Apply a previously computed normalization to NumPy or TensorFlow data."""

    if stats.mode == "unit_interval":
        return image
    if stats.mode != "zscore":
        raise ValueError(f"NormalizationStats contém modo desconhecido: {stats.mode!r}")
    if isinstance(image, np.ndarray):
        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        return (image.astype(np.float32, copy=False) - mean) / std
    tensorflow = require_tensorflow()
    mean_tensor = tensorflow.constant(stats.mean, dtype=tensorflow.float32)
    std_tensor = tensorflow.constant(stats.std, dtype=tensorflow.float32)
    return (tensorflow.cast(image, tensorflow.float32) - mean_tensor) / std_tensor


@dataclasses.dataclass(frozen=True)
class BalanceResult:
    """Training-only sampling/weighting output for one balance strategy."""

    mode: str
    samples: Sequence[Any] | np.ndarray
    labels: np.ndarray
    class_weights: dict[int, float] | None
    raw_class_counts: dict[int, int]
    output_class_counts: dict[int, int]
    is_noop: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "raw_class_counts": self.raw_class_counts,
            "output_class_counts": self.output_class_counts,
            "class_weights": self.class_weights,
            "is_noop": bool(self.is_noop),
        }


def _counts_are_balanced(counts: Mapping[int, int]) -> bool:
    values = list(counts.values())
    return bool(values) and len(set(values)) == 1


def balance_training_data(
    train_samples: Sequence[Any] | np.ndarray,
    train_labels: Sequence[int] | np.ndarray,
    *,
    mode: str,
    seed: int,
) -> BalanceResult:
    """Apply one mutually exclusive strategy to training data only.

    Validation and test collections are intentionally not accepted by this API;
    that prevents accidental resampling or weighting leakage into evaluation.
    """

    canonical_mode = canonical_balance_mode(mode)
    labels = _as_label_array(train_labels)
    if _sample_count(train_samples) != len(labels):
        raise ValueError("train_samples e train_labels possuem tamanhos diferentes.")
    raw_counts = class_counts(labels)
    classes = np.asarray(sorted(raw_counts), dtype=np.int64)
    balanced = _counts_are_balanced(raw_counts)

    if canonical_mode == "all_raw":
        return BalanceResult(
            mode=canonical_mode,
            samples=train_samples,
            labels=labels.astype(np.int32),
            class_weights=None,
            raw_class_counts=raw_counts,
            output_class_counts=raw_counts.copy(),
            is_noop=True,
        )

    if canonical_mode == "class_weight":
        total = len(labels)
        num_classes = len(classes)
        weights = {int(label): float(total / (num_classes * raw_counts[int(label)])) for label in classes}
        return BalanceResult(
            mode=canonical_mode,
            samples=train_samples,
            labels=labels.astype(np.int32),
            class_weights=weights,
            raw_class_counts=raw_counts,
            output_class_counts=raw_counts.copy(),
            is_noop=balanced,
        )

    rng = np.random.default_rng(int(seed))
    target = min(raw_counts.values()) if canonical_mode == "undersample" else max(raw_counts.values())
    sampled_parts: list[np.ndarray] = []
    for label in classes:
        class_indices = np.flatnonzero(labels == label).astype(np.int64, copy=False)
        if canonical_mode == "undersample":
            chosen = rng.choice(class_indices, size=target, replace=False)
        else:  # Preserve every original example, then repeat only the deficit.
            deficit = int(target - len(class_indices))
            if deficit:
                extra = rng.choice(class_indices, size=deficit, replace=True)
                chosen = np.concatenate((class_indices, extra))
            else:
                chosen = class_indices.copy()
        sampled_parts.append(chosen)
    selected = rng.permutation(np.concatenate(sampled_parts)).astype(np.int64, copy=False)
    selected_labels = labels[selected].astype(np.int32, copy=False)
    output_counts = class_counts(selected_labels)
    return BalanceResult(
        mode=canonical_mode,
        samples=select_samples(train_samples, selected),
        labels=selected_labels,
        class_weights=None,
        raw_class_counts=raw_counts,
        output_class_counts=output_counts,
        is_noop=balanced,
    )


@dataclasses.dataclass(frozen=True)
class AugmentationConfig:
    """The complete fixed augmentation policy inherited from the legacy CNN."""

    flip_lr: bool = False
    brightness_delta: float = 0.03
    contrast_lower: float = 0.92
    contrast_upper: float = 1.08
    translate_frac: float = 0.05
    zoom_min: float = 0.96
    zoom_max: float = 1.04
    noise_std: float = 0.005
    cutout_prob: float = 0.25
    cutout_max_frac: float = 0.08

    def validate(self) -> None:
        if self.brightness_delta < 0 or self.translate_frac < 0 or self.noise_std < 0:
            raise ValueError("Parâmetros de augmentação devem ser não negativos.")
        if not 0 <= self.cutout_prob <= 1 or self.cutout_max_frac < 0:
            raise ValueError("cutout_prob deve estar em [0,1] e cutout_max_frac deve ser não negativo.")
        if self.contrast_lower <= 0 or self.contrast_upper < self.contrast_lower:
            raise ValueError("Intervalo de contraste inválido.")
        if self.zoom_min <= 0 or self.zoom_max < self.zoom_min:
            raise ValueError("Intervalo de zoom inválido.")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _seed_for(base_seed: int, example_index: Any, salt: int) -> Any:
    tensorflow = require_tensorflow()
    return tensorflow.stack(
        (
            tensorflow.cast(int(base_seed) + int(salt), tensorflow.int32),
            tensorflow.cast(example_index, tensorflow.int32),
        )
    )


def _apply_translation(image: Any, config: AugmentationConfig, seed: Any) -> Any:
    tensorflow = require_tensorflow()
    if config.translate_frac <= 0:
        return image
    height = tensorflow.shape(image)[0]
    width = tensorflow.shape(image)[1]
    pad_y = tensorflow.cast(tensorflow.round(tensorflow.cast(height, tensorflow.float32) * config.translate_frac), tensorflow.int32)
    pad_x = tensorflow.cast(tensorflow.round(tensorflow.cast(width, tensorflow.float32) * config.translate_frac), tensorflow.int32)
    padded = tensorflow.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="REFLECT")
    offset_y = tensorflow.random.stateless_uniform(
        (), seed=seed, minval=0, maxval=pad_y * 2 + 1, dtype=tensorflow.int32
    )
    offset_x = tensorflow.random.stateless_uniform(
        (), seed=tensorflow.stack((seed[0], seed[1] + 1)), minval=0, maxval=pad_x * 2 + 1, dtype=tensorflow.int32
    )
    return tensorflow.image.crop_to_bounding_box(padded, offset_y, offset_x, height, width)


def _apply_zoom(image: Any, config: AugmentationConfig, seed: Any) -> Any:
    tensorflow = require_tensorflow()
    if config.zoom_min == 1.0 and config.zoom_max == 1.0:
        return image
    height = tensorflow.shape(image)[0]
    width = tensorflow.shape(image)[1]
    zoom = tensorflow.random.stateless_uniform((), seed=seed, minval=config.zoom_min, maxval=config.zoom_max)
    resized_height = tensorflow.maximum(
        tensorflow.cast(tensorflow.round(tensorflow.cast(height, tensorflow.float32) * zoom), tensorflow.int32), 1
    )
    resized_width = tensorflow.maximum(
        tensorflow.cast(tensorflow.round(tensorflow.cast(width, tensorflow.float32) * zoom), tensorflow.int32), 1
    )
    resized = tensorflow.image.resize(image, (resized_height, resized_width), method="bilinear")

    def zoom_in() -> Any:
        max_y = tensorflow.maximum(resized_height - height + 1, 1)
        max_x = tensorflow.maximum(resized_width - width + 1, 1)
        offset_y = tensorflow.random.stateless_uniform(
            (), seed=tensorflow.stack((seed[0], seed[1] + 2)), minval=0, maxval=max_y, dtype=tensorflow.int32
        )
        offset_x = tensorflow.random.stateless_uniform(
            (), seed=tensorflow.stack((seed[0], seed[1] + 3)), minval=0, maxval=max_x, dtype=tensorflow.int32
        )
        return tensorflow.image.crop_to_bounding_box(resized, offset_y, offset_x, height, width)

    def zoom_out() -> Any:
        pad_y = (height - resized_height) // 2
        pad_x = (width - resized_width) // 2
        return tensorflow.pad(
            resized,
            ((pad_y, height - resized_height - pad_y), (pad_x, width - resized_width - pad_x), (0, 0)),
            mode="REFLECT",
        )

    return tensorflow.cond(zoom > 1.0, zoom_in, zoom_out)


def _apply_cutout(image: Any, config: AugmentationConfig, seed: Any) -> Any:
    tensorflow = require_tensorflow()
    if config.cutout_prob <= 0 or config.cutout_max_frac <= 0:
        return image

    def cutout() -> Any:
        height = tensorflow.shape(image)[0]
        width = tensorflow.shape(image)[1]
        max_area = tensorflow.cast(
            tensorflow.round(tensorflow.cast(height * width, tensorflow.float32) * config.cutout_max_frac), tensorflow.int32
        )
        side = tensorflow.maximum(
            tensorflow.cast(tensorflow.round(tensorflow.sqrt(tensorflow.cast(max_area, tensorflow.float32))), tensorflow.int32), 1
        )
        offset_y = tensorflow.random.stateless_uniform(
            (), seed=tensorflow.stack((seed[0], seed[1] + 5)), minval=0,
            maxval=tensorflow.maximum(height - side + 1, 1), dtype=tensorflow.int32
        )
        offset_x = tensorflow.random.stateless_uniform(
            (), seed=tensorflow.stack((seed[0], seed[1] + 6)), minval=0,
            maxval=tensorflow.maximum(width - side + 1, 1), dtype=tensorflow.int32
        )
        yy = tensorflow.range(height)[:, tensorflow.newaxis]
        xx = tensorflow.range(width)[tensorflow.newaxis, :]
        in_y = tensorflow.logical_and(yy >= offset_y, yy < offset_y + side)
        in_x = tensorflow.logical_and(xx >= offset_x, xx < offset_x + side)
        keep = tensorflow.logical_not(tensorflow.logical_and(in_y, in_x))
        keep = tensorflow.cast(keep, tensorflow.float32)[..., tensorflow.newaxis]
        fill = tensorflow.reduce_mean(image)
        return image * keep + (1.0 - keep) * fill

    probability = tensorflow.random.stateless_uniform((), seed=seed)
    return tensorflow.cond(probability < config.cutout_prob, cutout, lambda: image)


def augment_image(image: Any, config: AugmentationConfig, *, base_seed: int, example_index: Any) -> Any:
    """Apply the fixed legacy augmentation sequence with stateless randomness."""

    tensorflow = require_tensorflow()
    config.validate()
    if config.flip_lr:
        image = tensorflow.image.stateless_random_flip_left_right(image, _seed_for(base_seed, example_index, 0))
    image = _apply_translation(image, config, _seed_for(base_seed, example_index, 10))
    image = _apply_zoom(image, config, _seed_for(base_seed, example_index, 20))
    if config.brightness_delta > 0:
        delta = tensorflow.random.stateless_uniform(
            (), seed=_seed_for(base_seed, example_index, 30), minval=-config.brightness_delta, maxval=config.brightness_delta
        )
        image = tensorflow.clip_by_value(image + delta, 0.0, 1.0)
    if config.contrast_upper > config.contrast_lower:
        image = tensorflow.image.stateless_random_contrast(
            image,
            lower=config.contrast_lower,
            upper=config.contrast_upper,
            seed=_seed_for(base_seed, example_index, 31),
        )
    if config.noise_std > 0:
        noise = tensorflow.random.stateless_normal(
            tensorflow.shape(image), seed=_seed_for(base_seed, example_index, 40), stddev=config.noise_std, dtype=tensorflow.float32
        )
        image = tensorflow.clip_by_value(image + noise, 0.0, 1.0)
    image = _apply_cutout(image, config, _seed_for(base_seed, example_index, 50))
    return tensorflow.clip_by_value(image, 0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class DatasetBuildInfo:
    """Cardinality metadata consumed by the runner and written to manifests."""

    source_examples: int
    extra_augmented_examples: int
    total_examples: int
    batches: int
    training: bool
    decoded_cache_enabled: bool
    decoded_cache_estimated_bytes: int
    shuffle_buffer_examples: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_tf_dataset(
    samples: Sequence[Any] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    image_size: int,
    channels: int,
    batch_size: int,
    normalization: NormalizationStats | None = None,
    training: bool = False,
    augmentation: AugmentationConfig | None = None,
    extra_fraction: float = 0.0,
    seed: int = 42,
    shuffle: bool | None = None,
    deterministic: bool = True,
    preprocess_cache_max_mib: int = 0,
    shuffle_buffer_max_mib: int = 0,
    output_dtype: str = "float16",
    repeat: bool | None = None,
) -> tuple[Any, DatasetBuildInfo]:
    """Construct a deterministic ``tf.data`` pipeline for one split.

    Training datasets contain every raw sample once and exactly
    ``floor(N * extra_fraction)`` additional, augmented samples.  Evaluation
    datasets never shuffle or augment.  Augmentation runs before z-score
    transformation so the legacy image-space parameters remain meaningful.
    """

    tensorflow = require_tensorflow()
    _validate_image_shape_arguments(image_size=image_size, channels=channels)
    if int(batch_size) < 1:
        raise ValueError("batch_size deve ser positivo.")
    if extra_fraction < 0:
        raise ValueError("extra_fraction não pode ser negativo.")
    if int(preprocess_cache_max_mib) < 0:
        raise ValueError("preprocess_cache_max_mib não pode ser negativo.")
    if int(shuffle_buffer_max_mib) < 0:
        raise ValueError("shuffle_buffer_max_mib não pode ser negativo.")
    try:
        feature_dtype = tensorflow.as_dtype(output_dtype)
    except TypeError as exc:
        raise ValueError(f"output_dtype invÃ¡lido: {output_dtype!r}.") from exc
    if not feature_dtype.is_floating:
        raise ValueError("output_dtype deve ser um dtype de ponto flutuante.")
    values = _as_label_array(labels)
    count = _sample_count(samples)
    if count != len(values):
        raise ValueError("samples e labels possuem tamanhos diferentes.")
    if normalization is None:
        normalization = unit_interval_stats(channels=channels, image_size=image_size)
    if normalization.channels != int(channels) or normalization.image_size != int(image_size):
        raise ValueError("As estatísticas de normalização não correspondem ao tamanho/canais da pipeline.")
    if augmentation is None:
        augmentation = AugmentationConfig()
    augmentation.validate()
    if shuffle is None:
        shuffle = bool(training)
    if repeat is None:
        repeat = bool(training)

    source, sample_is_path, source_channels = _make_source_dataset(samples, values)
    options = tensorflow.data.Options()
    options.experimental_deterministic = bool(deterministic)
    source = source.with_options(options)

    def decode(sample: Any, label: Any) -> tuple[Any, Any]:
        return (
            _decode_resize_image(
                sample,
                sample_is_path=sample_is_path,
                source_channels=source_channels,
                image_size=image_size,
                channels=channels,
            ),
            tensorflow.cast(label, tensorflow.int32),
        )

    # Decoding and resize are deterministic but expensive for folder-backed
    # datasets. Cache them before shuffle/augmentation so later epochs avoid
    # re-reading files while still receiving fresh order and random augmentation.
    decoded = source.map(decode, num_parallel_calls=tensorflow.data.AUTOTUNE, deterministic=bool(deterministic))
    decoded_cache_estimated_bytes = int(count * int(image_size) * int(image_size) * int(channels) * 4)
    cache_limit_bytes = int(preprocess_cache_max_mib) * 1024 * 1024
    decoded_cache_enabled = cache_limit_bytes > 0 and decoded_cache_estimated_bytes <= cache_limit_bytes
    if decoded_cache_enabled:
        decoded = decoded.cache()

    extra_count = int(math.floor(count * float(extra_fraction))) if training else 0
    shuffle_buffer_size = count
    if shuffle and int(shuffle_buffer_max_mib) > 0:
        # There are two independently shuffled sources when augmentation adds
        # extra samples. Split the configured memory budget between them so a
        # full-dataset shuffle cannot exhaust the host RAM.
        shuffled_sources = 2 if training and extra_count else 1
        per_source_limit = (int(shuffle_buffer_max_mib) * 1024 * 1024) // shuffled_sources
        decoded_example_bytes = int(image_size) * int(image_size) * int(channels) * 4 + 4
        shuffle_buffer_size = max(1, min(count, per_source_limit // decoded_example_bytes))

    raw = decoded
    if shuffle:
        raw = raw.shuffle(buffer_size=shuffle_buffer_size, seed=int(seed), reshuffle_each_iteration=True)

    dataset = raw
    if training and extra_count:
        augmented_source = decoded
        if shuffle:
            augmented_source = augmented_source.shuffle(
                buffer_size=shuffle_buffer_size, seed=int(seed) + 1, reshuffle_each_iteration=True
            )
        augmented_source = augmented_source.repeat().take(extra_count).enumerate()

        def decode_augment(index: Any, row: tuple[Any, Any]) -> tuple[Any, Any]:
            image, label = row
            return augment_image(image, augmentation, base_seed=int(seed), example_index=index), label

        augmented = augmented_source.map(
            decode_augment, num_parallel_calls=tensorflow.data.AUTOTUNE, deterministic=bool(deterministic)
        )
        dataset = raw.concatenate(augmented)

    def normalize(image: Any, label: Any) -> tuple[Any, Any]:
        # The model uses Keras' mixed_float16 policy.  Casting at the end of
        # preprocessing also transfers normalized batches to the GPU as FP16,
        # while resize/normalization themselves remain FP32 for numerical
        # stability.  Labels stay int32 for sparse cross-entropy.
        return tensorflow.cast(normalize_image(image, normalization), feature_dtype), label

    dataset = dataset.map(normalize, num_parallel_calls=tensorflow.data.AUTOTUNE, deterministic=bool(deterministic))
    dataset = dataset.batch(int(batch_size), drop_remainder=False)
    # Keras 3 cannot always infer the finite cardinality of the concatenated
    # raw+augmented pipeline.  Make the training stream explicitly renewable
    # and let the runner bound every epoch with DatasetBuildInfo.batches.
    # Evaluation remains finite so prediction/evaluation still stop naturally.
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tensorflow.data.AUTOTUNE)
    total = count + extra_count
    return dataset, DatasetBuildInfo(
        source_examples=count,
        extra_augmented_examples=extra_count,
        total_examples=total,
        batches=int(math.ceil(total / int(batch_size))),
        training=bool(training),
        decoded_cache_enabled=decoded_cache_enabled,
        decoded_cache_estimated_bytes=decoded_cache_estimated_bytes,
        shuffle_buffer_examples=int(shuffle_buffer_size) if shuffle else 0,
    )


@dataclasses.dataclass(frozen=True)
class PreparedRunDatasets:
    """All per-run data outputs, including manifest-ready reproducibility data."""

    train_ds: Any
    validation_ds: Any
    test_ds: Any
    split: SplitIndices
    normalization: NormalizationStats
    balancing: BalanceResult
    class_weights: dict[int, float] | None
    metadata: dict[str, Any]

    @property
    def val_ds(self) -> Any:
        """Short alias used by Keras training calls."""

        return self.validation_ds

    def save_manifest(self, path: str | Path) -> Path:
        """Salva o manifesto JSON de metadados da execução de maneira atômica."""
        from utils.experiment import atomic_write_json
        return atomic_write_json(path, self.metadata)


def iter_torch_batches(dataset: Any, *, device: str | None = None) -> Any:
    """Yield an NHWC ``tf.data.Dataset`` as PyTorch ``NCHW`` batches.

    This is the only bridge needed by a PennyLane/Torch model. Qiskit-based
    estimators can consume the same batches after calling ``.cpu().numpy()``.
    Torch is imported lazily so dataset auditing and TensorFlow-only use stay
    lightweight.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependent on environment
        raise RuntimeError("PyTorch não está instalado.") from exc
    for images, labels in dataset:
        images_np = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
        labels_np = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
        features = torch.as_tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
        targets = torch.as_tensor(labels_np, dtype=torch.long)
        if device is not None:
            features, targets = features.to(device), targets.to(device)
        yield features, targets


def prepare_run_datasets(
    samples: Sequence[Any] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    image_size: int,
    channels: int,
    batch_size: int,
    normalization_mode: str = "unit_interval",
    balance_mode: str = "all_raw",
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    extra_fraction: float,
    augmentation: AugmentationConfig,
    preprocess_cache_max_mib: int = 0,
    shuffle_buffer_max_mib: int = 0,
    output_dtype: str = "float16",
) -> PreparedRunDatasets:
    """Prepare the full leakage-safe data flow for one matrix cell.

    The sequence is intentionally fixed: split first, derive z-score values
    from raw training data, balance only training data, then construct the
    three datasets.  The returned ``metadata`` is ready for a run manifest.
    """

    values = _as_label_array(labels)
    if _sample_count(samples) != len(values):
        raise ValueError("samples e labels possuem tamanhos diferentes.")
    canonical_norm = canonical_normalization_mode(normalization_mode)
    canonical_balance = canonical_balance_mode(balance_mode)
    split = stratified_split_indices(
        values,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    raw_train_samples = select_samples(samples, split.train)
    raw_train_labels = values[split.train]
    validation_samples = select_samples(samples, split.validation)
    validation_labels = values[split.validation]
    test_samples = select_samples(samples, split.test)
    test_labels = values[split.test]

    normalization = compute_normalization_stats(
        raw_train_samples,
        image_size=image_size,
        channels=channels,
        mode=canonical_norm,
        batch_size=batch_size,
    )
    balancing = balance_training_data(raw_train_samples, raw_train_labels, mode=canonical_balance, seed=seed)
    train_ds, train_info = build_tf_dataset(
        balancing.samples,
        balancing.labels,
        image_size=image_size,
        channels=channels,
        batch_size=batch_size,
        normalization=normalization,
        training=True,
        augmentation=augmentation,
        extra_fraction=extra_fraction,
        seed=seed,
        preprocess_cache_max_mib=preprocess_cache_max_mib,
        shuffle_buffer_max_mib=shuffle_buffer_max_mib,
        output_dtype=output_dtype,
    )
    validation_ds, validation_info = build_tf_dataset(
        validation_samples,
        validation_labels,
        image_size=image_size,
        channels=channels,
        batch_size=batch_size,
        normalization=normalization,
        training=False,
        seed=seed,
        preprocess_cache_max_mib=preprocess_cache_max_mib,
        shuffle_buffer_max_mib=shuffle_buffer_max_mib,
        output_dtype=output_dtype,
    )
    test_ds, test_info = build_tf_dataset(
        test_samples,
        test_labels,
        image_size=image_size,
        channels=channels,
        batch_size=batch_size,
        normalization=normalization,
        training=False,
        seed=seed,
        preprocess_cache_max_mib=preprocess_cache_max_mib,
        shuffle_buffer_max_mib=shuffle_buffer_max_mib,
        output_dtype=output_dtype,
    )
    metadata = {
        "seed": int(seed),
        "image_size": int(image_size),
        "channels": int(channels),
        "batch_size": int(batch_size),
        "output_dtype": str(output_dtype),
        "fractions": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(test_fraction),
        },
        "split": split.to_dict(),
        "raw_split_class_counts": {
            "train": class_counts(raw_train_labels),
            "validation": class_counts(validation_labels),
            "test": class_counts(test_labels),
        },
        "normalization": normalization.to_dict(),
        "balancing": balancing.to_dict(),
        "augmentation": augmentation.to_dict(),
        "extra_fraction": float(extra_fraction),
        "datasets": {
            "train": train_info.to_dict(),
            "validation": validation_info.to_dict(),
            "test": test_info.to_dict(),
        },
        "num_classes": int(len(np.unique(values))),
        "class_ids": [int(item) for item in np.unique(values)],
    }
    return PreparedRunDatasets(
        train_ds=train_ds,
        validation_ds=validation_ds,
        test_ds=test_ds,
        split=split,
        normalization=normalization,
        balancing=balancing,
        class_weights=balancing.class_weights,
        metadata=metadata,
    )


# Compact aliases for callers migrating from a conventional data-pipeline API.
make_tf_dataset = build_tf_dataset
split_stratified = stratified_split_indices
