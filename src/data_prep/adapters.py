"""Local-only loaders for the image-classification benchmark datasets.

The module deliberately does not use ``tf.keras.datasets``: those helpers may
silently download data.  Every adapter accepts a user-provided local path and
raises a descriptive error when the expected files are absent or malformed.

Folder-backed datasets are represented by :class:`SampleRecord` instances so
the training pipeline can resize/decode lazily.  Compact binary/CSV datasets
are returned as in-memory NumPy arrays.  Both representations share the same
``LoadedDataset`` API.
"""

from __future__ import annotations

import csv
import gzip
import pickle
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


class DatasetAdapterError(RuntimeError):
    """Base exception for failures while reading a local benchmark dataset."""


class LocalDatasetNotFoundError(DatasetAdapterError):
    """Raised when the configured local path or a required file is absent."""


class DatasetFormatError(DatasetAdapterError):
    """Raised when a local dataset does not match a supported on-disk layout."""


class OptionalDependencyError(DatasetAdapterError):
    """Raised only when a selected local format needs an uninstalled package."""


IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"})


def _as_path(path: str | Path, dataset: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise LocalDatasetNotFoundError(
            f"Local data for '{dataset}' was not found at '{candidate}'. "
            "Downloads are disabled; configure a path that already contains the dataset."
        )
    return candidate.resolve()


def _find_first(root: Path, names: Sequence[str]) -> Path | None:
    """Find a direct or nested candidate without assuming a particular archive root."""
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        matches = sorted(candidate for candidate in root.rglob(name) if candidate.is_file())
        if matches:
            return matches[0]
    return None


def _normalise_dataset_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    aliases = {
        "fashion-mnist": "fashion_mnist",
        "fashionmnist": "fashion_mnist",
        "emnist-balanced": "emnist_balanced",
        "emnistbalanced": "emnist_balanced",
        "cifar-10": "cifar10",
        "cifar-100": "cifar100_coarse",
        "cifar100": "cifar100_coarse",
        "cifar-100-coarse": "cifar100_coarse",
        "cifar100-coarse": "cifar100_coarse",
        "cinic-10": "cinic10",
        "gtsrb": "gtsrb",
        "fer-2013": "fer2013",
        "fer-2013-csv": "fer2013",
        "svhn": "svhn",
        "kmnist": "kmnist",
        "mnist": "mnist",
    }
    return aliases.get(key, key.replace("-", "_"))


@dataclass(frozen=True)
class SampleRecord:
    """One label/path pair for a lazily decoded image sample."""

    path: Path
    label: int
    split: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "label": int(self.label),
            "split": self.split,
            "metadata": dict(self.metadata),
        }


@dataclass
class DatasetSplit:
    """A homogeneous split represented by arrays, paths, or (rarely) both."""

    images: np.ndarray | None = None
    labels: np.ndarray | None = None
    records: list[SampleRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (self.images is None) != (self.labels is None):
            raise DatasetFormatError("A DatasetSplit must provide both images and labels, or neither.")
        if self.images is not None:
            self.images = np.asarray(self.images)
            self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
            if self.images.ndim < 2:
                raise DatasetFormatError(f"Image array needs a batch axis, got shape {self.images.shape}.")
            if len(self.images) != len(self.labels):
                raise DatasetFormatError(
                    f"Images ({len(self.images)}) and labels ({len(self.labels)}) have different lengths."
                )
        self.records = list(self.records)

    def __len__(self) -> int:
        return (0 if self.labels is None else int(len(self.labels))) + len(self.records)

    @property
    def has_arrays(self) -> bool:
        return self.images is not None


@dataclass
class LoadedDataset:
    """Dataset returned by every adapter.

    Array data uses ``N,H,W,C`` (grayscale has ``C=1``).  Path records retain
    their native dimensions/channels and are decoded with :meth:`load_record`.
    This makes large folder datasets practical while leaving resize/padding to
    the shared training pipeline.
    """

    name: str
    class_names: list[str]
    splits: dict[str, DatasetSplit]
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _normalise_dataset_name(self.name)
        self.source_path = Path(self.source_path)
        self.class_names = [str(value) for value in self.class_names]
        self.splits = {str(key): value for key, value in self.splits.items() if len(value) > 0}
        if not self.splits:
            raise DatasetFormatError(f"'{self.name}' did not contain any samples.")
        if not self.class_names:
            raise DatasetFormatError(f"'{self.name}' did not expose any class names.")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self.splits)

    def sample_count(self, split: str | None = None) -> int:
        if split is not None:
            return len(self.splits[split])
        return sum(len(partition) for partition in self.splits.values())

    def labels_for(self, split: str | None = None) -> np.ndarray:
        selected = self.splits.items() if split is None else ((split, self.splits[split]),)
        pieces: list[np.ndarray] = []
        for _, partition in selected:
            if partition.labels is not None:
                pieces.append(partition.labels.astype(np.int64, copy=False))
            if partition.records:
                pieces.append(np.asarray([record.label for record in partition.records], dtype=np.int64))
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def iter_records(self, split: str | None = None) -> Iterator[SampleRecord]:
        selected = self.splits.items() if split is None else ((split, self.splits[split]),)
        for _, partition in selected:
            yield from partition.records

    def iter_samples(
        self, split: str | None = None, *, load_images: bool = False
    ) -> Iterator[tuple[np.ndarray | Path, int, SampleRecord | None]]:
        """Yield ``(image-or-path, label, record)`` across selected split(s)."""
        selected = self.splits.items() if split is None else ((split, self.splits[split]),)
        for split_name, partition in selected:
            if partition.images is not None and partition.labels is not None:
                for image, label in zip(partition.images, partition.labels, strict=True):
                    yield image, int(label), None
            for record in partition.records:
                image_or_path: np.ndarray | Path = load_image_file(record.path) if load_images else record.path
                yield image_or_path, int(record.label), record

    def materialize_split(self, split: str | None = None) -> tuple[list[np.ndarray], np.ndarray]:
        """Decode selected data into image list plus labels.

        A list is intentional: folder datasets can contain varying original
        image sizes.  The runner should resize/pad before ``np.stack``.
        """
        images: list[np.ndarray] = []
        labels: list[int] = []
        for image_or_path, label, _ in self.iter_samples(split, load_images=True):
            assert isinstance(image_or_path, np.ndarray)
            images.append(image_or_path)
            labels.append(label)
        return images, np.asarray(labels, dtype=np.int64)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": str(self.source_path),
            "class_names": list(self.class_names),
            "splits": {name: partition.__len__() for name, partition in self.splits.items()},
            "metadata": dict(self.metadata),
        }


def _import_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OptionalDependencyError("Decoding image files requires Pillow. Install 'Pillow'.") from exc
    return Image


def load_image_file(path: str | Path) -> np.ndarray:
    """Decode a local image into ``H,W,C`` uint/float array without TensorFlow.

    Paletted, alpha, CMYK and uncommon modes are converted to RGB.  L-like
    modes are converted to a single channel.  Decoder errors are deliberately
    propagated as ``DatasetFormatError`` so callers/audits can report the file.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise LocalDatasetNotFoundError(f"Image file is missing: '{image_path}'.")
    Image = _import_pillow()
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.mode in {"1", "L", "I", "I;16", "F"}:
                decoded = image.convert("L")
            elif image.mode == "LA":
                decoded = image.convert("L")
            else:
                decoded = image.convert("RGB")
            array = np.asarray(decoded)
    except Exception as exc:
        raise DatasetFormatError(f"Could not decode image '{image_path}': {exc}") from exc
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if array.ndim != 3 or array.shape[-1] not in {1, 3}:
        raise DatasetFormatError(f"Decoded image '{image_path}' has unsupported shape {array.shape}.")
    return np.ascontiguousarray(array)


def _ensure_nhwc(images: np.ndarray, *, dataset: str) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim == 3:  # N,H,W grayscale
        array = array[..., np.newaxis]
    elif array.ndim == 4:
        # Some legacy data is N,C,H,W.  Convert only when the layout is clear.
        if array.shape[-1] not in {1, 3} and array.shape[1] in {1, 3}:
            array = np.moveaxis(array, 1, -1)
    else:
        raise DatasetFormatError(f"'{dataset}' images need rank 3 or 4, got shape {array.shape}.")
    if array.shape[-1] not in {1, 3}:
        raise DatasetFormatError(
            f"'{dataset}' images must have one or three channels after conversion, got {array.shape}."
        )
    return np.ascontiguousarray(array)


def _make_array_split(images: np.ndarray, labels: np.ndarray, *, dataset: str) -> DatasetSplit:
    image_array = _ensure_nhwc(images, dataset=dataset)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(image_array) != len(label_array):
        raise DatasetFormatError(
            f"'{dataset}' image count ({len(image_array)}) does not match label count ({len(label_array)})."
        )
    if len(label_array) == 0:
        raise DatasetFormatError(f"'{dataset}' contains an empty split.")
    if np.any(label_array < 0):
        raise DatasetFormatError(f"'{dataset}' has negative class labels.")
    return DatasetSplit(images=image_array, labels=label_array)


def _read_idx(path: Path, *, expected_dims: int | None = None) -> np.ndarray:
    """Read uncompressed or gzip-compressed IDX data, validating its header."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rb") as stream:
            header = stream.read(4)
            if len(header) != 4:
                raise DatasetFormatError(f"IDX file '{path}' has a truncated header.")
            zero, data_type, dimensions = struct.unpack(">HBB", header)
            if zero != 0 or data_type != 0x08:
                raise DatasetFormatError(f"IDX file '{path}' has unsupported magic/type bytes.")
            if expected_dims is not None and dimensions != expected_dims:
                raise DatasetFormatError(
                    f"IDX file '{path}' has {dimensions} dimensions; expected {expected_dims}."
                )
            shape_raw = stream.read(4 * dimensions)
            if len(shape_raw) != 4 * dimensions:
                raise DatasetFormatError(f"IDX file '{path}' has a truncated shape header.")
            shape = struct.unpack(">" + "I" * dimensions, shape_raw)
            count = int(np.prod(shape))
            payload = stream.read()
    except OSError as exc:
        raise DatasetFormatError(f"Could not read IDX file '{path}': {exc}") from exc
    if len(payload) != count:
        raise DatasetFormatError(
            f"IDX file '{path}' has {len(payload)} data bytes; header requires {count}."
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


def _find_idx(root: Path, stem: str) -> Path | None:
    return _find_first(root, [stem, f"{stem}.gz"])


def _npz_arrays(path: Path, dataset: str) -> LoadedDataset | None:
    """Load common Keras and KMNIST NPZ layouts, returning None if not suitable."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            standard = {"x_train", "y_train", "x_test", "y_test"}
            if standard.issubset(keys):
                train = _make_array_split(archive["x_train"], archive["y_train"], dataset=dataset)
                test = _make_array_split(archive["x_test"], archive["y_test"], dataset=dataset)
            elif {"images", "labels"}.issubset(keys):
                train = _make_array_split(archive["images"], archive["labels"], dataset=dataset)
                test = None
            else:
                return None
    except (OSError, ValueError) as exc:
        raise DatasetFormatError(f"Could not read NPZ dataset '{path}': {exc}") from exc
    labels = train.labels if test is None else np.concatenate([train.labels, test.labels])
    names = [str(label) for label in range(int(labels.max()) + 1)]
    splits = {"train": train}
    if test is not None:
        splits["test"] = test
    return LoadedDataset(dataset, names, splits, path.parent, {"layout": "npz", "file": str(path)})


def _load_mnist_like(dataset: str, path: str | Path, *, npz_patterns: Sequence[str], idx_prefix: str = "") -> LoadedDataset:
    root = _as_path(path, dataset)
    if root.is_file():
        if root.suffix.lower() != ".npz":
            raise DatasetFormatError(f"'{dataset}' expects an NPZ file or a directory containing IDX files, got '{root}'.")
        loaded = _npz_arrays(root, dataset)
        if loaded is None:
            raise DatasetFormatError(f"NPZ file '{root}' lacks x_train/y_train/x_test/y_test arrays.")
        return loaded
    npz = _find_first(root, list(npz_patterns))
    if npz:
        loaded = _npz_arrays(npz, dataset)
        if loaded is not None:
            return loaded
    # Official names are sufficient for MNIST/Fashion/EMNIST.  A prefix lets
    # KMNIST's alternative filenames remain discoverable through its NPZ path.
    train_images = _find_idx(root, f"{idx_prefix}train-images-idx3-ubyte")
    train_labels = _find_idx(root, f"{idx_prefix}train-labels-idx1-ubyte")
    test_images = _find_idx(root, f"{idx_prefix}t10k-images-idx3-ubyte")
    test_labels = _find_idx(root, f"{idx_prefix}t10k-labels-idx1-ubyte")
    if not all((train_images, train_labels, test_images, test_labels)):
        expected = "train-images-idx3-ubyte[.gz], train-labels-idx1-ubyte[.gz], t10k-images-idx3-ubyte[.gz], t10k-labels-idx1-ubyte[.gz]"
        raise LocalDatasetNotFoundError(
            f"Could not find all local {dataset} IDX files below '{root}'. Expected {expected}, "
            f"or one of {list(npz_patterns)}. No download was attempted."
        )
    train = _make_array_split(_read_idx(train_images, expected_dims=3), _read_idx(train_labels, expected_dims=1), dataset=dataset)
    test = _make_array_split(_read_idx(test_images, expected_dims=3), _read_idx(test_labels, expected_dims=1), dataset=dataset)
    highest = int(max(train.labels.max(), test.labels.max()))
    return LoadedDataset(
        dataset,
        [str(label) for label in range(highest + 1)],
        {"train": train, "test": test},
        root,
        {"layout": "idx", "original_splits": ["train", "test"]},
    )


def load_mnist(path: str | Path, **_: Any) -> LoadedDataset:
    return _load_mnist_like("mnist", path, npz_patterns=("mnist.npz", "MNIST.npz"))


def load_fashion_mnist(path: str | Path, **_: Any) -> LoadedDataset:
    return _load_mnist_like("fashion_mnist", path, npz_patterns=("fashion-mnist.npz", "fashion_mnist.npz", "FashionMNIST.npz"))


def load_kmnist(path: str | Path, **_: Any) -> LoadedDataset:
    root = _as_path(path, "kmnist")
    if root.is_file():
        return _load_mnist_like("kmnist", root, npz_patterns=(root.name,))
    train_img = _find_first(root, ("kmnist-train-imgs.npz", "kmnist-train-images.npz"))
    train_lbl = _find_first(root, ("kmnist-train-labels.npz",))
    test_img = _find_first(root, ("kmnist-test-imgs.npz", "kmnist-test-images.npz"))
    test_lbl = _find_first(root, ("kmnist-test-labels.npz",))
    if all((train_img, train_lbl, test_img, test_lbl)):
        try:
            with np.load(train_img, allow_pickle=False) as source:
                x_train = source[source.files[0]]
            with np.load(train_lbl, allow_pickle=False) as source:
                y_train = source[source.files[0]]
            with np.load(test_img, allow_pickle=False) as source:
                x_test = source[source.files[0]]
            with np.load(test_lbl, allow_pickle=False) as source:
                y_test = source[source.files[0]]
        except (OSError, ValueError, IndexError) as exc:
            raise DatasetFormatError(f"Could not read KMNIST NPZ components below '{root}': {exc}") from exc
        train = _make_array_split(x_train, y_train, dataset="kmnist")
        test = _make_array_split(x_test, y_test, dataset="kmnist")
        return LoadedDataset("kmnist", [str(i) for i in range(int(max(train.labels.max(), test.labels.max())) + 1)], {"train": train, "test": test}, root, {"layout": "kmnist-component-npz"})
    return _load_mnist_like("kmnist", root, npz_patterns=("kmnist.npz", "KMNIST.npz"))


def load_emnist_balanced(path: str | Path, **_: Any) -> LoadedDataset:
    root = _as_path(path, "emnist_balanced")
    if root.is_file():
        if root.suffix.lower() == ".mat":
            return _load_emnist_balanced_mat(root)
        return _load_mnist_like("emnist_balanced", root, npz_patterns=(root.name,))
    npz = _find_first(root, ("emnist-balanced.npz", "emnist_balanced.npz"))
    if npz:
        loaded = _npz_arrays(npz, "emnist_balanced")
        if loaded:
            return loaded
    matlab = _find_first(root, ("emnist-balanced.mat", "emnist_balanced.mat"))
    if matlab:
        return _load_emnist_balanced_mat(matlab)
    train_images = _find_idx(root, "emnist-balanced-train-images-idx3-ubyte")
    train_labels = _find_idx(root, "emnist-balanced-train-labels-idx1-ubyte")
    test_images = _find_idx(root, "emnist-balanced-test-images-idx3-ubyte")
    test_labels = _find_idx(root, "emnist-balanced-test-labels-idx1-ubyte")
    if not all((train_images, train_labels, test_images, test_labels)):
        raise LocalDatasetNotFoundError(
            f"Could not find EMNIST Balanced IDX files below '{root}'. Expected names beginning "
            "emnist-balanced-{train,test}-{images,labels}-idx*. No download was attempted."
        )
    # Official EMNIST IDX files serialize each character with the axes swapped.
    # Normalise it here so the shared resize/padding pipeline sees the expected
    # visual orientation, without affecting other MNIST-like sources.
    train_pixels = np.swapaxes(_read_idx(train_images, expected_dims=3), 1, 2)
    test_pixels = np.swapaxes(_read_idx(test_images, expected_dims=3), 1, 2)
    train = _make_array_split(train_pixels, _read_idx(train_labels, expected_dims=1), dataset="emnist_balanced")
    test = _make_array_split(test_pixels, _read_idx(test_labels, expected_dims=1), dataset="emnist_balanced")
    # Official Balanced split has labels 0..46, but deriving this retains local variants.
    class_count = int(max(train.labels.max(), test.labels.max())) + 1
    return LoadedDataset(
        "emnist_balanced",
        [str(i) for i in range(class_count)],
        {"train": train, "test": test},
        root,
        {"layout": "emnist-idx", "orientation": "axes_swapped_to_row_major"},
    )


def _mat_field(value: Any, *names: str) -> Any:
    """Read a field from scipy's dict/mat_struct/structured-array variants."""
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
        if isinstance(value, np.ndarray) and value.dtype.names and name in value.dtype.names:
            return value[name]
    raise DatasetFormatError(f"MATLAB structure lacks expected field(s): {names}.")


def _emnist_mat_images(value: Any, *, source: Path) -> np.ndarray:
    images = np.asarray(value)
    if images.ndim == 1 and images.size == 28 * 28:
        images = images.reshape(1, -1)
    if images.ndim != 2:
        raise DatasetFormatError(f"EMNIST MATLAB images in '{source}' need shape N×784 or 784×N, got {images.shape}.")
    if images.shape[1] == 28 * 28:
        rows = images
    elif images.shape[0] == 28 * 28:
        rows = images.T
    else:
        raise DatasetFormatError(f"EMNIST MATLAB images in '{source}' do not have a 784-pixel axis: {images.shape}.")
    # MATLAB serializes each 28×28 image in column-major order.  The transpose
    # converts that representation into conventional row-major image geometry.
    return rows.reshape(-1, 28, 28).transpose(0, 2, 1)


def _load_emnist_balanced_mat(path: Path) -> LoadedDataset:
    """Load the official optional ``matlab/emnist-balanced.mat`` layout."""
    loadmat = _import_scipy_io()
    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        container = payload.get("dataset", payload)
        train_part = _mat_field(container, "train")
        test_part = _mat_field(container, "test")
        train_images = _emnist_mat_images(_mat_field(train_part, "images", "data"), source=path)
        test_images = _emnist_mat_images(_mat_field(test_part, "images", "data"), source=path)
        train_labels = np.asarray(_mat_field(train_part, "labels", "label"), dtype=np.int64).reshape(-1)
        test_labels = np.asarray(_mat_field(test_part, "labels", "label"), dtype=np.int64).reshape(-1)
    except DatasetFormatError:
        raise
    except Exception as exc:
        raise DatasetFormatError(f"Could not read EMNIST Balanced MATLAB file '{path}': {exc}") from exc
    train = _make_array_split(train_images, train_labels, dataset="emnist_balanced")
    test = _make_array_split(test_images, test_labels, dataset="emnist_balanced")
    class_count = int(max(train.labels.max(), test.labels.max())) + 1
    return LoadedDataset(
        "emnist_balanced",
        [str(index) for index in range(class_count)],
        {"train": train, "test": test},
        path.parent,
        {"layout": "emnist-matlab", "file": str(path), "matlab_column_major_transposed": True},
    )


def _load_pickle(path: Path, *, dataset: str) -> Mapping[Any, Any]:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream, encoding="latin1")
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
        raise DatasetFormatError(f"Could not read {dataset} pickle '{path}': {exc}") from exc
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{dataset} pickle '{path}' must contain a mapping.")
    return value


def _mapping_get(mapping: Mapping[Any, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
        binary = key.encode("utf-8")
        if binary in mapping:
            return mapping[binary]
    raise DatasetFormatError(f"Missing expected pickle field; tried {keys}.")


_CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
_CIFAR100_COARSE_NAMES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals",
    "trees", "vehicles_1", "vehicles_2",
]


def _cifar_images(data: Any, *, dataset: str) -> np.ndarray:
    raw = np.asarray(data)
    if raw.ndim == 2 and raw.shape[1] == 3072:
        return raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    if raw.ndim == 4:
        return _ensure_nhwc(raw, dataset=dataset)
    raise DatasetFormatError(f"{dataset} image payload has unsupported shape {raw.shape}; expected N x 3072.")


def load_cifar10(path: str | Path, **_: Any) -> LoadedDataset:
    root = _as_path(path, "cifar10")
    if root.is_file() and root.suffix.lower() == ".npz":
        loaded = _npz_arrays(root, "cifar10")
        if loaded is None:
            raise DatasetFormatError(f"NPZ file '{root}' does not have Keras-style CIFAR-10 arrays.")
        loaded.class_names = list(_CIFAR10_NAMES)
        return loaded
    base = root if root.is_dir() else root.parent
    train_files = [candidate for candidate in (_find_first(base, (f"data_batch_{i}",)) for i in range(1, 6)) if candidate]
    test_file = _find_first(base, ("test_batch",))
    if len(train_files) != 5 or test_file is None:
        npz = _find_first(base, ("cifar10.npz", "cifar-10.npz"))
        if npz:
            return load_cifar10(npz)
        raise LocalDatasetNotFoundError(
            f"Could not find CIFAR-10 data_batch_1..5 and test_batch below '{base}'. No download was attempted."
        )
    train_images: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for data_file in sorted(train_files, key=lambda item: item.name):
        entry = _load_pickle(data_file, dataset="CIFAR-10")
        train_images.append(_cifar_images(_mapping_get(entry, "data"), dataset="cifar10"))
        train_labels.append(np.asarray(_mapping_get(entry, "labels", "fine_labels"), dtype=np.int64))
    test_entry = _load_pickle(test_file, dataset="CIFAR-10")
    train = _make_array_split(np.concatenate(train_images), np.concatenate(train_labels), dataset="cifar10")
    test = _make_array_split(_cifar_images(_mapping_get(test_entry, "data"), dataset="cifar10"), np.asarray(_mapping_get(test_entry, "labels", "fine_labels"), dtype=np.int64), dataset="cifar10")
    return LoadedDataset("cifar10", list(_CIFAR10_NAMES), {"train": train, "test": test}, base, {"layout": "cifar-10-python"})


def load_cifar100_coarse(path: str | Path, **_: Any) -> LoadedDataset:
    root = _as_path(path, "cifar100_coarse")
    if root.is_file() and root.suffix.lower() == ".npz":
        try:
            with np.load(root, allow_pickle=False) as source:
                required = {"x_train", "y_train", "x_test", "y_test"}
                if not required.issubset(source.files):
                    raise DatasetFormatError(f"NPZ '{root}' needs x_train/y_train/x_test/y_test coarse labels.")
                train = _make_array_split(source["x_train"], source["y_train"], dataset="cifar100_coarse")
                test = _make_array_split(source["x_test"], source["y_test"], dataset="cifar100_coarse")
        except (OSError, ValueError) as exc:
            if isinstance(exc, DatasetFormatError):
                raise
            raise DatasetFormatError(f"Could not read CIFAR-100 NPZ '{root}': {exc}") from exc
        return LoadedDataset("cifar100_coarse", list(_CIFAR100_COARSE_NAMES), {"train": train, "test": test}, root.parent, {"layout": "npz", "label_level": "coarse"})
    base = root if root.is_dir() else root.parent
    train_file = _find_first(base, ("train",))
    test_file = _find_first(base, ("test",))
    if not train_file or not test_file:
        raise LocalDatasetNotFoundError(
            f"Could not find CIFAR-100 'train' and 'test' files below '{base}'. No download was attempted."
        )
    train_entry = _load_pickle(train_file, dataset="CIFAR-100")
    test_entry = _load_pickle(test_file, dataset="CIFAR-100")
    try:
        train_labels = _mapping_get(train_entry, "coarse_labels")
        test_labels = _mapping_get(test_entry, "coarse_labels")
    except DatasetFormatError as exc:
        raise DatasetFormatError(
            f"CIFAR-100 at '{base}' does not include coarse_labels. This benchmark intentionally uses superclasses."
        ) from exc
    train = _make_array_split(_cifar_images(_mapping_get(train_entry, "data"), dataset="cifar100_coarse"), train_labels, dataset="cifar100_coarse")
    test = _make_array_split(_cifar_images(_mapping_get(test_entry, "data"), dataset="cifar100_coarse"), test_labels, dataset="cifar100_coarse")
    return LoadedDataset("cifar100_coarse", list(_CIFAR100_COARSE_NAMES), {"train": train, "test": test}, base, {"layout": "cifar-100-python", "label_level": "coarse"})


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _class_folder_records(
    root: Path,
    split_dirs: Mapping[str, Path],
    *,
    dataset: str,
    class_order: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, DatasetSplit]]:
    """Build lazy samples from ``split/class/image`` directories."""
    class_set: set[str] = set()
    for split_dir in split_dirs.values():
        if split_dir.is_dir():
            class_set.update(child.name for child in split_dir.iterdir() if child.is_dir())
    if not class_set:
        raise DatasetFormatError(
            f"'{dataset}' has no class directories in {', '.join(str(value) for value in split_dirs.values())}."
        )
    classes = list(class_order) if class_order is not None else sorted(class_set, key=_natural_sort_key)
    missing = class_set.difference(classes)
    if missing:
        classes.extend(sorted(missing, key=_natural_sort_key))
    mapping = {name: index for index, name in enumerate(classes)}
    splits: dict[str, DatasetSplit] = {}
    for split_name, split_dir in split_dirs.items():
        if not split_dir.is_dir():
            continue
        records: list[SampleRecord] = []
        for class_name in classes:
            class_dir = split_dir / class_name
            for image_path in _image_files(class_dir):
                records.append(SampleRecord(image_path, mapping[class_name], split_name))
        if records:
            splits[split_name] = DatasetSplit(records=records)
    if not splits:
        raise DatasetFormatError(f"'{dataset}' contains no supported image files.")
    return classes, splits


def _natural_sort_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def load_cinic10(path: str | Path, **_: Any) -> LoadedDataset:
    root = _as_path(path, "cinic10")
    if not root.is_dir():
        raise DatasetFormatError(f"CINIC-10 path must be a directory, got '{root}'.")
    candidates = [root]
    candidates.extend(candidate for candidate in root.iterdir() if candidate.is_dir())
    selected: dict[str, Path] | None = None
    for base in candidates:
        detected = {name: base / name for name in ("train", "valid", "test") if (base / name).is_dir()}
        if detected:
            selected = detected
            root = base
            break
    if not selected:
        raise LocalDatasetNotFoundError(
            f"Could not find CINIC-10 train/valid/test directories below '{root}'. No download was attempted."
        )
    classes, splits = _class_folder_records(root, selected, dataset="cinic10", class_order=_CIFAR10_NAMES)
    return LoadedDataset("cinic10", classes, splits, root, {"layout": "class-folders"})


def _import_scipy_io() -> Any:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise OptionalDependencyError("Loading SVHN .mat files requires SciPy. Install 'scipy'.") from exc
    return loadmat


def _load_svhn_mat(path: Path) -> DatasetSplit:
    loadmat = _import_scipy_io()
    try:
        data = loadmat(path)
    except Exception as exc:
        raise DatasetFormatError(f"Could not read SVHN MAT file '{path}': {exc}") from exc
    if "X" not in data or "y" not in data:
        raise DatasetFormatError(f"SVHN MAT file '{path}' needs 'X' and 'y' arrays.")
    images = np.asarray(data["X"])
    labels = np.asarray(data["y"], dtype=np.int64).reshape(-1)
    if images.ndim != 4:
        raise DatasetFormatError(f"SVHN X in '{path}' needs four dimensions, got {images.shape}.")
    # Official SVHN is H,W,C,N; retain support for already-NHWC variants.
    if images.shape[0:3] == (32, 32, 3):
        images = np.moveaxis(images, -1, 0)
    images = _ensure_nhwc(images, dataset="svhn")
    labels[labels == 10] = 0
    return _make_array_split(images, labels, dataset="svhn")


def load_svhn(path: str | Path, *, include_extra: bool = False, **_: Any) -> LoadedDataset:
    root = _as_path(path, "svhn")
    if root.is_file():
        # A single MAT is permitted for diagnostics, though it becomes an unnamed split.
        split_name = "train" if "train" in root.name.lower() else "test" if "test" in root.name.lower() else "unspecified"
        return LoadedDataset("svhn", [str(i) for i in range(10)], {split_name: _load_svhn_mat(root)}, root.parent, {"layout": "svhn-mat", "include_extra": False})
    train_file = _find_first(root, ("train_32x32.mat",))
    test_file = _find_first(root, ("test_32x32.mat",))
    if not train_file or not test_file:
        raise LocalDatasetNotFoundError(
            f"Could not find train_32x32.mat and test_32x32.mat below '{root}'. No download was attempted."
        )
    splits = {"train": _load_svhn_mat(train_file), "test": _load_svhn_mat(test_file)}
    extra_file = _find_first(root, ("extra_32x32.mat",))
    if include_extra and extra_file:
        splits["extra"] = _load_svhn_mat(extra_file)
    return LoadedDataset("svhn", [str(i) for i in range(10)], splits, root, {"layout": "svhn-mat", "include_extra": bool(include_extra and extra_file)})


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read UTF-8/Latin-1 CSVs with comma or semicolon delimiters."""
    text: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise DatasetFormatError(f"Could not read CSV '{path}': {exc}") from exc
    if text is None:
        raise DatasetFormatError(f"Could not decode CSV '{path}' as UTF-8 or Latin-1.")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;")
    except csv.Error:
        dialect = csv.excel
        dialect.delimiter = ";" if text.partition("\n")[0].count(";") > text.partition("\n")[0].count(",") else ","
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        raise DatasetFormatError(f"CSV '{path}' has no header row.")
    rows = [{str(key).strip(): ("" if value is None else str(value).strip()) for key, value in row.items()} for row in reader]
    return rows, [str(name).strip() for name in reader.fieldnames]


def _field_lookup(fields: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalised = {re.sub(r"[^a-z0-9]", "", field.lower()): field for field in fields}
    for candidate in candidates:
        result = normalised.get(re.sub(r"[^a-z0-9]", "", candidate.lower()))
        if result:
            return result
    return None


def _resolve_csv_image_path(csv_path: Path, raw: str, roots: Sequence[Path]) -> Path:
    raw_path = Path(raw.replace("\\", "/"))
    if raw_path.is_absolute():
        return raw_path
    candidates = [csv_path.parent / raw_path] + [root / raw_path for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # Preserve the primary candidate so audit can identify absent paths rather than hiding them.
    return candidates[0].resolve()


def _gtsrb_csv_records(csv_path: Path, *, split: str, roots: Sequence[Path]) -> list[SampleRecord]:
    rows, fields = _read_csv_rows(csv_path)
    path_field = _field_lookup(fields, ("Path", "Filename", "FileName", "image", "image_path"))
    label_field = _field_lookup(fields, ("ClassId", "ClassID", "class", "label", "target"))
    if not path_field or not label_field:
        raise DatasetFormatError(
            f"GTSRB CSV '{csv_path}' must include a path column and ClassId/label column; found {fields}."
        )
    records: list[SampleRecord] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            label = int(float(row[label_field]))
        except ValueError as exc:
            raise DatasetFormatError(f"Invalid GTSRB label at {csv_path}:{row_number}: {row[label_field]!r}.") from exc
        records.append(
            SampleRecord(
                _resolve_csv_image_path(csv_path, row[path_field], roots),
                label,
                split,
                {"csv": str(csv_path), "row": row_number},
            )
        )
    return records


def _gtsrb_folder_records(train_root: Path, *, split: str = "train") -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for class_dir in sorted((item for item in train_root.iterdir() if item.is_dir()), key=lambda item: _natural_sort_key(item.name)):
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        # Per-class GT-xxxxx.csv is authoritative if present; folder label is the fallback.
        local_csv = next((entry for entry in class_dir.glob("*.csv") if "gt" in entry.name.lower()), None)
        if local_csv:
            records.extend(_gtsrb_csv_records(local_csv, split=split, roots=(train_root, class_dir)))
        else:
            records.extend(SampleRecord(image, label, split) for image in _image_files(class_dir))
    return records


def _first_labelled_gtsrb_csv(candidates: Iterable[Path]) -> Path | None:
    """Ignore image-metadata CSVs that do not contain a class label.

    The official GTSRB test-images archive includes
    ``GT-final_test.test.csv`` with geometry only.  The separate ground-truth
    archive supplies ``GT-final_test.csv`` with ``ClassId``.  Selecting by
    filename alone can therefore make a complete local dataset look invalid.
    """

    for candidate in candidates:
        try:
            _, fields = _read_csv_rows(candidate)
        except DatasetAdapterError:
            continue
        if _field_lookup(fields, ("ClassId", "ClassID", "class", "label", "target")):
            return candidate
    return None


def load_gtsrb(path: str | Path, **_: Any) -> LoadedDataset:
    root = _as_path(path, "gtsrb")
    if not root.is_dir():
        raise DatasetFormatError(f"GTSRB path must be a directory, got '{root}'.")
    directories = [root] + [entry for entry in root.rglob("*") if entry.is_dir() and entry.relative_to(root).parts.__len__() <= 3]
    train_root = next((entry for entry in directories if entry.name.lower() in {"train", "final_training", "training"}), None)
    # Some archives put images directly below Final_Training/Images.
    if train_root and (train_root / "Images").is_dir():
        train_root = train_root / "Images"
    train_records: list[SampleRecord] = []
    if train_root:
        train_csv = next((entry for entry in train_root.glob("*.csv") if "train" in entry.name.lower()), None)
        if train_csv:
            train_records = _gtsrb_csv_records(train_csv, split="train", roots=(root, train_root))
        else:
            train_records = _gtsrb_folder_records(train_root)
    # Kaggle layout often provides Train.csv at archive root even when Train directory exists.
    root_train_csv = next((entry for entry in root.glob("*.csv") if entry.name.lower() in {"train.csv", "training.csv"}), None)
    if root_train_csv:
        train_records = _gtsrb_csv_records(root_train_csv, split="train", roots=(root, train_root or root))
    test_root = next((entry for entry in directories if entry.name.lower() in {"test", "final_test"}), None)
    if test_root and (test_root / "Images").is_dir():
        test_root = test_root / "Images"
    test_candidates = sorted(
        (
            entry
            for entry in root.rglob("*.csv")
            if "test" in entry.name.lower() and ("gt" in entry.name.lower() or entry.name.lower() == "test.csv")
        ),
        key=lambda entry: (entry.name.lower() != "gt-final_test.csv", str(entry).lower()),
    )
    test_csv = _first_labelled_gtsrb_csv(test_candidates)
    test_records = _gtsrb_csv_records(test_csv, split="test", roots=(root, test_root or root)) if test_csv else []
    if not test_records and test_root:
        # Folder-only test data is valid only if its immediate folders encode labels.
        test_records = _gtsrb_folder_records(test_root, split="test")
    if not train_records:
        raise LocalDatasetNotFoundError(
            f"Could not find labelled GTSRB training images below '{root}'. Supported layouts are Train/<class>/, "
            "Final_Training/Images/<class>/, or a Train.csv with Path and ClassId. No download was attempted."
        )
    # The official dataset has 43 numeric labels.  Retain any larger local labels so validation can report them.
    largest = max(record.label for record in train_records + test_records)
    class_names = [str(index) for index in range(max(43, largest + 1))]
    splits = {"train": DatasetSplit(records=train_records)}
    if test_records:
        splits["test"] = DatasetSplit(records=test_records)
    return LoadedDataset("gtsrb", class_names, splits, root, {"layout": "gtsrb-folders-or-csv"})


_FER_CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_fer2013(path: str | Path, **_: Any) -> LoadedDataset:
    configured = _as_path(path, "fer2013")
    csv_path = configured if configured.is_file() else _find_first(configured, ("fer2013.csv", "FER2013.csv", "fer2013new.csv"))
    if csv_path is None:
        raise LocalDatasetNotFoundError(
            f"Could not find a FER2013 CSV below '{configured}'. Expected fer2013.csv with emotion and pixels columns."
        )
    rows, fields = _read_csv_rows(csv_path)
    label_field = _field_lookup(fields, ("emotion", "label", "class", "target"))
    pixels_field = _field_lookup(fields, ("pixels", "pixel", "image"))
    usage_field = _field_lookup(fields, ("Usage", "split", "set"))
    if not label_field or not pixels_field:
        raise DatasetFormatError(
            f"FER2013 CSV '{csv_path}' needs emotion/label and pixels columns; found {fields}."
        )
    grouped_images: dict[str, list[np.ndarray]] = {}
    grouped_labels: dict[str, list[int]] = {}
    usage_map = {"training": "train", "train": "train", "publictest": "validation", "validation": "validation", "val": "validation", "privatetest": "test", "test": "test"}
    for row_number, row in enumerate(rows, start=2):
        try:
            label = int(float(row[label_field]))
        except ValueError as exc:
            raise DatasetFormatError(f"Invalid FER2013 emotion at {csv_path}:{row_number}: {row[label_field]!r}.") from exc
        if label < 0:
            raise DatasetFormatError(f"FER2013 label at {csv_path}:{row_number} is negative ({label}).")
        values = np.fromstring(row[pixels_field], dtype=np.uint8, sep=" ")
        if values.size != 48 * 48:
            raise DatasetFormatError(
                f"FER2013 pixels at {csv_path}:{row_number} contain {values.size} values; expected 2304."
            )
        raw_usage = row[usage_field].strip().lower() if usage_field else "unspecified"
        split = usage_map.get(raw_usage, raw_usage or "unspecified")
        grouped_images.setdefault(split, []).append(values.reshape(48, 48))
        grouped_labels.setdefault(split, []).append(label)
    if not grouped_images:
        raise DatasetFormatError(f"FER2013 CSV '{csv_path}' contains no data rows.")
    max_label = max(max(values) for values in grouped_labels.values())
    names = list(_FER_CLASS_NAMES)
    if max_label >= len(names):
        names.extend(str(index) for index in range(len(names), max_label + 1))
    splits = {
        split: _make_array_split(np.stack(images), np.asarray(grouped_labels[split]), dataset="fer2013")
        for split, images in grouped_images.items()
    }
    return LoadedDataset("fer2013", names, splits, csv_path.parent, {"layout": "fer2013-csv", "file": str(csv_path)})


Adapter = Callable[..., LoadedDataset]

ADAPTER_REGISTRY: dict[str, Adapter] = {
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "kmnist": load_kmnist,
    "emnist_balanced": load_emnist_balanced,
    "cifar10": load_cifar10,
    "cifar100_coarse": load_cifar100_coarse,
    "cinic10": load_cinic10,
    "svhn": load_svhn,
    "gtsrb": load_gtsrb,
    "fer2013": load_fer2013,
}


def available_adapters() -> tuple[str, ...]:
    return tuple(ADAPTER_REGISTRY)


def get_adapter(name: str) -> Adapter:
    canonical = _normalise_dataset_name(name)
    try:
        return ADAPTER_REGISTRY[canonical]
    except KeyError as exc:
        raise DatasetAdapterError(
            f"Unsupported dataset '{name}'. Available local adapters: {', '.join(available_adapters())}."
        ) from exc


def load_local_dataset(name: str, path: str | Path, **options: Any) -> LoadedDataset:
    """Load one supported dataset from an already-present local path.

    ``options`` currently supports ``include_extra=True`` for SVHN.  Unknown
    options are tolerated by adapters to keep manifest handling forward-safe.
    """
    canonical = _normalise_dataset_name(name)
    dataset = get_adapter(canonical)(path, **options)
    if dataset.name != canonical:
        dataset.name = canonical
    return dataset


def load_dataset(name: str, path: str | Path, **options: Any) -> LoadedDataset:
    """Compatibility alias for :func:`load_local_dataset`."""
    return load_local_dataset(name, path, **options)


def load_manifest_entry(entry: Mapping[str, Any]) -> LoadedDataset:
    """Load a minimal manifest item: ``{'name': 'mnist', 'path': 'D:/...'}``."""
    name = entry.get("name", entry.get("dataset"))
    path = entry.get("path", entry.get("source_path"))
    if not name or not path:
        raise DatasetFormatError("Dataset manifest entry requires both 'name' (or 'dataset') and 'path'.")
    options = {key: value for key, value in entry.items() if key not in {"name", "dataset", "path", "source_path"}}
    return load_local_dataset(str(name), str(path), **options)


__all__ = [
    "ADAPTER_REGISTRY",
    "DatasetAdapterError",
    "DatasetFormatError",
    "DatasetSplit",
    "LoadedDataset",
    "LocalDatasetNotFoundError",
    "OptionalDependencyError",
    "SampleRecord",
    "available_adapters",
    "get_adapter",
    "load_dataset",
    "load_image_file",
    "load_local_dataset",
    "load_manifest_entry",
]
