"""Read-only, provenance-checked feature vectors shared by all heads."""
from __future__ import annotations
import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from utils.experiment_config import DEFAULT_FEATURES_ROOT, INPUT_FEATURES, NUM_CLASSES
from utils.state import atomic_write_bytes

REQUIRED_FILES = tuple(f"{split}_{kind}.npy"
                      for split in ("train", "val", "test")
                      for kind in ("features", "labels"))

class FeatureVectorsError(RuntimeError):
    """Feature arrays or their provenance do not satisfy the shared contract."""

@dataclasses.dataclass(frozen=True)
class FeatureBundle:
    dataset: str
    num_classes: int
    directory: Path
    features: Mapping[str, np.ndarray]
    labels: Mapping[str, np.ndarray]
    statistics: Mapping[str, Mapping[str, Any]]
    hashes: Mapping[str, str]
    source_manifest: Mapping[str, Any]
    split_fingerprint: str | None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FeatureVectorsError(f"JSON deveria conter um objeto: {path}")
    return payload


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    atomic_write_bytes(path, buffer.getvalue())


def _looks_like_feature_directory(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def discover_features128(dataset: str, features_root: str | Path = DEFAULT_FEATURES_ROOT) -> Path:
    """Resolve exactly one complete features128 directory for ``dataset``."""

    root = Path(features_root).expanduser().resolve()
    candidates: list[Path] = []
    direct = (root, root / dataset, root / "features128" / dataset)
    candidates.extend(path for path in direct if _looks_like_feature_directory(path))
    candidates.extend(
        path for path in (root / dataset / "runs").glob("*/artifacts/features128")
        if _looks_like_feature_directory(path)
    )
    unique = sorted({path.resolve() for path in candidates}, key=str)
    if len(unique) != 1:
        rendered = "\n".join(f"  - {path}" for path in unique) or "  (nenhum)"
        raise FeatureVectorsError(
            f"Esperava exatamente um diretório features128 completo para {dataset}; "
            f"encontrei {len(unique)} em {root}:\n{rendered}"
        )
    return unique[0]


def _feature_statistics(values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(map(int, values.shape)),
        "dtype": str(values.dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def _source_metadata(directory: Path) -> tuple[dict[str, Any], str | None]:
    feature_manifest = _read_json(directory / "manifest.json")
    run_root = directory.parent.parent
    run_manifest = _read_json(run_root / "manifest.json") if run_root.name != directory.name else {}
    split_fingerprint = run_manifest.get("split_fingerprint") or feature_manifest.get("split_fingerprint")
    return {
        "features_manifest": feature_manifest,
        "features_manifest_path": str((directory / "manifest.json").resolve())
        if (directory / "manifest.json").is_file() else None,
        "source_run_manifest": str((run_root / "manifest.json").resolve())
        if (run_root / "manifest.json").is_file() else None,
        "source_run_id": run_manifest.get("run_id"),
    }, str(split_fingerprint) if split_fingerprint else None


def load_feature_bundle(
    dataset: str,
    features_root: str | Path = DEFAULT_FEATURES_ROOT,
) -> FeatureBundle:
    """Load and strictly validate the six saved arrays without transforming them."""

    if dataset not in NUM_CLASSES:
        raise FeatureVectorsError(f"Dataset não suportado: {dataset}")
    directory = discover_features128(dataset, features_root)
    num_classes = int(NUM_CLASSES[dataset])
    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    statistics: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {}

    for split in ("train", "val", "test"):
        feature_path = directory / f"{split}_features.npy"
        label_path = directory / f"{split}_labels.npy"
        x_values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        y_values = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if x_values.ndim != 2 or tuple(x_values.shape[1:]) != (INPUT_FEATURES,):
            raise FeatureVectorsError(
                f"{dataset}/{split}: esperado shape (N,{INPUT_FEATURES}); recebido {x_values.shape}."
            )
        if x_values.dtype != np.float32:
            raise FeatureVectorsError(
                f"{dataset}/{split}: features devem ser float32; recebido {x_values.dtype}."
            )
        if y_values.ndim != 1 or y_values.dtype.kind not in {"i", "u"}:
            raise FeatureVectorsError(
                f"{dataset}/{split}: labels devem ser um vetor inteiro; recebido {y_values.shape}/{y_values.dtype}."
            )
        if len(x_values) != len(y_values) or len(x_values) == 0:
            raise FeatureVectorsError(f"{dataset}/{split}: features e labels possuem tamanhos incompatíveis.")
        if not np.isfinite(x_values).all():
            raise FeatureVectorsError(f"{dataset}/{split}: features contêm NaN ou infinito.")
        integer_labels = np.asarray(y_values, dtype=np.int64)
        if np.any(integer_labels < 0) or np.any(integer_labels >= num_classes):
            raise FeatureVectorsError(
                f"{dataset}/{split}: labels fora do intervalo [0,{num_classes})."
            )
        observed = set(map(int, np.unique(integer_labels)))
        expected = set(range(num_classes))
        if observed != expected:
            raise FeatureVectorsError(
                f"{dataset}/{split}: classes presentes {sorted(observed)} diferem de {sorted(expected)}."
            )
        features[split] = x_values
        labels[split] = integer_labels
        statistics[split] = {
            **_feature_statistics(x_values),
            "examples": int(len(x_values)),
            "label_dtype": str(y_values.dtype),
            "label_min": int(integer_labels.min()),
            "label_max": int(integer_labels.max()),
        }
        hashes[feature_path.name] = _sha256_file(feature_path)
        hashes[label_path.name] = _sha256_file(label_path)

    source_manifest, split_fingerprint = _source_metadata(directory)
    feature_manifest = source_manifest["features_manifest"]
    if feature_manifest:
        if feature_manifest.get("status") not in (None, "completed"):
            raise FeatureVectorsError("Exportação de features não está concluída.")
        if feature_manifest.get("dataset", dataset) != dataset:
            raise FeatureVectorsError("Manifest de features pertence a outro dataset.")
        if int(feature_manifest.get("feature_width", INPUT_FEATURES)) != INPUT_FEATURES:
            raise FeatureVectorsError("Largura declarada no manifest diverge.")
        if feature_manifest.get("split_fingerprint") not in (None, split_fingerprint):
            raise FeatureVectorsError("Fingerprint dos splits diverge entre manifests.")
        for split in ("train", "val", "test"):
            declared = feature_manifest.get("splits", {}).get(split, {})
            if declared.get("examples") not in (None, len(labels[split])):
                raise FeatureVectorsError(f"Tamanho declarado diverge: {dataset}/{split}")
            for kind in ("features", "labels"):
                name = f"{split}_{kind}.npy"
                if declared.get(f"{kind}_sha256") not in (None, hashes[name]):
                    raise FeatureVectorsError(f"Hash declarado diverge: {dataset}/{name}")
                values = features[split] if kind == "features" else labels[split]
                if declared.get(f"{kind}_shape") not in (None, list(values.shape)):
                    raise FeatureVectorsError(f"Shape declarado diverge: {dataset}/{name}")
                if declared.get(f"{kind}_dtype") not in (None, statistics[split]["dtype" if kind == "features" else "label_dtype"]):
                    raise FeatureVectorsError(f"Dtype declarado diverge: {dataset}/{name}")
    return FeatureBundle(
        dataset=dataset,
        num_classes=num_classes,
        directory=directory,
        features=features,
        labels=labels,
        statistics=statistics,
        hashes=hashes,
        source_manifest=source_manifest,
        split_fingerprint=split_fingerprint,
    )
