"""Atomic persistence primitives shared by classical experiment runners."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FILE = "manifest.json"


def utc_now() -> str:
    """Return a timezone-aware timestamp suitable for persisted artifacts."""

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_ready(item) for item in value), key=repr)
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_ready(scalar())
        except (TypeError, ValueError):
            pass
    as_list = getattr(value, "tolist", None)
    if callable(as_list):
        try:
            return _json_ready(as_list())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    """Render a stable JSON representation for hashes and JSONL records."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def config_fingerprint(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class RunPaths:
    """Canonical paths created for one classical training run."""

    root: Path
    manifest: Path
    status: Path
    checkpoints: Path
    logs: Path
    artifacts: Path
    telemetry: Path

    @classmethod
    def from_root(cls, run_dir: str | Path) -> "RunPaths":
        root = Path(run_dir)
        return cls(
            root=root,
            manifest=root / MANIFEST_FILE,
            status=root / "status.json",
            checkpoints=root / "checkpoints",
            logs=root / "logs",
            artifacts=root / "artifacts",
            telemetry=root / "telemetry",
        )

    def ensure(self) -> "RunPaths":
        for directory in (self.root, self.checkpoints, self.logs, self.artifacts, self.telemetry):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write bytes atomically, leaving no partial artifact after interruption."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    rendered = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)
    return atomic_write_text(path, f"{rendered}\n")
