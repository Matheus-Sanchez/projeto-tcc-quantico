"""Persistência atômica e estado retomável de execuções de benchmark."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

MANIFEST_FILE = "manifest.json"
STATE_EVENTS_FILE = "state_events.jsonl"
VALID_RUN_STATUSES = frozenset({"pending", "running", "interrupted", "completed", "failed"})

class StateError(RuntimeError): pass
class ManifestCompatibilityError(StateError): pass
class InvalidStateTransition(StateError): pass

def utc_now() -> str:
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
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value

def canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)

def config_fingerprint(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()

def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "unknown"

@dataclasses.dataclass(frozen=True)
class RunIdentity:
    dataset: str
    normalization: str
    balance_mode: str
    seed: int

    @property
    def run_id(self) -> str:
        return stable_run_id(self.dataset, self.normalization, self.balance_mode, self.seed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "normalization": self.normalization,
            "balance_mode": self.balance_mode,
            "seed": int(self.seed),
        }

def stable_run_id(dataset: str, normalization: str, balance_mode: str, seed: int) -> str:
    return "__".join((_slug(dataset), _slug(normalization), _slug(balance_mode), f"seed-{int(seed)}"))

@dataclasses.dataclass(frozen=True)
class RunPaths:
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            try: Path(temporary_name).unlink(missing_ok=True)
            except OSError: pass
    return destination

def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))

def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    rendered = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)
    return atomic_write_text(path, f"{rendered}\n")
