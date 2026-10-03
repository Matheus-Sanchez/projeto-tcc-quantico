"""Lightweight, file-backed hardware telemetry for long training runs."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from utils.state import atomic_write_json, canonical_json, utc_now

try:  # ``psutil`` is optional at import time so non-training utilities still work.
    import psutil
except ImportError:  # pragma: no cover - depends on the environment
    psutil = None


_MEBIBYTE = 1024.0 * 1024.0
_PSUTIL_ERRORS = (OSError,) if psutil is None else (psutil.Error, OSError)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


class TelemetrySampler:
    """Sample CPU, RAM, process, disk and (when present) NVIDIA GPU usage.

    Samples are appended to ``samples.jsonl`` as they are collected, so a long
    experiment remains inspectable even if it is interrupted.  ``summary.json``
    is written when :meth:`stop` is called.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        interval_seconds: float = 5.0,
        disk_paths: Mapping[str, str | Path] | None = None,
        data_path: str | Path | None = None,
        process_id: int | None = None,
        include_children: bool = False,
    ) -> None:
        if float(interval_seconds) <= 0:
            raise ValueError("interval_seconds deve ser positivo.")
        self.output_dir = Path(output_dir)
        self.interval_seconds = float(interval_seconds)
        self.process_id = int(process_id if process_id is not None else os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._samples_lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._started_at: float | None = None
        self._process: Any | None = None
        self.include_children = bool(include_children)
        self._child_processes: dict[int, Any] = {}
        self._nvidia_smi = shutil.which("nvidia-smi")
        paths = {"output": self.output_dir}
        if data_path is not None:
            paths["data"] = Path(data_path)
        if disk_paths:
            paths.update({str(name): Path(path) for name, path in disk_paths.items()})
        self._disk_paths = paths

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def samples_path(self) -> Path:
        return self.output_dir / "samples.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def hardware_path(self) -> Path:
        return self.output_dir / "hardware.json"

    def _prime_psutil(self) -> None:
        if psutil is None:
            return
        try:
            self._process = psutil.Process(self.process_id)
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)
        except (psutil.Error, OSError):
            self._process = None

    def _hardware_metadata(self) -> dict[str, Any]:
        physical_cores = logical_cores = None
        if psutil is not None:
            try:
                physical_cores = psutil.cpu_count(logical=False)
                logical_cores = psutil.cpu_count(logical=True)
            except psutil.Error:
                pass
        return {
            "timestamp": utc_now(),
            "platform": platform.platform(),
            "python": sys.version,
            "process_id": self.process_id,
            "cpu": {
                "processor": platform.processor() or None,
                "physical_cores": physical_cores,
                "logical_cores": logical_cores,
            },
            "telemetry": {
                "psutil_available": psutil is not None,
                "nvidia_smi_available": self._nvidia_smi is not None,
                "interval_seconds": self.interval_seconds,
            },
        }

    def start(self) -> "TelemetrySampler":
        with self._lifecycle_lock:
            if self.running:
                return self
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stop_event.clear()
            self._samples.clear()
            self._started_at = time.perf_counter()
            self._prime_psutil()
            atomic_write_json(self.hardware_path, self._hardware_metadata())
            self.snapshot(event="run_start", phase="initialization")
            self._thread = threading.Thread(target=self._loop, name="tcc-telemetry", daemon=True)
            self._thread.start()
        return self

    def _disk_snapshot(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, requested_path in self._disk_paths.items():
            path = Path(requested_path)
            target = path if path.is_dir() else path.parent
            try:
                usage = shutil.disk_usage(target)
            except OSError:
                result[name] = {"path": str(path), "available": False}
                continue
            result[name] = {
                "path": str(path),
                "available": True,
                "total_mb": usage.total / _MEBIBYTE,
                "used_mb": usage.used / _MEBIBYTE,
                "free_mb": usage.free / _MEBIBYTE,
                "used_percent": 100.0 * usage.used / usage.total if usage.total else None,
            }
        return result

    def _gpu_snapshot(self) -> dict[str, Any]:
        if self._nvidia_smi is None:
            return {"available": False, "devices": []}
        command = [
            self._nvidia_smi,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=2.0, check=False)
        except (OSError, subprocess.SubprocessError):
            return {"available": False, "devices": []}
        if completed.returncode != 0:
            return {"available": False, "devices": []}
        devices: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) != 7:
                continue
            devices.append(
                {
                    "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                    "name": parts[1],
                    "utilization_percent": _safe_float(parts[2]),
                    "memory_used_mb": _safe_float(parts[3]),
                    "memory_total_mb": _safe_float(parts[4]),
                    "temperature_c": _safe_float(parts[5]),
                    "power_w": _safe_float(parts[6]),
                }
            )
        return {"available": bool(devices), "devices": devices}

    def _process_snapshot(self) -> dict[str, Any]:
        if self._process is None:
            return {"available": False}
        try:
            memory = self._process.memory_info()
            io = self._process.io_counters() if hasattr(self._process, "io_counters") else None
            result = {
                "available": True,
                "cpu_percent": _safe_float(self._process.cpu_percent(None)),
                "rss_mb": memory.rss / _MEBIBYTE,
                "vms_mb": memory.vms / _MEBIBYTE,
                "memory_percent": _safe_float(self._process.memory_percent()),
                "threads": int(self._process.num_threads()),
                "read_mb": getattr(io, "read_bytes", 0) / _MEBIBYTE if io is not None else None,
                "write_mb": getattr(io, "write_bytes", 0) / _MEBIBYTE if io is not None else None,
            }
            if self.include_children:
                children = []
                alive = set()
                for child in self._process.children(recursive=True):
                    try:
                        alive.add(child.pid)
                        cached = self._child_processes.get(child.pid)
                        if cached is None or cached.create_time() != child.create_time():
                            self._child_processes[child.pid] = child
                            cached = child
                        memory_info = cached.memory_info()
                        children.append({"pid": cached.pid, "name": cached.name(),
                                         "cpu_percent": _safe_float(cached.cpu_percent(None)),
                                         "rss_mb": memory_info.rss / _MEBIBYTE,
                                         "threads": int(cached.num_threads())})
                    except _PSUTIL_ERRORS:
                        continue
                self._child_processes = {pid: value for pid, value in self._child_processes.items()
                                         if pid in alive}
                result["children"] = children
                result["children_count"] = len(children)
                result["tree_rss_mb"] = result["rss_mb"] + sum(c["rss_mb"] for c in children)
                result["tree_cpu_percent"] = (result["cpu_percent"] or 0) + sum(
                    c["cpu_percent"] or 0 for c in children)
            return result
        except _PSUTIL_ERRORS:
            return {"available": False}

    def _system_snapshot(self) -> dict[str, Any]:
        if psutil is None:
            return {"available": False}
        try:
            memory = psutil.virtual_memory()
            frequency = psutil.cpu_freq()
            return {
                "available": True,
                "cpu_percent": _safe_float(psutil.cpu_percent(None)),
                "cpu_frequency_mhz": _safe_float(getattr(frequency, "current", None)),
                "memory_total_mb": memory.total / _MEBIBYTE,
                "memory_used_mb": memory.used / _MEBIBYTE,
                "memory_available_mb": memory.available / _MEBIBYTE,
                "memory_percent": _safe_float(memory.percent),
            }
        except psutil.Error:
            return {"available": False}

    def _capture(self, *, event: str, phase: str | None = None) -> dict[str, Any]:
        elapsed = time.perf_counter() - self._started_at if self._started_at is not None else None
        return {
            "timestamp": utc_now(),
            "event": str(event),
            "phase": phase,
            "elapsed_seconds": elapsed,
            "system": self._system_snapshot(),
            "process": self._process_snapshot(),
            "gpu": self._gpu_snapshot(),
            "disk": self._disk_snapshot(),
        }

    def snapshot(self, *, event: str = "sample", phase: str | None = None) -> dict[str, Any]:
        """Persist one immediate sample and return its serialisable payload."""

        sample = self._capture(event=event, phase=phase)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self._samples_lock:
            self._samples.append(sample)
            with self.samples_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(sample))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        return sample

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.snapshot(event="interval")
            except OSError:
                # Training must continue even if a removable output volume disappears.
                continue

    @staticmethod
    def _summary_values(samples: list[dict[str, Any]], *path: str) -> list[float]:
        values: list[float] = []
        for sample in samples:
            current: Any = sample
            for key in path:
                if not isinstance(current, Mapping):
                    current = None
                    break
                current = current.get(key)
            value = _safe_float(current)
            if value is not None:
                values.append(value)
        return values

    def _summary(self) -> dict[str, Any]:
        with self._samples_lock:
            samples = list(self._samples)
        summary: dict[str, Any] = {
            "sample_count": len(samples),
            "duration_seconds": (time.perf_counter() - self._started_at) if self._started_at is not None else None,
            "metrics": {},
        }
        metric_paths = {
            "system_cpu_percent": ("system", "cpu_percent"),
            "system_memory_percent": ("system", "memory_percent"),
            "process_cpu_percent": ("process", "cpu_percent"),
            "process_rss_mb": ("process", "rss_mb"),
        }
        for name, path in metric_paths.items():
            values = self._summary_values(samples, *path)
            summary["metrics"][name] = {
                "mean": _mean(values),
                "max": max(values) if values else None,
                "min": min(values) if values else None,
            }
        gpu_utilization = [
            value
            for sample in samples
            for device in sample.get("gpu", {}).get("devices", [])
            for value in [_safe_float(device.get("utilization_percent"))]
            if value is not None
        ]
        summary["metrics"]["gpu_utilization_percent"] = {
            "mean": _mean(gpu_utilization),
            "max": max(gpu_utilization) if gpu_utilization else None,
            "min": min(gpu_utilization) if gpu_utilization else None,
        }
        return summary

    def stop(self, *, final_event: str = "train_end") -> dict[str, Any]:
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
        try:
            self.snapshot(event=final_event, phase="completed")
        except OSError:
            pass
        summary = self._summary()
        atomic_write_json(self.summary_path, summary)
        return {
            "hardware": str(self.hardware_path.resolve()),
            "samples": str(self.samples_path.resolve()),
            "summary": str(self.summary_path.resolve()),
            "sample_count": summary["sample_count"],
        }


__all__ = ["TelemetrySampler"]
