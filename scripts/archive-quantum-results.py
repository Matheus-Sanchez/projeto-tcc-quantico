"""Archive completed local runs and verify every archived file against its source."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs" / "quantum-parallel-128-20"
DESTINATION = PROJECT / "results" / "quantum-parallel-128-20"
DATASETS = ("svhn", "gtsrb", "fer2013")
BACKENDS = ("pennylane", "qiskit")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def snapshot(directory: Path) -> dict:
    files = {}
    for parent, directories, names in os.walk(directory):
        for name in directories + names:
            if (Path(parent) / name).is_symlink():
                raise RuntimeError(f"Unexpected symlink: {Path(parent) / name}")
        for name in names:
            path = Path(parent) / name
            stat = path.stat()
            files[path.relative_to(SOURCE).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def archive_run(relative: Path, name: str) -> dict:
    source = SOURCE / relative
    expected = snapshot(source)
    if not expected:
        raise RuntimeError(f"No source files in {source}")
    archive = DESTINATION / "archives" / f"{name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".partial")
    print(f"Archiving {relative}: {len(expected):,} files, "
          f"{sum(size for size, _ in expected.values()):,} bytes", flush=True)
    if not archive.exists():
        subprocess.run(["tar", "-czf", str(temporary), "-C", str(SOURCE),
                        relative.as_posix()], check=True)
        candidate = temporary
    else:
        print("Reusing existing archive and verifying its contents.", flush=True)
        candidate = archive
    if candidate.stat().st_size >= 2 * 1024**3:
        raise RuntimeError(f"Archive exceeds the 2 GiB LFS limit: {candidate}")
    inventory = DESTINATION / "inventories" / f"{name}.jsonl.gz"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory_pending = inventory.with_name(inventory.name + ".partial")
    seen = set()
    with tarfile.open(candidate, "r|gz") as packed, gzip.open(
        inventory_pending, "wt", encoding="utf-8", newline="\n"
    ) as records:
        for member in packed:
            if member.isdir():
                continue
            key = PurePosixPath(member.name).as_posix()
            if not member.isfile() or key not in expected or key in seen:
                raise RuntimeError(f"Unexpected archive entry: {key}")
            size, modified = expected[key]
            path = SOURCE / key
            if member.size != size:
                raise RuntimeError(f"Size mismatch: {key}")
            with packed.extractfile(member) as stream:
                archived_hash = sha256_stream(stream)
            if sha256_file(path) != archived_hash:
                raise RuntimeError(f"Content mismatch: {key}")
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (size, modified):
                raise RuntimeError(f"Source changed while archiving: {key}")
            records.write(json.dumps({"path": key, "bytes": size,
                                      "sha256": archived_hash}) + "\n")
            seen.add(key)
            if len(seen) % 10000 == 0:
                print(f"Verified {len(seen):,}/{len(expected):,} files in {name}", flush=True)
    if seen != set(expected) or snapshot(source) != expected:
        raise RuntimeError(f"Incomplete archive or changed source tree: {relative}")
    if candidate != archive:
        candidate.replace(archive)
    inventory_pending.replace(inventory)
    record = {
        "archive": archive.relative_to(DESTINATION).as_posix(),
        "source": relative.as_posix(),
        "source_files": len(expected),
        "source_bytes": sum(size for size, _ in expected.values()),
        "archive_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "inventory": inventory.relative_to(DESTINATION).as_posix(),
        "inventory_sha256": sha256_file(inventory),
        "verification": "Every archived file matched its source SHA-256 and size.",
    }
    print(f"Verified archive {archive.name}: {record['archive_bytes']:,} bytes", flush=True)
    return record


def completed_run(dataset: str, backend: str) -> dict:
    run = SOURCE / "runs" / dataset / backend / "seed-42"
    result = read_json(run / "result.json")
    status = read_json(run / "status.json")
    state = read_json(run / "checkpoints" / "state.json")
    config = read_json(run / "config.json")
    with (run / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    if not (result["status"] == status["status"] == "completed"
            and result["mode"] == status["mode"] == "full"
            and state["epoch"] == config["effective_epochs"] == len(history) == 100
            and state["best_epoch"] == result["selected_epoch"]):
        raise RuntimeError(f"Run is incomplete: {dataset}/{backend}")
    for checkpoint_key, hash_key in (("checkpoint", "sha256"),
                                      ("best_checkpoint", "best_sha256")):
        if sha256_file(run / "checkpoints" / state[checkpoint_key]) != state[hash_key]:
            raise RuntimeError(f"Checkpoint hash mismatch: {dataset}/{backend}")
    summary = DESTINATION / "summaries" / dataset / backend
    summary.mkdir(parents=True, exist_ok=True)
    for path in run.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".txt"}:
            shutil.copyfile(path, summary / path.name)
    shutil.copyfile(run / "checkpoints" / "state.json", summary / "checkpoint-state.json")
    return {
        "dataset": dataset, "backend": backend, "seed": result["seed"],
        "status": result["status"], "epochs": state["epoch"],
        "selected_epoch": result["selected_epoch"],
        "selected_val_macro_f1": result["selected_val_macro_f1"],
        "test_accuracy": result["test_accuracy"],
        "test_macro_f1": result["test_macro_f1"],
        "elapsed_seconds": result["elapsed_this_session_seconds"],
        "completed_at": status["completed_at"],
        "trainable_parameters": result["trainable_parameters"],
        "noise_study_conditions": result["noise_study_conditions"],
    }


def main() -> None:
    if (SOURCE / "runs" / ".matrix.lock").exists():
        raise RuntimeError("A matrix lock exists; do not snapshot an active training run.")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_commit": source_commit, "source_root": SOURCE.relative_to(PROJECT).as_posix(),
                "full_runs": [], "archives": [], "original_matrix_files": []}
    for dataset in DATASETS:
        for backend in BACKENDS:
            row = completed_run(dataset, backend)
            record = archive_run(Path("runs") / dataset / backend / "seed-42",
                                 f"{dataset}-{backend}-seed-42")
            manifest["full_runs"].append(row)
            manifest["archives"].append(record)
            (DESTINATION / "manifest.partial.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    smoke = Path("smoke") / "mnist" / "pennylane" / "seed-42"
    if (SOURCE / smoke).is_dir():
        manifest["archives"].append(archive_run(smoke, "smoke-mnist-pennylane-seed-42"))
    for mode in ("runs", "smoke"):
        directory = SOURCE / mode
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.name.endswith(".lock"):
                raise RuntimeError(f"Unexpected active lock: {path}")
            target = DESTINATION / "original-matrices" / mode / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest["original_matrix_files"].append({
                "source": path.relative_to(SOURCE).as_posix(),
                "path": target.relative_to(DESTINATION).as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    manifest["total_source_files"] = (sum(x["source_files"] for x in manifest["archives"])
                                     + len(manifest["original_matrix_files"]))
    manifest["total_source_bytes"] = (sum(x["source_bytes"] for x in manifest["archives"])
                                     + sum(x["bytes"] for x in manifest["original_matrix_files"]))
    manifest["total_archive_bytes"] = sum(x["archive_bytes"] for x in manifest["archives"])
    (DESTINATION / "manifest.partial.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (DESTINATION / "manifest.partial.json").replace(DESTINATION / "manifest.json")
    with (DESTINATION / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest["full_runs"][0]))
        writer.writeheader()
        writer.writerows(manifest["full_runs"])
    print(json.dumps({key: value for key, value in manifest.items()
                      if key.startswith("total_")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
