#!/usr/bin/env python3
"""Resumably hash the four preregistered PI2S intermediate checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GOAL_UUID = "01a0fd61-4bdd-7850-a9c1-f6ea9b66dad8"
SOURCE_COMMIT = "6602c05d78ff97a150255cc37aae83e1f15c9df6"
PI2S_RELATIVE = Path("simulation/s4_3_pi2s")
CHECKPOINTS = (
    (
        "B_HVA_seed42_step10000",
        Path("simulation/s4_3_pi2m/bhva/pinch_tongs/s43_pi2m_bhva_seed42/10000"),
    ),
    (
        "B_HVA_seed42_step20000",
        Path("simulation/s4_3_pi2m/bhva/pinch_tongs/s43_pi2m_bhva_seed42/20000"),
    ),
    (
        "B_HVA_seed43_step10000",
        Path(
            "simulation/s4_3_pi2b_policy/experiments/runs/B_HVA/seed43/pinch_tongs/"
            "s43_pi2b_b_hva_seed43/10000"
        ),
    ),
    (
        "B_HVA_seed43_step20000",
        Path(
            "simulation/s4_3_pi2b_policy/experiments/runs/B_HVA/seed43/pinch_tongs/"
            "s43_pi2b_b_hva_seed43/20000"
        ),
    ),
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def canonical_tree_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        digest.update(f"{row['path']}\0{row['sha256']}\0{row['bytes']}\n".encode())
    return digest.hexdigest()


def load_partial(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid partial manifest {path}:{line_number}") from exc
            rows[row["path"]] = row
    return rows


def append_partial(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def unchanged(row: dict[str, Any], stat: os.stat_result) -> bool:
    return (
        row.get("bytes") == stat.st_size
        and row.get("mtime_ns") == stat.st_mtime_ns
        and row.get("inode") == stat.st_ino
        and row.get("device") == stat.st_dev
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=(ROOT / ".local/experiments").resolve(),
    )
    args = parser.parse_args()
    experiments = args.experiment_root.resolve(strict=True)
    pi2s_root = experiments / PI2S_RELATIVE
    artifact = pi2s_root / "artifacts/intermediate_checkpoint_hashes.json"
    progress_path = pi2s_root / "logs/intermediate_checkpoint_hash_progress.json"
    cache_root = pi2s_root / "cache/intermediate_checkpoint_hashes"
    script_sha = sha256_file(Path(__file__))
    protocol_path = ROOT / "configs/simulation/pi2s/diagnostic_protocol.json"
    protocol_sha = sha256_file(protocol_path)
    started_wall = time.time()
    started_at = now_utc()

    inventory: dict[str, list[Path]] = {}
    total_bytes = 0
    total_files = 0
    for label, relative in CHECKPOINTS:
        checkpoint = experiments / relative
        if not checkpoint.is_dir():
            raise RuntimeError(f"missing intermediate checkpoint {checkpoint}")
        if not (checkpoint / "params/manifest.ocdbt").is_file():
            raise RuntimeError(f"missing params manifest {checkpoint}")
        if not (checkpoint / "train_state/manifest.ocdbt").is_file():
            raise RuntimeError(f"missing train_state manifest {checkpoint}")
        files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
        inventory[label] = files
        total_files += len(files)
        total_bytes += sum(path.stat().st_size for path in files)

    completed_bytes = 0
    completed_files = 0
    results = []
    for label, relative in CHECKPOINTS:
        checkpoint = experiments / relative
        partial_path = cache_root / f"{label}.jsonl"
        previous = load_partial(partial_path)
        rows = []
        for path in inventory[label]:
            relative_path = path.relative_to(checkpoint).as_posix()
            before = path.stat()
            prior = previous.get(relative_path)
            if prior is not None and unchanged(prior, before):
                row = prior
                reused = True
            else:
                digest = sha256_file(path)
                after = path.stat()
                if (
                    before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_ino != after.st_ino
                    or before.st_dev != after.st_dev
                ):
                    raise RuntimeError(f"checkpoint file changed while hashing: {path}")
                row = {
                    "path": relative_path,
                    "bytes": after.st_size,
                    "sha256": digest,
                    "mtime_ns": after.st_mtime_ns,
                    "inode": after.st_ino,
                    "device": after.st_dev,
                }
                append_partial(partial_path, row)
                reused = False
            rows.append(row)
            completed_files += 1
            completed_bytes += int(row["bytes"])
            elapsed = max(time.time() - started_wall, 1e-9)
            rate = completed_bytes / elapsed
            remaining = max(0, total_bytes - completed_bytes)
            progress = {
                "schema": "tactile3d-unit.s4-3-pi2s-intermediate-hash-progress.v1",
                "status": "RUNNING",
                "goal_uuid": GOAL_UUID,
                "pid": os.getpid(),
                "source_commit": SOURCE_COMMIT,
                "script_sha256": script_sha,
                "protocol_sha256": protocol_sha,
                "started_at_utc": started_at,
                "updated_at_utc": now_utc(),
                "current_checkpoint": label,
                "current_file": relative_path,
                "current_file_reused": reused,
                "completed_files": completed_files,
                "total_files": total_files,
                "completed_bytes": completed_bytes,
                "total_bytes": total_bytes,
                "rate_bytes_per_second": rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "output": "$PI2S_ROOT/artifacts/intermediate_checkpoint_hashes.json",
            }
            atomic_json(progress_path, progress)
            print(
                f"HASH_PROGRESS files={completed_files}/{total_files} "
                f"bytes={completed_bytes}/{total_bytes} checkpoint={label} file={relative_path}",
                flush=True,
            )
        results.append(
            {
                "label": label,
                "path": "$EXPERIMENT_ROOT/" + relative.as_posix(),
                "directory_step": int(relative.name),
                "files": len(rows),
                "bytes": sum(int(row["bytes"]) for row in rows),
                "tree_sha256": canonical_tree_sha(rows),
                "manifest": [
                    {key: row[key] for key in ("path", "bytes", "sha256")}
                    for row in rows
                ],
            }
        )

    payload = {
        "schema": "tactile3d-unit.s4-3-pi2s-intermediate-checkpoint-hashes.v1",
        "status": "PASS",
        "goal_uuid": GOAL_UUID,
        "source_commit": SOURCE_COMMIT,
        "script_sha256": script_sha,
        "protocol_draft_sha256": protocol_sha,
        "hash_algorithm": "PI1/PI2M/PI2N path\\0sha256\\0bytes newline tree SHA-256",
        "created_at_utc": now_utc(),
        "started_at_utc": started_at,
        "elapsed_seconds": time.time() - started_wall,
        "checkpoints": results,
        "gates": {
            "four_checkpoints": len(results) == 4,
            "all_files_stable_while_hashed": True,
            "params_and_train_state_manifests_present": True,
            "read_only": True,
        },
    }
    atomic_json(artifact, payload)
    atomic_json(
        progress_path,
        {
            **progress,
            "status": "COMPLETE",
            "updated_at_utc": now_utc(),
            "completed_files": total_files,
            "completed_bytes": total_bytes,
            "eta_seconds": 0.0,
            "result_sha256": sha256_file(artifact),
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "checkpoints": len(results),
                "bytes": total_bytes,
                "artifact": str(artifact),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
