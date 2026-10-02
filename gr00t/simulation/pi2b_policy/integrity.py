"""Content hashing helpers shared by Track A audit and completion gates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def tree_sha256(rows: Iterable[dict[str, object]]) -> str:
    """Historical checkpoint hash algorithm used by PI1/PI2M/PI2N."""

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"])):
        digest.update(
            f"{row['path']}\0{row['sha256']}\0{row['bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()
