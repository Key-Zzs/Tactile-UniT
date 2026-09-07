#!/usr/bin/env python3
"""Validate and manifest the official DexJoCo pinch_tongs pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05" / "pi05_dexjoco_ckpt/pinch_tongs"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_checkpoint_manifest.json"

OFFICIAL_REPOSITORY = "DexJoCo/DexJoCo-Pi05"
OFFICIAL_REVISION = "8d253e04e1b82c452c5804273939ff7f002f217f"
OFFICIAL_SUBTREE = "pi05_dexjoco_ckpt/pinch_tongs/**"
EXPECTED_FILE_COUNT = 57
EXPECTED_BYTES = 9_561_967_083


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(checkpoint: Path) -> dict[str, Any]:
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    relative_files = [path.relative_to(checkpoint).as_posix() for path in files]
    sizes = {relative: path.stat().st_size for relative, path in zip(relative_files, files, strict=True)}
    hashes = {relative: sha256_file(path) for relative, path in zip(relative_files, files, strict=True)}

    norm_path = checkpoint / "assets/local_repo/norm_stats.json"
    norm_stats = json.loads(norm_path.read_text(encoding="utf-8"))["norm_stats"]
    norm_dimensions = {
        key: {stat: len(values) for stat, values in payload.items()} for key, payload in norm_stats.items()
    }
    gates = {
        "exact_file_count": len(files) == EXPECTED_FILE_COUNT,
        "exact_total_bytes": sum(sizes.values()) == EXPECTED_BYTES,
        "checkpoint_metadata": (checkpoint / "_CHECKPOINT_METADATA").is_file(),
        "params_metadata": (checkpoint / "params/_METADATA").is_file(),
        "params_manifest": (checkpoint / "params/manifest.ocdbt").is_file(),
        "norm_stats": norm_path.is_file(),
        "state_norm_23d": set(norm_dimensions["state"].values()) == {23},
        "action_norm_22d": set(norm_dimensions["actions"].values()) == {22},
    }
    return {
        "schema": "tactile3d-unit.s4-3-pi0-official-checkpoint-manifest.v1",
        "stage": "PI0-2",
        "repository": OFFICIAL_REPOSITORY,
        "revision": OFFICIAL_REVISION,
        "subtree": OFFICIAL_SUBTREE,
        "checkpoint_path": "$REPO_ROOT/.local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_dexjoco_ckpt/pinch_tongs",
        "file_count": len(files),
        "total_bytes": sum(sizes.values()),
        "hash_algorithm": "sha256",
        "files": {relative: {"bytes": sizes[relative], "sha256": hashes[relative]} for relative in relative_files},
        "norm_dimensions": norm_dimensions,
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = build_manifest(args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "file_count": manifest["file_count"],
                "status": manifest["status"],
                "total_bytes": manifest["total_bytes"],
            },
            sort_keys=True,
        )
    )
    if manifest["status"] != "PASS":
        raise SystemExit("S4_3_PI0_OFFICIAL_CHECKPOINT_MANIFEST_FAIL")


if __name__ == "__main__":
    main()
