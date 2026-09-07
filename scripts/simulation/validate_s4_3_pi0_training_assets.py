#!/usr/bin/env python3
"""Validate the selectively downloaded official S4.3-PI0 training assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
DATASET_REPOSITORY = "DexJoCo/DexJoCo-Datasets-LeRobot"
DATASET_REVISION = "5a57c54e55dc5858dd9fb949c5f67c0c9716e6b3"
DATASET_SUBTREE = "dexjoco_lerobot_datasets/pinch_tongs/"
DATASET_ROOT = ROOT / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot" / DATASET_SUBTREE
DATASET_TREE = (
    ROOT
    / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot"
    / ".cache/huggingface/trees"
    / f"{DATASET_REVISION}.json"
)
EXPECTED_DATASET_FILE_COUNT = 10
EXPECTED_DATASET_BYTES = 865_831_099

BASE_MODEL_REPOSITORY = "DexJoCo/DexJoCo-Pi05"
BASE_MODEL_REVISION = "8d253e04e1b82c452c5804273939ff7f002f217f"
BASE_MODEL_SUBTREE = "pi05_base/"
BASE_MODEL_ROOT = ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base"
BASE_MODEL_TREE = (
    ROOT
    / ".local/external/s4_3_pi0/models/DexJoCo-Pi05"
    / ".cache/huggingface/trees"
    / f"{BASE_MODEL_REVISION}.json"
)
EXPECTED_BASE_MODEL_FILE_COUNT = 29
EXPECTED_BASE_MODEL_BYTES = 12_441_749_581

OFFICIAL_PROMPT = "Grasp the tongs and perform three consecutive open-close motions."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - Git object identity, not security.
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, payload: Any) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    target = ARTIFACT_ROOT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def expected_files(tree_path: Path, prefix: str) -> dict[str, dict[str, Any]]:
    tree = json.loads(tree_path.read_text(encoding="utf-8"))["files"]
    return {path.removeprefix(prefix): metadata for path, metadata in tree.items() if path.startswith(prefix)}


def validate_tree(
    *, root: Path, tree_path: Path, prefix: str, expected_count: int, expected_bytes: int
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    expected = expected_files(tree_path, prefix)
    actual = {str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file()}
    files = []
    content_matches = True
    for relative, metadata in sorted(expected.items()):
        path = actual.get(relative)
        item: dict[str, Any] = {
            "path": relative,
            "expected_bytes": metadata["size"],
            "exists": path is not None,
        }
        if path is not None:
            item["bytes"] = path.stat().st_size
            item["sha256"] = sha256_file(path)
            item["size_match"] = item["bytes"] == metadata["size"]
            if "lfs_sha256" in metadata:
                item["expected_sha256"] = metadata["lfs_sha256"]
                item["content_match"] = item["sha256"] == metadata["lfs_sha256"]
            else:
                item["git_blob_sha1"] = git_blob_sha1(path)
                item["expected_git_blob_sha1"] = metadata["blob_id"]
                item["content_match"] = item["git_blob_sha1"] == metadata["blob_id"]
        else:
            item["size_match"] = False
            item["content_match"] = False
        content_matches = content_matches and bool(item["content_match"])
        files.append(item)

    gates = {
        "tree_file_count": len(expected) == expected_count,
        "tree_total_bytes": sum(item["size"] for item in expected.values()) == expected_bytes,
        "actual_file_set": set(actual) == set(expected),
        "actual_total_bytes": sum(path.stat().st_size for path in actual.values()) == expected_bytes,
        "all_content_hashes": content_matches,
    }
    return files, gates


def ffprobe(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def validate_dataset() -> dict[str, Any]:
    files, gates = validate_tree(
        root=DATASET_ROOT,
        tree_path=DATASET_TREE,
        prefix=DATASET_SUBTREE,
        expected_count=EXPECTED_DATASET_FILE_COUNT,
        expected_bytes=EXPECTED_DATASET_BYTES,
    )
    info = json.loads((DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    features = info["features"]
    video_paths = sorted(DATASET_ROOT.glob("videos/**/*.mp4"))
    gates.update(
        {
            "lerobot_version": info["codebase_version"] == "v3.0",
            "episodes": info["total_episodes"] == 100,
            "frames": info["total_frames"] == 40_065,
            "single_task": info["total_tasks"] == 1,
            "fps": info["fps"] == 30,
            "state_23d": features["observation.state"]["shape"] == [23],
            "action_22d": features["action"]["shape"] == [22],
            "front_camera": features["observation.images.front"]["shape"] == [640, 640, 3],
            "wrist_camera": features["observation.images.wrist"]["shape"] == [640, 640, 3],
            "video_count": len(video_paths) == 5,
            "videos_decodable": all(ffprobe(path) for path in video_paths),
        }
    )

    # LeRobot v3 stores task strings in the per-episode table for this dataset.
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet",
            columns=["tasks"],
        )
        task_rows = table.column("tasks").to_pylist()
        prompts = sorted({prompt for row in task_rows for prompt in row})
    except ImportError:
        prompts = []
    gates["official_prompt"] = prompts == [OFFICIAL_PROMPT]

    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-official-dataset-manifest.v1",
        "stage": "PI0-4",
        "repository": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "selective_subtree": DATASET_SUBTREE + "**",
        "dataset_path": "$REPO_ROOT/.local/external/s4_3_pi0/datasets/"
        "DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs",
        "regime": "rand_obj",
        "task": "pinch_tongs",
        "files": files,
        "metadata": {
            "lerobot_version": info["codebase_version"],
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "fps": info["fps"],
            "state_dimensions": 23,
            "action_dimensions": 22,
            "image_keys": ["observation.images.front", "observation.images.wrist"],
            "image_shape": [640, 640, 3],
            "prompts": prompts,
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json("official_dataset_manifest.json", payload)
    return payload


def validate_base_model() -> dict[str, Any]:
    files, gates = validate_tree(
        root=BASE_MODEL_ROOT,
        tree_path=BASE_MODEL_TREE,
        prefix=BASE_MODEL_SUBTREE,
        expected_count=EXPECTED_BASE_MODEL_FILE_COUNT,
        expected_bytes=EXPECTED_BASE_MODEL_BYTES,
    )
    gates.update(
        {
            "params_directory": (BASE_MODEL_ROOT / "params").is_dir(),
            "orbax_commit_marker": (BASE_MODEL_ROOT / "params/commit_success.txt").is_file(),
            "single_arm_base_only": not (BASE_MODEL_ROOT.parent / "pi05_base_action_dim_44").exists(),
        }
    )
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-official-base-model-manifest.v1",
        "stage": "PI0-4",
        "repository": BASE_MODEL_REPOSITORY,
        "revision": BASE_MODEL_REVISION,
        "selective_subtree": BASE_MODEL_SUBTREE + "**",
        "official_primary_source": "gs://openpi-assets/checkpoints/pi05_base",
        "mirror_path": "$REPO_ROOT/.local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base",
        "official_and_mirror_tree_file_count": EXPECTED_BASE_MODEL_FILE_COUNT,
        "official_and_mirror_tree_bytes": EXPECTED_BASE_MODEL_BYTES,
        "files": files,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json("official_base_model_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    dataset = validate_dataset()
    result = {"dataset": dataset["status"]}
    if not args.dataset_only:
        base_model = validate_base_model()
        result["base_model"] = base_model["status"]
    print(json.dumps(result, sort_keys=True))
    if any(status != "PASS" for status in result.values()):
        raise SystemExit("S4_3_PI0_TRAINING_ASSET_GATE_FAIL")


if __name__ == "__main__":
    main()
