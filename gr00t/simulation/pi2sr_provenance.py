"""Prospective source, environment, data, and PRNG provenance for PI2S-R."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "tactile3d-unit.pi2sr-provenance-prng.v1"
SAFE_ENVIRONMENT_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_provenance_manifest(
    *,
    repo_root: Path,
    config_paths: Sequence[Path],
    base_model_checkpoint_tree_sha256: str | None,
    teacher_or_contact_encoder_sha256: str,
    data_identity: Mapping[str, Any],
    randomness: Mapping[str, Any],
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Capture prospective identities without inventing unavailable randomness."""

    root = Path(repo_root).resolve()
    config_hashes = {
        str(Path(path).resolve().relative_to(root)): sha256_file(path) for path in config_paths
    }
    dirty_rows = _git(root, "status", "--porcelain=v1").splitlines()
    submodules = {}
    for row in _git(root, "submodule", "status", "--recursive").splitlines():
        if not row:
            continue
        fields = row.lstrip(" +-U").split()
        if len(fields) >= 2:
            submodules[fields[1]] = fields[0]
    safe_environment = {name: os.environ.get(name) for name in SAFE_ENVIRONMENT_NAMES}
    package_versions = {
        name: _package_version(name)
        for name in ("numpy", "torch", "jax", "jaxlib", "scipy", "mujoco", "pytest")
    }
    try:
        import torch

        cuda_version = torch.version.cuda
    except ImportError:  # pragma: no cover - recorded as unavailable in minimal environments
        cuda_version = None
    normalized_environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": package_versions,
        "deterministic_environment": safe_environment,
        "device": device,
        "dtype": dtype,
    }
    randomness_contract = {
        name: randomness.get(name)
        for name in (
            "root_training_seed",
            "initialization_seed_namespace",
            "data_loader_seed",
            "augmentation_seed",
            "noise_time_sampling_seed",
            "evaluator_reset_seed",
            "policy_sampling_root_key",
            "policy_request_counter",
            "policy_key_index",
        )
    }
    missing_randomness = [name for name, value in randomness_contract.items() if value is None]
    manifest = {
        "schema": SCHEMA,
        "source_identity": {
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_branch": _git(root, "branch", "--show-current"),
            "git_dirty": bool(dirty_rows),
            "dirty_path_count": len(dirty_rows),
            "submodule_commits": submodules,
            "config_sha256": config_hashes,
            "base_model_checkpoint_tree_sha256": base_model_checkpoint_tree_sha256,
            "teacher_or_contact_encoder_sha256": teacher_or_contact_encoder_sha256,
        },
        "environment": {
            "python_version": sys.version,
            "package_versions": package_versions,
            "cuda_version": cuda_version,
            "jax_version": package_versions["jax"],
            "torch_version": package_versions["torch"],
            "deterministic_environment": safe_environment,
            "visible_gpu_indices_and_uuids": [],
            "device": device,
            "dtype": dtype,
            "thread_counts": {
                "OMP_NUM_THREADS": safe_environment["OMP_NUM_THREADS"],
                "MKL_NUM_THREADS": safe_environment["MKL_NUM_THREADS"],
                "OPENBLAS_NUM_THREADS": safe_environment["OPENBLAS_NUM_THREADS"],
                "NUMEXPR_NUM_THREADS": safe_environment["NUMEXPR_NUM_THREADS"],
            },
            "normalized_environment_digest": canonical_sha256(normalized_environment),
        },
        "data": dict(data_identity),
        "randomness": randomness_contract,
        "missing_randomness": missing_randomness,
        "missing_value_policy": "null means unavailable and is never inferred",
        "historical_retroactive_prng_reconstruction_claimed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def compare_provenance(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    left_source = left["source_identity"]
    right_source = right["source_identity"]
    comparable = all(
        value is not None
        for value in (
            left_source.get("git_commit"),
            right_source.get("git_commit"),
            left_source.get("config_sha256"),
            right_source.get("config_sha256"),
        )
    )
    if not comparable:
        return "insufficient_provenance"
    same_source = all(
        left_source.get(name) == right_source.get(name)
        for name in (
            "git_commit",
            "submodule_commits",
            "config_sha256",
            "base_model_checkpoint_tree_sha256",
            "teacher_or_contact_encoder_sha256",
        )
    ) and left.get("data") == right.get("data")
    if not same_source:
        return "different_source_or_config"
    if left.get("randomness") == right.get("randomness"):
        return "same_source_same_config_same_seed"
    return "same_source_same_config_different_seed"
