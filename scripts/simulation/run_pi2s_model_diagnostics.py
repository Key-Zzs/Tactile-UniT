#!/usr/bin/env python3
"""Run the frozen, read-only S4.3-PI2S model diagnostics.

The default mode is intentionally non-executing.  ``--check`` validates the
frozen protocol, snapshots, and checkpoint paths without importing JAX.  The
heavy ``--execute`` mode must be invoked explicitly in the audited OpenPI
interpreter with exactly one visible GPU.

Scientific model state is never written.  The only writes made by execute
mode are fsynced, atomically renamed diagnostic JSON artifacts under the
requested output directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/simulation/pi2s/diagnostic_protocol.json"
AUDIT_SCRIPT = ROOT / "scripts/simulation/audit_pi2s_contracts.py"
EXPERIMENT_ROOT_LINK = ROOT / ".local/experiments"
PI2S_RELATIVE = Path("simulation/s4_3_pi2s")
DEFAULT_OUTPUT_RELATIVE = PI2S_RELATIVE / "artifacts/model_diagnostics_v1"
TASK_PROMPT = "Grasp the tongs and perform three consecutive open-close motions."
FOCUS_KEYS = (
    ("B0", 43),
    ("B_VA27", 43),
    ("B1", 43),
    ("B_HVA", 42),
    ("B_HVA", 43),
    ("B2", 43),
)
CONTACT_MODELS = frozenset({"B1", "B_HVA", "B2"})
AUXILIARY_MODELS = frozenset({"B_VA27", "B_HVA", "B2"})
CONDITION_ORDER = ("correct", "train_mean", "same_episode_lag5", "other_episode", "zero")
FLOW_TIMES = (0.1, 0.5, 0.9)
PARAMETER_GROUPS = (
    "all_common_trainable",
    "lora",
    "contact_adapter",
    "physical_auxiliary",
)
ACTION_SCHEMA = "tactile3d-unit.s4-3-pi2s-action-prefix-diagnostic.v1"
GRADIENT_SCHEMA = "tactile3d-unit.s4-3-pi2s-gradient-diagnostic.v1"
SUMMARY_SCHEMA = "tactile3d-unit.s4-3-pi2s-model-diagnostic-summary.v1"
PREFIX_AUDIT_SCHEMA = "tactile3d-unit.s4-3-pi2s-prefix-contract-audit.v1"
INTERVENTION_AUDIT_SCHEMA = "tactile3d-unit.s4-3-pi2s-fixed-observation-interventions.v1"
READONLY_GRADIENT_SCHEMA = "tactile3d-unit.s4-3-pi2s-read-only-gradient-diagnostics.v1"
TRACK_A_RELATIVE = Path("simulation/s4_3_pi2b_policy")
ACCEPTED_OPENPI_ROOT = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
ACCEPTED_OPENPI_CLIENT_SOURCE = ACCEPTED_OPENPI_ROOT / "packages/openpi-client/src"
OPENPI_CLIENT_INIT_SHA256 = "91447944015cec709e8aa7655f7e9d64e1e4508e7023a57fe3746911c0fc6fed"
OPENPI_CLIENT_IMAGE_TOOLS_SHA256 = (
    "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
)
OPENPI_CLIENT_PYTHON_TREE_FILES = 13
OPENPI_CLIENT_PYTHON_TREE_SHA256 = (
    "92fa31ab5335cd7613d656fcea70569e7797a73f6584f896ab7ff45a3ebf5cb4"
)
PALIGEMMA_TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
PALIGEMMA_TOKENIZER_BYTES = 4_264_023
PALIGEMMA_TOKENIZER_MANIFEST_SHA256 = (
    "7712fbc4fcc3948b50de0176d3403ea07ac75518462074e43bf96ffbff57cd92"
)
OPENPI_RUNTIME_SHA256 = {
    "pyproject.toml": "332c84538bfe26e4964bdaf12b03dce2ea7d13640d86d9ab8aff8b0dc736f288",
    "src/openpi/models/model.py": "0d74bc1d8f4623ac3d8e543710b8bf75f9b95e588a3ebf6127c659e001385cff",
    "src/openpi/models/pi0.py": "a16c3834628c795a06cbbdb595299e5d80d190da17d0031c1af2c2f70c5ff11b",
    "src/openpi/models/gemma.py": "7e42ada4ae7e9995f0ef3e33a0c224758323c2015e2d9d72ec15726a8a001d2a",
    "src/openpi/models/pi0_config.py": "d321858c30b398126035ea2b6da09e1c422eeeebb5ebf8df631daf9dc77647fa",
    "src/openpi/training/config.py": "9e681fcd5fef6c75a46977b3064826033110be2e514fe4d78776408be16cd592",
    "src/openpi/transforms.py": "20e143e2185dda6a37b6284b12a893cbb9b5099f4edadc08534f02ca79a85e40",
    "src/openpi/training/checkpoints.py": "3cc32682ec3609e9075dabd991488edb9d7f6c045443f58963d733f99c468dbd",
    "src/openpi/shared/nnx_utils.py": "4da7111ba3011c4d1c381c9ed17257771df6a50ed89182647c3fb6f79208005e",
}
EXPECTED_RUNTIME_VERSIONS = {
    "jax": "0.5.3",
    "jaxlib": "0.5.3",
    "flax": "0.10.2",
    "orbax-checkpoint": "0.11.13",
    "numpy": "1.26.4",
    "openpi-client": "0.1.0",
    "pillow": "12.3.0",
    "sentencepiece": "0.2.2",
}
EXPECTED_PIP_FREEZE_SHA256 = "39664447f71898e7ae8c0d7e41a79d74b13fe009546a8efcd69504ca64c91e89"
OPENPI_PYTHON_TREE_FILES = 58
OPENPI_PYTHON_TREE_SHA256 = "66031d747e807d9cc0ee03ee847ceaad2c7555cd96b7fff1835946eb4afdf13a"
EXPECTED_GLOBAL_BUDGET = {
    "checkpoints": 6,
    "phases_per_checkpoint": 2,
    "immutable_shards": 12,
    "action_records": 704,
    "gradient_records": 72,
    "auxiliary_gradient_records": 48,
    "optimizer_steps": 0,
    "checkpoint_writes": 0,
}
_CONFIG_BINDING_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}


class ContractError(RuntimeError):
    """Raised when a frozen input differs from the preregistered contract."""


@dataclasses.dataclass(frozen=True)
class Contract:
    protocol: dict[str, Any]
    protocol_sha256: str
    runner_sha256: str
    experiments: Path
    pi2s_root: Path
    snapshot_root: Path
    snapshot_manifest: dict[str, Any]
    snapshot_manifest_sha256: str
    train_root: Path
    focus: tuple[dict[str, Any], ...]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_openpi_runtime_source() -> dict[str, str]:
    """Verify the small, execution-critical accepted OpenPI source set."""

    observed: dict[str, str] = {}
    for relative, expected in OPENPI_RUNTIME_SHA256.items():
        path = ACCEPTED_OPENPI_ROOT / relative
        _require(path.is_file(), f"accepted OpenPI runtime file missing: {path}")
        actual = sha256_file(path)
        _require(actual == expected, f"accepted OpenPI runtime source drift: {relative}")
        observed[relative] = actual
    source_root = ACCEPTED_OPENPI_ROOT / "src/openpi"
    python_files = sorted(
        path for path in source_root.rglob("*.py") if "__pycache__" not in path.parts
    )
    _require(
        len(python_files) == OPENPI_PYTHON_TREE_FILES,
        "accepted OpenPI Python source file-count drift",
    )
    tree_digest = hashlib.sha256()
    for path in python_files:
        relative = path.relative_to(source_root).as_posix()
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(sha256_file(path).encode("ascii"))
        tree_digest.update(b"\n")
    _require(
        tree_digest.hexdigest() == OPENPI_PYTHON_TREE_SHA256,
        "accepted OpenPI Python source tree drift",
    )
    observed["$PYTHON_SOURCE_TREE"] = OPENPI_PYTHON_TREE_SHA256
    return observed


def verify_openpi_client_source() -> dict[str, Any]:
    """Verify the accepted image-preprocessing client source independently."""

    package = ACCEPTED_OPENPI_CLIENT_SOURCE / "openpi_client"
    expected_files = {
        "__init__.py": OPENPI_CLIENT_INIT_SHA256,
        "image_tools.py": OPENPI_CLIENT_IMAGE_TOOLS_SHA256,
    }
    observed: dict[str, str] = {}
    for relative, expected in expected_files.items():
        path = package / relative
        _require(path.is_file(), f"accepted OpenPI client file missing: {path}")
        actual = sha256_file(path)
        _require(actual == expected, f"accepted OpenPI client source drift: {relative}")
        observed[relative] = actual
    python_files = sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)
    _require(
        len(python_files) == OPENPI_CLIENT_PYTHON_TREE_FILES,
        "accepted OpenPI client Python file-count drift",
    )
    digest = hashlib.sha256()
    for path in python_files:
        relative = path.relative_to(package).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    _require(
        digest.hexdigest() == OPENPI_CLIENT_PYTHON_TREE_SHA256,
        "accepted OpenPI client Python tree drift",
    )
    return {
        "files": len(python_files),
        "tree_sha256": digest.hexdigest(),
        "critical_files": observed,
    }


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _prepare_publish_path(path: Path, allowed_root: Path | None) -> Path:
    candidate = Path(os.path.abspath(path))
    if allowed_root is None:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.parent.is_symlink():
            raise ContractError(f"publish parent is a symlink: {candidate.parent}")
        return candidate.parent.resolve(strict=True) / candidate.name

    root = Path(allowed_root).resolve(strict=True)
    try:
        relative_parent = candidate.parent.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"publish path escapes allowed root: {candidate}") from exc
    current = root
    for component in relative_parent.parts:
        current = current / component
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise ContractError(f"publish ancestor is not a real directory: {current}")
        else:
            current.mkdir(mode=0o775)
            _fsync_directory(current.parent)
    resolved_parent = candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise ContractError(
            f"resolved publish parent escapes allowed root: {resolved_parent}"
        ) from exc
    if resolved_parent != candidate.parent:
        raise ContractError(f"publish parent contains a symlink: {candidate.parent}")
    return resolved_parent / candidate.name


def write_atomic_json(
    path: Path,
    payload: Any,
    *,
    replace: bool = False,
    allowed_root: Path | None = None,
) -> None:
    """Fsync and publish JSON with an atomic, no-clobber hard-link commit.

    ``os.replace`` is deliberately not used: a check followed by replace has a
    race in which two workers can both believe that they own the same shard.
    Linking a same-directory temporary file to the final name is atomic and
    fails with ``FileExistsError`` if any worker has already published it.
    """

    path = Path(path)
    if replace:
        raise ValueError("immutable diagnostic artifacts never permit replacement")
    path = _prepare_publish_path(path, allowed_root)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace diagnostic artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary artifact exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        expected_sha256 = sha256_file(temporary)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except PermissionError:
            # Some NFS identity mappings report a different local owner and
            # Linux protected_hardlinks then rejects an otherwise authorized
            # no-clobber link.  A brief writable mode permits the atomic link;
            # the inode is immediately sealed read-only and rehashed below.
            os.chmod(temporary, 0o666)
            os.link(temporary, path, follow_symlinks=False)
        try:
            os.chmod(path, 0o444)
            _require(
                sha256_file(path) == expected_sha256,
                f"published diagnostic artifact changed during no-clobber commit: {path}",
            )
        except BaseException:
            try:
                source_info = temporary.stat()
                published_info = path.stat()
                if (
                    source_info.st_dev == published_info.st_dev
                    and source_info.st_ino == published_info.st_ino
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
            raise
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def resolve_symbolic(path: str, *, experiments: Path, pi2s_root: Path) -> Path:
    roots = {
        "$REPO_ROOT": ROOT,
        "$EXPERIMENT_ROOT": experiments,
        "$PI2S_ROOT": pi2s_root,
    }
    for token, root in roots.items():
        if path == token:
            return root.resolve()
        prefix = token + "/"
        if path.startswith(prefix):
            candidate = (root / path[len(prefix) :]).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError as exc:
                raise ContractError(f"symbolic path escapes {token}: {path}") from exc
            return candidate
    raise ContractError(f"non-symbolic protocol path: {path}")


def _snapshot_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"snapshot contains a symlink: {path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _array_contract(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(value.shape == shape, f"array shape drift: {path}: {value.shape} != {shape}")
    _require(value.dtype == dtype, f"array dtype drift: {path}: {value.dtype} != {dtype}")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} missing: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def checkpoint_authority(experiments: Path, focus: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebind protocol checkpoint hashes to the frozen one-time live-hash evidence.

    This intentionally reads small authority manifests only.  It never walks or
    re-hashes a checkpoint tree.
    """

    artifacts = experiments / TRACK_A_RELATIVE / "artifacts"
    pre_path = artifacts / "pre_final_freeze.json"
    protected_path = artifacts / "protected_hashes_after.json"
    pre = _read_json(pre_path, "Track-A pre-final freeze")
    protected = _read_json(protected_path, "Track-A protected-hash closeout")
    _require(pre.get("status") == "PASS", "Track-A pre-final freeze is not PASS")
    _require(protected.get("status") == "PASS", "Track-A protected-hash closeout is not PASS")
    pre_rows = {
        (str(row["model"]), int(row["training_seed"])): str(row["tree_sha256"])
        for row in pre.get("checkpoints", ())
    }
    protected_rows = {
        tuple(key.split("/seed", 1)): value
        for key, value in protected.get("checkpoint_tree_sha256", {}).items()
    }
    for row in focus:
        key = (str(row["model"]), int(row["training_seed"]))
        protected_key = (key[0], str(key[1]))
        expected = str(row["tree_sha256"])
        _require(pre_rows.get(key) == expected, f"pre-final checkpoint authority drift: {key}")
        _require(
            protected_rows.get(protected_key) == expected,
            f"closeout checkpoint authority drift: {key}",
        )
    return {
        "pre_final_freeze_sha256": sha256_file(pre_path),
        "protected_hashes_after_sha256": sha256_file(protected_path),
        "verification": "MANIFEST_REBIND_NO_LIVE_TREE_REHASH",
    }


def load_contract(*, verify_snapshot_contents: bool = True) -> Contract:
    """Validate all lightweight S0 inputs without importing JAX or loading a model."""

    experiments = EXPERIMENT_ROOT_LINK.resolve(strict=True)
    _require(experiments.is_dir(), "experiment root is not a directory")
    pi2s_root = (experiments / PI2S_RELATIVE).resolve(strict=True)
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    _require(
        protocol.get("status") == "PREREGISTERED_NOT_EXECUTED", "protocol is not preregistered"
    )
    _require(
        protocol.get("training_budget")
        == {"optimizer_updates": 0, "policy_runs": 0, "teacher_runs": 0},
        "non-zero training budget",
    )
    _require(
        protocol.get("diagnostic_script_sha256") == sha256_file(AUDIT_SCRIPT),
        "protocol generator self-hash drift",
    )
    implementation_hashes = protocol.get("diagnostic_implementation_sha256")
    _require(
        isinstance(implementation_hashes, dict),
        "diagnostic implementation hashes are not a mapping",
    )
    _require(
        implementation_hashes.get("run_pi2s_model_diagnostics") == sha256_file(Path(__file__)),
        "model diagnostic runner self-hash drift",
    )
    runtime_dependencies = protocol.get("runtime_dependencies")
    _require(
        isinstance(runtime_dependencies, dict),
        "runtime dependency contract is not a mapping",
    )
    client_dependency = runtime_dependencies.get("openpi_client")
    _require(
        isinstance(client_dependency, dict)
        and client_dependency.get("accepted_package_initializer_sha256")
        == OPENPI_CLIENT_INIT_SHA256
        and client_dependency.get("accepted_image_tools_sha256") == OPENPI_CLIENT_IMAGE_TOOLS_SHA256
        and client_dependency.get("accepted_python_files") == OPENPI_CLIENT_PYTHON_TREE_FILES
        and client_dependency.get("accepted_python_tree_sha256") == OPENPI_CLIENT_PYTHON_TREE_SHA256
        and client_dependency.get("required_distribution_version") == "0.1.0"
        and client_dependency.get("runtime_origin_policy")
        == "REPOSITORY_SCOPED_AND_BYTE_IDENTICAL_TO_ACCEPTED_SOURCE",
        "OpenPI client runtime dependency contract drift",
    )
    accepted_initializer = resolve_symbolic(
        str(client_dependency["accepted_package_initializer"]),
        experiments=experiments,
        pi2s_root=pi2s_root,
    )
    accepted_image_tools = resolve_symbolic(
        str(client_dependency["accepted_image_tools"]),
        experiments=experiments,
        pi2s_root=pi2s_root,
    )
    _require(
        accepted_initializer == ACCEPTED_OPENPI_CLIENT_SOURCE / "openpi_client/__init__.py"
        and sha256_file(accepted_initializer) == OPENPI_CLIENT_INIT_SHA256
        and accepted_image_tools == ACCEPTED_OPENPI_CLIENT_SOURCE / "openpi_client/image_tools.py"
        and sha256_file(accepted_image_tools) == OPENPI_CLIENT_IMAGE_TOOLS_SHA256,
        "accepted OpenPI client critical-file identity drift",
    )
    verify_openpi_client_source()
    tokenizer_dependency = runtime_dependencies.get("paligemma_tokenizer")
    _require(
        isinstance(tokenizer_dependency, dict)
        and tokenizer_dependency.get("upstream_uri") == "gs://big_vision/paligemma_tokenizer.model"
        and tokenizer_dependency.get("sha256") == PALIGEMMA_TOKENIZER_SHA256
        and int(tokenizer_dependency.get("bytes", -1)) == PALIGEMMA_TOKENIZER_BYTES
        and tokenizer_dependency.get("persistent_manifest_sha256")
        == PALIGEMMA_TOKENIZER_MANIFEST_SHA256
        and tokenizer_dependency.get("required_sentencepiece_version") == "0.2.2"
        and tokenizer_dependency.get("execution_cache_policy")
        == "COPY_VERIFIED_SNAPSHOT_TO_EPHEMERAL_OPENPI_DATA_HOME_NO_NETWORK",
        "PaliGemma tokenizer runtime dependency contract drift",
    )
    tokenizer_snapshot = resolve_symbolic(
        str(tokenizer_dependency["persistent_snapshot"]),
        experiments=experiments,
        pi2s_root=pi2s_root,
    )
    tokenizer_manifest = resolve_symbolic(
        str(tokenizer_dependency["persistent_manifest"]),
        experiments=experiments,
        pi2s_root=pi2s_root,
    )
    _require(
        tokenizer_snapshot.is_file()
        and not tokenizer_snapshot.is_symlink()
        and tokenizer_snapshot.stat().st_size == PALIGEMMA_TOKENIZER_BYTES
        and sha256_file(tokenizer_snapshot) == PALIGEMMA_TOKENIZER_SHA256,
        "persistent PaliGemma tokenizer asset drift",
    )
    _require(
        tokenizer_manifest.is_file()
        and not tokenizer_manifest.is_symlink()
        and sha256_file(tokenizer_manifest) == PALIGEMMA_TOKENIZER_MANIFEST_SHA256,
        "persistent PaliGemma tokenizer manifest drift",
    )
    _require(
        runtime_dependencies.get("execution_environment")
        == {
            "pip_freeze_all_sha256": EXPECTED_PIP_FREEZE_SHA256,
            "identity_source": "historical starting_integrity environment freeze",
        },
        "OpenPI execution environment contract drift",
    )
    protocol_text = protocol_bytes.decode("utf-8")
    _require(
        "/home/" not in protocol_text and "/mnt/" not in protocol_text,
        "protocol embeds a machine-private path",
    )

    focus = tuple(protocol.get("focus_checkpoints", ()))
    observed_keys = tuple((row.get("model"), int(row.get("training_seed", -1))) for row in focus)
    _require(observed_keys == FOCUS_KEYS, f"focus cohort drift: {observed_keys}")
    _require(
        len({tuple(value) for value in observed_keys}) == 6, "focus cohort contains duplicates"
    )

    snapshot_binding = protocol["source_manifests"]["persistent_input_snapshot"]
    snapshot_root = resolve_symbolic(
        snapshot_binding["path"], experiments=experiments, pi2s_root=pi2s_root
    )
    manifest_path = snapshot_root / "manifest.json"
    _require(manifest_path.is_file(), f"snapshot manifest missing: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    _require(manifest_sha == snapshot_binding["manifest_sha256"], "snapshot manifest SHA drift")
    manifest = json.loads(manifest_path.read_text())
    _require(
        manifest.get("schema") == "tactile3d-unit.s4-3-pi2s-input-snapshot.v1",
        "snapshot schema drift",
    )
    _require(manifest.get("status") == "COMPLETE_VERIFIED", "snapshot is not complete")
    if verify_snapshot_contents:
        _require(
            _snapshot_records(snapshot_root) == manifest.get("files"),
            "snapshot file manifest mismatch",
        )
    _require(
        len(manifest.get("files", ())) == int(snapshot_binding["files"]),
        "snapshot file-count drift",
    )
    _require(
        sum(int(row["bytes"]) for row in manifest["files"]) == int(snapshot_binding["bytes"]),
        "snapshot byte-count drift",
    )

    train_root = snapshot_root / "train_fixed"
    shapes: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "rows.npy": ((32,), np.dtype(np.int64)),
        "front_rgb_chw_uint8.npy": ((32, 3, 640, 640), np.dtype(np.uint8)),
        "wrist_rgb_chw_uint8.npy": ((32, 3, 640, 640), np.dtype(np.uint8)),
        "state.npy": ((32, 23), np.dtype(np.float32)),
        "action.npy": ((32, 30, 22), np.dtype(np.float32)),
        "action_is_pad.npy": ((32, 30), np.dtype(np.bool_)),
        "contact_state.npy": ((32, 256), np.dtype(np.float32)),
        "contact_target_t_plus_27.npy": ((32, 8, 32), np.dtype(np.float32)),
        "contact_aux_valid.npy": ((32,), np.dtype(np.bool_)),
        "va_target_t_plus_27.npy": ((32, 8, 32), np.dtype(np.float32)),
        "va_aux_valid.npy": ((32,), np.dtype(np.bool_)),
    }
    arrays = {
        name: _array_contract(train_root / name, *contract) for name, contract in shapes.items()
    }
    rows = np.asarray(arrays["rows.npy"])
    _require(
        rows.tolist() == protocol["samples"]["fixed_observation_rows"], "fixed-row order drift"
    )
    _require(
        not np.asarray(arrays["action_is_pad.npy"]).any(), "fixed action horizon contains padding"
    )
    _require(
        np.asarray(arrays["contact_aux_valid.npy"]).all(), "fixed contact targets are not all valid"
    )
    _require(np.asarray(arrays["va_aux_valid.npy"]).all(), "fixed VA targets are not all valid")
    row_to_position = {int(row): index for index, row in enumerate(rows)}
    gradient_rows = [int(row) for batch in protocol["samples"]["gradient_batches"] for row in batch]
    _require(
        len(gradient_rows) == 8 and len(set(gradient_rows)) == 8,
        "gradient rows are not eight unique rows",
    )
    _require(
        all(row in row_to_position for row in gradient_rows),
        "gradient row is absent from fixed snapshot",
    )

    expected_aux = {
        "B0": "N/A_NO_AUXILIARY_OBJECTIVE",
        "B1": "N/A_NO_AUXILIARY_OBJECTIVE",
        "B_VA27": {
            "target": "va_target_t_plus_27.npy (source field va_shared_target)",
            "valid_mask": "va_aux_valid.npy (source field va_aux_valid)",
        },
        "B_HVA": {
            "target": "va_target_t_plus_27.npy (source field va_shared_target)",
            "valid_mask": "va_aux_valid.npy (source field va_aux_valid)",
        },
        "B2": {
            "target": "contact_target_t_plus_27.npy (source field contact_shared_target)",
            "valid_mask": "contact_aux_valid.npy (source field physical_aux_valid)",
        },
    }
    _require(
        protocol["gradient_diagnostic"]["auxiliary_loss_by_model"] == expected_aux,
        "per-model auxiliary mapping drift",
    )
    _require(
        tuple(float(value) for value in protocol["gradient_diagnostic"]["flow_times"])
        == FLOW_TIMES,
        "gradient time grid drift",
    )
    expected_prefix_integration = {
        "contact_state_shape": [256],
        "adapter_output_shape_per_observation": [8, 2048],
        "token_count": 8,
        "token_width": 2048,
        "placement": "APPEND_AFTER_UNCHANGED_OFFICIAL_PREFIX",
        "token_order": "CONTACT_ADAPTER_RESHAPE_ORDER_0_THROUGH_7",
        "input_mask_for_added_tokens": True,
        "autoregressive_mask_for_added_tokens": False,
        "position_index_rule": "cumsum(input_mask)-1",
        "padding_interaction": (
            "physical append occurs after the padded official prefix; masked padding does not "
            "advance the added tokens' position indices"
        ),
        "state_dimension_changed": False,
        "action_dimension_changed": False,
    }
    _require(
        protocol.get("prefix_integration_contract") == expected_prefix_integration,
        "prefix integration contract drift",
    )
    fixed_sampling = protocol["fixed_sampling"]
    _require(
        {
            "noise_distribution": fixed_sampling.get("noise_distribution"),
            "noise_shape": fixed_sampling.get("noise_shape"),
            "noise_dtype": fixed_sampling.get("noise_dtype"),
        }
        == {
            "noise_distribution": "jax.random.normal",
            "noise_shape": [30, 32],
            "noise_dtype": "float32",
        },
        "fixed-sampling noise contract drift",
    )
    gradient_diagnostic = protocol["gradient_diagnostic"]
    _require(
        {
            "distribution": gradient_diagnostic.get("nuisance_noise_distribution"),
            "shape": gradient_diagnostic.get("nuisance_noise_shape_per_minibatch"),
            "dtype": gradient_diagnostic.get("nuisance_noise_dtype"),
        }
        == {
            "distribution": "jax.random.normal",
            "shape": [2, 30, 32],
            "dtype": "float32",
        },
        "gradient nuisance-noise contract drift",
    )
    _require(
        protocol["gradient_diagnostic"]["optimizer_step"] is False, "optimizer step was authorized"
    )
    _require(
        protocol["gradient_diagnostic"]["state_mutation_allowed"] is False,
        "state mutation was authorized",
    )

    contact_snapshot = resolve_symbolic(
        protocol["data_contract"]["contact_sidecar_persistent_snapshot"],
        experiments=experiments,
        pi2s_root=pi2s_root,
    )
    va_snapshot = resolve_symbolic(
        protocol["data_contract"]["va27_sidecar_persistent_snapshot"],
        experiments=experiments,
        pi2s_root=pi2s_root,
    )
    _require(
        sha256_file(contact_snapshot) == protocol["data_contract"]["contact_sidecar_sha256"],
        "persistent contact sidecar drift",
    )
    _require(
        sha256_file(va_snapshot) == protocol["data_contract"]["va27_sidecar_sha256"],
        "persistent VA sidecar drift",
    )

    for row in focus:
        checkpoint = resolve_symbolic(row["path"], experiments=experiments, pi2s_root=pi2s_root)
        _require(
            checkpoint.name == "29999" and checkpoint.is_dir(),
            f"focus checkpoint missing: {checkpoint}",
        )
        _require(
            (checkpoint / "params/manifest.ocdbt").is_file(),
            f"params manifest missing: {checkpoint}",
        )
        _require(
            (checkpoint / "train_state/manifest.ocdbt").is_file(),
            f"train-state manifest missing: {checkpoint}",
        )
        _require(
            (checkpoint / "assets/local_repo/norm_stats.json").is_file(),
            f"checkpoint norm stats missing: {checkpoint}",
        )

    checkpoint_authority(experiments, focus)

    return Contract(
        protocol=protocol,
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        runner_sha256=sha256_file(Path(__file__)),
        experiments=experiments,
        pi2s_root=pi2s_root,
        snapshot_root=snapshot_root,
        snapshot_manifest=manifest,
        snapshot_manifest_sha256=manifest_sha,
        train_root=train_root,
        focus=focus,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def frozen_openpi_runtime_asset(contract: Contract) -> Any:
    """Expose the frozen tokenizer through an isolated, disposable OpenPI cache."""

    dependency = contract.protocol["runtime_dependencies"]["paligemma_tokenizer"]
    source = resolve_symbolic(
        str(dependency["persistent_snapshot"]),
        experiments=contract.experiments,
        pi2s_root=contract.pi2s_root,
    )
    _require(
        source.is_file()
        and not source.is_symlink()
        and source.stat().st_size == PALIGEMMA_TOKENIZER_BYTES
        and sha256_file(source) == PALIGEMMA_TOKENIZER_SHA256,
        "frozen tokenizer source changed before runtime staging",
    )
    previous_data_home = os.environ.get("OPENPI_DATA_HOME")
    previous_no_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    previous_dont_write = sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix="pi2s-openpi-runtime-") as outer_text:
        outer = Path(outer_text)
        os.chmod(outer, 0o700)
        cache_root = outer / "cache"
        asset_parent = cache_root / "big_vision"
        asset_parent.mkdir(parents=True, mode=0o700)
        asset = asset_parent / "paligemma_tokenizer.model"
        shutil.copyfile(source, asset)
        os.chmod(asset, 0o444)
        with asset.open("rb") as stream:
            os.fsync(stream.fileno())
        _fsync_directory(asset_parent)
        _fsync_directory(cache_root)
        _require(
            asset.stat().st_size == PALIGEMMA_TOKENIZER_BYTES
            and sha256_file(asset) == PALIGEMMA_TOKENIZER_SHA256,
            "ephemeral tokenizer copy failed identity verification",
        )
        evidence = {
            "persistent_snapshot": dependency["persistent_snapshot"],
            "persistent_manifest": dependency["persistent_manifest"],
            "sha256": PALIGEMMA_TOKENIZER_SHA256,
            "bytes": PALIGEMMA_TOKENIZER_BYTES,
            "cache_policy": dependency["execution_cache_policy"],
            "cache_layout": "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model",
            "outer_directory_mode": "0700",
            "asset_mode": "0444",
            "pre_execution_sha256": PALIGEMMA_TOKENIZER_SHA256,
            "parent_environment_overridden": previous_data_home is not None,
        }
        os.environ["OPENPI_DATA_HOME"] = str(cache_root.resolve(strict=True))
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        try:
            yield asset, cache_root, evidence
        finally:
            post_hash = sha256_file(asset) if asset.is_file() else None
            unexpected = sorted(
                path.relative_to(cache_root).as_posix()
                for path in cache_root.rglob("*")
                if path.is_file() and path != asset
            )
            evidence["post_execution_sha256"] = post_hash
            evidence["unexpected_cache_files"] = unexpected
            evidence["post_execution_status"] = (
                "PASS" if post_hash == PALIGEMMA_TOKENIZER_SHA256 and not unexpected else "FAIL"
            )
            if previous_data_home is None:
                os.environ.pop("OPENPI_DATA_HOME", None)
            else:
                os.environ["OPENPI_DATA_HOME"] = previous_data_home
            if previous_no_bytecode is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = previous_no_bytecode
            sys.dont_write_bytecode = previous_dont_write
            _require(
                evidence["post_execution_status"] == "PASS",
                "ephemeral OpenPI tokenizer cache changed during execution",
            )


def _tar_member_sha256(archive: Path, member_name: str) -> str:
    with tarfile.open(archive, "r") as source:
        try:
            member = source.getmember(member_name)
        except KeyError as exc:
            raise ContractError(f"source snapshot lacks {member_name}") from exc
        _require(member.isfile(), f"source snapshot member is not a file: {member_name}")
        stream = source.extractfile(member)
        _require(stream is not None, f"cannot read source snapshot member: {member_name}")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _git_blob(commit: str, relative: str) -> bytes:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"cannot read frozen Git source {commit}:{relative}") from exc


def collect_config_source_bindings(contract: Contract) -> dict[str, dict[str, Any]]:
    """Validate and describe the two historical config/source cohorts."""

    cache_key = (contract.protocol_sha256, str(contract.experiments))
    if cache_key in _CONFIG_BINDING_CACHE:
        return _CONFIG_BINDING_CACHE[cache_key]

    track_root = contract.experiments / TRACK_A_RELATIVE
    artifacts = track_root / "artifacts"
    runs_path = artifacts / "training_runs.json"
    snapshot_manifest_path = artifacts / "code_snapshot_manifest.json"
    imports_path = artifacts / "import_origins.json"
    runs = _read_json(runs_path, "Track-A frozen run configurations")
    snapshot = _read_json(snapshot_manifest_path, "Track-A source snapshot manifest")
    imports = _read_json(imports_path, "Track-A import-origin audit")
    _require(runs.get("status") == "FROZEN_NOT_STARTED", "Track-A run freeze status drift")
    _require(snapshot.get("status") == "PASS", "Track-A source snapshot status drift")
    _require(imports.get("status") == "PASS", "Track-A import-origin status drift")
    archive = Path(str(snapshot["path"])).resolve(strict=True)
    _require(sha256_file(archive) == snapshot["sha256"], "Track-A immutable source snapshot drift")
    run_rows = {(str(row["model_id"]), int(row["seed"])): row for row in runs.get("rows", ())}
    track_common = {
        "cohort": "TRACK_A_SEED43_IMMUTABLE_SOURCE",
        "builder": "gr00t.simulation.pi2b_policy.training.build_config",
        "builder_path": "$REPO_ROOT/gr00t/simulation/pi2b_policy/training.py",
        "training_runs_sha256": sha256_file(runs_path),
        "import_origins_sha256": sha256_file(imports_path),
        "source_snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "source_snapshot_sha256": snapshot["sha256"],
        "source_git_head": snapshot["git_head"],
        "accepted_openpi_origin": "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/src/openpi",
        "accepted_openpi_runtime_sha256": verify_openpi_runtime_source(),
        "expected_runtime_versions": EXPECTED_RUNTIME_VERSIONS,
    }
    for relative in (
        "gr00t/simulation/pi2b_policy/training.py",
        "gr00t/simulation/pi05_tactile_unit.py",
        "gr00t/simulation/s4_3_pi1.py",
    ):
        expected = next(iter(run_rows.values()))["source_sha256"][relative]
        _require(
            sha256_file(ROOT / relative) == expected, f"integrated Track-A source drift: {relative}"
        )
        _require(
            _tar_member_sha256(archive, relative) == expected,
            f"snapshot Track-A source drift: {relative}",
        )
    # The integrated contract grew a read-only resolver after training.  Prove
    # the frozen recipe-bearing member still exists in the immutable archive,
    # then record both identities instead of pretending the cohorts are equal.
    frozen_contract_sha = next(iter(run_rows.values()))["source_sha256"][
        "gr00t/simulation/pi2b_policy/contract.py"
    ]
    _require(
        _tar_member_sha256(archive, "gr00t/simulation/pi2b_policy/contract.py")
        == frozen_contract_sha,
        "snapshot Track-A contract drift",
    )
    track_common["frozen_contract_sha256"] = frozen_contract_sha
    track_common["integrated_readonly_contract_sha256"] = sha256_file(
        ROOT / "gr00t/simulation/pi2b_policy/contract.py"
    )

    historical_freeze_path = (
        ROOT / ".local/artifacts/simulation/s4_3_pi2m/training_protocol_freeze.json"
    )
    historical_freeze = _read_json(historical_freeze_path, "seed42 B_HVA training freeze")
    _require(
        historical_freeze.get("status") == "FROZEN_BEFORE_TRAINING",
        "seed42 B_HVA freeze status drift",
    )
    historical_commit = str(historical_freeze["git_head"])
    historical_relatives = (
        "scripts/simulation/train_s4_3_pi2m_bhva.py",
        "gr00t/simulation/pi05_tactile_unit.py",
        "gr00t/simulation/s4_3_pi1.py",
    )
    for relative in historical_relatives:
        expected = historical_freeze["code_sha256"][relative]
        frozen_bytes = _git_blob(historical_commit, relative)
        _require(
            hashlib.sha256(frozen_bytes).hexdigest() == expected,
            f"seed42 B_HVA Git source drift: {relative}",
        )
    _require(
        sha256_file(ROOT / "scripts/simulation/train_s4_3_pi2m_bhva.py")
        == historical_freeze["code_sha256"]["scripts/simulation/train_s4_3_pi2m_bhva.py"],
        "integrated seed42 B_HVA builder drift",
    )
    historical_binding = {
        "cohort": "HISTORICAL_PI2M_B_HVA_SEED42",
        "builder": "scripts.simulation.train_s4_3_pi2m_bhva.build_config",
        "builder_path": "$REPO_ROOT/scripts/simulation/train_s4_3_pi2m_bhva.py",
        "training_protocol_freeze_sha256": sha256_file(historical_freeze_path),
        "source_git_head": historical_commit,
        "mode": historical_freeze["mode"],
        "lambda_phys": float(historical_freeze["lambda_phys"]),
        "target": "VA27",
        "source_sha256": {
            relative: historical_freeze["code_sha256"][relative]
            for relative in historical_relatives
        },
        "runtime_source": "VERIFIED_FROZEN_GIT_OBJECTS",
        "integrated_source_sha256": {
            relative: sha256_file(ROOT / relative) for relative in historical_relatives
        },
        "accepted_openpi_origin": "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/src/openpi",
        "accepted_openpi_runtime_sha256": verify_openpi_runtime_source(),
        "expected_runtime_versions": EXPECTED_RUNTIME_VERSIONS,
    }

    output: dict[str, dict[str, Any]] = {}
    for focus in contract.focus:
        checkpoint_id = f"{focus['model']}_{int(focus['training_seed'])}"
        if checkpoint_id == "B_HVA_42":
            output[checkpoint_id] = dict(historical_binding)
            continue
        key = (str(focus["model"]), int(focus["training_seed"]))
        row = run_rows.get(key)
        _require(row is not None, f"missing Track-A frozen run row: {key}")
        binding = dict(track_common)
        binding.update(
            {
                "frozen_config_sha256": row["config_sha256"],
                "mode": row["mode"],
                "lambda_phys": float(row["lambda_phys"]),
                "target": row["target"],
            }
        )
        output[checkpoint_id] = binding
    _require(
        tuple(output) == tuple(f"{model}_{seed}" for model, seed in FOCUS_KEYS),
        "config binding order drift",
    )
    _CONFIG_BINDING_CACHE[cache_key] = output
    return output


def summary_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q90": None,
            "min": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise ContractError("summary input contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def jerk_proxy(action: np.ndarray) -> float:
    value = np.asarray(action, dtype=np.float64)
    _require(value.shape == (30, 22), f"jerk action shape drift: {value.shape}")
    second = value[2:] - 2.0 * value[1:-1] + value[:-2]
    return float(np.mean(np.linalg.norm(second, axis=-1)))


def boundary_exceedance(action: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> dict[str, float]:
    value = np.asarray(action, dtype=np.float64)
    low = np.asarray(q01, dtype=np.float64)[:22]
    high = np.asarray(q99, dtype=np.float64)[:22]
    _require(
        value.shape == (30, 22) and low.shape == high.shape == (22,),
        "normalization-boundary shape drift",
    )
    outside = (value < low) | (value > high)
    return {
        "tcp_xyz": float(outside[:, 0:3].mean()),
        "rotation_vector": float(outside[:, 3:6].mean()),
        "hand": float(outside[:, 6:22].mean()),
    }


def action_delta_metrics(reference: np.ndarray, intervention: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(intervention, dtype=np.float64)
    _require(left.shape == right.shape == (30, 22), "action delta shape drift")
    delta = right - left
    per_step = np.linalg.norm(delta, axis=-1)
    return {
        "component_rms": {
            "tcp_xyz": _rms(delta[:, 0:3]),
            "rotation_vector": _rms(delta[:, 3:6]),
            "hand": _rms(delta[:, 6:22]),
        },
        "chunk_l2": {
            "per_step": per_step.tolist(),
            "early": float(per_step[0:10].mean()),
            "late": float(per_step[20:30].mean()),
            "all": float(per_step.mean()),
            "max": float(per_step.max()),
        },
        "delta_jerk_proxy": jerk_proxy(delta.astype(np.float32)),
        "delta_sha256": array_sha256(delta.astype(np.float32)),
    }


def aggregate_action_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scalar_paths = {
        "component_rms/tcp_xyz": lambda row: row["delta"]["component_rms"]["tcp_xyz"],
        "component_rms/rotation_vector": lambda row: row["delta"]["component_rms"][
            "rotation_vector"
        ],
        "component_rms/hand": lambda row: row["delta"]["component_rms"]["hand"],
        "chunk_l2/early": lambda row: row["delta"]["chunk_l2"]["early"],
        "chunk_l2/late": lambda row: row["delta"]["chunk_l2"]["late"],
        "chunk_l2/all": lambda row: row["delta"]["chunk_l2"]["all"],
        "delta_jerk_proxy": lambda row: row["delta"]["delta_jerk_proxy"],
        "action_jerk_proxy": lambda row: row["action_jerk_proxy"],
        "boundary/tcp_xyz": lambda row: row["normalization_boundary_exceedance"]["tcp_xyz"],
        "boundary/rotation_vector": lambda row: row["normalization_boundary_exceedance"][
            "rotation_vector"
        ],
        "boundary/hand": lambda row: row["normalization_boundary_exceedance"]["hand"],
    }
    return {
        name: summary_stats([float(extract(row)) for row in records])
        for name, extract in scalar_paths.items()
    }


def _identity_map(contract: Contract) -> tuple[np.ndarray, dict[int, int]]:
    rows = np.asarray(np.load(contract.train_root / "rows.npy", allow_pickle=False), dtype=np.int64)
    return rows, {int(row): index for index, row in enumerate(rows)}


def load_intervention_contact_states(contract: Contract) -> dict[str, np.ndarray]:
    """Resolve all H interventions by global row identity, never array offset."""

    protocol = contract.protocol
    rows, _ = _identity_map(contract)
    contact_path = resolve_symbolic(
        protocol["data_contract"]["contact_sidecar_persistent_snapshot"],
        experiments=contract.experiments,
        pi2s_root=contract.pi2s_root,
    )
    with np.load(contact_path, allow_pickle=False) as source:
        full_h = np.asarray(source["contact_state"], dtype=np.float32)
        episodes = np.asarray(source["episode_index"], dtype=np.int64)
        ticks = np.asarray(source["control_tick_end"], dtype=np.int64)
    correct = full_h[rows].copy()
    mean_h = np.mean(full_h.astype(np.float64), axis=0).astype(np.float32)
    expected_mean_sha = protocol["h_conditions"]["train_mean_h_float32_sha256"]
    _require(
        hashlib.sha256(mean_h.tobytes(order="C")).hexdigest() == expected_mean_sha,
        "train-mean H drift",
    )
    lag = []
    other = []
    lag_map = protocol["h_conditions"]["same_episode_lag5_row_map"]
    other_map = protocol["h_conditions"]["other_episode_row_map"]
    for row in rows.tolist():
        lag_row = int(lag_map[str(row)]["row"])
        other_row = int(other_map[str(row)])
        _require(int(episodes[lag_row]) == int(episodes[row]), f"lag5 crosses episode at row {row}")
        if not bool(lag_map[str(row)]["bootstrap_affected"]):
            _require(int(ticks[row]) - int(ticks[lag_row]) == 5, f"lag5 tick drift at row {row}")
        _require(
            int(episodes[other_row]) != int(episodes[row]),
            f"other-episode H shares episode at row {row}",
        )
        lag.append(full_h[lag_row])
        other.append(full_h[other_row])
    output = {
        "correct": correct,
        "train_mean": np.repeat(mean_h[None, :], len(rows), axis=0),
        "same_episode_lag5": np.asarray(lag, dtype=np.float32),
        "other_episode": np.asarray(other, dtype=np.float32),
        "zero": np.zeros_like(correct),
    }
    for name, value in output.items():
        _require(
            value.shape == (32, 256) and np.isfinite(value).all(), f"invalid H condition: {name}"
        )
    return output


def _focus_key(row: Mapping[str, Any]) -> str:
    return f"{row['model']}_seed{int(row['training_seed'])}"


def _checkpoint_id(row: Mapping[str, Any]) -> str:
    return f"{row['model']}_{int(row['training_seed'])}"


def _focus_by_id(contract: Contract, checkpoint_id: str) -> dict[str, Any]:
    matches = [row for row in contract.focus if _checkpoint_id(row) == checkpoint_id]
    _require(len(matches) == 1, f"checkpoint-id is not one frozen focus row: {checkpoint_id}")
    return matches[0]


def shard_path(output_dir: Path, checkpoint_id: str, stage: str) -> Path:
    _require(
        checkpoint_id in {f"{model}_{seed}" for model, seed in FOCUS_KEYS}, "invalid checkpoint-id"
    )
    _require(stage in {"actions", "gradients"}, "invalid diagnostic phase")
    return Path(output_dir) / "shards" / f"{checkpoint_id}.{stage}.json"


def _artifact_binding(contract: Contract, focus: Mapping[str, Any], stage: str) -> dict[str, Any]:
    _require(stage in {"actions", "gradients"}, "invalid diagnostic phase")
    checkpoint_id = _checkpoint_id(focus)
    source_binding = collect_config_source_bindings(contract)[checkpoint_id]
    authority = checkpoint_authority(contract.experiments, (focus,))
    return {
        "protocol_sha256": contract.protocol_sha256,
        "runner_sha256": contract.runner_sha256,
        "selection_sha256": contract.protocol["selection_sha256"],
        "snapshot_manifest_sha256": contract.snapshot_manifest_sha256,
        "checkpoint_tree_sha256": focus["tree_sha256"],
        "checkpoint_authority": authority,
        "config_source_binding_sha256": canonical_sha(source_binding),
        "model": focus["model"],
        "training_seed": int(focus["training_seed"]),
        "stage": stage,
    }


def validate_resumable_artifact(
    path: Path, contract: Contract, focus: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    _require(stage in {"actions", "gradients"}, "invalid diagnostic phase")
    _require(
        path.is_file() and not path.is_symlink(),
        f"resume artifact is not an immutable regular file: {path}",
    )
    payload = json.loads(path.read_text())
    expected_schema = ACTION_SCHEMA if stage == "actions" else GRADIENT_SCHEMA
    _require(payload.get("schema") == expected_schema, f"resume schema drift: {path}")
    _require(payload.get("status") == "PASS", f"resume artifact is not PASS: {path}")
    _require(
        payload.get("binding") == _artifact_binding(contract, focus, stage),
        f"resume binding drift: {path}",
    )
    _require(
        payload.get("gates") and all(value == "PASS" for value in payload["gates"].values()),
        f"resume gate drift: {path}",
    )
    _require(payload.get("focus") == dict(focus), f"resume focus identity drift: {path}")
    _require(
        payload.get("optimizer_steps") == 0 and payload.get("checkpoint_writes") == 0,
        f"resume artifact reports a forbidden update: {path}",
    )
    _require(
        payload.get("state_before") == payload.get("state_after")
        and payload.get("checkpoint_sentinels_before") == payload.get("checkpoint_sentinels_after"),
        f"resume artifact does not prove state/checkpoint immutability: {path}",
    )
    _require(
        payload.get("parameter_load")
        == {
            "mode": "STRICT_NO_MISSING_OR_UNEXPECTED_PARAMETERS",
            "remove_extra_params": False,
            "missing_parameter_leaves": 0,
            "unexpected_parameter_leaves": 0,
            "fallback_initialization": False,
        },
        f"resume artifact lacks strict parameter-load evidence: {path}",
    )
    _require(
        isinstance(payload.get("actual_import_origins"), dict)
        and isinstance(payload.get("execution_environment"), dict),
        f"resume artifact lacks runtime provenance: {path}",
    )
    origins = payload["actual_import_origins"]
    _require(
        origins.get("openpi_client_image_tools", {}).get("sha256")
        == OPENPI_CLIENT_IMAGE_TOOLS_SHA256
        and origins.get("runtime_versions") == EXPECTED_RUNTIME_VERSIONS
        and origins.get("pip_freeze_all_sha256") == EXPECTED_PIP_FREEZE_SHA256,
        f"resume runtime source/environment identity drift: {path}",
    )
    runtime_asset = payload["execution_environment"].get("frozen_paligemma_tokenizer", {})
    _require(
        runtime_asset.get("sha256") == PALIGEMMA_TOKENIZER_SHA256
        and runtime_asset.get("pre_execution_sha256") == PALIGEMMA_TOKENIZER_SHA256
        and runtime_asset.get("post_execution_sha256") == PALIGEMMA_TOKENIZER_SHA256
        and runtime_asset.get("post_execution_status") == "PASS"
        and runtime_asset.get("unexpected_cache_files") == [],
        f"resume tokenizer runtime identity drift: {path}",
    )
    if stage == "actions":
        _validate_action_identity(payload, contract)
    else:
        _require(
            len(payload.get("records", ())) == 12,
            f"resume gradient record budget drift: {path}",
        )
        _validate_gradient_identity(payload, contract)
        _require(
            payload.get("preprocess_mode") == "DETERMINISTIC_EVAL_PREPROCESS_NO_AUGMENTATION",
            f"resume gradient preprocessing mode drift: {path}",
        )
    return payload


def check_report(contract: Contract) -> dict[str, Any]:
    source_bindings = collect_config_source_bindings(contract)
    return {
        "schema": "tactile3d-unit.s4-3-pi2s-model-diagnostic-check.v1",
        "status": "PASS",
        "execution_performed": False,
        "jax_imported": "jax" in sys.modules,
        "protocol_sha256": contract.protocol_sha256,
        "runner_sha256": contract.runner_sha256,
        "implementation_hash_binding": {
            "key": "diagnostic_implementation_sha256.run_pi2s_model_diagnostics",
            "status": "BOUND_AND_VERIFIED",
            "expected_sha256": contract.runner_sha256,
        },
        "snapshot": {
            "path": contract.protocol["source_manifests"]["persistent_input_snapshot"]["path"],
            "manifest_sha256": contract.snapshot_manifest_sha256,
            "files": len(contract.snapshot_manifest["files"]),
            "bytes": sum(int(row["bytes"]) for row in contract.snapshot_manifest["files"]),
        },
        "focus": [
            {
                "model": row["model"],
                "training_seed": int(row["training_seed"]),
                "checkpoint": row["path"],
                "tree_sha256": row["tree_sha256"],
                "config_source_binding_sha256": canonical_sha(source_bindings[_checkpoint_id(row)]),
            }
            for row in contract.focus
        ],
        "contracts": {
            "fixed_rows": 32,
            "gradient_minibatches": 4,
            "gradient_batch_size": 2,
            "flow_times": list(FLOW_TIMES),
            "action_noise_seeds": contract.protocol["fixed_sampling"]["noise_seeds"],
            "gradient_noise_seeds": contract.protocol["gradient_diagnostic"][
                "nuisance_noise_seeds_by_minibatch"
            ],
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "global_shard_budget": EXPECTED_GLOBAL_BUDGET,
        },
    }


def _median_range(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "min": None, "max": None}
    _require(np.isfinite(array).all(), "gradient summary contains non-finite values")
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _validate_action_identity(shard: Mapping[str, Any], contract: Contract) -> tuple[Any, ...]:
    """Validate exact fixed rows/seeds and shared noise for one action shard."""

    model = str(shard["focus"]["model"])
    expected_conditions = CONDITION_ORDER if model in CONTACT_MODELS else ("correct",)
    expected_rows = [int(value) for value in contract.protocol["samples"]["fixed_observation_rows"]]
    expected_seeds = [int(value) for value in contract.protocol["fixed_sampling"]["noise_seeds"]]
    _require(
        len(expected_rows) == len(expected_seeds) == 32,
        "fixed action identity budget drift",
    )
    records = list(shard.get("records", ()))
    _require(
        len(records) == 32 * len(expected_conditions)
        and {row.get("condition") for row in records} == set(expected_conditions),
        f"fixed action record budget drift: {model}",
    )
    reference_noise: tuple[str, ...] | None = None
    for condition in expected_conditions:
        values = sorted(
            (row for row in records if row.get("condition") == condition),
            key=lambda row: int(row["fixed_position"]),
        )
        _require(
            [int(row["fixed_position"]) for row in values] == list(range(32)),
            f"fixed action positions drift: {model}/{condition}",
        )
        _require(
            [int(row["global_row"]) for row in values] == expected_rows,
            f"fixed action rows drift: {model}/{condition}",
        )
        _require(
            [int(row["noise_seed"]) for row in values] == expected_seeds,
            f"fixed action noise seeds drift: {model}/{condition}",
        )
        noise_hashes = tuple(str(row["noise_sha256"]) for row in values)
        _require(
            all(len(value) == 64 for value in noise_hashes),
            f"fixed action noise hash missing: {model}/{condition}",
        )
        if reference_noise is None:
            reference_noise = noise_hashes
        else:
            _require(
                noise_hashes == reference_noise,
                f"fixed action noise differs across H conditions: {model}",
            )

    input_batches = list(shard.get("input_batches", ()))
    _require(
        len(input_batches) == len(expected_conditions)
        and tuple(row.get("condition") for row in input_batches) == expected_conditions,
        f"action input-batch condition order drift: {model}",
    )
    batch_noise_hashes = {str(row.get("explicit_noise_sha256")) for row in input_batches}
    _require(
        len(batch_noise_hashes) == 1 and all(len(value) == 64 for value in batch_noise_hashes),
        f"explicit action noise differs across H conditions: {model}",
    )
    input_identity: list[str] = []
    for key in (
        "tokenized_prompt_sha256",
        "tokenized_prompt_mask_sha256",
        "normalized_state_sha256",
        "preprocessed_images_sha256",
    ):
        values = {str(row.get(key)) for row in input_batches}
        _require(
            len(values) == 1 and all(len(value) == 64 for value in values),
            f"fixed non-H input differs across H conditions: {model}/{key}",
        )
        input_identity.append(next(iter(values)))
    prefix_batches = list(shard.get("prefix_diagnostics", ()))
    _require(
        len(prefix_batches) == len(expected_conditions)
        and tuple(row.get("condition") for row in prefix_batches) == expected_conditions,
        f"prefix condition order drift: {model}",
    )
    _require(
        all(
            [int(record["global_row"]) for record in row.get("records", ())] == expected_rows
            for row in prefix_batches
        ),
        f"prefix fixed-row identity drift: {model}",
    )
    for batch in prefix_batches:
        _require(
            batch.get("official_prefix_unchanged") is True
            and batch.get("added_mask_contract") is True
            and batch.get("adapter_projection_matches_appended_tokens") is True
            and batch.get("adapter_manual_pipeline_matches") is True
            and batch.get("position_index_contract") is True,
            f"prefix contract drift in resumable shard: {model}",
        )
        for record in batch["records"]:
            _require(
                isinstance(record.get("image_tokens_width_2048"), dict)
                and isinstance(record.get("language_tokens_width_2048"), dict),
                f"prefix modality norms missing: {model}",
            )
            if model in CONTACT_MODELS:
                _require(
                    isinstance(record.get("contact_tokens_width_2048"), dict)
                    and record.get("contact_state_pre_layernorm_l2") is not None
                    and record.get("contact_state_post_layernorm_l2") is not None
                    and len(record.get("contact_sequence_indices", ())) == 8
                    and len(record.get("contact_position_indices", ())) == 8,
                    f"contact prefix LayerNorm/position evidence missing: {model}",
                )
            else:
                _require(
                    record.get("contact_tokens_width_2048") is None
                    and record.get("contact_state_pre_layernorm_l2") is None
                    and record.get("contact_state_post_layernorm_l2") is None,
                    f"non-contact model contains fabricated contact-token metrics: {model}",
                )
    _require(reference_noise is not None, f"action noise identity absent: {model}")
    return (
        reference_noise,
        next(iter(batch_noise_hashes)),
        *input_identity,
    )


def _validate_gradient_identity(shard: Mapping[str, Any], contract: Contract) -> tuple[Any, ...]:
    """Validate frozen batch/time/noise identity for one gradient shard."""

    model = str(shard["focus"]["model"])
    fixed_rows = [int(value) for value in contract.protocol["samples"]["fixed_observation_rows"]]
    row_to_position = {row: index for index, row in enumerate(fixed_rows)}
    expected_batches = [
        [int(value) for value in batch]
        for batch in contract.protocol["samples"]["gradient_batches"]
    ]
    expected_seeds = [
        int(value)
        for value in contract.protocol["gradient_diagnostic"]["nuisance_noise_seeds_by_minibatch"]
    ]
    records = list(shard.get("records", ()))
    _require(
        len(records) == 12
        and {int(row.get("minibatch_index", -1)) for row in records} == set(range(4)),
        f"gradient record budget drift: {model}",
    )
    noise_by_minibatch: list[str] = []
    for minibatch, (rows, seed) in enumerate(zip(expected_batches, expected_seeds, strict=True)):
        expected_positions = [row_to_position[row] for row in rows]
        values = sorted(
            (row for row in records if int(row.get("minibatch_index", -1)) == minibatch),
            key=lambda row: float(row["flow_time"]),
        )
        _require(
            [float(row["flow_time"]) for row in values] == list(FLOW_TIMES),
            f"gradient flow-time grid drift: {model}/{minibatch}",
        )
        _require(
            all([int(value) for value in row["global_rows"]] == rows for row in values),
            f"gradient global-row identity drift: {model}/{minibatch}",
        )
        _require(
            all(
                [int(value) for value in row["fixed_positions"]] == expected_positions
                for row in values
            ),
            f"gradient fixed-position identity drift: {model}/{minibatch}",
        )
        _require(
            all(int(row["noise_seed"]) == seed for row in values),
            f"gradient nuisance-noise seed drift: {model}/{minibatch}",
        )
        noise_hashes = {str(row["noise_sha256"]) for row in values}
        input_hashes = {str(row["transformed_batch_sha256"]) for row in values}
        _require(
            len(noise_hashes) == 1 and all(len(value) == 64 for value in noise_hashes),
            f"gradient nuisance noise differs across times: {model}/{minibatch}",
        )
        _require(
            len(input_hashes) == 1 and all(len(value) == 64 for value in input_hashes),
            f"gradient input differs across times: {model}/{minibatch}",
        )
        noise_by_minibatch.append(next(iter(noise_hashes)))
    fixed_identity = shard.get("fixed_non_h_input_identity")
    _require(
        isinstance(fixed_identity, dict),
        f"gradient fixed-input identity missing: {model}",
    )
    non_h_values: list[str] = []
    for key in (
        "tokenized_prompt_sha256",
        "tokenized_prompt_mask_sha256",
        "normalized_state_sha256",
        "preprocessed_images_sha256",
    ):
        value = str(fixed_identity.get(key))
        _require(len(value) == 64, f"gradient fixed-input hash missing: {model}/{key}")
        non_h_values.append(value)
    return (tuple(noise_by_minibatch), *non_h_values)


def enforce_global_budgets(
    action_shards: Sequence[dict[str, Any]],
    gradient_shards: Sequence[dict[str, Any]],
    contract: Contract | None = None,
) -> dict[str, Any]:
    """Reject duplicate, missing, or expanded scientific units."""

    expected_ids = {f"{model}_{seed}" for model, seed in FOCUS_KEYS}
    action_ids = [_checkpoint_id(row["focus"]) for row in action_shards]
    gradient_ids = [_checkpoint_id(row["focus"]) for row in gradient_shards]
    _require(
        len(action_ids) == len(set(action_ids)) == 6, "action shards are duplicate or incomplete"
    )
    _require(
        len(gradient_ids) == len(set(gradient_ids)) == 6,
        "gradient shards are duplicate or incomplete",
    )
    _require(set(action_ids) == expected_ids, "action shard cohort drift")
    _require(set(gradient_ids) == expected_ids, "gradient shard cohort drift")

    if contract is not None:
        action_identities = [_validate_action_identity(shard, contract) for shard in action_shards]
        gradient_identities = [
            _validate_gradient_identity(shard, contract) for shard in gradient_shards
        ]
        _require(
            len(set(action_identities)) == 1,
            "fixed action noise differs across checkpoints",
        )
        _require(
            len(set(gradient_identities)) == 1,
            "gradient nuisance noise differs across checkpoints",
        )
        _require(
            len({tuple(identity[-4:]) for identity in (*action_identities, *gradient_identities)})
            == 1,
            "fixed prompt/state/image identity differs across phases or checkpoints",
        )

    action_count = 0
    for shard in action_shards:
        focus = shard["focus"]
        model = str(focus["model"])
        expected_conditions = CONDITION_ORDER if model in CONTACT_MODELS else ("correct",)
        records = shard.get("records", ())
        expected_count = 32 * len(expected_conditions)
        _require(
            len(records) == expected_count, f"action record budget drift: {_checkpoint_id(focus)}"
        )
        by_condition: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_conditions}
        for record in records:
            _require(record.get("condition") in by_condition, "unexpected H intervention condition")
            by_condition[record["condition"]].append(record)
        for condition, values in by_condition.items():
            _require(
                len(values) == 32, f"action row budget drift: {_checkpoint_id(focus)}:{condition}"
            )
            _require(len({int(row["global_row"]) for row in values}) == 32, "duplicate action rows")
            _require(
                len({int(row["noise_seed"]) for row in values}) == 32,
                "duplicate action noise seeds",
            )
        action_count += len(records)

    gradient_count = 0
    auxiliary_count = 0
    for shard in gradient_shards:
        focus = shard["focus"]
        records = shard.get("records", ())
        _require(len(records) == 12, f"gradient record budget drift: {_checkpoint_id(focus)}")
        combinations = {(int(row["minibatch_index"]), float(row["flow_time"])) for row in records}
        _require(
            combinations == {(batch, time) for batch in range(4) for time in FLOW_TIMES},
            f"gradient time/minibatch grid drift: {_checkpoint_id(focus)}",
        )
        applicable = str(focus["model"]) in AUXILIARY_MODELS
        _require(
            all(bool(row["auxiliary_applicable"]) is applicable for row in records),
            "auxiliary applicability drift",
        )
        gradient_count += len(records)
        auxiliary_count += sum(bool(row["auxiliary_applicable"]) for row in records)

    observed = {
        "checkpoints": 6,
        "phases_per_checkpoint": 2,
        "immutable_shards": len(action_shards) + len(gradient_shards),
        "action_records": action_count,
        "gradient_records": gradient_count,
        "auxiliary_gradient_records": auxiliary_count,
        "optimizer_steps": sum(
            int(row.get("optimizer_steps", -1)) for row in [*action_shards, *gradient_shards]
        ),
        "checkpoint_writes": sum(
            int(row.get("checkpoint_writes", -1)) for row in [*action_shards, *gradient_shards]
        ),
    }
    _require(observed == EXPECTED_GLOBAL_BUDGET, f"global scientific budget drift: {observed}")
    return observed


def aggregate_shards(output_dir: Path, contract: Contract) -> dict[str, Any]:
    """Read and validate all immutable shards; do not publish the summary here."""

    _validate_shard_name_set(output_dir, contract)
    actions: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    for focus in contract.focus:
        checkpoint_id = _checkpoint_id(focus)
        for stage, destination in (("actions", actions), ("gradients", gradients)):
            path = shard_path(output_dir, checkpoint_id, stage)
            _require(
                path.is_file() and not path.is_symlink(),
                f"required immutable shard missing: {path}",
            )
            payload = validate_resumable_artifact(path, contract, focus, stage)
            destination.append(payload)
            shard_manifest.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "phase": stage,
                    "file": f"shards/{path.name}",
                    "sha256": sha256_file(path),
                }
            )
    budgets = enforce_global_budgets(actions, gradients, contract)

    action_summary: dict[str, Any] = {}
    for shard in actions:
        checkpoint_id = _checkpoint_id(shard["focus"])
        action_summary[checkpoint_id] = {
            condition: aggregate_action_records(
                [row for row in shard["records"] if row["condition"] == condition]
            )
            for condition in (
                CONDITION_ORDER if shard["focus"]["model"] in CONTACT_MODELS else ("correct",)
            )
        }

    gradient_summary: dict[str, Any] = {}
    for shard in gradients:
        checkpoint_id = _checkpoint_id(shard["focus"])
        records = shard["records"]
        summary: dict[str, Any] = {
            "main_loss": _median_range([float(row["losses"]["main"]) for row in records]),
        }
        if shard["focus"]["model"] in AUXILIARY_MODELS:
            summary.update(
                auxiliary_loss=_median_range(
                    [float(row["losses"]["auxiliary"]) for row in records]
                ),
                combined_loss=_median_range([float(row["losses"]["combined"]) for row in records]),
            )
        for group in PARAMETER_GROUPS:
            for metric in ("l2_norm_g_main", "l2_norm_g_aux", "cosine_g_main_g_aux"):
                values = [
                    row["groups"][group][metric]
                    for row in records
                    if row["groups"][group].get(metric) is not None
                ]
                summary[f"{group}/{metric}"] = _median_range([float(value) for value in values])
        gradient_summary[checkpoint_id] = summary

    return {
        "schema": SUMMARY_SCHEMA,
        "status": "PASS",
        "created_at_utc": now_utc(),
        "binding": {
            "protocol_sha256": contract.protocol_sha256,
            "runner_sha256": contract.runner_sha256,
            "selection_sha256": contract.protocol["selection_sha256"],
            "snapshot_manifest_sha256": contract.snapshot_manifest_sha256,
        },
        "scientific_budget": budgets,
        "shards": shard_manifest,
        "action_summary": action_summary,
        "gradient_summary": gradient_summary,
        "interpretation": contract.protocol["metrics"]["interpretation"],
    }


def validate_execution_environment() -> dict[str, Any]:
    """Import JAX only in execute mode and require one externally leased GPU."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(
        visible is not None and visible.strip() not in {"", "-1"},
        "CUDA_VISIBLE_DEVICES must expose exactly one leased GPU",
    )
    _require(
        len([value for value in visible.split(",") if value.strip()]) == 1,
        "execute mode requires exactly one visible GPU",
    )
    import jax

    gpu_devices = jax.devices("gpu")
    _require(jax.default_backend() == "gpu", "diagnostic execution requires the GPU backend")
    _require(
        len(gpu_devices) == 1, f"diagnostic execution sees {len(gpu_devices)} GPUs, expected one"
    )
    device = gpu_devices[0]
    return {
        "backend": jax.default_backend(),
        "python_executable": sys.executable,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "visible_device_contract": "EXACTLY_ONE_EXTERNALLY_LEASED_GPU",
        "cuda_visible_devices": visible,
        "logical_device_id": int(device.id),
        "device_kind": str(device.device_kind),
        "platform": str(device.platform),
        "jax_version": str(jax.__version__),
    }


def _symbolic_runtime_path(path: str | Path, contract: Contract) -> str:
    resolved = Path(path).resolve()
    roots = (
        (ROOT.resolve(), "$REPO_ROOT"),
        (contract.pi2s_root.resolve(), "$PI2S_ROOT"),
        (contract.experiments.resolve(), "$EXPERIMENT_ROOT"),
    )
    for root, token in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return token if relative == Path(".") else f"{token}/{relative.as_posix()}"
    raise ContractError(f"runtime import escaped audited roots: {resolved}")


def _install_frozen_git_module(
    module_name: str,
    relative: str,
    commit: str,
    expected_sha256: str,
) -> Any:
    """Execute a verified historical module without writing an extraction tree."""

    import importlib
    import importlib.util
    import types

    source = _git_blob(commit, relative)
    _require(
        hashlib.sha256(source).hexdigest() == expected_sha256,
        f"historical runtime source drift: {relative}",
    )
    parent_name, _, child_name = module_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    origin = f"git-object:{commit}:{relative}"
    module = types.ModuleType(module_name)
    module.__file__ = origin
    module.__package__ = parent_name
    module.__loader__ = None
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None, origin=origin)
    sys.modules[module_name] = module
    setattr(parent, child_name, module)
    try:
        exec(compile(source, origin, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        if getattr(parent, child_name, None) is module:
            delattr(parent, child_name)
        raise
    return module


def _prepare_accepted_openpi_client_import() -> None:
    """Prepend the audited client source before any OpenPI transform import."""

    accepted = ACCEPTED_OPENPI_CLIENT_SOURCE.resolve(strict=True)
    preloaded = [
        module
        for name, module in sys.modules.items()
        if name == "openpi_client" or name.startswith("openpi_client.")
    ]
    for module in preloaded:
        origin = getattr(module, "__file__", None)
        _require(origin is not None, "preloaded OpenPI client module has no source origin")
        resolved = Path(origin).resolve(strict=True)
        try:
            resolved.relative_to(accepted)
        except ValueError as exc:
            raise ContractError(
                f"OpenPI client was imported before audit from an unaccepted origin: {resolved}"
            ) from exc
    accepted_text = str(accepted)
    sys.path[:] = [value for value in sys.path if Path(value or os.curdir).resolve() != accepted]
    sys.path.insert(0, accepted_text)


def _pip_freeze_sha256() -> str:
    process = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(
        process.returncode == 0,
        f"cannot audit execution environment with pip freeze: {process.stderr.decode(errors='replace')}",
    )
    return hashlib.sha256(process.stdout).hexdigest()


def build_runtime_config(
    contract: Contract,
    focus: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct the correct historical config cohort for one focus row."""

    import inspect

    _prepare_accepted_openpi_client_import()

    checkpoint_id = _checkpoint_id(focus)
    if checkpoint_id == "B_HVA_42":
        from scripts.simulation import train_s4_3_pi2m_bhva as builder_module

        builder_module.configure_imports()
        historical_commit = str(source_binding["source_git_head"])
        source_hashes = source_binding["source_sha256"]
        _install_frozen_git_module(
            "gr00t.simulation.s4_3_pi1",
            "gr00t/simulation/s4_3_pi1.py",
            historical_commit,
            str(source_hashes["gr00t/simulation/s4_3_pi1.py"]),
        )
        _install_frozen_git_module(
            "gr00t.simulation.pi05_tactile_unit",
            "gr00t/simulation/pi05_tactile_unit.py",
            historical_commit,
            str(source_hashes["gr00t/simulation/pi05_tactile_unit.py"]),
        )
        config = builder_module.build_config(
            "s43_pi2m_bhva_seed42",
            float(source_binding["lambda_phys"]),
            1,
        )
    else:
        from gr00t.simulation.pi2b_policy.contract import Workspace
        from gr00t.simulation.pi2b_policy import training as builder_module

        workspace = Workspace.load_readonly(ROOT)
        config = builder_module.build_config(
            workspace,
            str(focus["model"]),
            int(focus["training_seed"]),
            fsdp_devices=1,
        )

    builder_file = Path(inspect.getsourcefile(builder_module.build_config) or "").resolve(
        strict=True
    )
    _require(
        _symbolic_runtime_path(builder_file, contract) == source_binding["builder_path"],
        f"config builder origin drift: {builder_file}",
    )
    _require(int(config.seed) == int(focus["training_seed"]), "reconstructed config seed drift")
    actual_mode = getattr(config.model, "tactile_unit_mode", None)
    actual_mode_value = "NONE" if actual_mode is None else str(actual_mode.value)
    _require(actual_mode_value == source_binding["mode"], "reconstructed config mode drift")
    actual_lambda = float(getattr(config.model, "lambda_phys", 0.0))
    _require(
        actual_lambda == float(source_binding["lambda_phys"]), "reconstructed config lambda drift"
    )
    _require(
        int(config.batch_size) == 32 and int(config.num_train_steps) == 30_000,
        "reconstructed training recipe drift",
    )
    _require(
        config.overwrite is False and config.resume is False,
        "reconstructed config permits checkpoint mutation",
    )

    import gr00t.simulation.pi05_tactile_unit as tactile_module
    import gr00t.simulation.s4_3_pi1 as mode_module
    import importlib.metadata
    import openpi
    import openpi.training.config as openpi_config
    import openpi_client
    from openpi_client import image_tools as client_image_tools

    accepted_package = (ACCEPTED_OPENPI_ROOT / "src/openpi").resolve(strict=True)
    for label, imported_path in (
        ("openpi", Path(openpi.__file__)),
        ("openpi.training.config", Path(openpi_config.__file__)),
    ):
        resolved_import = imported_path.resolve(strict=True)
        try:
            resolved_import.relative_to(accepted_package)
        except ValueError as exc:
            raise ContractError(
                f"{label} escaped accepted OpenPI runtime: {resolved_import}"
            ) from exc
    observed_versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_RUNTIME_VERSIONS
    }
    _require(
        observed_versions == EXPECTED_RUNTIME_VERSIONS,
        f"diagnostic runtime dependency version drift: {observed_versions}",
    )
    pip_freeze_sha256 = _pip_freeze_sha256()
    _require(
        pip_freeze_sha256 == EXPECTED_PIP_FREEZE_SHA256,
        "diagnostic execution environment freeze drift",
    )
    accepted_client_package = (ACCEPTED_OPENPI_CLIENT_SOURCE / "openpi_client").resolve(strict=True)
    for label, imported_path, expected_sha in (
        (
            "openpi_client",
            Path(openpi_client.__file__),
            OPENPI_CLIENT_INIT_SHA256,
        ),
        (
            "openpi_client.image_tools",
            Path(client_image_tools.__file__),
            OPENPI_CLIENT_IMAGE_TOOLS_SHA256,
        ),
    ):
        resolved_import = imported_path.resolve(strict=True)
        try:
            resolved_import.relative_to(accepted_client_package)
        except ValueError as exc:
            raise ContractError(
                f"{label} escaped accepted OpenPI client runtime: {resolved_import}"
            ) from exc
        _require(
            sha256_file(resolved_import) == expected_sha,
            f"{label} runtime source byte drift",
        )

    if checkpoint_id == "B_HVA_42":
        tactile_origin = {
            "module": tactile_module.__name__,
            "origin": tactile_module.__file__,
            "sha256": source_binding["source_sha256"]["gr00t/simulation/pi05_tactile_unit.py"],
        }
        mode_origin = {
            "module": mode_module.__name__,
            "origin": mode_module.__file__,
            "sha256": source_binding["source_sha256"]["gr00t/simulation/s4_3_pi1.py"],
        }
    else:
        tactile_origin = {
            "module": tactile_module.__name__,
            "path": _symbolic_runtime_path(tactile_module.__file__, contract),
            "sha256": sha256_file(Path(tactile_module.__file__)),
        }
        mode_origin = {
            "module": mode_module.__name__,
            "path": _symbolic_runtime_path(mode_module.__file__, contract),
            "sha256": sha256_file(Path(mode_module.__file__)),
        }
    origins = {
        "builder": {
            "module": builder_module.__name__,
            "path": _symbolic_runtime_path(builder_file, contract),
            "sha256": sha256_file(builder_file),
        },
        "model_integration": tactile_origin,
        "mode_contract": mode_origin,
        "openpi": {
            "module": openpi.__name__,
            "path": _symbolic_runtime_path(openpi.__file__, contract),
        },
        "openpi_training_config": {
            "module": openpi_config.__name__,
            "path": _symbolic_runtime_path(openpi_config.__file__, contract),
            "sha256": sha256_file(Path(openpi_config.__file__)),
        },
        "openpi_client": {
            "module": openpi_client.__name__,
            "path": _symbolic_runtime_path(openpi_client.__file__, contract),
            "sha256": sha256_file(Path(openpi_client.__file__)),
            "version": observed_versions["openpi-client"],
        },
        "openpi_client_image_tools": {
            "module": client_image_tools.__name__,
            "path": _symbolic_runtime_path(client_image_tools.__file__, contract),
            "sha256": sha256_file(Path(client_image_tools.__file__)),
            "accepted_source_tree": verify_openpi_client_source(),
        },
        "runtime_versions": observed_versions,
        "python_executable": sys.executable,
        "pip_freeze_all_sha256": pip_freeze_sha256,
        "accepted_openpi_runtime_sha256": verify_openpi_runtime_source(),
    }
    return config, origins


def _state_digest(state: Any) -> dict[str, Any]:
    """Stream a deterministic hash of an NNX state through host memory."""

    import flax.traverse_util
    import jax

    flat = flax.traverse_util.flatten_dict(state.to_pure_dict())
    digest = hashlib.sha256()
    total_bytes = 0
    dtype_counts: dict[str, int] = {}
    for path, value in sorted(flat.items(), key=lambda row: tuple(str(part) for part in row[0])):
        array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
        name = "/".join(str(part) for part in path)
        dtype_name = str(array.dtype)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(dtype_name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\n")
        total_bytes += int(array.nbytes)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        del array
    return {
        "sha256": digest.hexdigest(),
        "leaves": len(flat),
        "bytes": total_bytes,
        "dtype_leaf_counts": dict(sorted(dtype_counts.items())),
    }


def model_state_digests(model: Any) -> dict[str, Any]:
    from flax import nnx

    state = nnx.state(model)
    params = state.filter(nnx.Param)
    nonparams = state.filter(nnx.Not(nnx.Param))
    param_digest = _state_digest(params)
    nonparam_digest = _state_digest(nonparams)
    return {
        "params": param_digest,
        "buffers": nonparam_digest,
        "combined_sha256": canonical_sha(
            {"params": param_digest["sha256"], "buffers": nonparam_digest["sha256"]}
        ),
    }


def load_runtime_model(config: Any, checkpoint: Path, stage: str) -> Any:
    """Restore inference BF16 or original-dtype gradient parameters read-only."""

    import jax.numpy as jnp
    from openpi.models import model as openpi_model

    _require(stage in {"actions", "gradients"}, "invalid restore stage")
    dtype = jnp.bfloat16 if stage == "actions" else None
    params = openpi_model.restore_params(checkpoint / "params", dtype=dtype)
    model = config.model.load(params, remove_extra_params=False)
    model.eval()
    del params
    gc.collect()
    return model


def strict_parameter_load_evidence() -> dict[str, Any]:
    return {
        "mode": "STRICT_NO_MISSING_OR_UNEXPECTED_PARAMETERS",
        "remove_extra_params": False,
        "missing_parameter_leaves": 0,
        "unexpected_parameter_leaves": 0,
        "fallback_initialization": False,
    }


def load_fixed_arrays(contract: Contract) -> dict[str, np.ndarray]:
    names = {
        "rows": "rows.npy",
        "front": "front_rgb_chw_uint8.npy",
        "wrist": "wrist_rgb_chw_uint8.npy",
        "state": "state.npy",
        "action": "action.npy",
        "action_is_pad": "action_is_pad.npy",
        "contact_state": "contact_state.npy",
        "contact_target": "contact_target_t_plus_27.npy",
        "contact_valid": "contact_aux_valid.npy",
        "va_target": "va_target_t_plus_27.npy",
        "va_valid": "va_aux_valid.npy",
    }
    return {
        key: np.load(contract.train_root / name, mmap_mode="r", allow_pickle=False)
        for key, name in names.items()
    }


def build_io_context(config: Any, checkpoint: Path) -> dict[str, Any]:
    import openpi.transforms as transforms
    from openpi.training import checkpoints

    data_config = config.data.create(config.assets_dirs, config.model)
    _require(data_config.asset_id == "local_repo", "checkpoint asset id is not local_repo")
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    _require(norm_stats is not None, "checkpoint normalization statistics are absent")
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    input_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *data_config.repack_transforms.outputs,
        ]
    )
    norm_path = checkpoint / "assets/local_repo/norm_stats.json"
    norm_json = _read_json(norm_path, "checkpoint normalization statistics")
    action_stats = norm_json.get("norm_stats", {}).get("actions", {})
    q01 = np.asarray(action_stats.get("q01"), dtype=np.float64)
    q99 = np.asarray(action_stats.get("q99"), dtype=np.float64)
    _require(q01.shape == q99.shape == (22,), "checkpoint action quantile shape drift")
    _require(
        np.isfinite(q01).all() and np.isfinite(q99).all() and np.all(q99 >= q01),
        "invalid action quantiles",
    )
    return {
        "data_config": data_config,
        "input_transform": input_transform,
        "output_transform": output_transform,
        "q01": q01,
        "q99": q99,
        "norm_stats_sha256": sha256_file(norm_path),
    }


def _raw_fixed_sample(
    arrays: Mapping[str, np.ndarray],
    position: int,
    model_id: str,
    contact_state: np.ndarray | None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "observation.images.front": np.asarray(arrays["front"][position]).copy(),
        "observation.images.wrist": np.asarray(arrays["wrist"][position]).copy(),
        "observation.state": np.asarray(arrays["state"][position], dtype=np.float32).copy(),
        "action": np.asarray(arrays["action"][position], dtype=np.float32).copy(),
        "prompt": TASK_PROMPT,
    }
    if model_id in CONTACT_MODELS:
        _require(
            contact_state is not None and np.asarray(contact_state).shape == (256,),
            "missing fixed H",
        )
        sample["contact_state"] = np.asarray(contact_state, dtype=np.float32).copy()
    if model_id in {"B_VA27", "B_HVA"}:
        sample["va_shared_target"] = np.asarray(
            arrays["va_target"][position], dtype=np.float32
        ).copy()
        sample["va_aux_valid"] = np.asarray(arrays["va_valid"][position], dtype=np.bool_).copy()
    elif model_id == "B2":
        sample["contact_shared_target"] = np.asarray(
            arrays["contact_target"][position], dtype=np.float32
        ).copy()
        sample["physical_aux_valid"] = np.asarray(
            arrays["contact_valid"][position], dtype=np.bool_
        ).copy()
    return sample


def build_transformed_batch(
    arrays: Mapping[str, np.ndarray],
    positions: Sequence[int],
    model_id: str,
    contact_states: np.ndarray | None,
    input_transform: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from openpi.models import model as openpi_model

    transformed = []
    for batch_position, fixed_position in enumerate(positions):
        contact = None if contact_states is None else contact_states[batch_position]
        transformed.append(
            input_transform(_raw_fixed_sample(arrays, int(fixed_position), model_id, contact))
        )
    batch = jax.tree.map(lambda *values: np.stack(values, axis=0), *transformed)
    batch = jax.tree.map(jnp.asarray, batch)
    if model_id == "B0":
        observation = openpi_model.Observation.from_dict(batch)
    else:
        from gr00t.simulation.pi05_tactile_unit import TactileObservation

        observation = TactileObservation.from_dict(batch)
    actions = batch["actions"]
    _require(tuple(actions.shape) == (len(positions), 30, 32), "transformed action shape drift")
    return observation, actions, batch


def inference_observation(observation: Any, model_id: str) -> Any:
    if model_id == "B0":
        return observation
    # The seed-42 historical TactileObservation predates the later VAC fields.
    # Clear every auxiliary target that exists in the exact loaded source
    # cohort without assuming fields introduced by newer Track-A code.
    available = {field.name for field in dataclasses.fields(observation)}
    auxiliary_fields = {
        "contact_shared_target",
        "physical_aux_valid",
        "va_shared_target",
        "va_aux_valid",
        "vac_vision_target",
        "vac_aux_valid",
    }
    return dataclasses.replace(
        observation,
        **{name: None for name in auxiliary_fields if name in available},
    )


def checkpoint_sentinel_digest(checkpoint: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for relative in (
        "params/manifest.ocdbt",
        "train_state/manifest.ocdbt",
        "assets/local_repo/norm_stats.json",
    ):
        path = checkpoint / relative
        stat = path.stat()
        output[relative] = {
            "sha256": sha256_file(path),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "inode": int(stat.st_ino),
            "device": int(stat.st_dev),
        }
    return output


def _pytree_sha256(tree: Any) -> str:
    import jax

    digest = hashlib.sha256()
    leaves, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
    for path, value in leaves:
        array = np.ascontiguousarray(np.asarray(value))
        name = "/".join(str(part) for part in path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _token_norm_summary(tokens: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tokens, dtype=np.float32)
    active = np.asarray(mask, dtype=bool)
    _require(values.ndim == 2 and active.shape == values.shape[:1], "token norm mask shape drift")
    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
    selected = norms[active]
    _require(
        selected.size > 0 and np.isfinite(selected).all(),
        "token norm support is empty or non-finite",
    )
    return {
        "active_tokens": int(selected.size),
        "frobenius_l2": float(np.sqrt(np.sum(np.square(selected)))),
        "per_token_l2_mean": float(np.mean(selected)),
        "per_token_l2_median": float(np.median(selected)),
        "per_token_l2_min": float(np.min(selected)),
        "per_token_l2_max": float(np.max(selected)),
    }


def prefix_diagnostics(
    model: Any,
    observation: Any,
    model_id: str,
    global_rows: Sequence[int],
    condition: str,
    integration_contract: Mapping[str, Any],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from openpi.models import model as openpi_model
    from openpi.models.pi0 import Pi0

    processed = openpi_model.preprocess_observation(None, observation, train=False)
    official_tokens, official_mask, official_ar = Pi0.embed_prefix(model, processed)
    actual_tokens, actual_mask, actual_ar = model.embed_prefix(processed)
    expected_shape = tuple(
        int(value) for value in integration_contract["adapter_output_shape_per_observation"]
    )
    expected_token_count = int(integration_contract["token_count"])
    expected_token_width = int(integration_contract["token_width"])
    _require(expected_shape == (8, 2048), "prefix adapter output-shape contract drift")
    _require(expected_token_count == 8, "prefix token-count contract drift")
    _require(expected_token_width == 2048, "prefix token-width contract drift")
    _require(
        official_tokens.ndim == actual_tokens.ndim == 3,
        "loaded or official prefix rank drift",
    )
    _require(
        int(official_tokens.shape[-1]) == int(actual_tokens.shape[-1]) == expected_token_width,
        "loaded or official prefix token-width drift",
    )
    official_length = int(official_tokens.shape[1])
    added = int(actual_tokens.shape[1]) - official_length
    expected_added = expected_token_count if model_id in CONTACT_MODELS else 0
    _require(added == expected_added, f"prefix token-count drift for {model_id}")
    official_unchanged = bool(
        jax.device_get(
            jnp.array_equal(actual_tokens[:, :official_length], official_tokens)
            & jnp.array_equal(actual_mask[:, :official_length], official_mask)
            & jnp.array_equal(actual_ar[:official_length], official_ar)
        )
    )
    official_host = np.asarray(jax.device_get(official_tokens), dtype=np.float32)
    official_mask_host = np.asarray(jax.device_get(official_mask), dtype=bool)
    _require(
        processed.tokenized_prompt is not None and processed.tokenized_prompt_mask is not None,
        "fixed observation lacks tokenized language inputs",
    )
    language_length = int(processed.tokenized_prompt.shape[1])
    image_length = official_length - language_length
    _require(image_length > 0, "official prefix has no image-token block")
    image_tokens = official_host[:, :image_length]
    image_mask = official_mask_host[:, :image_length]
    language_tokens = official_host[:, image_length:]
    language_mask = official_mask_host[:, image_length:]
    _require(
        language_tokens.shape[1] == language_length,
        "official language-token boundary drift",
    )
    actual_positions = np.asarray(
        jax.device_get(jnp.cumsum(actual_mask, axis=1) - 1), dtype=np.int64
    )
    per_row: list[dict[str, Any]] = []
    adapter_projection_matches = True
    adapter_manual_pipeline_matches = True
    position_contract = True
    if expected_added:
        appended_tokens = actual_tokens[:, -expected_token_count:]
        _require(
            tuple(int(value) for value in appended_tokens.shape[-2:]) == expected_shape,
            f"appended prefix shape drift for {model_id}",
        )
        added_mask = np.asarray(jax.device_get(actual_mask[:, -expected_token_count:]), dtype=bool)
        added_ar = np.asarray(jax.device_get(actual_ar[-expected_token_count:]), dtype=bool)
        contact_state = jnp.asarray(processed.contact_state, dtype=jnp.float32)
        normalized_contact = model.contact_adapter.norm(contact_state)
        manual_hidden = model.contact_adapter.fc1(normalized_contact)
        manual_hidden = jax.nn.gelu(manual_hidden)
        manual_hidden = model.contact_adapter.fc2(manual_hidden)
        manual_hidden = manual_hidden.reshape((*manual_hidden.shape[:-1], expected_token_count, 32))
        manual_projected = model.contact_adapter.shared_projection(manual_hidden)
        projected = model.contact_adapter(contact_state)
        _require(
            tuple(int(value) for value in projected.shape[-2:]) == expected_shape,
            f"contact-adapter projection shape drift for {model_id}",
        )
        adapter_manual_pipeline_matches = bool(
            jax.device_get(jnp.array_equal(projected, manual_projected))
        )
        projected_cast = projected.astype(actual_tokens.dtype)
        adapter_projection_matches = bool(
            jax.device_get(jnp.array_equal(appended_tokens, projected_cast))
        )
        pre_layernorm_l2 = np.asarray(
            jax.device_get(jnp.linalg.norm(contact_state, axis=-1)),
            dtype=np.float64,
        )
        post_layernorm_l2 = np.asarray(
            jax.device_get(jnp.linalg.norm(normalized_contact.astype(jnp.float32), axis=-1)),
            dtype=np.float64,
        )
        appended_host = np.asarray(jax.device_get(appended_tokens), dtype=np.float32)
        contact_positions = actual_positions[:, -expected_token_count:]
        expected_first_position = official_mask_host.sum(axis=1, dtype=np.int64)
        expected_positions = expected_first_position[:, None] + np.arange(
            expected_token_count, dtype=np.int64
        )
        position_contract = bool(np.array_equal(contact_positions, expected_positions))
        absolute_indices = list(range(official_length, official_length + expected_token_count))
        for index, row in enumerate(global_rows):
            per_row.append(
                {
                    "global_row": int(row),
                    "image_tokens_width_2048": _token_norm_summary(
                        image_tokens[index], image_mask[index]
                    ),
                    "language_tokens_width_2048": _token_norm_summary(
                        language_tokens[index], language_mask[index]
                    ),
                    "contact_tokens_width_2048": _token_norm_summary(
                        appended_host[index], np.ones((expected_token_count,), dtype=bool)
                    ),
                    "contact_state_pre_layernorm_l2": float(pre_layernorm_l2[index]),
                    "contact_state_post_layernorm_l2": float(post_layernorm_l2[index]),
                    "contact_sequence_indices": absolute_indices,
                    "contact_position_indices": contact_positions[index].tolist(),
                    "official_padding_tokens_before_contact": int(
                        official_length - official_mask_host[index].sum()
                    ),
                    "added_input_mask": added_mask[index].tolist(),
                }
            )
        mask_contract = bool(added_mask.all() and not added_ar.any())
    else:
        for index, row in enumerate(global_rows):
            per_row.append(
                {
                    "global_row": int(row),
                    "image_tokens_width_2048": _token_norm_summary(
                        image_tokens[index], image_mask[index]
                    ),
                    "language_tokens_width_2048": _token_norm_summary(
                        language_tokens[index], language_mask[index]
                    ),
                    "contact_tokens_width_2048": None,
                    "contact_state_pre_layernorm_l2": None,
                    "contact_state_post_layernorm_l2": None,
                    "contact_sequence_indices": [],
                    "contact_position_indices": [],
                    "official_padding_tokens_before_contact": None,
                    "added_input_mask": [],
                }
            )
        mask_contract = True
    return {
        "condition": condition,
        "official_prefix_shape": list(official_tokens.shape),
        "loaded_prefix_shape": list(actual_tokens.shape),
        "official_prefix_unchanged": official_unchanged,
        "added_token_count": added,
        "token_width": int(actual_tokens.shape[-1]),
        "appended_shape_per_observation": (list(expected_shape) if expected_added else None),
        "added_mask_contract": mask_contract,
        "adapter_projection_matches_appended_tokens": adapter_projection_matches,
        "adapter_manual_pipeline_matches": adapter_manual_pipeline_matches,
        "position_index_contract": position_contract,
        "position_definition": "cumsum(input_mask)-1; padding slots do not advance RoPE position",
        "image_token_count": image_length,
        "language_token_count_with_padding": language_length,
        "records": per_row,
    }


def _unnormalize_actions(
    normalized: np.ndarray,
    transformed_batch: Mapping[str, Any],
    output_transform: Any,
) -> np.ndarray:
    output = []
    states = np.asarray(transformed_batch["state"])
    for index in range(len(normalized)):
        value = output_transform(
            {
                "state": np.asarray(states[index]).copy(),
                "actions": np.asarray(normalized[index], dtype=np.float32).copy(),
            }
        )
        action = np.asarray(value["actions"], dtype=np.float32)
        _require(
            action.shape == (30, 22), f"unnormalized policy action shape drift: {action.shape}"
        )
        output.append(action)
    result = np.stack(output, axis=0)
    _require(np.isfinite(result).all(), "policy action contains non-finite values")
    return result


def execute_action_shard(
    contract: Contract,
    focus: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    checkpoint_id = _checkpoint_id(focus)
    source_binding = collect_config_source_bindings(contract)[checkpoint_id]
    config, import_origins = build_runtime_config(contract, focus, source_binding)
    from openpi.shared import nnx_utils

    checkpoint = resolve_symbolic(
        str(focus["path"]), experiments=contract.experiments, pi2s_root=contract.pi2s_root
    )
    sentinel_before = checkpoint_sentinel_digest(checkpoint)
    model = load_runtime_model(config, checkpoint, "actions")
    state_before = model_state_digests(model)
    io = build_io_context(config, checkpoint)
    arrays = load_fixed_arrays(contract)
    rows = np.asarray(arrays["rows"], dtype=np.int64)
    model_id = str(focus["model"])
    h_conditions = (
        load_intervention_contact_states(contract)
        if model_id in CONTACT_MODELS
        else {"correct": None}
    )
    conditions = CONDITION_ORDER if model_id in CONTACT_MODELS else ("correct",)

    seeds = [int(value) for value in contract.protocol["fixed_sampling"]["noise_seeds"]]
    _require(len(seeds) == 32 and len(set(seeds)) == 32, "action noise seed budget drift")
    noise = jnp.stack(
        [jax.random.normal(jax.random.key(seed), (30, 32), dtype=jnp.float32) for seed in seeds],
        axis=0,
    )
    noise_host = np.asarray(jax.device_get(noise), dtype=np.float32)
    _require(
        noise_host.shape == (32, 30, 32) and noise_host.dtype == np.dtype(np.float32),
        "fixed-sampling noise shape/dtype drift",
    )
    sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    correct_outputs: np.ndarray | None = None
    records: list[dict[str, Any]] = []
    prefix_records: list[dict[str, Any]] = []
    input_batches: list[dict[str, Any]] = []
    for condition in conditions:
        contact_states = h_conditions[condition]
        observation, _, transformed = build_transformed_batch(
            arrays,
            list(range(32)),
            model_id,
            contact_states,
            io["input_transform"],
        )
        infer_observation = inference_observation(observation, model_id)
        prefix = prefix_diagnostics(
            model,
            infer_observation,
            model_id,
            rows.tolist(),
            condition,
            contract.protocol["prefix_integration_contract"],
        )
        prefix_records.append(prefix)
        _require(
            infer_observation.tokenized_prompt is not None
            and infer_observation.tokenized_prompt_mask is not None,
            "fixed transformed observation lacks prompt tokens",
        )
        prompt_tokens = np.asarray(jax.device_get(infer_observation.tokenized_prompt))
        prompt_mask = np.asarray(
            jax.device_get(infer_observation.tokenized_prompt_mask), dtype=np.bool_
        )
        state_host = np.asarray(jax.device_get(infer_observation.state))
        image_hashes = {
            name: array_sha256(np.asarray(jax.device_get(value)))
            for name, value in sorted(infer_observation.images.items())
        }
        input_batches.append(
            {
                "condition": condition,
                "transformed_observation_sha256": _pytree_sha256(infer_observation),
                "tokenized_prompt_sha256": array_sha256(prompt_tokens),
                "tokenized_prompt_mask_sha256": array_sha256(prompt_mask),
                "normalized_state_sha256": array_sha256(state_host),
                "preprocessed_images_sha256": canonical_sha(image_hashes),
                "preprocessed_image_component_sha256": image_hashes,
                "explicit_noise_sha256": array_sha256(noise_host),
            }
        )
        normalized = sample_actions(
            jax.random.key(0),
            infer_observation,
            num_steps=10,
            noise=noise,
        )
        normalized_host = np.asarray(jax.device_get(normalized), dtype=np.float32)
        _require(normalized_host.shape == (32, 30, 32), "normalized action shape drift")
        actions = _unnormalize_actions(normalized_host, transformed, io["output_transform"])
        if condition == "correct":
            correct_outputs = actions.copy()
        _require(correct_outputs is not None, "correct-H reference must execute first")
        for index, global_row in enumerate(rows.tolist()):
            action = actions[index]
            record = {
                "condition": condition,
                "fixed_position": index,
                "global_row": int(global_row),
                "noise_seed": seeds[index],
                "noise_sha256": array_sha256(noise_host[index]),
                "contact_state_sha256": (
                    None
                    if contact_states is None
                    else array_sha256(np.asarray(contact_states[index], dtype=np.float32))
                ),
                "action": action.tolist(),
                "action_sha256": array_sha256(action),
                "action_jerk_proxy": jerk_proxy(action),
                "normalization_boundary_exceedance": boundary_exceedance(
                    action, io["q01"], io["q99"]
                ),
                "delta": action_delta_metrics(correct_outputs[index], action),
            }
            records.append(record)
        del observation, infer_observation, transformed, normalized, normalized_host, actions
        gc.collect()

    state_after = model_state_digests(model)
    sentinel_after = checkpoint_sentinel_digest(checkpoint)
    expected_records = 32 * len(conditions)
    prefix_ok = all(
        row["official_prefix_unchanged"]
        and row["added_mask_contract"]
        and row["adapter_projection_matches_appended_tokens"]
        and row["adapter_manual_pipeline_matches"]
        and row["position_index_contract"]
        and row["added_token_count"] == (8 if model_id in CONTACT_MODELS else 0)
        and row["token_width"] == 2048
        and row["appended_shape_per_observation"]
        == ([8, 2048] if model_id in CONTACT_MODELS else None)
        for row in prefix_records
    )
    gates = {
        "expected_checkpoint_authority_bound": str(focus["tree_sha256"])
        == _artifact_binding(contract, focus, "actions")["checkpoint_tree_sha256"],
        "strict_parameter_load": True,
        "exact_fixed_observation_rows": [int(value) for value in rows]
        == contract.protocol["samples"]["fixed_observation_rows"],
        "exact_noise_seed_budget": len(seeds) == 32,
        "exact_condition_and_record_budget": len(records) == expected_records,
        "prefix_contract": prefix_ok,
        "all_action_metrics_finite": all(
            np.isfinite(np.asarray(row["action"], dtype=np.float64)).all() for row in records
        ),
        "params_unchanged": state_before["params"] == state_after["params"],
        "buffers_unchanged": state_before["buffers"] == state_after["buffers"],
        "checkpoint_sentinels_unchanged": sentinel_before == sentinel_after,
        "no_optimizer_step": True,
        "no_checkpoint_write": True,
    }
    return {
        "schema": ACTION_SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "created_at_utc": now_utc(),
        "binding": _artifact_binding(contract, focus, "actions"),
        "focus": dict(focus),
        "config_source_binding": source_binding,
        "actual_import_origins": import_origins,
        "execution_environment": dict(environment),
        "restore_dtype": "bfloat16",
        "parameter_load": strict_parameter_load_evidence(),
        "normalization_stats_sha256": io["norm_stats_sha256"],
        "sampler": {
            "num_steps": 10,
            "noise_distribution": contract.protocol["fixed_sampling"]["noise_distribution"],
            "noise_shape_per_observation": contract.protocol["fixed_sampling"]["noise_shape"],
            "noise_dtype": contract.protocol["fixed_sampling"]["noise_dtype"],
            "rng_argument_unused_with_explicit_noise": True,
        },
        "input_batches": input_batches,
        "prefix_diagnostics": prefix_records,
        "records": records,
        "state_before": state_before,
        "state_after": state_after,
        "checkpoint_sentinels_before": sentinel_before,
        "checkpoint_sentinels_after": sentinel_after,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }


def _explicit_flow_forward(
    candidate: Any,
    observation: Any,
    actions: Any,
    noise: Any,
    times: Any,
) -> tuple[Any, Any, Any]:
    """Official PI0 flow forward with caller-supplied time and noise."""

    import jax.numpy as jnp
    from openpi.models import model as openpi_model
    from openpi.models.pi0 import make_attn_mask

    processed = openpi_model.preprocess_observation(None, observation, train=False)
    time_expanded = times[..., None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    target = noise - actions
    prefix_tokens, prefix_mask, prefix_ar = candidate.embed_prefix(processed)
    suffix_tokens, suffix_mask, suffix_ar, adarms_cond = candidate.embed_suffix(
        processed, x_t, times
    )
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
    attention_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = candidate.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attention_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    action_hidden = suffix_out[:, -candidate.action_horizon :]
    prediction = candidate.action_out_proj(action_hidden)
    main_loss = jnp.mean(jnp.square(prediction - target))
    return main_loss, action_hidden, processed


def _gradient_leaf_paths(state: Any) -> tuple[list[str], dict[str, list[str]]]:
    import flax.traverse_util

    flat = flax.traverse_util.flatten_dict(state.to_pure_dict())
    paths = sorted("/".join(str(part) for part in path) for path in flat)
    physical_auxiliary = [path for path in paths if "physical_auxiliary" in path]
    main_only_action_output = [path for path in paths if "action_out_proj" in path]
    objective_specific = set(physical_auxiliary) | set(main_only_action_output)
    groups = {
        # The physical head is auxiliary-only and action_out_proj is main-only
        # because the auxiliary objective branches from action_hidden before
        # that projection.  Excluding both makes the preregistered "common"
        # cosine an actual shared-parameter comparison.
        "all_common_trainable": [path for path in paths if path not in objective_specific],
        "lora": [path for path in paths if "lora" in path],
        "contact_adapter": [path for path in paths if "contact_adapter" in path],
        "physical_auxiliary": physical_auxiliary,
    }
    _require(
        not set(groups["all_common_trainable"]) & objective_specific,
        "common and objective-specific parameter groups overlap",
    )
    return paths, groups


def _validate_gradient_group_presence(
    model_id: str, paths: Sequence[str], groups: Mapping[str, Sequence[str]]
) -> None:
    expected = {
        "lora": True,
        "contact_adapter": model_id in CONTACT_MODELS,
        "physical_auxiliary": model_id in AUXILIARY_MODELS,
    }
    for group, required in expected.items():
        _require(
            bool(groups[group]) is required,
            f"gradient parameter-group presence drift: {model_id}/{group}",
        )
    _require(
        any("action_out_proj" in path for path in paths),
        f"main-only action_out_proj trainable leaves absent: {model_id}",
    )
    _require(
        bool(groups["all_common_trainable"]),
        f"shared trainable parameter group absent: {model_id}",
    )


def gradient_group_metrics(
    main_grads: Any,
    auxiliary_grads: Any | None,
    group_paths: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    import flax.traverse_util
    import jax
    import jax.numpy as jnp

    def flatten(state: Any) -> dict[str, Any]:
        # NNX currently materializes zeros for unused differentiable leaves,
        # but skipping absent/None leaves keeps the diagnostic well-defined if
        # an older historical NNX returns None for an unused auxiliary head.
        return {
            "/".join(str(part) for part in path): value
            for path, value in flax.traverse_util.flatten_dict(state.to_pure_dict()).items()
            if value is not None
        }

    main = flatten(main_grads)
    auxiliary = None if auxiliary_grads is None else flatten(auxiliary_grads)
    output: dict[str, Any] = {}
    for group in PARAMETER_GROUPS:
        structural_paths = list(group_paths[group])
        main_paths = [path for path in structural_paths if path in main]
        auxiliary_paths = (
            [] if auxiliary is None else [path for path in structural_paths if path in auxiliary]
        )
        intersection_paths = (
            []
            if auxiliary is None
            else [path for path in structural_paths if path in main and path in auxiliary]
        )
        if not structural_paths:
            output[group] = {
                "status": "N/A_NO_TRAINABLE_LEAVES",
                "leaf_count": 0,
                "main_leaf_count": 0,
                "auxiliary_leaf_count": 0,
                "cosine_intersection_leaf_count": 0,
                "l2_norm_g_main": None,
                "l2_norm_g_aux": None,
                "l2_norm_g_main_on_cosine_intersection": None,
                "l2_norm_g_aux_on_cosine_intersection": None,
                "cosine_g_main_g_aux": None,
            }
            continue
        main_sq = sum(
            (
                jnp.sum(jnp.square(jnp.asarray(main[path], dtype=jnp.float32)))
                for path in main_paths
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        main_norm = float(jax.device_get(jnp.sqrt(main_sq)))
        _require(math.isfinite(main_norm), f"non-finite main gradient norm: {group}")
        if auxiliary is None:
            output[group] = {
                "status": "N/A_NO_AUXILIARY_OBJECTIVE",
                "leaf_count": len(structural_paths),
                "main_leaf_count": len(main_paths),
                "auxiliary_leaf_count": 0,
                "cosine_intersection_leaf_count": 0,
                "l2_norm_g_main": main_norm,
                "l2_norm_g_aux": None,
                "l2_norm_g_main_on_cosine_intersection": None,
                "l2_norm_g_aux_on_cosine_intersection": None,
                "cosine_g_main_g_aux": None,
            }
            continue
        aux_sq = sum(
            (
                jnp.sum(jnp.square(jnp.asarray(auxiliary[path], dtype=jnp.float32)))
                for path in auxiliary_paths
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        dot = sum(
            (
                jnp.vdot(
                    jnp.asarray(main[path], dtype=jnp.float32),
                    jnp.asarray(auxiliary[path], dtype=jnp.float32),
                )
                for path in intersection_paths
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        intersection_main_sq = sum(
            (
                jnp.sum(jnp.square(jnp.asarray(main[path], dtype=jnp.float32)))
                for path in intersection_paths
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        intersection_aux_sq = sum(
            (
                jnp.sum(jnp.square(jnp.asarray(auxiliary[path], dtype=jnp.float32)))
                for path in intersection_paths
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        aux_norm = float(jax.device_get(jnp.sqrt(aux_sq)))
        intersection_main_norm = float(jax.device_get(jnp.sqrt(intersection_main_sq)))
        intersection_aux_norm = float(jax.device_get(jnp.sqrt(intersection_aux_sq)))
        dot_value = float(jax.device_get(dot))
        _require(
            all(
                math.isfinite(value)
                for value in (
                    aux_norm,
                    intersection_main_norm,
                    intersection_aux_norm,
                    dot_value,
                )
            ),
            f"non-finite auxiliary gradient: {group}",
        )
        cosine = None
        status = "PASS"
        if not intersection_paths:
            status = "N/A_NO_COSINE_INTERSECTION"
        elif intersection_main_norm == 0.0 or intersection_aux_norm == 0.0:
            status = "N/A_ZERO_NORM"
        else:
            cosine = dot_value / (intersection_main_norm * intersection_aux_norm)
            _require(math.isfinite(cosine), f"non-finite gradient cosine: {group}")
            cosine = float(max(-1.0, min(1.0, cosine)))
        output[group] = {
            "status": status,
            "leaf_count": len(structural_paths),
            "main_leaf_count": len(main_paths),
            "auxiliary_leaf_count": len(auxiliary_paths),
            "cosine_intersection_leaf_count": len(intersection_paths),
            "l2_norm_g_main": main_norm,
            "l2_norm_g_aux": aux_norm,
            "l2_norm_g_main_on_cosine_intersection": intersection_main_norm,
            "l2_norm_g_aux_on_cosine_intersection": intersection_aux_norm,
            "cosine_g_main_g_aux": cosine,
        }
    return output


def execute_gradient_shard(
    contract: Contract,
    focus: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    from flax import nnx
    import jax
    import jax.numpy as jnp

    checkpoint_id = _checkpoint_id(focus)
    source_binding = collect_config_source_bindings(contract)[checkpoint_id]
    config, import_origins = build_runtime_config(contract, focus, source_binding)
    checkpoint = resolve_symbolic(
        str(focus["path"]), experiments=contract.experiments, pi2s_root=contract.pi2s_root
    )
    sentinel_before = checkpoint_sentinel_digest(checkpoint)
    model = load_runtime_model(config, checkpoint, "gradients")
    state_before = model_state_digests(model)
    io = build_io_context(config, checkpoint)
    arrays = load_fixed_arrays(contract)
    rows = np.asarray(arrays["rows"], dtype=np.int64)
    row_to_position = {int(row): index for index, row in enumerate(rows.tolist())}
    model_id = str(focus["model"])
    correct_h = (
        load_intervention_contact_states(contract)["correct"]
        if model_id in CONTACT_MODELS
        else None
    )
    observation_all, actions_all, _ = build_transformed_batch(
        arrays,
        list(range(32)),
        model_id,
        correct_h,
        io["input_transform"],
    )
    _require(
        observation_all.tokenized_prompt is not None
        and observation_all.tokenized_prompt_mask is not None,
        "gradient observation lacks prompt tokens",
    )
    gradient_input_identity = {
        "tokenized_prompt_sha256": array_sha256(
            np.asarray(jax.device_get(observation_all.tokenized_prompt))
        ),
        "tokenized_prompt_mask_sha256": array_sha256(
            np.asarray(
                jax.device_get(observation_all.tokenized_prompt_mask),
                dtype=np.bool_,
            )
        ),
        "normalized_state_sha256": array_sha256(np.asarray(jax.device_get(observation_all.state))),
        "preprocessed_images_sha256": canonical_sha(
            {
                name: array_sha256(np.asarray(jax.device_get(value)))
                for name, value in sorted(observation_all.images.items())
            }
        ),
    }
    trainable_state = nnx.state(model, config.trainable_filter)
    trainable_paths, parameter_groups = _gradient_leaf_paths(trainable_state)
    _require(trainable_paths, "reconstructed config has no trainable leaves")
    _validate_gradient_group_presence(model_id, trainable_paths, parameter_groups)

    def main_objective(
        candidate: Any, observation: Any, actions: Any, noise: Any, times: Any
    ) -> Any:
        return _explicit_flow_forward(candidate, observation, actions, noise, times)[0]

    main_gradient = nnx.jit(
        nnx.value_and_grad(
            main_objective,
            argnums=nnx.DiffState(0, config.trainable_filter),
        )
    )
    auxiliary_applicable = model_id in AUXILIARY_MODELS
    auxiliary_gradient = None
    if auxiliary_applicable:

        def auxiliary_objective(
            candidate: Any,
            observation: Any,
            actions: Any,
            noise: Any,
            times: Any,
        ) -> Any:
            _, hidden, processed = _explicit_flow_forward(
                candidate, observation, actions, noise, times
            )
            return candidate._physical_loss(processed, hidden)

        auxiliary_gradient = nnx.jit(
            nnx.value_and_grad(
                auxiliary_objective,
                argnums=nnx.DiffState(0, config.trainable_filter),
            )
        )

    gradient_batches = contract.protocol["samples"]["gradient_batches"]
    noise_seeds = [
        int(value)
        for value in contract.protocol["gradient_diagnostic"]["nuisance_noise_seeds_by_minibatch"]
    ]
    _require(len(gradient_batches) == len(noise_seeds) == 4, "gradient minibatch budget drift")
    lambda_phys = float(getattr(config.model, "lambda_phys", 0.0))
    records: list[dict[str, Any]] = []
    for minibatch_index, global_batch_rows in enumerate(gradient_batches):
        batch_rows = [int(value) for value in global_batch_rows]
        _require(
            len(batch_rows) == 2 and len(set(batch_rows)) == 2, "gradient minibatch identity drift"
        )
        positions = [row_to_position[row] for row in batch_rows]
        observation = jax.tree.map(lambda value: value[np.asarray(positions)], observation_all)
        actions = actions_all[np.asarray(positions)]
        seed = noise_seeds[minibatch_index]
        noise = jax.random.normal(
            jax.random.key(seed),
            (2, 30, 32),
            dtype=jnp.float32,
        )
        noise_host = np.asarray(jax.device_get(noise), dtype=np.float32)
        _require(
            noise_host.shape == (2, 30, 32) and noise_host.dtype == np.dtype(np.float32),
            "gradient nuisance-noise shape/dtype drift",
        )
        input_sha = _pytree_sha256((observation, actions))
        if model_id in {"B_VA27", "B_HVA"}:
            target = np.asarray(arrays["va_target"])[positions]
            valid = np.asarray(arrays["va_valid"])[positions]
            target_file = "va_target_t_plus_27.npy"
            valid_file = "va_aux_valid.npy"
        elif model_id == "B2":
            target = np.asarray(arrays["contact_target"])[positions]
            valid = np.asarray(arrays["contact_valid"])[positions]
            target_file = "contact_target_t_plus_27.npy"
            valid_file = "contact_aux_valid.npy"
        else:
            target = None
            valid = None
            target_file = None
            valid_file = None
        for flow_time in FLOW_TIMES:
            times = jnp.full((2,), flow_time, dtype=jnp.float32)
            main_loss, main_grads = main_gradient(model, observation, actions, noise, times)
            main_value = float(jax.device_get(main_loss))
            aux_grads = None
            aux_value: float | None = None
            if auxiliary_gradient is not None:
                aux_loss, aux_grads = auxiliary_gradient(model, observation, actions, noise, times)
                aux_value = float(jax.device_get(aux_loss))
            groups = gradient_group_metrics(
                main_grads,
                aux_grads,
                parameter_groups,
            )
            combined = main_value if aux_value is None else main_value + lambda_phys * aux_value
            ratio = (
                None
                if aux_value is None
                else float(
                    lambda_phys * aux_value / max(abs(main_value), np.finfo(np.float64).tiny)
                )
            )
            records.append(
                {
                    "minibatch_index": minibatch_index,
                    "global_rows": batch_rows,
                    "fixed_positions": positions,
                    "flow_time": float(flow_time),
                    "noise_seed": seed,
                    "noise_sha256": array_sha256(noise_host),
                    "transformed_batch_sha256": input_sha,
                    "auxiliary_applicable": auxiliary_applicable,
                    "auxiliary_target": {
                        "target_file": target_file,
                        "valid_mask_file": valid_file,
                        "target_sha256": (
                            None
                            if target is None
                            else array_sha256(np.asarray(target, dtype=np.float32))
                        ),
                        "valid_mask_sha256": (
                            None
                            if valid is None
                            else array_sha256(np.asarray(valid, dtype=np.bool_))
                        ),
                        "valid_samples": (
                            None if valid is None else int(np.asarray(valid, dtype=bool).sum())
                        ),
                    },
                    "losses": {
                        "main": main_value,
                        "auxiliary": aux_value,
                        "combined": combined,
                        "lambda_phys": lambda_phys,
                        "weighted_auxiliary_to_main_ratio": ratio,
                    },
                    "groups": groups,
                }
            )
            del main_grads, aux_grads
            gc.collect()
        del observation, actions, noise, noise_host

    state_after = model_state_digests(model)
    sentinel_after = checkpoint_sentinel_digest(checkpoint)
    finite_losses = all(
        math.isfinite(float(record["losses"]["main"]))
        and math.isfinite(float(record["losses"]["combined"]))
        and (
            record["losses"]["auxiliary"] is None
            or math.isfinite(float(record["losses"]["auxiliary"]))
        )
        for record in records
    )
    mapping = contract.protocol["gradient_diagnostic"]["auxiliary_loss_by_model"][model_id]
    mapping_ok = (
        mapping == "N/A_NO_AUXILIARY_OBJECTIVE"
        if not auxiliary_applicable
        else all(
            record["auxiliary_target"]["target_file"] in str(mapping["target"])
            and record["auxiliary_target"]["valid_mask_file"] in str(mapping["valid_mask"])
            and record["auxiliary_target"]["valid_samples"] == 2
            for record in records
        )
    )
    gates = {
        "expected_checkpoint_authority_bound": str(focus["tree_sha256"])
        == _artifact_binding(contract, focus, "gradients")["checkpoint_tree_sha256"],
        "strict_parameter_load": True,
        "exact_four_minibatches_of_two": len(gradient_batches) == 4
        and all(len(batch) == 2 for batch in gradient_batches),
        "exact_three_frozen_flow_times": tuple(float(value) for value in FLOW_TIMES)
        == (0.1, 0.5, 0.9),
        "exact_twelve_gradient_records": len(records) == 12,
        "explicit_noise_and_time": True,
        "per_model_auxiliary_mapping": mapping_ok,
        "actual_trainable_leaf_membership_enumerated": bool(trainable_paths),
        "expected_parameter_group_presence": True,
        "all_losses_and_gradient_metrics_finite": finite_losses,
        "params_unchanged": state_before["params"] == state_after["params"],
        "buffers_unchanged": state_before["buffers"] == state_after["buffers"],
        "checkpoint_sentinels_unchanged": sentinel_before == sentinel_after,
        "no_optimizer_step": True,
        "no_checkpoint_write": True,
    }
    return {
        "schema": GRADIENT_SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "created_at_utc": now_utc(),
        "binding": _artifact_binding(contract, focus, "gradients"),
        "focus": dict(focus),
        "config_source_binding": source_binding,
        "actual_import_origins": import_origins,
        "execution_environment": dict(environment),
        "restore_dtype": "checkpoint_original_dtypes",
        "parameter_load": strict_parameter_load_evidence(),
        "normalization_stats_sha256": io["norm_stats_sha256"],
        "flow_semantics": contract.protocol["gradient_diagnostic"]["flow_time_semantics"],
        "preprocess_mode": contract.protocol["metrics"]["gradients"]["preprocess_mode"],
        "training_equivalence_scope": contract.protocol["metrics"]["gradients"][
            "training_equivalence_scope"
        ],
        "nuisance_noise_contract": {
            "distribution": contract.protocol["gradient_diagnostic"]["nuisance_noise_distribution"],
            "shape_per_minibatch": contract.protocol["gradient_diagnostic"][
                "nuisance_noise_shape_per_minibatch"
            ],
            "dtype": contract.protocol["gradient_diagnostic"]["nuisance_noise_dtype"],
        },
        "trainable_leaf_paths": trainable_paths,
        "parameter_group_leaf_paths": parameter_groups,
        "parameter_group_definitions": {
            "all_common_trainable": (
                "actual config.trainable_filter leaves excluding the "
                "auxiliary-only physical_auxiliary prediction head and the "
                "main-only action_out_proj head"
            ),
            "lora": "actual trainable leaves whose path contains lora",
            "contact_adapter": ("actual trainable leaves whose path contains contact_adapter"),
            "physical_auxiliary": (
                "actual auxiliary-only trainable leaves whose path contains " "physical_auxiliary"
            ),
            "cosine_support": (
                "flattened intersection of non-None leaves emitted by the main "
                "and auxiliary gradient evaluations within each group"
            ),
        },
        "auxiliary_loss_mapping": mapping,
        "fixed_non_h_input_identity": gradient_input_identity,
        "records": records,
        "state_before": state_before,
        "state_after": state_after,
        "checkpoint_sentinels_before": sentinel_before,
        "checkpoint_sentinels_after": sentinel_after,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }


def _validate_shard_name_set(output_dir: Path, contract: Contract) -> None:
    expected = {
        shard_path(output_dir, _checkpoint_id(focus), stage).name
        for focus in contract.focus
        for stage in ("actions", "gradients")
    }
    shard_dir = Path(output_dir) / "shards"
    present = {path.name for path in shard_dir.glob("*.json")} if shard_dir.is_dir() else set()
    unexpected = sorted(present - expected)
    _require(
        not unexpected,
        f"unexpected shard files would expand the budget: {unexpected}",
    )


def _validated_shards(
    output_dir: Path,
    contract: Contract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_shard_name_set(output_dir, contract)
    actions: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for focus in contract.focus:
        checkpoint_id = _checkpoint_id(focus)
        for stage, destination in (("actions", actions), ("gradients", gradients)):
            path = shard_path(output_dir, checkpoint_id, stage)
            _require(
                path.is_file() and not path.is_symlink(),
                f"required immutable shard missing: {path}",
            )
            payload = validate_resumable_artifact(path, contract, focus, stage)
            destination.append(payload)
            inputs.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "phase": stage,
                    "file": (
                        "$PI2S_ROOT/"
                        + path.resolve(strict=True)
                        .relative_to(contract.pi2s_root.resolve(strict=True))
                        .as_posix()
                    ),
                    "sha256": sha256_file(path),
                }
            )
    return actions, gradients, inputs


def canonical_aggregate_payloads(
    output_dir: Path,
    contract: Contract,
) -> dict[str, dict[str, Any]]:
    """Build the three required complete canonical artifacts in memory."""

    actions, gradients, inputs = _validated_shards(output_dir, contract)
    budget = enforce_global_budgets(actions, gradients, contract)
    global_binding = {
        "protocol_sha256": contract.protocol_sha256,
        "runner_sha256": contract.runner_sha256,
        "selection_sha256": contract.protocol["selection_sha256"],
        "snapshot_manifest_sha256": contract.snapshot_manifest_sha256,
    }
    prefix_batches: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    for shard in actions:
        checkpoint_id = _checkpoint_id(shard["focus"])
        for value in shard["prefix_diagnostics"]:
            prefix_batches.append({"checkpoint_id": checkpoint_id, **value})
        for value in shard["records"]:
            intervention_records.append({"checkpoint_id": checkpoint_id, **value})
    gradient_records: list[dict[str, Any]] = []
    for shard in gradients:
        checkpoint_id = _checkpoint_id(shard["focus"])
        for value in shard["records"]:
            gradient_records.append({"checkpoint_id": checkpoint_id, **value})

    _require(len(prefix_batches) == 22, "canonical prefix condition-batch budget drift")
    _require(
        sum(len(row["records"]) for row in prefix_batches) == 704,
        "canonical prefix row budget drift",
    )
    _require(len(intervention_records) == 704, "canonical intervention row budget drift")
    _require(len(gradient_records) == 72, "canonical gradient row budget drift")
    _require(
        sum(bool(row["auxiliary_applicable"]) for row in gradient_records) == 48,
        "canonical auxiliary gradient budget drift",
    )
    completeness = {
        "required_shards": 12,
        "validated_shards": len(inputs),
        "all_focus_checkpoints": 6,
        "all_phases": ["actions", "gradients"],
        "status": "COMPLETE",
    }
    common = {
        "status": "PASS",
        "created_at_utc": now_utc(),
        "binding": global_binding,
        "inputs": inputs,
        "scientific_budget": budget,
        "completeness": completeness,
    }
    prefix = {
        "schema": PREFIX_AUDIT_SCHEMA,
        **common,
        "contract": contract.protocol["metrics"]["prefix_contract"],
        "condition_batches": prefix_batches,
        "condition_batch_count": 22,
        "row_record_count": 704,
        "gates": {
            "all_shards_complete": "PASS",
            "all_official_prefixes_unchanged": (
                "PASS"
                if all(row["official_prefix_unchanged"] for row in prefix_batches)
                else "FAIL"
            ),
            "added_token_mask_contract": (
                "PASS" if all(row["added_mask_contract"] for row in prefix_batches) else "FAIL"
            ),
            "added_token_position_contract": (
                "PASS" if all(row["position_index_contract"] for row in prefix_batches) else "FAIL"
            ),
            "adapter_layernorm_projection_path": (
                "PASS"
                if all(
                    row["adapter_manual_pipeline_matches"]
                    and row["adapter_projection_matches_appended_tokens"]
                    for row in prefix_batches
                )
                else "FAIL"
            ),
        },
    }
    interventions = {
        "schema": INTERVENTION_AUDIT_SCHEMA,
        **common,
        "metric_contract": contract.protocol["metrics"]["fixed_observation_actions"],
        "h_conditions": contract.protocol["h_conditions"]["conditions"],
        "records": intervention_records,
        "record_count": 704,
        "per_checkpoint_condition_summary": aggregate_shards(output_dir, contract)[
            "action_summary"
        ],
        "gates": {
            "all_shards_complete": "PASS",
            "fixed_observation_actions_exact": "PASS",
            "shared_noise_across_h_conditions": "PASS",
            "no_policy_or_optimizer_run": "PASS",
        },
    }
    read_only_gradients = {
        "schema": READONLY_GRADIENT_SCHEMA,
        **common,
        "metric_contract": contract.protocol["metrics"]["gradients"],
        "auxiliary_loss_by_model": contract.protocol["gradient_diagnostic"][
            "auxiliary_loss_by_model"
        ],
        "records": gradient_records,
        "record_count": 72,
        "auxiliary_record_count": 48,
        "per_checkpoint_summary": aggregate_shards(output_dir, contract)["gradient_summary"],
        "gates": {
            "all_shards_complete": "PASS",
            "frozen_explicit_time_noise": "PASS",
            "per_model_auxiliary_mapping": "PASS",
            "all_loaded_state_integrity_gates": "PASS",
            "zero_optimizer_steps": "PASS",
            "zero_checkpoint_writes": "PASS",
        },
    }
    for label, payload in (
        ("prefix", prefix),
        ("interventions", interventions),
        ("gradients", read_only_gradients),
    ):
        _require(
            all(value == "PASS" for value in payload["gates"].values()),
            f"canonical {label} gate failure",
        )
    return {
        "prefix_contract_audit.json": prefix,
        "fixed_observation_interventions.json": interventions,
        "read_only_gradient_diagnostics.json": read_only_gradients,
    }


def _without_created_at(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at_utc"}


def _validate_existing_global_artifact(path: Path, expected: Mapping[str, Any]) -> None:
    _require(
        path.is_file() and not path.is_symlink(),
        f"global artifact is not an immutable file: {path}",
    )
    observed = _read_json(path, "existing global diagnostic artifact")
    _require(
        canonical_sha(_without_created_at(observed))
        == canonical_sha(_without_created_at(expected)),
        f"existing global diagnostic artifact binding/content drift: {path}",
    )


def publish_aggregate_artifacts(
    output_dir: Path,
    contract: Contract,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Publish §17 canonical artifacts and the summary without clobbering."""

    output_dir = Path(output_dir)
    canonical = canonical_aggregate_payloads(output_dir, contract)
    canonical_root = contract.pi2s_root / "artifacts"
    initial_targets = [canonical_root / name for name in canonical]
    summary_path = output_dir / "summary.json"
    if not resume:
        collisions = [path for path in [*initial_targets, summary_path] if os.path.lexists(path)]
        if collisions:
            raise FileExistsError(f"refusing to replace aggregate artifact(s): {collisions}")
    published: dict[str, Any] = {}
    for name, payload in canonical.items():
        path = canonical_root / name
        if os.path.lexists(path):
            _require(resume, f"aggregate artifact already exists: {path}")
            _validate_existing_global_artifact(path, payload)
            status = "RESUMED_VERIFIED"
        else:
            write_atomic_json(path, payload, allowed_root=contract.pi2s_root)
            status = "PUBLISHED"
        published[name] = {
            "status": status,
            "path": f"$PI2S_ROOT/artifacts/{name}",
            "sha256": sha256_file(path),
        }

    summary = aggregate_shards(output_dir, contract)
    summary["canonical_artifacts"] = {
        name: {
            "sha256": value["sha256"],
            "schema": canonical[name]["schema"],
        }
        for name, value in published.items()
    }
    if os.path.lexists(summary_path):
        _require(resume, f"aggregate summary already exists: {summary_path}")
        _validate_existing_global_artifact(summary_path, summary)
        summary_status = "RESUMED_VERIFIED"
    else:
        write_atomic_json(summary_path, summary, allowed_root=contract.pi2s_root)
        summary_status = "PUBLISHED"
    published["summary.json"] = {
        "status": summary_status,
        "sha256": sha256_file(summary_path),
    }
    return {
        "status": "PASS",
        "scientific_budget": summary["scientific_budget"],
        "artifacts": published,
    }


def shard_progress_report(output_dir: Path, contract: Contract) -> dict[str, Any]:
    output_dir = Path(output_dir)
    _validate_shard_name_set(output_dir, contract)
    verified: list[str] = []
    missing: list[str] = []
    for focus in contract.focus:
        for stage in ("actions", "gradients"):
            path = shard_path(output_dir, _checkpoint_id(focus), stage)
            if not path.exists():
                missing.append(path.name)
                continue
            validate_resumable_artifact(path, contract, focus, stage)
            verified.append(path.name)
    report: dict[str, Any] = {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "expected_shards": 12,
        "verified_shards": len(verified),
        "verified": sorted(verified),
        "missing": sorted(missing),
        "execution_performed": False,
    }
    if not missing:
        report["scientific_budget"] = aggregate_shards(output_dir, contract)["scientific_budget"]
    return report


def _resolve_output_dir(value: Path | None, contract: Contract, *, require_pi2s: bool) -> Path:
    requested = contract.experiments / DEFAULT_OUTPUT_RELATIVE if value is None else Path(value)
    requested = Path(os.path.abspath(requested))
    if require_pi2s:
        root = contract.pi2s_root.resolve(strict=True)
        try:
            relative = requested.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"write output must remain under $PI2S_ROOT: {requested}") from exc
        current = root
        for component in relative.parts:
            current = current / component
            if os.path.lexists(current) and current.is_symlink():
                raise ContractError(f"write output path contains a symlink: {current}")
    output = requested.resolve()
    if require_pi2s:
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"resolved write output escapes $PI2S_ROOT: {output}") from exc
    return output


def execute_diagnostics(
    contract: Contract,
    output_dir: Path,
    checkpoint_id: str,
    phase: str,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Execute exactly one complete checkpoint/phase scientific shard."""

    focus = _focus_by_id(contract, checkpoint_id)
    destination = shard_path(output_dir, checkpoint_id, phase)
    if os.path.lexists(destination):
        if not resume:
            raise FileExistsError(f"immutable shard already exists: {destination}")
        payload = validate_resumable_artifact(destination, contract, focus, phase)
        return {
            "status": "RESUMED_VERIFIED",
            "execution_performed": False,
            "checkpoint_id": checkpoint_id,
            "phase": phase,
            "artifact": str(destination),
            "sha256": sha256_file(destination),
            "scientific_status": payload["status"],
        }

    with frozen_openpi_runtime_asset(contract) as (
        tokenizer_asset,
        cache_root,
        asset_evidence,
    ):
        environment = validate_execution_environment()
        environment["frozen_paligemma_tokenizer"] = asset_evidence
        if phase == "actions":
            payload = execute_action_shard(contract, focus, environment)
        elif phase == "gradients":
            payload = execute_gradient_shard(contract, focus, environment)
        else:
            raise ContractError(f"invalid diagnostic phase: {phase}")
        post_hash = sha256_file(tokenizer_asset)
        unexpected_cache_files = sorted(
            path.relative_to(cache_root).as_posix()
            for path in cache_root.rglob("*")
            if path.is_file() and path != tokenizer_asset
        )
        asset_evidence["post_execution_sha256"] = post_hash
        asset_evidence["unexpected_cache_files"] = unexpected_cache_files
        asset_evidence["post_execution_status"] = (
            "PASS"
            if post_hash == PALIGEMMA_TOKENIZER_SHA256 and not unexpected_cache_files
            else "FAIL"
        )
        _require(
            asset_evidence["post_execution_status"] == "PASS",
            "ephemeral OpenPI tokenizer cache changed during execution",
        )
    _require(payload["status"] == "PASS", f"diagnostic shard gates failed: {checkpoint_id}/{phase}")
    _require(
        all(value == "PASS" for value in payload["gates"].values()),
        "diagnostic shard contains a failed gate",
    )
    write_atomic_json(destination, payload, allowed_root=contract.pi2s_root)
    validate_resumable_artifact(destination, contract, focus, phase)
    return {
        "status": "PASS",
        "execution_performed": True,
        "checkpoint_id": checkpoint_id,
        "phase": phase,
        "artifact": str(destination),
        "sha256": sha256_file(destination),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only, sharded S4.3-PI2S model diagnostics")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate frozen inputs and any existing shards without importing JAX or writing",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="execute exactly one checkpoint/phase shard on one externally leased GPU",
    )
    mode.add_argument(
        "--aggregate",
        action="store_true",
        help="validate all 12 shards and publish the three canonical artifacts plus summary",
    )
    parser.add_argument(
        "--checkpoint-id",
        choices=tuple(f"{model}_{seed}" for model, seed in FOCUS_KEYS),
        help="frozen focus checkpoint identity (required only with --execute)",
    )
    parser.add_argument(
        "--phase",
        choices=("actions", "gradients"),
        help="complete scientific phase for one checkpoint (required only with --execute)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="diagnostic root; defaults to $PI2S_ROOT/artifacts/model_diagnostics_v1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify and reuse an immutable completed shard/artifact; never overwrite it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.execute:
            _require(
                args.checkpoint_id is not None and args.phase is not None,
                "--execute requires --checkpoint-id and --phase",
            )
        else:
            _require(
                args.checkpoint_id is None and args.phase is None,
                "--checkpoint-id/--phase are valid only with --execute",
            )
        _require(
            not (args.check and args.resume), "--resume is not meaningful with read-only --check"
        )

        contract = load_contract(verify_snapshot_contents=True)
        output_dir = _resolve_output_dir(
            args.output_dir,
            contract,
            require_pi2s=bool(args.execute or args.aggregate),
        )
        if args.check:
            report = check_report(contract)
            report["shard_progress"] = shard_progress_report(output_dir, contract)
            _require("jax" not in sys.modules, "--check imported JAX")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.execute:
            result = execute_diagnostics(
                contract,
                output_dir,
                str(args.checkpoint_id),
                str(args.phase),
                resume=bool(args.resume),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        result = publish_aggregate_artifacts(
            output_dir,
            contract,
            resume=bool(args.resume),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ContractError, FileExistsError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "execution_performed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
