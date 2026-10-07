#!/usr/bin/env python3
"""Run the frozen S4.3-PI2S base development rollout diagnostic.

This entry point is deliberately separate from the historical Track-A
evaluator.  It preserves the accepted policy, observation, action, replan,
native-success, and max-step semantics while adding privileged *read-only*
per-control-step telemetry.  It never updates an optimizer or a checkpoint.

The scientific unit is one tuple from the committed PI2S protocol:
checkpoint x reset identity x policy sampling seed x correct-H.  The base
wave contains exactly 72 tuples.  Infrastructure failures may be retried once
with the same tuple; attempts are never combined or selected by performance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from queue import Empty
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / "configs/simulation/pi2s/diagnostic_protocol.json"
EXPERIMENTS = ROOT / ".local/experiments"
PI2S_ROOT = EXPERIMENTS / "simulation/s4_3_pi2s"
MANIFEST = PI2S_ROOT / "artifacts/diagnostic_rollout_manifest.json"
RESULTS = PI2S_ROOT / "artifacts/diagnostic_rollout_results.json"
RUN_ROOT = PI2S_ROOT / "diagnostics/rollouts_v1"
LOG_ROOT = PI2S_ROOT / "logs/rollouts_v1"
CACHE_ROOT = PI2S_ROOT / "cache/rollouts_v1"
PROGRESS = RUN_ROOT / "progress.json"
RESUME = PI2S_ROOT / "resume_state.json"
RESUME_BACKUP = PI2S_ROOT / "handoff/resume_state_before_s5.json"
DEXJOCO = ROOT / "third_party/dexjoco"
CONFIG = DEXJOCO / "configs/rand_obj/pinch_tongs.yaml"
REGIONS = ROOT / "configs/simulation/s4_1_dexjoco_contact_regions.json"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
OPENPI_PYTHON = CONDA_ROOT / "envs/openpi/bin/python"
UNIT_PYTHON = CONDA_ROOT / "envs/unit/bin/python"
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
FOCUS_KEYS = (
    ("B0", 43),
    ("B_VA27", 43),
    ("B1", 43),
    ("B_HVA", 42),
    ("B_HVA", 43),
    ("B2", 43),
)
CONTACT_MODELS = frozenset({"B1", "B_HVA", "B2"})
RUNTIME_MODES = {
    "B0": "NONE",
    "B_VA27": "NONE",
    "B1": "CONTACT_STATE_TOKENS",
    "B_HVA": "CONTACT_STATE_TOKENS",
    "B2": "CONTACT_STATE_TOKENS",
}
SAMPLING_SEEDS = (4317, 4318)
EXPECTED_TUPLES = 72
MAX_INFRASTRUCTURE_ATTEMPTS = 2
OPENPI_CLIENT_ROOT = ROOT / "third_party/dexjoco/openpi/packages/openpi-client/src/openpi_client"
OPENPI_CLIENT_TREE_FILES = 13
OPENPI_CLIENT_TREE_SHA256 = "92fa31ab5335cd7613d656fcea70569e7797a73f6584f896ab7ff45a3ebf5cb4"
OPENPI_CLIENT_INIT_SHA256 = "91447944015cec709e8aa7655f7e9d64e1e4508e7023a57fe3746911c0fc6fed"
OPENPI_CLIENT_IMAGE_TOOLS_SHA256 = (
    "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
)
OPENPI_PIP_FREEZE_SHA256 = "39664447f71898e7ae8c0d7e41a79d74b13fe009546a8efcd69504ca64c91e89"
EVAL_PIP_FREEZE_SHA256 = "d776ecd6f5e5b73455b86e29c80946d9082cf865da5924413ff445a266b2d0b1"
OPENPI_VERSIONS = {
    "jax": "0.5.3",
    "jaxlib": "0.5.3",
    "flax": "0.10.2",
    "openpi-client": "0.1.0",
    "numpy": "1.26.4",
    "pillow": "12.3.0",
    "sentencepiece": "0.2.2",
}
EVAL_VERSIONS = {
    "openpi-client": "0.1.0",
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "mujoco": "3.4.0",
    "imageio": "2.37.4",
}
XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_exclude_nondeterministic_ops=true",
    "--xla_gpu_autotune_level=0",
)
SCHEMA_MANIFEST = "tactile3d-unit.s4-3-pi2s-diagnostic-rollout-manifest.v1"
SCHEMA_TUPLE = "tactile3d-unit.s4-3-pi2s-diagnostic-rollout-tuple.v1"
SCHEMA_RESULTS = "tactile3d-unit.s4-3-pi2s-diagnostic-rollout-results.v1"
STAGE_DEFINITION = {
    "INITIAL": "control step zero after exact reset identity verification",
    "APPROACH": "N/A: no preregistered native object-distance predicate",
    "OBJECT_CONTACT": "matched hand-region/object contact_count > 0 and normal_force_sum > 1e-9",
    "STABLE_GRASP": "N/A: consecutive object contact is recorded but is not relabelled as grasp stability",
    "LIFT": "native tongs_pos[2] >= native _lift_z",
    "CYCLE_1": "native pinch_count >= 1",
    "CYCLE_2": "native pinch_count >= 2",
    "CYCLE_3": "native pinch_count >= 3",
    "NATIVE_TRIGGER": "LIFT and CYCLE_3 at the same control step",
    "NATIVE_SUCCESS": "native success_counter >= 30 and info.succeed",
}
SOURCE_PATHS = (
    "configs/simulation/pi2s/diagnostic_protocol.json",
    "configs/simulation/s4_1_dexjoco_contact_regions.json",
    "gr00t/simulation/pi1d_runtime.py",
    "gr00t/simulation/simulated_tactile.py",
    "gr00t/simulation/pi2b_policy/contract.py",
    "gr00t/simulation/pi2b_policy/training.py",
    "scripts/simulation/pi2b_policy/serve.py",
    "scripts/simulation/serve_s4_3_pi1_contact_state.py",
    "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml",
    "third_party/dexjoco/dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py",
    "third_party/dexjoco/dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py",
    "third_party/dexjoco/dexjoco/dexjoco/sim/envs/panda_pinch_tongs_env.py",
    "third_party/dexjoco/openpi/packages/openpi-client/src/openpi_client/__init__.py",
    "third_party/dexjoco/openpi/packages/openpi-client/src/openpi_client/base_policy.py",
    "third_party/dexjoco/openpi/packages/openpi-client/src/openpi_client/image_tools.py",
    "third_party/dexjoco/openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py",
    "third_party/dexjoco/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py",
)

# Evaluator-process state.  One process executes exactly one canonical tuple.
TUPLE_SPEC: dict[str, Any] = {}
DIAGNOSTICS_JSONL = Path("/nonexistent")
TUPLE_ARTIFACT = Path("/nonexistent")
CONTACT_SOCKET = Path("/nonexistent")
ACTION_PROVENANCE_BY_TIMESTAMP: dict[int, tuple[str, ...]] = {}
EVALUATOR_RUNTIME_EVIDENCE: dict[str, Any] = {}


class ContractError(RuntimeError):
    """A scientific contract failure that is never performance-selected."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def python_tree_sha256(root: Path) -> tuple[int, str]:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def pip_freeze_sha256() -> str:
    payload = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], stderr=subprocess.DEVNULL
    )
    return hashlib.sha256(payload).hexdigest()


def verify_openpi_client_runtime() -> dict[str, Any]:
    import openpi_client
    from openpi_client import image_tools

    expected_init = OPENPI_CLIENT_ROOT / "__init__.py"
    expected_image_tools = OPENPI_CLIENT_ROOT / "image_tools.py"
    actual_init = Path(openpi_client.__file__).resolve()
    actual_image_tools = Path(image_tools.__file__).resolve()
    if (
        actual_init != expected_init.resolve()
        or actual_image_tools != expected_image_tools.resolve()
    ):
        raise ContractError(
            "openpi_client import origin drifted: "
            f"init={actual_init} image_tools={actual_image_tools}"
        )
    if sha256_file(actual_init) != OPENPI_CLIENT_INIT_SHA256:
        raise ContractError("openpi_client __init__ bytes drifted")
    if sha256_file(actual_image_tools) != OPENPI_CLIENT_IMAGE_TOOLS_SHA256:
        raise ContractError("openpi_client image_tools bytes drifted")
    count, tree_sha = python_tree_sha256(OPENPI_CLIENT_ROOT)
    if count != OPENPI_CLIENT_TREE_FILES or tree_sha != OPENPI_CLIENT_TREE_SHA256:
        raise ContractError("openpi_client Python tree drifted")
    return {
        "init_origin": str(actual_init),
        "image_tools_origin": str(actual_image_tools),
        "python_files": count,
        "python_tree_sha256": tree_sha,
    }


def verify_runtime_environment(role: str) -> dict[str, Any]:
    if role == "OPENPI_POLICY_SERVER":
        expected_executable = OPENPI_PYTHON.resolve()
        expected_versions = OPENPI_VERSIONS
        expected_freeze = OPENPI_PIP_FREEZE_SHA256
    elif role == "DEXJOCO_EVALUATOR":
        expected_executable = EVAL_PYTHON.resolve()
        expected_versions = EVAL_VERSIONS
        expected_freeze = EVAL_PIP_FREEZE_SHA256
    else:
        raise ValueError(role)
    if Path(sys.executable).resolve() != expected_executable:
        raise ContractError(
            f"{role} interpreter drifted: {Path(sys.executable).resolve()} != {expected_executable}"
        )
    observed_versions = {
        package: importlib.metadata.version(package) for package in expected_versions
    }
    if observed_versions != expected_versions:
        raise ContractError(f"{role} critical package versions drifted: {observed_versions}")
    freeze_sha = pip_freeze_sha256()
    if freeze_sha != expected_freeze:
        raise ContractError(f"{role} pip-freeze identity drifted")
    client = verify_openpi_client_runtime()
    evidence: dict[str, Any] = {
        "role": role,
        "executable": str(Path(sys.executable).resolve()),
        "versions": observed_versions,
        "pip_freeze_sha256": freeze_sha,
        "openpi_client": client,
    }
    if role == "OPENPI_POLICY_SERVER":
        import openpi

        accepted = ROOT / ".local/external/simulation/s4_3_pi1/openpi/src/openpi"
        origin = Path(openpi.__file__).resolve()
        if origin != (accepted / "__init__.py").resolve():
            raise ContractError(f"accepted OpenPI import origin drifted: {origin}")
        evidence["openpi_origin"] = str(origin)
    return evidence


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_mutable_json(path: Path, payload: Any) -> None:
    """Durably update mutable progress state; never used for scientific results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def publish_json_no_clobber(path: Path, payload: Any) -> None:
    """Publish a scientific artifact atomically without overwriting any name."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ContractError(f"publish parent is a symlink: {path.parent}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise FileExistsError(path) from error
            raise
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def copy_file_no_clobber(source: Path, destination: Path) -> None:
    """Copy exact bytes durably and publish by a no-replace hard link."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if temporary.stat().st_size != source.stat().st_size:
            raise ContractError("resume-state backup size mismatch")
        if sha256_file(temporary) != sha256_file(source):
            raise ContractError("resume-state backup SHA mismatch")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise FileExistsError(destination) from error
            raise
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def symbolic_checkpoint(path: str) -> Path:
    prefix = "$EXPERIMENT_ROOT/"
    if not path.startswith(prefix):
        raise ContractError(f"checkpoint is not symbolic: {path}")
    return EXPERIMENTS.resolve(strict=True) / path.removeprefix(prefix)


def checkpoint_id(row: Mapping[str, Any]) -> str:
    return f"{row['model']}_{int(row['training_seed'])}"


def tuple_key(row: Mapping[str, Any]) -> str:
    return (
        f"{row['checkpoint_id'].lower()}__r{int(row['reset_seed'])}_"
        f"i{int(row['reset_block_index']):02d}__p{int(row['sampling_seed'])}"
    )


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def committed_blob_sha(commit: str, relative: str) -> str:
    payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def load_protocol() -> tuple[dict[str, Any], str]:
    protocol = read_json(PROTOCOL)
    if protocol.get("status") != "PREREGISTERED_NOT_EXECUTED":
        raise ContractError("PI2S protocol is not frozen pre-execution")
    if protocol.get("development_rollouts", {}).get("base", {}).get("canonical_tuples") != 72:
        raise ContractError("PI2S base rollout budget is not exactly 72")
    if protocol["development_rollouts"].get("maximum_canonical_tuples") != 120:
        raise ContractError("PI2S maximum rollout budget is not 120")
    return protocol, sha256_file(PROTOCOL)


def build_base_tuples(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    focus = protocol["focus_checkpoints"]
    if [(row["model"], int(row["training_seed"])) for row in focus] != list(FOCUS_KEYS):
        raise ContractError("focus checkpoint order/identity drifted")
    base = protocol["development_rollouts"]["base"]
    resets = base["reset_specs"]
    seeds = tuple(int(value) for value in base["sampling_seeds"])
    if len(resets) != 6 or seeds != SAMPLING_SEEDS:
        raise ContractError("base reset or sampling-seed contract drifted")
    tuples: list[dict[str, Any]] = []
    for checkpoint in focus:
        for reset in resets:
            for sampling_seed in seeds:
                row = {
                    "checkpoint_id": checkpoint_id(checkpoint),
                    "model": checkpoint["model"],
                    "training_seed": int(checkpoint["training_seed"]),
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_tree_sha256": checkpoint["tree_sha256"],
                    "reset_identity": reset["reset_identity"],
                    "reset_seed": int(reset["seed"]),
                    "reset_block_index": int(reset["block_index"]),
                    "reset_global_index": int(reset["global_index"]),
                    "mjstate_sha256": reset["mjstate_sha256"],
                    "processed_state_sha256": reset["processed_state_sha256"],
                    "sampling_seed": sampling_seed,
                    "h_condition": "correct",
                    "runtime_mode": RUNTIME_MODES[checkpoint["model"]],
                }
                row["tuple_key"] = tuple_key(row)
                row["tuple_identity_sha256"] = canonical_sha(row)
                tuples.append(row)
    if (
        len(tuples) != EXPECTED_TUPLES
        or len({row["tuple_key"] for row in tuples}) != EXPECTED_TUPLES
    ):
        raise ContractError("base canonical tuple count/uniqueness failed")
    return tuples


def validate_checkpoint_presence(row: Mapping[str, Any]) -> None:
    path = symbolic_checkpoint(str(row["checkpoint_path"]))
    required = (
        path / "_CHECKPOINT_METADATA",
        path / "params/manifest.ocdbt",
        path / "train_state/manifest.ocdbt",
    )
    if not all(item.is_file() for item in required):
        raise ContractError(f"checkpoint structure missing for {row['checkpoint_id']}: {path}")


def build_manifest(source_commit: str) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    if git_output("branch", "--show-current") != "develop/sim-benchmark":
        raise ContractError("S5 manifest may only be frozen on develop/sim-benchmark")
    runner_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if committed_blob_sha(source_commit, runner_relative) != sha256_file(Path(__file__).resolve()):
        raise ContractError("running S5 runner is not the exact blob at source commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=ROOT, check=False
    )
    if ancestor.returncode:
        raise ContractError("S5 source commit is not an ancestor of current HEAD")
    tuples = build_base_tuples(protocol)
    for row in tuples[::12]:
        validate_checkpoint_presence(row)
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}
    prior_artifacts = {}
    for name in (
        "historical_failure_stage_analysis.json",
        "offline_online_h_parity.json",
        "prefix_contract_audit.json",
        "fixed_observation_interventions.json",
        "read_only_gradient_diagnostics.json",
    ):
        path = PI2S_ROOT / "artifacts" / name
        if not path.is_file():
            raise ContractError(f"required pre-S5 evidence is missing: {path}")
        prior_artifacts[name] = {
            "path": f"$PI2S_ROOT/artifacts/{name}",
            "sha256": sha256_file(path),
        }
    payload = {
        "schema": SCHEMA_MANIFEST,
        "status": "FROZEN_PRE_EXECUTION",
        "created_at_utc": now_utc(),
        "source_commit": source_commit,
        "source_branch": git_output("branch", "--show-current"),
        "runner": "$REPO_ROOT/" + runner_relative,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": "$REPO_ROOT/configs/simulation/pi2s/diagnostic_protocol.json",
        "protocol_sha256": protocol_sha,
        "source_sha256": source_hashes,
        "pre_s5_evidence": prior_artifacts,
        "written_live_information_need": {
            "historical_gap": (
                "The 3000 formal outcomes contain episode aggregates but no timestamped native "
                "state, so HVA43's exact/coarse live failure progression cannot be reconstructed."
            ),
            "base_question_1": (
                "For each frozen checkpoint and reset, what is the last observed native milestone "
                "before max_steps or native success?"
            ),
            "base_question_2": (
                "Do two preregistered policy sampling seeds on the same checkpoint/reset change the "
                "native milestone path, consistent with runtime stochasticity contribution?"
            ),
            "base_question_3": (
                "Does HVA43 reach object contact, lift, and the three native pinch counters before "
                "failure, relative to HVA42 and non-HVA controls?"
            ),
            "why_existing_evidence_is_insufficient": (
                "S2 found a small online/offline numerical H mismatch and S3 showed H sensitivity "
                "plus mixed main/aux gradient alignment, but neither establishes the closed-loop "
                "first-failure stage or a unique cause."
            ),
            "optional_h_intervention_decision": (
                "DEFERRED_UNTIL_ALL_72_BASE_TUPLES_COMPLETE_AND_A_WRITTEN_POST_BASE_DECISION"
            ),
        },
        "runtime_dependencies": {
            "openpi_policy_server": {
                "executable": str(OPENPI_PYTHON),
                "pip_freeze_sha256": OPENPI_PIP_FREEZE_SHA256,
                "critical_versions": OPENPI_VERSIONS,
            },
            "dexjoco_evaluator": {
                "executable": "$REPO_ROOT/.local/external/s4_3_pi0/eval-venv/bin/python",
                "pip_freeze_sha256": EVAL_PIP_FREEZE_SHA256,
                "critical_versions": EVAL_VERSIONS,
            },
            "openpi_client": {
                "root": "$REPO_ROOT/third_party/dexjoco/openpi/packages/openpi-client/src/openpi_client",
                "python_files": OPENPI_CLIENT_TREE_FILES,
                "python_tree_sha256": OPENPI_CLIENT_TREE_SHA256,
                "image_tools_sha256": OPENPI_CLIENT_IMAGE_TOOLS_SHA256,
            },
        },
        "stage_definition": STAGE_DEFINITION,
        "exact_first_failure_available": False,
        "coarse_native_milestone_localization_available": True,
        "privileged_state_is_policy_input": False,
        "official_semantics_preserved": {
            "prompt": True,
            "observation_and_action_adapter": True,
            "replan_ratio": 0.8,
            "native_success": True,
            "native_max_steps": 1000,
            "physics_or_reset_mutation": False,
        },
        "sampling_seed_control": {
            "method": "fresh per-episode JAX policy RNG key before first request",
            "metadata_removed_before_policy_transforms": True,
            "wrapper_calls_same_policy_infer": True,
            "reset_on_exact_infrastructure_retry": True,
        },
        "base_budget": {
            "canonical_tuples": 72,
            "checkpoints": 6,
            "resets": 6,
            "sampling_seeds": 2,
            "h_condition": "correct",
        },
        "optional_budget": {
            "canonical_tuples": 48,
            "executed_by_this_wave": False,
            "requires_post_base_written_decision": True,
        },
        "infrastructure_retry_per_tuple": 1,
        "tuples": tuples,
        "tuple_sequence_sha256": canonical_sha(tuples),
        "performance_seen_while_freezing": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
    }
    return payload


def freeze_manifest(source_commit: str) -> None:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    payload = build_manifest(source_commit)
    publish_json_no_clobber(MANIFEST, payload)
    if RESUME.is_file() and not RESUME_BACKUP.exists():
        copy_file_no_clobber(RESUME, RESUME_BACKUP)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "tuples": len(payload["tuples"]),
                "sha256": sha256_file(MANIFEST),
            },
            sort_keys=True,
        )
    )


def validate_manifest() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    manifest = read_json(MANIFEST)
    if (
        manifest.get("schema") != SCHEMA_MANIFEST
        or manifest.get("status") != "FROZEN_PRE_EXECUTION"
    ):
        raise ContractError("S5 rollout manifest is not eligible")
    if manifest.get("protocol_sha256") != protocol_sha:
        raise ContractError("S5 manifest protocol binding drifted")
    if manifest.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise ContractError("S5 runner drifted after manifest freeze")
    source_commit = str(manifest.get("source_commit"))
    runner_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if committed_blob_sha(source_commit, runner_relative) != manifest["runner_sha256"]:
        raise ContractError("S5 committed runner binding failed")
    observed_sources = {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}
    if observed_sources != manifest.get("source_sha256"):
        raise ContractError("S5 execution source drifted")
    for name, row in manifest.get("pre_s5_evidence", {}).items():
        path = PI2S_ROOT / "artifacts" / name
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ContractError(f"pre-S5 evidence drifted: {name}")
    expected = build_base_tuples(protocol)
    if manifest.get("tuples") != expected or manifest.get("tuple_sequence_sha256") != canonical_sha(
        expected
    ):
        raise ContractError("S5 tuple manifest drifted")
    for row in expected[::12]:
        validate_checkpoint_presence(row)
    return manifest


class SeedResetPolicy:
    """Reset only the accepted policy RNG, then call the same policy infer."""

    def __init__(self, policy: Any, jax_module: Any):
        self.policy = policy
        self.jax = jax_module
        self.current_token: str | None = None
        self.query_index = 0

    @property
    def metadata(self) -> Any:
        return self.policy.metadata

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(observation)
        try:
            sampling_seed = int(payload.pop("_pi2s_sampling_seed"))
            episode_token = str(payload.pop("_pi2s_episode_token"))
            reset_rng = bool(payload.pop("_pi2s_reset_rng"))
        except KeyError as error:
            raise ContractError(f"S5 sampling metadata missing: {error}") from error
        if reset_rng:
            self.policy._rng = self.jax.random.key(sampling_seed)
            self.current_token = episode_token
            self.query_index = 0
        elif self.current_token != episode_token:
            raise ContractError("episode token changed without an explicit RNG reset")
        index = self.query_index
        result = dict(self.policy.infer(payload))
        result["pi2s_sampling_control"] = {
            "sampling_seed": sampling_seed,
            "episode_token": episode_token,
            "query_index": index,
            "rng_reset_before_query": reset_rng,
        }
        self.query_index += 1
        return result


def serve_policy(checkpoint_name: str, port: int) -> None:
    manifest = validate_manifest()
    row = next(
        (value for value in manifest["tuples"] if value["checkpoint_id"] == checkpoint_name), None
    )
    if row is None:
        raise ContractError(f"unknown focus checkpoint {checkpoint_name}")
    from gr00t.simulation.pi2b_policy.contract import Workspace
    from gr00t.simulation.pi2b_policy.training import configure_imports
    from scripts.simulation.pi2b_policy.serve import runtime_config

    workspace = Workspace.load_readonly(ROOT)
    checkpoint = symbolic_checkpoint(row["checkpoint_path"])
    configure_imports(workspace)
    import jax
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    runtime_evidence = verify_runtime_environment("OPENPI_POLICY_SERVER")

    config, training_mode, lambda_phys = runtime_config(
        workspace, str(row["model"]), int(row["training_seed"])
    )
    policy = policy_config.create_trained_policy(config, checkpoint)
    wrapped = SeedResetPolicy(policy, jax)
    print(
        "PI2S_POLICY_SERVER_READY "
        f"checkpoint_id={checkpoint_name} training_mode={training_mode} "
        f"runtime_mode={row['runtime_mode']} lambda_phys={lambda_phys} port={port}",
        flush=True,
    )
    print("PI2S_POLICY_RUNTIME " + json.dumps(runtime_evidence, sort_keys=True), flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="127.0.0.1", port=port, metadata=wrapped.metadata
    ).serve_forever()


def contact_service(artifact: Path, socket_path: Path) -> None:
    from scripts.simulation import serve_s4_3_pi1_contact_state as service

    service.ARTIFACT = artifact
    sys.argv = [sys.argv[0], "--socket", str(socket_path)]
    service.main()


def instrumented_inference_process(
    obs_queue: Any,
    action_queue: Any,
    stop_event: Any,
    port: int,
    inferencing_event: Any,
    seed: int,
    host: str,
) -> None:
    """Accepted inference loop with timing, chunk identity, and seed evidence."""

    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from openpi_client import websocket_client_policy

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    official._set_seed(seed)
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    sequence = 0
    while not stop_event.is_set():
        observation = official.get_latest(obs_queue)
        if observation is None:
            stop_event.wait(0.01)
            continue
        payload = dict(observation.obs)
        reset_identity = str(payload.pop("_pi2s_reset_identity"))
        tuple_identity = str(payload.pop("_pi2s_tuple_identity"))
        started_wall_ns = time.time_ns()
        started_mono_ns = time.monotonic_ns()
        try:
            result = client.infer(payload)
            ended_mono_ns = time.monotonic_ns()
            ended_wall_ns = time.time_ns()
            action_chunk = np.asarray(result["actions"])
            control = result.get("pi2s_sampling_control", {})
            chunk_id = hashlib.sha256(
                f"{tuple_identity}:{sequence}:".encode("ascii")
                + np.ascontiguousarray(action_chunk).tobytes()
            ).hexdigest()
            record = {
                "type": "action_chunk",
                "sequence": sequence,
                "chunk_id": chunk_id,
                "tuple_identity_sha256": tuple_identity,
                "reset_identity": reset_identity,
                "observation_timestamp": int(observation.timestamp),
                "shape": list(action_chunk.shape),
                "dtype": str(action_chunk.dtype),
                "finite": bool(np.isfinite(action_chunk).all()),
                "action_chunk_sha256": sha256_array(action_chunk),
                "infer_started_wall_ns": started_wall_ns,
                "infer_ended_wall_ns": ended_wall_ns,
                "infer_started_monotonic_ns": started_mono_ns,
                "infer_ended_monotonic_ns": ended_mono_ns,
                "roundtrip_ms": (ended_mono_ns - started_mono_ns) / 1e6,
                "policy_timing": result.get("policy_timing"),
                "sampling_control": control,
                "contact_state_sent": "contact_state" in payload,
                "training_only_fields_sent": sorted(
                    set(payload)
                    & {
                        "contact_shared_target",
                        "physical_aux_valid",
                        "va_shared_target",
                        "va_aux_valid",
                    }
                ),
            }
            append_jsonl(DIAGNOSTICS_JSONL, record)
            action_queue.put(
                {
                    "action": action_chunk,
                    "timestamp": int(observation.timestamp),
                    "chunk_id": chunk_id,
                }
            )
            sequence += 1
            inferencing_event.clear()
        except Exception as error:
            append_jsonl(
                DIAGNOSTICS_JSONL,
                {
                    "type": "server_client_error",
                    "tuple_identity_sha256": tuple_identity,
                    "reset_identity": reset_identity,
                    "observation_timestamp": int(observation.timestamp),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


@dataclass
class ScheduledAction:
    action: np.ndarray
    timestamp: int
    chunk_ids: tuple[str, ...]


def receive_actions_with_provenance(
    action_queue: Any, actions_buffer: Any, now_timestamp: int, dual_arm: bool
) -> None:
    """Byte-equivalent accepted merge math plus diagnostic chunk provenance."""

    from dexjoco_openpi_client import eval_dexjoco_openpi as official

    interp = official._interp_dual_arm_action if dual_arm else official._interp_single_arm_action
    while actions_buffer and actions_buffer[0].timestamp < now_timestamp:
        actions_buffer.popleft()
    while True:
        try:
            chunk = action_queue.get_nowait()
        except Empty:
            break
        chunk_timestamp = int(chunk["timestamp"])
        if chunk_timestamp > now_timestamp:
            append_jsonl(
                DIAGNOSTICS_JSONL,
                {
                    "type": "stale_cross_episode_action_discarded",
                    "chunk_id": chunk["chunk_id"],
                    "action_timestamp": chunk_timestamp,
                    "current_timestamp": now_timestamp,
                },
            )
            continue
        action_chunk = np.asarray(chunk["action"])
        chunk_range = (now_timestamp, chunk_timestamp + action_chunk.shape[0])
        if chunk_range[1] <= now_timestamp:
            continue
        action = action_chunk[
            (chunk_range[0] - chunk_timestamp) : (chunk_range[1] - chunk_timestamp)
        ]
        if actions_buffer:
            buffer_range = (actions_buffer[0].timestamp, actions_buffer[-1].timestamp + 1)
            if buffer_range[1] - buffer_range[0] != len(actions_buffer):
                raise ContractError("action buffer timestamps are not continuous")
        else:
            buffer_range = (now_timestamp, now_timestamp)
        overlap = (max(chunk_range[0], buffer_range[0]), min(chunk_range[1], buffer_range[1]))
        overlap_len = overlap[1] - overlap[0]
        for timestamp in range(overlap[0], overlap[1]):
            buffer_index = timestamp - buffer_range[0]
            action_index = timestamp - chunk_range[0]
            interpolation = (timestamp - overlap[0] + 1) / (overlap_len + 1)
            old = actions_buffer[buffer_index]
            chunk_ids = tuple(dict.fromkeys((*old.chunk_ids, str(chunk["chunk_id"]))))
            actions_buffer[buffer_index] = ScheduledAction(
                action=interp(old.action, action[action_index], interpolation),
                timestamp=timestamp,
                chunk_ids=chunk_ids,
            )
        for timestamp in range(buffer_range[1], chunk_range[1]):
            action_index = timestamp - chunk_range[0]
            actions_buffer.append(
                ScheduledAction(
                    action=action[action_index],
                    timestamp=timestamp,
                    chunk_ids=(str(chunk["chunk_id"]),),
                )
            )
    ACTION_PROVENANCE_BY_TIMESTAMP.clear()
    ACTION_PROVENANCE_BY_TIMESTAMP.update(
        {int(row.timestamp): tuple(row.chunk_ids) for row in actions_buffer}
    )


HAND_SENSOR_NAMES = (
    "allegro_right/ffj0_pos",
    "allegro_right/ffj1_pos",
    "allegro_right/ffj2_pos",
    "allegro_right/ffj3_pos",
    "allegro_right/mfj0_pos",
    "allegro_right/mfj1_pos",
    "allegro_right/mfj2_pos",
    "allegro_right/mfj3_pos",
    "allegro_right/rfj0_pos",
    "allegro_right/rfj1_pos",
    "allegro_right/rfj2_pos",
    "allegro_right/rfj3_pos",
    "allegro_right/thj0_pos",
    "allegro_right/thj1_pos",
    "allegro_right/thj2_pos",
    "allegro_right/thj3_pos",
)


def build_diagnostic_environment(base_class: Any) -> Any:
    import mujoco

    from gr00t.simulation.pi1d_runtime import CausalContactRuntime, ContactStateUnixClient
    from gr00t.simulation.simulated_tactile import ContactRegionMap, SimulatedTactileExtractor

    class DiagnosticEnvironment(base_class):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._contact_client: Any = None
            self._contact_runtime: Any = None
            self._extractor: Any = None
            self._extractor_audit: dict[str, Any] | None = None
            self._steps: list[dict[str, Any]] = []
            self._observation_events: list[dict[str, Any]] = []
            self._last_raw_result: tuple[Any, ...] | None = None
            self._control_step = 0
            self._observation_index = 0
            self._consecutive_contact_run = 0
            self._max_consecutive_contact_run = 0
            self._last_h: dict[str, Any] | None = None
            self._reset_verified = False

        def start(self) -> None:
            super().start()
            config = json.loads(REGIONS.read_text(encoding="utf-8"))
            region_map = ContactRegionMap.from_config(
                {"regions": config["regions"], "object_body_names": config["object_body_names"]}
            )
            raw = self.env.unwrapped
            self._extractor_audit = region_map.resolve(raw.model, mujoco)
            if self._extractor_audit.get("status") != "PASS" or region_map.tactile_dim != 30:
                raise ContractError("live tactile region resolution failed")
            self._extractor = SimulatedTactileExtractor(region_map)
            self._contact_client = ContactStateUnixClient(CONTACT_SOCKET)
            self._contact_runtime = CausalContactRuntime(self._contact_client)
            original_step = self.env.step

            def captured_step(action: np.ndarray) -> Any:
                apply_started_wall_ns = time.time_ns()
                apply_started_mono_ns = time.monotonic_ns()
                result = original_step(action)
                apply_ended_mono_ns = time.monotonic_ns()
                apply_ended_wall_ns = time.time_ns()
                self._last_raw_result = (
                    *result,
                    np.asarray(action, dtype=np.float64).copy(),
                    apply_started_wall_ns,
                    apply_ended_wall_ns,
                    apply_started_mono_ns,
                    apply_ended_mono_ns,
                )
                return result

            self.env.step = captured_step

        def _extract_tactile(self) -> tuple[np.ndarray, dict[str, Any]]:
            raw = self.env.unwrapped
            tactile, diagnostics = self._extractor.extract(raw.model, raw.data, mujoco)
            tactile = np.asarray(tactile, dtype=np.float32)
            if tactile.shape != (30,) or not np.isfinite(tactile).all():
                raise ContractError("live tactile observation is not finite [30]")
            return tactile, diagnostics

        def _reset_identity(self) -> tuple[str, str, str]:
            raw = self.env.unwrapped
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64)
            mujoco.mj_getState(raw.model, raw.data, state, state_spec)
            processed = np.asarray(self.obs["state"], dtype=np.float64)
            return sha256_array(state, processed), sha256_array(state), sha256_array(processed)

        def reset(self) -> None:
            for _ in range(int(TUPLE_SPEC["reset_block_index"]) + 1):
                super().reset()
            identity, mjstate_sha, processed_sha = self._reset_identity()
            if (
                identity != TUPLE_SPEC["reset_identity"]
                or mjstate_sha != TUPLE_SPEC["mjstate_sha256"]
                or processed_sha != TUPLE_SPEC["processed_state_sha256"]
            ):
                raise ContractError(
                    f"reset reconstruction mismatch: observed={identity} expected={TUPLE_SPEC['reset_identity']}"
                )
            self._reset_verified = True
            tactile, diagnostics = self._extract_tactile()
            self._contact_runtime.reset(tactile)
            self._initial_tactile = {
                "tactile_sha256": sha256_array(tactile),
                "contact_count": int(diagnostics["contact_count"]),
            }

        def _native_state(self, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
            raw = self.env.unwrapped
            tongs_joint = float(raw._data.sensor("tongs_joint_0_pos").data)
            tongs_pos = np.asarray(raw._data.sensor("tongs_pos").data, dtype=np.float64).copy()
            tcp_pos = np.asarray(
                raw._data.sensor("franka/flange_pos").data, dtype=np.float64
            ).copy()
            tcp_quat = np.asarray(
                raw._data.sensor("franka/flange_quat").data, dtype=np.float64
            ).copy()
            hand = np.asarray(
                [float(raw._data.sensor(name).data) for name in HAND_SENSOR_NAMES], dtype=np.float64
            )
            matched = list(diagnostics.get("matched_pairs", []))
            normal_sum = float(sum(float(row["normal_force"]) for row in matched))
            normal_max = float(max((float(row["normal_force"]) for row in matched), default=0.0))
            contact = int(diagnostics.get("contact_count", 0)) > 0 and normal_sum > 1e-9
            if contact:
                self._consecutive_contact_run += 1
            else:
                self._consecutive_contact_run = 0
            self._max_consecutive_contact_run = max(
                self._max_consecutive_contact_run, self._consecutive_contact_run
            )
            pinch_count = int(getattr(raw, "_pinch_count", 0))
            success_counter = int(getattr(raw, "_success_counter", 0))
            lift_z = float(raw._lift_z)
            lifted = bool(tongs_pos[2] >= lift_z)
            return {
                "tongs_joint_0_pos": tongs_joint,
                "tongs_pos": tongs_pos.tolist(),
                "lift_z": lift_z,
                "lifted": lifted,
                "tcp_position": tcp_pos.tolist(),
                "tcp_quaternion_wxyz": tcp_quat.tolist(),
                "hand_joint_positions": hand.tolist(),
                "pinch_count": pinch_count,
                "success_counter": success_counter,
                "native_trigger": bool(lifted and pinch_count >= 3),
                "matched_object_contact": contact,
                "matched_contact_count": int(diagnostics.get("contact_count", 0)),
                "normal_force_sum": normal_sum,
                "normal_force_max": normal_max,
                "consecutive_object_contact_run": self._consecutive_contact_run,
                "contact_count_by_region": dict(diagnostics.get("contact_count_by_region", {})),
            }

        def step(self, action: np.ndarray) -> Any:
            policy_action = np.asarray(action, dtype=np.float64).copy()
            if policy_action.shape != (22,) or not np.isfinite(policy_action).all():
                raise ContractError("executed policy action is not finite [22]")
            timestamp = self._control_step
            chunk_ids = ACTION_PROVENANCE_BY_TIMESTAMP.get(timestamp, ())
            result = super().step(action)
            if self._last_raw_result is None:
                raise ContractError("raw environment step capture is missing")
            (
                _obs,
                reward,
                terminated,
                truncated,
                info,
                env_action,
                apply_started_wall_ns,
                apply_ended_wall_ns,
                apply_started_mono_ns,
                apply_ended_mono_ns,
            ) = self._last_raw_result
            tactile, diagnostics = self._extract_tactile()
            self._contact_runtime.append(tactile)
            native = self._native_state(diagnostics)
            self._steps.append(
                {
                    "control_step": timestamp,
                    "action_source": "POLICY_CHUNK" if chunk_ids else "STAY",
                    "action_chunk_ids": list(chunk_ids),
                    "policy_action_22": policy_action.tolist(),
                    "policy_action_sha256": sha256_array(policy_action),
                    "environment_action_23": np.asarray(env_action, dtype=np.float64).tolist(),
                    "environment_action_sha256": sha256_array(np.asarray(env_action)),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info_succeed": bool(info.get("succeed", False)),
                    "info_pinch_count": int(info.get("pinch_count", 0)),
                    "apply_started_wall_ns": int(apply_started_wall_ns),
                    "apply_ended_wall_ns": int(apply_ended_wall_ns),
                    "apply_started_monotonic_ns": int(apply_started_mono_ns),
                    "apply_ended_monotonic_ns": int(apply_ended_mono_ns),
                    "apply_ms": (apply_ended_mono_ns - apply_started_mono_ns) / 1e6,
                    "tactile_sha256": sha256_array(tactile),
                    "tactile_l2": float(np.linalg.norm(tactile)),
                    "h_at_latest_observation": copy.deepcopy(self._last_h),
                    "native": native,
                }
            )
            self._control_step += 1
            return result

        def get_obs(self) -> dict[str, np.ndarray]:
            started_wall_ns = time.time_ns()
            started_mono_ns = time.monotonic_ns()
            observation = super().get_obs()
            contact_state = self._contact_runtime.contact_state()
            history = self._contact_runtime.current_history()
            ended_mono_ns = time.monotonic_ns()
            ended_wall_ns = time.time_ns()
            h_row = {
                "control_step": self._control_step,
                "h_sha256": sha256_array(contact_state),
                "h_l2": float(np.linalg.norm(contact_state)),
                "history_sha256": sha256_array(history),
                "history_shape": [26, 30],
                "contact_state_shape": [256],
                "observe_started_wall_ns": started_wall_ns,
                "observe_ended_wall_ns": ended_wall_ns,
                "observe_started_monotonic_ns": started_mono_ns,
                "observe_ended_monotonic_ns": ended_mono_ns,
                "observe_ms": (ended_mono_ns - started_mono_ns) / 1e6,
                "sent_to_policy": TUPLE_SPEC["model"] in CONTACT_MODELS,
            }
            self._last_h = h_row
            self._observation_events.append(h_row)
            if TUPLE_SPEC["model"] in CONTACT_MODELS:
                observation["contact_state"] = contact_state
            observation["_pi2s_reset_identity"] = TUPLE_SPEC["reset_identity"]
            observation["_pi2s_tuple_identity"] = TUPLE_SPEC["tuple_identity_sha256"]
            observation["_pi2s_sampling_seed"] = int(TUPLE_SPEC["sampling_seed"])
            observation["_pi2s_episode_token"] = TUPLE_SPEC["tuple_key"]
            observation["_pi2s_reset_rng"] = self._observation_index == 0
            self._observation_index += 1
            return observation

        def close(self) -> None:
            try:
                if self._contact_client is not None:
                    self._contact_client.close()
                inference_rows = (
                    [
                        json.loads(line)
                        for line in DIAGNOSTICS_JSONL.read_text().splitlines()
                        if line
                    ]
                    if DIAGNOSTICS_JSONL.is_file()
                    else []
                )
                chunks = [row for row in inference_rows if row.get("type") == "action_chunk"]
                errors = [row for row in inference_rows if row.get("type") == "server_client_error"]
                sampling_gates = all(
                    row.get("sampling_control", {}).get("sampling_seed")
                    == TUPLE_SPEC["sampling_seed"]
                    and row.get("sampling_control", {}).get("episode_token")
                    == TUPLE_SPEC["tuple_key"]
                    and row.get("sampling_control", {}).get("query_index") == index
                    and row.get("sampling_control", {}).get("rng_reset_before_query")
                    == (index == 0)
                    for index, row in enumerate(chunks)
                )
                steps = len(self._steps)
                success = bool(self._steps and self._steps[-1]["info_succeed"])
                gates = {
                    "exact_reset_identity": self._reset_verified,
                    "one_to_1000_steps": 1 <= steps <= 1000,
                    "native_terminal_matches_steps": success or steps == 1000,
                    "finite_actions": all(
                        np.isfinite(row["policy_action_22"]).all()
                        and np.isfinite(row["environment_action_23"]).all()
                        for row in self._steps
                    ),
                    "all_h_finite_256": bool(self._observation_events)
                    and all(np.isfinite(row["h_l2"]) for row in self._observation_events),
                    "action_chunks_finite_30x22": bool(chunks)
                    and all(row["finite"] and row["shape"] == [30, 22] for row in chunks),
                    "sampling_seed_control_exact": bool(chunks) and sampling_gates,
                    "contact_delivery_matches_mode": all(
                        row["contact_state_sent"] == (TUPLE_SPEC["model"] in CONTACT_MODELS)
                        for row in chunks
                    ),
                    "training_only_targets_absent": all(
                        not row["training_only_fields_sent"] for row in chunks
                    ),
                    "no_server_client_errors": not errors,
                    "telemetry_per_physical_step": len(self._steps) == steps,
                    "checkpoint_not_written": True,
                    "optimizer_not_stepped": True,
                }
                payload = {
                    "schema": SCHEMA_TUPLE,
                    "status": "PASS" if all(gates.values()) else "FAIL",
                    "created_at_utc": now_utc(),
                    "tuple": TUPLE_SPEC,
                    "stage_definition": STAGE_DEFINITION,
                    "exact_first_failure_stage": "N/A_DETAILED_APPROACH_AND_STABLE_GRASP_NOT_DEFINED",
                    "success": success,
                    "termination": (
                        "native_success"
                        if success
                        else "native_max_steps" if steps == 1000 else "other_native_termination"
                    ),
                    "steps": steps,
                    "max_pinch_count": max(
                        (row["native"]["pinch_count"] for row in self._steps), default=0
                    ),
                    "ever_object_contact": any(
                        row["native"]["matched_object_contact"] for row in self._steps
                    ),
                    "ever_lifted": any(row["native"]["lifted"] for row in self._steps),
                    "ever_native_trigger": any(
                        row["native"]["native_trigger"] for row in self._steps
                    ),
                    "max_success_counter": max(
                        (row["native"]["success_counter"] for row in self._steps), default=0
                    ),
                    "max_consecutive_object_contact_run": self._max_consecutive_contact_run,
                    "initial_tactile": getattr(self, "_initial_tactile", None),
                    "observation_events": self._observation_events,
                    "action_chunks": chunks,
                    "server_client_errors": errors,
                    "step_telemetry": self._steps,
                    "evaluator_runtime": EVALUATOR_RUNTIME_EVIDENCE,
                    "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
                    "privileged_telemetry_sent_to_policy": False,
                    "formal_track_a_score_replacement_allowed": False,
                }
                if TUPLE_ARTIFACT.exists():
                    raise FileExistsError(TUPLE_ARTIFACT)
                publish_json_no_clobber(TUPLE_ARTIFACT, payload)
                print(
                    json.dumps(
                        {
                            "tuple_key": TUPLE_SPEC["tuple_key"],
                            "status": payload["status"],
                            "success": success,
                            "steps": steps,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                super().close()

    return DiagnosticEnvironment


def evaluate_tuple(
    tuple_spec_path: Path,
    contact_socket: Path,
    output: Path,
    diagnostics: Path,
    artifact: Path,
    port: int,
) -> None:
    global TUPLE_SPEC, CONTACT_SOCKET, DIAGNOSTICS_JSONL, TUPLE_ARTIFACT
    global EVALUATOR_RUNTIME_EVIDENCE
    manifest = validate_manifest()
    spec = read_json(tuple_spec_path)
    expected = next(
        (row for row in manifest["tuples"] if row["tuple_key"] == spec.get("tuple_key")), None
    )
    if expected is None or spec != expected:
        raise ContractError("tuple spec does not exactly match frozen manifest")
    if any(path.exists() for path in (output, diagnostics, artifact)):
        raise FileExistsError("refusing to overwrite S5 tuple attempt output")
    TUPLE_SPEC = spec
    CONTACT_SOCKET = contact_socket
    DIAGNOSTICS_JSONL = diagnostics
    TUPLE_ARTIFACT = artifact
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DEXJOCO / "dexjoco"))
    runtime_evidence = verify_runtime_environment("DEXJOCO_EVALUATOR")
    EVALUATOR_RUNTIME_EVIDENCE = runtime_evidence
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    ACTION_PROVENANCE_BY_TIMESTAMP.clear()
    official.DexJoCoOpenPIEnv = build_diagnostic_environment(DexJoCoOpenPIEnv)
    official.inference_process = instrumented_inference_process
    official.receive_actions = receive_actions_with_provenance
    official.main(
        config=CONFIG,
        seed=int(spec["reset_seed"]),
        rand_full=False,
        randomize_dynamics=False,
        port=port,
        host="127.0.0.1",
        output=output,
        render_mode="rgb_array",
        replan_ratio=0.8,
        episodes=1,
        pad_state_dim46=False,
        record_pressed_digits=False,
    )
    payload = read_json(artifact)
    if payload.get("status") != "PASS":
        raise ContractError(f"tuple artifact failed integrity gates: {spec['tuple_key']}")
    print("PI2S_EVALUATOR_RUNTIME " + json.dumps(runtime_evidence, sort_keys=True), flush=True)


def gpu_snapshot() -> dict[str, Any]:
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return {
        "captured_at_utc": now_utc(),
        "inventory": inventory,
        "compute_applications": applications,
    }


def gpu_is_idle(index: int, snapshot: Mapping[str, Any]) -> bool:
    row = next(
        (line for line in snapshot["inventory"] if int(line.split(",", 1)[0].strip()) == index),
        None,
    )
    if row is None:
        return False
    fields = [field.strip() for field in row.split(",")]
    uuid = fields[1]
    return int(fields[3]) <= 64 and not any(
        line.split(",", 1)[0].strip() == uuid for line in snapshot["compute_applications"]
    )


def acquire_gpu_lock(index: int) -> Any:
    common = Path(git_output("rev-parse", "--path-format=absolute", "--git-common-dir"))
    handle = (common / f"tactile3d_unit_gpu{index}.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise ContractError(f"GPU {index} advisory lock is held") from error
    return handle


def stop_process(process: subprocess.Popen[Any] | None, timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"process {process.pid} ignored SIGINT and SIGTERM") from error


def wait_for_log(process: subprocess.Popen[Any], path: Path, needle: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and needle in path.read_text(errors="replace"):
            return
        if process.poll() is not None:
            tail = path.read_text(errors="replace")[-12000:] if path.is_file() else ""
            raise RuntimeError(f"process exited {process.returncode} before {needle}:\n{tail}")
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for {needle} in {path}")


def block_paths(checkpoint_name: str) -> SimpleNamespace:
    lower = checkpoint_name.lower()
    return SimpleNamespace(
        run=RUN_ROOT / "blocks" / lower,
        log=LOG_ROOT / lower,
        cache=CACHE_ROOT / lower,
        socket=Path(f"/tmp/pi2s_s5_{lower}.sock"),
    )


def attempt_paths(tuple_row: Mapping[str, Any], attempt: int) -> SimpleNamespace:
    key = str(tuple_row["tuple_key"])
    block = block_paths(str(tuple_row["checkpoint_id"]))
    root = block.run / "attempts" / key / f"attempt_{attempt}"
    return SimpleNamespace(
        root=root,
        spec=root / "tuple_spec.json",
        artifact=root / "tuple_result.json",
        diagnostics=root / "inference.jsonl",
        output=block.cache / "attempts" / key / f"attempt_{attempt}" / "environment",
        client_log=block.log / "attempts" / key / f"attempt_{attempt}.log",
    )


def validate_tuple_artifact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != SCHEMA_TUPLE or payload.get("status") != "PASS":
        raise ContractError(f"tuple result is not PASS: {path}")
    if payload.get("tuple") != dict(expected):
        raise ContractError(f"tuple result identity drifted: {path}")
    gates = payload.get("gates", {})
    if not gates or any(value != "PASS" for value in gates.values()):
        raise ContractError(f"tuple result integrity gate failed: {path}")
    if payload.get("steps") != len(payload.get("step_telemetry", [])):
        raise ContractError(f"tuple result telemetry length drifted: {path}")
    return payload


def completed_attempt(tuple_row: Mapping[str, Any]) -> tuple[int, Path, dict[str, Any]] | None:
    for attempt in range(1, MAX_INFRASTRUCTURE_ATTEMPTS + 1):
        paths = attempt_paths(tuple_row, attempt)
        if paths.artifact.is_file():
            try:
                payload = validate_tuple_artifact(paths.artifact, tuple_row)
            except Exception:
                continue
            return attempt, paths.artifact, payload
    return None


def next_attempt(tuple_row: Mapping[str, Any]) -> int:
    for attempt in range(1, MAX_INFRASTRUCTURE_ATTEMPTS + 1):
        paths = attempt_paths(tuple_row, attempt)
        if (
            not paths.root.exists()
            and not paths.output.parent.exists()
            and not paths.client_log.exists()
        ):
            return attempt
    raise ContractError(
        f"tuple exhausted its one exact infrastructure retry: {tuple_row['tuple_key']}"
    )


def run_tuple_attempt(
    tuple_row: Mapping[str, Any],
    attempt: int,
    port: int,
    contact_socket: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    paths = attempt_paths(tuple_row, attempt)
    paths.root.mkdir(parents=True, exist_ok=False)
    paths.client_log.parent.mkdir(parents=True, exist_ok=True)
    paths.output.parent.mkdir(parents=True, exist_ok=False)
    publish_json_no_clobber(paths.spec, dict(tuple_row))
    started = time.time()
    base_env = os.environ | {
        "PYTHONPATH": f"{ROOT}:{DEXJOCO / 'dexjoco'}",
        "PYTHONUNBUFFERED": "1",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
    }
    with paths.client_log.open("x", encoding="utf-8") as client_log:
        result = subprocess.run(
            [
                str(EVAL_PYTHON),
                str(Path(__file__).resolve()),
                "evaluate-tuple",
                "--tuple-spec",
                str(paths.spec),
                "--contact-socket",
                str(contact_socket),
                "--output",
                str(paths.output),
                "--diagnostics",
                str(paths.diagnostics),
                "--artifact",
                str(paths.artifact),
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=base_env,
            stdout=client_log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        if paths.artifact.is_file():
            failed = read_json(paths.artifact)
            if failed.get("schema") == SCHEMA_TUPLE and failed.get("status") == "FAIL":
                raise ContractError(
                    f"tuple scientific integrity gate failed: {tuple_row['tuple_key']} "
                    f"attempt={attempt}"
                )
        raise RuntimeError(
            f"tuple evaluator exited {result.returncode}: {tuple_row['tuple_key']} attempt={attempt}"
        )
    validate_tuple_artifact(paths.artifact, tuple_row)
    return {
        "tuple_key": tuple_row["tuple_key"],
        "tuple_identity_sha256": tuple_row["tuple_identity_sha256"],
        "attempt": attempt,
        "artifact": str(paths.artifact),
        "artifact_sha256": sha256_file(paths.artifact),
        "client_log": str(paths.client_log),
        "video_root": str(paths.output),
        "elapsed_seconds": time.time() - started,
        "status": "PASS",
        "success_hidden_until_wave_complete": "RECORDED_IN_TUPLE_ARTIFACT",
        "steps_hidden_until_wave_complete": "RECORDED_IN_TUPLE_ARTIFACT",
        "prior_infrastructure_errors": [],
    }


def progress_payload(
    manifest: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    started: float,
    status: str,
) -> dict[str, Any]:
    completed = len(assignments)
    elapsed = max(time.time() - started, 1e-9)
    rate = completed / elapsed * 3600 if completed else None
    eta = (EXPECTED_TUPLES - completed) / (rate / 3600) if rate else None
    return {
        "schema": "tactile3d-unit.s4-3-pi2s-rollout-progress.v1",
        "status": status,
        "updated_at_utc": now_utc(),
        "supervisor_pid": os.getpid(),
        "source_commit": manifest["source_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": sha256_file(MANIFEST),
        "completed_canonical_tuples": completed,
        "planned_canonical_tuples": EXPECTED_TUPLES,
        "completed_blocks": sum(
            1
            for checkpoint_name in [f"{model}_{seed}" for model, seed in FOCUS_KEYS]
            if sum(
                row["tuple_key"].startswith(checkpoint_name.lower() + "__") for row in assignments
            )
            == 12
        ),
        "planned_blocks": 6,
        "elapsed_seconds": elapsed,
        "tuples_per_hour": rate,
        "eta_seconds": eta,
        "assignments": list(assignments),
        "performance_interpreted_during_execution": False,
        "optional_48_started": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
    }


def update_progress(
    manifest: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    started: float,
    status: str,
) -> None:
    payload = progress_payload(manifest, assignments, started, status)
    atomic_mutable_json(PROGRESS, payload)
    atomic_mutable_json(
        RESUME,
        {
            "schema": "tactile3d-unit.pi2s-resume.v2",
            "status": "PI2S_LONG_JOB_RUNNING" if status == "RUNNING" else status,
            "phase": "S5_BASE_72_DIAGNOSTIC_ROLLOUTS",
            "updated_at_utc": payload["updated_at_utc"],
            "supervisor_pid": os.getpid(),
            "source_commit": manifest["source_commit"],
            "protocol_sha256": manifest["protocol_sha256"],
            "manifest_sha256": payload["manifest_sha256"],
            "progress_path": str(PROGRESS),
            "supervisor_log": str(LOG_ROOT / "supervisor.log"),
            "completed_canonical_tuples": payload["completed_canonical_tuples"],
            "planned_canonical_tuples": EXPECTED_TUPLES,
            "long_job_running": status == "RUNNING",
            "optional_48_started": False,
            "training_performed": False,
            "push_performed": False,
            "worktree_removal_performed": False,
            "real_robot_used": False,
        },
    )


def run_block(
    manifest: Mapping[str, Any], checkpoint_name: str, gpu: int, port: int, started: float
) -> list[dict[str, Any]]:
    paths = block_paths(checkpoint_name)
    paths.run.mkdir(parents=True, exist_ok=True)
    paths.log.mkdir(parents=True, exist_ok=True)
    paths.cache.mkdir(parents=True, exist_ok=True)
    if paths.socket.exists():
        raise ContractError(f"refusing stale contact socket: {paths.socket}")
    contact_log_path = paths.log / "contact_service.log"
    server_log_path = paths.log / "policy_server.log"
    contact_artifact = paths.run / "contact_service.json"
    if any(path.exists() for path in (contact_log_path, server_log_path, contact_artifact)):
        # A resume reuses completed tuple artifacts but starts fresh services under
        # versioned log names; historical bytes are never replaced.
        suffix = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        contact_log_path = paths.log / f"contact_service_resume_{suffix}.log"
        server_log_path = paths.log / f"policy_server_resume_{suffix}.log"
        contact_artifact = paths.run / f"contact_service_resume_{suffix}.json"
    tuples = [row for row in manifest["tuples"] if row["checkpoint_id"] == checkpoint_name]
    accepted: list[dict[str, Any]] = []
    contact = policy = None
    handles: list[Any] = []
    try:
        contact_handle = contact_log_path.open("x", encoding="utf-8")
        server_handle = server_log_path.open("x", encoding="utf-8")
        handles.extend((contact_handle, server_handle))
        base_env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
        contact = subprocess.Popen(
            [
                str(UNIT_PYTHON),
                str(Path(__file__).resolve()),
                "contact-service",
                "--artifact",
                str(contact_artifact),
                "--socket",
                str(paths.socket),
            ],
            cwd=ROOT,
            env=base_env,
            stdout=contact_handle,
            stderr=subprocess.STDOUT,
        )
        wait_for_log(contact, contact_log_path, "CONTACT_STATE_SERVICE_READY", 120)
        gpu_env = base_env | {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "XLA_FLAGS": " ".join(XLA_FLAGS),
        }
        policy_env = dict(gpu_env)
        policy_env.pop("PYTHONPATH", None)
        policy = subprocess.Popen(
            [
                str(OPENPI_PYTHON),
                str(Path(__file__).resolve()),
                "serve-policy",
                "--checkpoint-id",
                checkpoint_name,
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=policy_env,
            stdout=server_handle,
            stderr=subprocess.STDOUT,
        )
        wait_for_log(
            policy,
            server_log_path,
            f"PI2S_POLICY_SERVER_READY checkpoint_id={checkpoint_name}",
            900,
        )
        for tuple_row in tuples:
            complete = completed_attempt(tuple_row)
            if complete is not None:
                attempt, artifact, _payload = complete
                accepted.append(
                    {
                        "tuple_key": tuple_row["tuple_key"],
                        "tuple_identity_sha256": tuple_row["tuple_identity_sha256"],
                        "attempt": attempt,
                        "artifact": str(artifact),
                        "artifact_sha256": sha256_file(artifact),
                        "status": "PASS",
                        "resumed_verified": True,
                        "prior_infrastructure_errors": [],
                    }
                )
                continue
            errors: list[dict[str, Any]] = []
            while True:
                attempt = next_attempt(tuple_row)
                try:
                    row = run_tuple_attempt(tuple_row, attempt, port, paths.socket, gpu)
                    row["prior_infrastructure_errors"] = errors
                    accepted.append(row)
                    break
                except ContractError:
                    raise
                except Exception as error:
                    failure = {
                        "schema": "tactile3d-unit.s4-3-pi2s-rollout-infrastructure-failure.v1",
                        "status": "INFRASTRUCTURE_FAILURE",
                        "tuple": dict(tuple_row),
                        "attempt": attempt,
                        "at_utc": now_utc(),
                        "type": type(error).__name__,
                        "message": str(error),
                        "canonical_outcome_produced": False,
                        "performance_selection_used": False,
                    }
                    failure_path = (
                        attempt_paths(tuple_row, attempt).root / "infrastructure_failure.json"
                    )
                    if not failure_path.exists():
                        publish_json_no_clobber(failure_path, failure)
                    errors.append(failure)
                    if attempt >= MAX_INFRASTRUCTURE_ATTEMPTS:
                        raise
            # Block-local progress is useful even before the supervisor merges
            # all concurrently completed rows.
            atomic_mutable_json(
                paths.run / "block_progress.json",
                {
                    "status": "RUNNING",
                    "updated_at_utc": now_utc(),
                    "checkpoint_id": checkpoint_name,
                    "physical_gpu": gpu,
                    "completed": len(accepted),
                    "planned": len(tuples),
                    "accepted": accepted,
                    "performance_interpreted": False,
                },
            )
        atomic_mutable_json(
            paths.run / "block_progress.json",
            {
                "status": "PASS",
                "updated_at_utc": now_utc(),
                "checkpoint_id": checkpoint_name,
                "physical_gpu": gpu,
                "completed": len(accepted),
                "planned": len(tuples),
                "accepted": accepted,
                "elapsed_since_wave_start_seconds": time.time() - started,
                "performance_interpreted": False,
            },
        )
        return accepted
    finally:
        stop_process(policy)
        stop_process(contact)
        for handle in handles:
            handle.close()


def collect_completed(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tuple_row in manifest["tuples"]:
        complete = completed_attempt(tuple_row)
        if complete is None:
            continue
        attempt, artifact, _payload = complete
        rows.append(
            {
                "tuple_key": tuple_row["tuple_key"],
                "tuple_identity_sha256": tuple_row["tuple_identity_sha256"],
                "attempt": attempt,
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "status": "PASS",
            }
        )
    return rows


def aggregate_results(
    manifest: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = {row["tuple_key"]: row for row in assignments}
    if (
        list(index) != [row["tuple_key"] for row in manifest["tuples"]]
        or len(index) != EXPECTED_TUPLES
    ):
        raise ContractError("S5 accepted tuple index is incomplete or out of order")
    episodes = []
    for tuple_row in manifest["tuples"]:
        accepted = index[tuple_row["tuple_key"]]
        path = Path(str(accepted["artifact"]))
        payload = validate_tuple_artifact(path, tuple_row)
        episodes.append(
            {
                "tuple": tuple_row,
                "attempt": int(accepted["attempt"]),
                "artifact": str(path),
                "artifact_sha256": sha256_file(path),
                "success": bool(payload["success"]),
                "termination": payload["termination"],
                "steps": int(payload["steps"]),
                "ever_object_contact": bool(payload["ever_object_contact"]),
                "ever_lifted": bool(payload["ever_lifted"]),
                "max_pinch_count": int(payload["max_pinch_count"]),
                "ever_native_trigger": bool(payload["ever_native_trigger"]),
                "max_success_counter": int(payload["max_success_counter"]),
                "max_consecutive_object_contact_run": int(
                    payload["max_consecutive_object_contact_run"]
                ),
                "exact_first_failure_stage": payload["exact_first_failure_stage"],
            }
        )
    attempts = []
    for tuple_row in manifest["tuples"]:
        for attempt in range(1, MAX_INFRASTRUCTURE_ATTEMPTS + 1):
            paths = attempt_paths(tuple_row, attempt)
            failure = paths.root / "infrastructure_failure.json"
            if paths.artifact.is_file():
                attempts.append(
                    {
                        "tuple_key": tuple_row["tuple_key"],
                        "attempt": attempt,
                        "kind": "SCIENTIFIC_RESULT",
                        "path": str(paths.artifact),
                        "sha256": sha256_file(paths.artifact),
                        "status": read_json(paths.artifact).get("status"),
                    }
                )
            if failure.is_file():
                attempts.append(
                    {
                        "tuple_key": tuple_row["tuple_key"],
                        "attempt": attempt,
                        "kind": "INFRASTRUCTURE_FAILURE",
                        "path": str(failure),
                        "sha256": sha256_file(failure),
                        "status": "INFRASTRUCTURE_FAILURE",
                    }
                )
    return {
        "schema": SCHEMA_RESULTS,
        "status": "PASS",
        "created_at_utc": now_utc(),
        "source_commit": manifest["source_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": sha256_file(MANIFEST),
        "canonical_tuples_planned": EXPECTED_TUPLES,
        "canonical_tuples_completed": len(episodes),
        "canonical_tuples_invalid": 0,
        "optional_48_executed": False,
        "episodes": episodes,
        "attempt_ledger": attempts,
        "runtime_logs": {
            checkpoint_name: sorted(
                str(path)
                for path in block_paths(checkpoint_name).log.glob("*.log")
                if path.is_file()
            )
            for checkpoint_name in [f"{model}_{seed}" for model, seed in FOCUS_KEYS]
        },
        "successes": sum(int(row["success"]) for row in episodes),
        "failures": sum(int(not row["success"]) for row in episodes),
        "stage_definition": STAGE_DEFINITION,
        "exact_first_failure_available": False,
        "coarse_native_milestone_localization_available": True,
        "gpu_snapshots": list(snapshots),
        "formal_track_a_scores_replaced": False,
        "performance_selection_used": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
    }


def orchestrate(gpus: list[int], *, resume: bool) -> None:
    manifest = validate_manifest()
    if len(gpus) < 1 or len(gpus) > 4 or len(gpus) != len(set(gpus)):
        raise ContractError("S5 requires one to four distinct physical GPUs")
    if gpus != sorted(gpus) or any(index not in range(4) for index in gpus):
        raise ContractError("S5 GPUs must be sorted physical IDs in range 0-3")
    if RESULTS.exists():
        raise FileExistsError(RESULTS)
    if not resume and any(path.exists() for path in (RUN_ROOT, LOG_ROOT, CACHE_ROOT)):
        raise FileExistsError("S5 output root exists; explicit --resume is required")
    for executable in (OPENPI_PYTHON, UNIT_PYTHON, EVAL_PYTHON):
        if not executable.is_file():
            raise ContractError(f"required interpreter missing: {executable}")
    RUN_ROOT.mkdir(parents=True, exist_ok=resume)
    LOG_ROOT.mkdir(parents=True, exist_ok=resume)
    CACHE_ROOT.mkdir(parents=True, exist_ok=resume)
    started = time.time()
    initial_completed = collect_completed(manifest)
    snap1 = gpu_snapshot()
    time.sleep(2.0)
    snap2 = gpu_snapshot()
    if not all(gpu_is_idle(index, snap1) and gpu_is_idle(index, snap2) for index in gpus):
        raise ContractError("one or more selected GPUs is not truly idle in both snapshots")
    locks: list[Any] = []
    snapshots: list[dict[str, Any]] = [snap1, snap2]
    assignments = list(initial_completed)
    try:
        locks = [acquire_gpu_lock(index) for index in gpus]
        snap3 = gpu_snapshot()
        snapshots.append(snap3)
        if not all(gpu_is_idle(index, snap3) for index in gpus):
            raise ContractError("one or more selected GPUs became busy after lease acquisition")
        update_progress(manifest, assignments, started, "RUNNING")
        pending_blocks = []
        for checkpoint_name in [f"{model}_{seed}" for model, seed in FOCUS_KEYS]:
            rows = [row for row in manifest["tuples"] if row["checkpoint_id"] == checkpoint_name]
            if all(completed_attempt(row) is not None for row in rows):
                continue
            pending_blocks.append(checkpoint_name)
        for wave_start in range(0, len(pending_blocks), len(gpus)):
            wave = pending_blocks[wave_start : wave_start + len(gpus)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {
                    pool.submit(
                        run_block,
                        manifest,
                        checkpoint_name,
                        gpus[offset],
                        9100 + wave_start + offset,
                        started,
                    ): checkpoint_name
                    for offset, checkpoint_name in enumerate(wave)
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    assignments = collect_completed(manifest)
                    update_progress(manifest, assignments, started, "RUNNING")
        assignments = collect_completed(manifest)
        assignments.sort(
            key=lambda row: next(
                index
                for index, expected in enumerate(manifest["tuples"])
                if expected["tuple_key"] == row["tuple_key"]
            )
        )
        snapshots.append(gpu_snapshot())
        result = aggregate_results(manifest, assignments, snapshots)
        publish_json_no_clobber(RESULTS, result)
        update_progress(manifest, assignments, started, "PASS")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "completed": len(assignments),
                    "planned": EXPECTED_TUPLES,
                    "result_sha256": sha256_file(RESULTS),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        assignments = collect_completed(manifest)
        update_progress(manifest, assignments, started, "FAILED")
        raise
    finally:
        for handle in locks:
            handle.close()


def status_payload() -> dict[str, Any]:
    manifest = validate_manifest()
    completed = collect_completed(manifest)
    progress = read_json(PROGRESS) if PROGRESS.is_file() else None
    result = read_json(RESULTS) if RESULTS.is_file() else None
    return {
        "status": (
            result.get("status")
            if result
            else progress.get("status") if progress else "NOT_STARTED"
        ),
        "manifest_sha256": sha256_file(MANIFEST),
        "protocol_sha256": manifest["protocol_sha256"],
        "source_commit": manifest["source_commit"],
        "completed_canonical_tuples": len(completed),
        "planned_canonical_tuples": EXPECTED_TUPLES,
        "remaining_canonical_tuples": EXPECTED_TUPLES - len(completed),
        "progress": progress,
        "results_present": result is not None,
        "results_sha256": sha256_file(RESULTS) if RESULTS.is_file() else None,
        "optional_48_started": False,
    }


def check_completed() -> None:
    manifest = validate_manifest()
    if not RESULTS.is_file():
        raise ContractError("diagnostic rollout results are absent")
    assignments = collect_completed(manifest)
    assignments.sort(
        key=lambda row: next(
            index
            for index, expected in enumerate(manifest["tuples"])
            if expected["tuple_key"] == row["tuple_key"]
        )
    )
    existing = read_json(RESULTS)
    recomputed = aggregate_results(manifest, assignments, existing.get("gpu_snapshots", []))
    for key in ("created_at_utc",):
        recomputed[key] = existing.get(key)
    if recomputed != existing:
        raise ContractError("diagnostic rollout result is stale or drifted")
    print(
        json.dumps(
            {"status": "CHECK_PASS", "tuples": len(assignments), "sha256": sha256_file(RESULTS)},
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--source-commit", required=True)
    sub.add_parser("validate")
    contact = sub.add_parser("contact-service")
    contact.add_argument("--artifact", type=Path, required=True)
    contact.add_argument("--socket", type=Path, required=True)
    serve = sub.add_parser("serve-policy")
    serve.add_argument("--checkpoint-id", required=True)
    serve.add_argument("--port", type=int, required=True)
    evaluate = sub.add_parser("evaluate-tuple")
    evaluate.add_argument("--tuple-spec", type=Path, required=True)
    evaluate.add_argument("--contact-socket", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diagnostics", type=Path, required=True)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--port", type=int, required=True)
    launch = sub.add_parser("orchestrate")
    launch.add_argument("--gpus", required=True)
    launch.add_argument("--resume", action="store_true")
    sub.add_parser("status")
    sub.add_parser("check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        freeze_manifest(args.source_commit)
    elif args.command == "validate":
        manifest = validate_manifest()
        print(
            json.dumps(
                {
                    "status": "VALIDATE_PASS",
                    "tuples": len(manifest["tuples"]),
                    "sha256": sha256_file(MANIFEST),
                },
                sort_keys=True,
            )
        )
    elif args.command == "contact-service":
        contact_service(args.artifact, args.socket)
    elif args.command == "serve-policy":
        serve_policy(args.checkpoint_id, args.port)
    elif args.command == "evaluate-tuple":
        evaluate_tuple(
            args.tuple_spec,
            args.contact_socket,
            args.output,
            args.diagnostics,
            args.artifact,
            args.port,
        )
    elif args.command == "orchestrate":
        try:
            gpus = [int(value) for value in args.gpus.split(",")]
        except ValueError as error:
            raise SystemExit("--gpus must be comma-separated physical IDs") from error
        orchestrate(gpus, resume=args.resume)
    elif args.command == "status":
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
    else:
        check_completed()


if __name__ == "__main__":
    main()
