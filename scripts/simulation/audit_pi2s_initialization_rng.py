#!/usr/bin/env python3
"""Audit PI2S initialization, RNG, data, loader, and source provenance on CPU.

This audit consumes only frozen JSON manifests, small source snapshots, source
files, and data/base identity manifests.  It never imports JAX/OpenPI, restores
model weights, opens a simulator, or uses a GPU.  In particular, source-level
RNG control flow is kept separate from values that were actually persisted by
the historical training processes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import secrets
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True

PI2B = (ROOT / ".local/experiments/simulation/s4_3_pi2b_policy").resolve()
PI2S = (ROOT / ".local/experiments/simulation/s4_3_pi2s").resolve()
# The original canonical filename is a preserved v1 artifact produced before
# publication hardening.  Publish the hardened schema under an explicit v2
# name so a successful rerun never requires replacing or deleting that history.
OUTPUT = PI2S / "artifacts/initialization_rng_audit_v2.json"
CHECKPOINT_MATRIX = PI2S / "artifacts/checkpoint_recipe_matrix.json"
SCHEMA = "tactile3d-unit.s4-3-pi2s-initialization-rng-audit.v2"

TRACK_A_ARTIFACTS = PI2B / "artifacts"
TRACK_A_RUNS = TRACK_A_ARTIFACTS / "training_runs.json"
TRACK_A_SNAPSHOT_MANIFEST = TRACK_A_ARTIFACTS / "code_snapshot_manifest.json"
TRACK_A_START = TRACK_A_ARTIFACTS / "starting_integrity.json"
TRACK_A_IMPORTS = TRACK_A_ARTIFACTS / "import_origins.json"
TRACK_A_PROTOCOL = ROOT / "configs/simulation/pi2b_policy/protocol.json"
TRACK_A_RECIPES = ROOT / "configs/simulation/pi2b_policy/model_recipes.json"

PI0 = ROOT / ".local/artifacts/simulation/s4_3_pi0"
PI1 = ROOT / ".local/artifacts/simulation/s4_3_pi1"
PI2M = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
PI2N = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
OPENPI_ROOT = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
BASE_MANIFEST = PI0 / "official_base_model_manifest.json"
DATASET_MANIFEST = PI0 / "official_dataset_manifest.json"
PI2S_DATA_SNAPSHOTS = PI2S / "source_snapshots/data"
CONTACT_SIDECAR = PI2S_DATA_SNAPSHOTS / "pi1_contact_sidecar.npz"
VA27_SIDECAR = PI2S_DATA_SNAPSHOTS / "pi2m_va27_sidecar.npz"
BVA_SOURCE_SNAPSHOT = PI2S.parent / "s4_3_pi2n/code_snapshots/B_VA27/prelaunch_source.tar.gz"

MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2")
SEEDS = (42, 43, 44)
TRACK_A_SEEDS = (43, 44)

SEED42_INPUTS: dict[str, dict[str, Path]] = {
    "B0": {
        "freeze": PI0 / "official_training_config.json",
        "completion": PI0 / "training_completion.json",
        "launch": PI0 / "training_launch.json",
    },
    "B_VA27": {
        "freeze": PI2N / "b_va27_training_freeze.json",
        "completion": PI2N / "b_va27_training_completion.json",
    },
    "B1": {
        "freeze": PI1 / "training_protocol_freeze.json",
        "completion": PI1 / "pi1b_training_completion.json",
        "launch": PI1 / "pi1b_launch.json",
        "base_gate": PI1 / "pi1b_loaded_base_gradient.json",
    },
    "B_HVA": {
        "freeze": PI2M / "training_protocol_freeze.json",
        "completion": PI2M / "training_completion.json",
        "launch": PI2M / "training_launch.json",
    },
    "B2": {
        "freeze": PI1 / "training_protocol_freeze.json",
        "completion": PI1 / "pi1c_training_completion.json",
        "launch": PI1 / "pi1c_launch.json",
        "calibration": PI1 / "pi1c_lambda_calibration.json",
    },
}


class AuditError(RuntimeError):
    """Raised when frozen evidence violates an audit invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


class InputRegistry:
    """Read every persistent input once and retain exact byte provenance."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._data: dict[str, bytes] = {}

    def bytes(self, path: Path, role: str, expected_sha256: str | None = None) -> bytes:
        require(path.is_file(), f"missing input for {role}: {path}")
        resolved = str(path.resolve())
        if resolved not in self._data:
            self._data[resolved] = path.read_bytes()
        data = self._data[resolved]
        observed = sha256_bytes(data)
        if expected_sha256 is not None:
            require(observed == expected_sha256, f"input SHA mismatch for {role}: {path}")

        if resolved in self._records:
            record = self._records[resolved]
            require(record["sha256"] == observed, f"input changed while auditing: {path}")
            if role not in record["roles"]:
                record["roles"].append(role)
                record["roles"].sort()
            previous_expected = record["expected_sha256"]
            if expected_sha256 is not None:
                require(
                    previous_expected in {None, expected_sha256},
                    f"conflicting expected SHA values for {path}",
                )
                record["expected_sha256"] = expected_sha256
                record["expected_sha256_status"] = "MATCH"
            return data

        self._records[resolved] = {
            "path": resolved,
            "bytes": len(data),
            "sha256": observed,
            "roles": [role],
            "expected_sha256": expected_sha256,
            "expected_sha256_status": "MATCH" if expected_sha256 is not None else "NOT_PROVIDED",
        }
        return data

    def json(self, path: Path, role: str, expected_sha256: str | None = None) -> dict[str, Any]:
        data = self.bytes(path, role, expected_sha256)
        try:
            value = json.loads(data)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSON input {path}: {error}") from error
        require(isinstance(value, dict), f"JSON input is not an object: {path}")
        return value

    def records(self) -> list[dict[str, Any]]:
        return [json_copy(self._records[key]) for key in sorted(self._records)]

    def manifest_sha256(self) -> str:
        return canonical_sha256(self.records())

    def observed_sha256(self, path: Path) -> str:
        resolved = str(path.resolve())
        require(resolved in self._records, f"input not registered: {path}")
        return str(self._records[resolved]["sha256"])

    def verify_unchanged(self) -> None:
        for record in self._records.values():
            path = Path(record["path"])
            require(path.is_file(), f"input disappeared while auditing: {path}")
            data = path.read_bytes()
            require(len(data) == record["bytes"], f"input size changed while auditing: {path}")
            require(
                sha256_bytes(data) == record["sha256"],
                f"input bytes changed while auditing: {path}",
            )


def verify_semantic_artifact(payload: Mapping[str, Any], label: str) -> None:
    semantic = payload.get("semantic_sha256")
    require(isinstance(semantic, str) and len(semantic) == 64, f"{label} lacks semantic SHA")
    view = json_copy(payload)
    view.pop("created_at_utc", None)
    view.pop("semantic_sha256", None)
    require(canonical_sha256(view) == semantic, f"{label} semantic SHA mismatch")


def require_all_pass(gates: Mapping[str, Any], label: str) -> None:
    require(bool(gates), f"{label} has no gates")
    failures = {key: value for key, value in gates.items() if value != "PASS"}
    require(not failures, f"{label} gate failures: {failures}")


def manifest_projection(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {"path": str(row["path"]), "bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
            for row in rows
        ),
        key=lambda row: row["path"],
    )


def read_snapshot_members(
    inputs: InputRegistry,
    path: Path,
    role: str,
    expected_archive_sha256: str,
    expected_members: Mapping[str, str],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    archive_bytes = inputs.bytes(path, role, expected_archive_sha256)
    payloads: dict[str, bytes] = {}
    identities: dict[str, dict[str, Any]] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive:
            names = archive.getnames()
            require(len(names) == len(set(names)), f"duplicate member names in {role}")
            for name, expected_sha in sorted(expected_members.items()):
                try:
                    member = archive.getmember(name)
                except KeyError as error:
                    raise AuditError(f"missing {role} member: {name}") from error
                require(member.isfile(), f"non-regular {role} member: {name}")
                stream = archive.extractfile(member)
                require(stream is not None, f"unreadable {role} member: {name}")
                data = stream.read()
                observed = sha256_bytes(data)
                require(observed == expected_sha, f"{role} member SHA mismatch: {name}")
                payloads[name] = data
                identities[name] = {
                    "bytes": len(data),
                    "sha256": observed,
                    "expected_sha256": expected_sha,
                    "status": "MATCH",
                }
    except tarfile.TarError as error:
        raise AuditError(f"invalid source snapshot {path}: {error}") from error
    return payloads, identities


def source_statements(source: bytes, statements: Iterable[str], label: str) -> list[dict[str, str]]:
    text = source.decode("utf-8")
    output = []
    for statement in statements:
        require(statement in text, f"missing frozen source statement in {label}: {statement}")
        output.append(
            {
                "statement": statement,
                "evidence_kind": "EXACT_STATEMENT_IN_SHA256_BOUND_SOURCE",
                "status": "OBSERVED",
            }
        )
    return output


def output_artifact(body: dict[str, Any], inputs: InputRegistry) -> dict[str, Any]:
    records = inputs.records()
    semantic = {
        "schema": SCHEMA,
        **body,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
            "execution": "CPU_ONLY_READ_ONLY_INPUTS_NO_MODEL_LOAD_NO_SIMULATOR_NO_GPU",
        },
        "input_manifest": records,
        "input_manifest_sha256": inputs.manifest_sha256(),
        "input_manifest_summary": {
            "observed_sha256_recorded": len(records),
            "provided_expected_sha256_match": sum(
                row["expected_sha256_status"] == "MATCH" for row in records
            ),
            "expected_sha256_not_provided": sum(
                row["expected_sha256_status"] == "NOT_PROVIDED" for row in records
            ),
        },
    }
    return {
        **semantic,
        "created_at_utc": now_utc(),
        "semantic_sha256": canonical_sha256(semantic),
    }


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def publish_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically install a new JSON file without ever replacing a directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(12)}")
    linked = False
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # POSIX hard-link creation is atomic and fails with EEXIST for every
            # pre-existing destination directory entry, including symlinks.
            try:
                os.link(temporary, path, follow_symlinks=False)
            except PermissionError:
                # NFS identity mapping can expose the newly-created inode with
                # a different local owner and trigger protected_hardlinks.
                # Briefly make the temporary inode writable, link it with the
                # same atomic no-clobber primitive, then seal and revalidate it.
                os.chmod(temporary, 0o666)
                os.link(temporary, path, follow_symlinks=False)
            linked = True
        except FileExistsError as error:
            raise AuditError(f"refusing to overwrite existing output: {path}") from error
        except OSError as error:
            raise AuditError(f"atomic no-clobber publish failed for {path}: {error}") from error
        try:
            os.chmod(path, 0o444)
            observed = path.read_bytes()
            require(observed == data, f"atomic publish readback mismatch: {path}")
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
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)
    require(linked, f"output was not published: {path}")
    return sha256_bytes(data)


def check_artifact(path: Path, expected: Mapping[str, Any]) -> str:
    require(path.is_file(), f"missing output for --check: {path}")
    try:
        current = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid output JSON {path}: {error}") from error
    require(isinstance(current, dict), f"output is not a JSON object: {path}")
    created = current.pop("created_at_utc", None)
    observed_semantic = current.pop("semantic_sha256", None)
    expected_view = json_copy(expected)
    expected_view.pop("created_at_utc", None)
    expected_semantic = expected_view.pop("semantic_sha256", None)
    require(isinstance(created, str) and created, f"missing created_at_utc: {path}")
    require(observed_semantic == canonical_sha256(current), f"bad semantic SHA: {path}")
    require(expected_semantic == canonical_sha256(expected_view), "internal semantic SHA error")
    require(current == expected_view, f"output is stale or drifted: {path}")
    return sha256_bytes(path.read_bytes())


def execute_artifact(path: Path, artifact: Mapping[str, Any], *, check: bool) -> dict[str, Any]:
    if check:
        output_sha = check_artifact(path, artifact)
        return {
            "status": "CHECK_PASS",
            "operation": "READ_ONLY_RECOMPUTE_AND_COMPARE",
            "write_attempted": False,
            "overwrite_allowed": False,
            "output_sha256": output_sha,
        }
    output_sha = publish_json_no_clobber(path, artifact)
    require(check_artifact(path, artifact) == output_sha, "published output check mismatch")
    return {
        "status": "PUBLISH_NO_CLOBBER_AND_READBACK_PASS",
        "operation": "ATOMIC_SAME_DIRECTORY_HARD_LINK_NO_CLOBBER_PUBLISH",
        "write_attempted": True,
        "overwrite_allowed": False,
        "output_sha256": output_sha,
    }


def _load_seed42(inputs: InputRegistry) -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for model, paths in SEED42_INPUTS.items():
        evidence: dict[str, dict[str, Any]] = {}
        for kind, path in paths.items():
            payload = inputs.json(path, f"seed42 {model} {kind} evidence")
            require(
                payload.get("status") in {"PASS", "FROZEN_BEFORE_TRAINING", "LAUNCHING"},
                f"bad seed42 {model} {kind} status",
            )
            evidence[kind] = payload
        completion = evidence["completion"]
        if "seed" in completion:
            require(completion["seed"] == 42, f"seed42 {model} completion seed drift")
        gates = completion.get("gates")
        if isinstance(gates, dict):
            require_all_pass(gates, f"seed42 {model} completion")
        loaded[model] = evidence
    return loaded


def _seed42_direct_fields(
    model: str,
    evidence: Mapping[str, Mapping[str, Any]],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = evidence["freeze"]
    completion = evidence["completion"]
    launch = evidence.get("launch")
    if model == "B0":
        recorded_seed = int(freeze["training"]["seed"])
        recorded_batch = int(freeze["training"]["runtime_parallelism"]["global_batch_size"])
        recorded_steps = int(freeze["training"]["num_train_steps"])
        recorded_mode = "NONE"
        mode_evidence = "STATIC_INTERPRETATION_OF_OFFICIAL_NONTACTILE_CONFIG"
        initialization = f"official {freeze['model']['base_model']}"
        initialization_evidence = "DIRECT_FREEZE_FIELD"
        resume = "UNVERIFIABLE_NOT_EXPLICIT_IN_SELECTED_FREEZE"
    else:
        recorded_seed = int(completion["seed"])
        recorded_batch = int(completion["global_batch_size"])
        recorded_steps = int(completion["optimizer_steps"])
        recorded_mode = str(completion["mode"])
        mode_evidence = "DIRECT_COMPLETION_FIELD"
        resume = freeze.get("resume", "UNVERIFIABLE_NOT_EXPLICIT_IN_SELECTED_FREEZE")
        if model == "B_VA27":
            initialization = str(freeze["initialization"])
            initialization_evidence = "DIRECT_FREEZE_FIELD"
        elif model == "B_HVA":
            initialization = str(freeze["initialization"])
            initialization_evidence = "DIRECT_FREEZE_FIELD"
        elif model == "B1":
            initialization = str(evidence["base_gate"]["initialization"])
            initialization_evidence = "DIRECT_LOADED_BASE_GATE_FIELD"
        else:
            initialization = str(evidence["calibration"]["model_state"])
            initialization_evidence = "DIRECT_PRETRAINING_CALIBRATION_FIELD"

    require(recorded_seed == 42, f"seed42 {model} recorded seed drift")
    require(recorded_batch == 32, f"seed42 {model} batch drift")
    require(recorded_steps == 30_000, f"seed42 {model} steps drift")
    require(recorded_mode == recipe["mode"], f"seed42 {model} mode drift")
    if completion.get("lambda_phys") is not None:
        require(
            float(completion["lambda_phys"]) == float(recipe["lambda_phys"]),
            f"seed42 {model} lambda drift",
        )
    if launch is not None:
        if launch.get("seed") is not None:
            require(int(launch["seed"]) == 42, f"seed42 {model} launch seed drift")
        if launch.get("mode") is not None:
            require(launch["mode"] == recipe["mode"], f"seed42 {model} launch mode drift")

    return {
        "numeric_training_seed": recorded_seed,
        "mode": recorded_mode,
        "mode_evidence_kind": mode_evidence,
        "global_batch_size": recorded_batch,
        "optimizer_steps": recorded_steps,
        "lambda_phys": recipe["lambda_phys"],
        "initialization_declaration": initialization,
        "initialization_evidence_kind": initialization_evidence,
        "resume": resume,
    }


def build_artifact() -> dict[str, Any]:
    inputs = InputRegistry()

    matrix = inputs.json(CHECKPOINT_MATRIX, "current PI2S checkpoint/recipe matrix v2")
    require(
        matrix.get("schema") == "tactile3d-unit.s4-3-pi2s-checkpoint-recipe-matrix.v2",
        "unexpected checkpoint matrix schema",
    )
    require(
        matrix.get("status") == "COMPLETE_WITH_DECLARED_UNVERIFIABLE_FIELDS",
        "checkpoint matrix status drift",
    )
    verify_semantic_artifact(matrix, "checkpoint recipe matrix")
    require(
        matrix["source_cohort_comparison"]["conclusion"]
        == "NOT_FULL_SOURCE_PARITY_SEED42_VS_SEEDS43_44",
        "checkpoint matrix source-cohort conclusion drift",
    )
    matrix_rows = {
        (str(row["model"]), int(row["training_seed"])): row for row in matrix["final_checkpoints"]
    }
    require(
        set(matrix_rows) == {(model, seed) for model in MODELS for seed in SEEDS},
        "checkpoint matrix shape drift",
    )

    protocol_bytes = inputs.bytes(TRACK_A_PROTOCOL, "Track A frozen protocol")
    recipe_bytes = inputs.bytes(TRACK_A_RECIPES, "Track A frozen model recipes")
    protocol = json.loads(protocol_bytes)
    recipes = json.loads(recipe_bytes)
    runs = inputs.json(TRACK_A_RUNS, "Track A frozen per-run configurations")
    snapshot_manifest = inputs.json(
        TRACK_A_SNAPSHOT_MANIFEST, "Track A immutable source snapshot manifest"
    )
    starting = inputs.json(TRACK_A_START, "Track A starting base/data/source integrity")
    imports = inputs.json(TRACK_A_IMPORTS, "Track A historical pretraining import origins")
    require(protocol.get("status") == "FROZEN_BEFORE_NEW_TRAINING", "Track A protocol not frozen")
    require(recipes.get("status") == "FROZEN_BEFORE_NEW_TRAINING", "Track A recipes not frozen")
    require(runs.get("status") == "FROZEN_NOT_STARTED", "Track A run freeze status drift")
    require(snapshot_manifest.get("status") == "PASS", "Track A source snapshot not PASS")
    require(starting.get("status") == "PASS", "Track A starting integrity not PASS")
    require(imports.get("status") == "PASS", "Track A imports not PASS")
    require_all_pass(starting["gates"], "Track A starting integrity")

    run_rows = {(str(row["model_id"]), int(row["seed"])): row for row in runs["rows"]}
    require(
        set(run_rows) == {(model, seed) for model in MODELS for seed in TRACK_A_SEEDS},
        "Track A run matrix drift",
    )
    protocol_sha = sha256_bytes(protocol_bytes)
    recipes_sha = sha256_bytes(recipe_bytes)
    source_maps = {canonical_sha256(row["source_sha256"]) for row in run_rows.values()}
    require(len(source_maps) == 1, "Track A rows do not share one source map")
    source_map = json_copy(next(iter(run_rows.values()))["source_sha256"])
    for key, row in run_rows.items():
        require(row["protocol_sha256"] == protocol_sha, f"Track A protocol SHA drift: {key}")
        require(row["recipes_sha256"] == recipes_sha, f"Track A recipes SHA drift: {key}")
        require(row["source_sha256"] == source_map, f"Track A source map drift: {key}")
        require(row["git_head"] == snapshot_manifest["git_head"], f"Track A git head drift: {key}")
        require(row["base"] == "official pi05_base", f"Track A base drift: {key}")
        require(
            row["dataset"] == "official pinch_tongs 100 episodes / 40065 rows",
            f"Track A dataset drift: {key}",
        )

    snapshot_path = Path(snapshot_manifest["path"])
    track_sources, track_source_identities = read_snapshot_members(
        inputs,
        snapshot_path,
        "immutable Track A source snapshot tar",
        str(snapshot_manifest["sha256"]),
        source_map,
    )

    openpi_expected = starting["source"]["openpi_hashes"]
    official_train_path = OPENPI_ROOT / "scripts/train.py"
    official_config_path = OPENPI_ROOT / "src/openpi/training/config.py"
    official_train = inputs.bytes(
        official_train_path,
        "accepted OpenPI trainer used by Track A",
        openpi_expected["scripts/train.py"],
    )
    inputs.bytes(
        official_config_path,
        "accepted OpenPI training config used by Track A",
        openpi_expected["src/openpi/training/config.py"],
    )

    builder_statements = source_statements(
        track_sources["gr00t/simulation/pi2b_policy/training.py"],
        (
            "loader = weight_loaders.CheckpointWeightLoader(str(workspace.base_params))",
            "loader = TactileCheckpointWeightLoader(str(workspace.base_params))",
            "seed=seed",
            "resume=False",
            "overwrite=False",
        ),
        "Track A config builder",
    )
    rng_statements = source_statements(
        official_train,
        (
            "rng = jax.random.key(config.seed)",
            "train_rng, init_rng = jax.random.split(rng)",
            "rng, model_rng = jax.random.split(rng)",
            "model = config.model.create(model_rng)",
            "partial_params = _load_weights_and_validate(config.weight_loader",
            "train_rng = jax.random.fold_in(rng, state.step)",
            "shuffle=True",
        ),
        "accepted OpenPI trainer",
    )

    seed42 = _load_seed42(inputs)
    bva_freeze = seed42["B_VA27"]["freeze"]
    base_manifest = inputs.json(
        BASE_MANIFEST,
        "official pi05_base file-identity manifest",
        str(bva_freeze["base_manifest_sha256"]),
    )
    dataset_manifest = inputs.json(
        DATASET_MANIFEST,
        "official pinch_tongs file-identity manifest",
        str(starting["dataset"]["manifest_sha256"]),
    )
    require(base_manifest.get("status") == "PASS", "official base manifest not PASS")
    require(dataset_manifest.get("status") == "PASS", "official dataset manifest not PASS")
    require(
        manifest_projection(base_manifest["files"])
        == manifest_projection(starting["base"]["files"]),
        "Track A base projection differs from official base manifest",
    )
    require(
        manifest_projection(dataset_manifest["files"])
        == manifest_projection(starting["dataset"]["files"]),
        "Track A dataset projection differs from official dataset manifest",
    )
    contact_sha = str(starting["sidecars"]["contact"]["sha256"])
    va27_sha = str(starting["sidecars"]["va27"]["sha256"])
    inputs.bytes(CONTACT_SIDECAR, "frozen contact-state training sidecar", contact_sha)
    inputs.bytes(VA27_SIDECAR, "frozen VA27 training sidecar", va27_sha)

    bva_sources, bva_source_identities = read_snapshot_members(
        inputs,
        BVA_SOURCE_SNAPSHOT,
        "seed42 B_VA27 prelaunch source snapshot",
        str(bva_freeze["source_snapshot_sha256"]),
        bva_freeze["source_files_sha256"],
    )
    source_statements(
        bva_sources["scripts/simulation/train_s4_3_pi2n.py"],
        (
            "weight_loader=TactileCheckpointWeightLoader(str(BASE_PARAMS))",
            "seed=42",
            "resume=False",
            "overwrite=False",
        ),
        "seed42 B_VA27 entrypoint",
    )

    b0_freeze = seed42["B0"]["freeze"]
    b0_train_path = ROOT / "third_party/dexjoco/openpi/scripts/train.py"
    b0_config_path = ROOT / "third_party/dexjoco/openpi/src/openpi/training/dexjoco_configs.py"
    b0_train = inputs.bytes(
        b0_train_path,
        "seed42 B0 frozen official trainer",
        b0_freeze["source_file_sha256"]["openpi/scripts/train.py"],
    )
    inputs.bytes(
        b0_config_path,
        "seed42 B0 frozen official DexJoCo config",
        b0_freeze["source_file_sha256"]["openpi/src/openpi/training/dexjoco_configs.py"],
    )
    b0_rng_statements = source_statements(
        b0_train,
        (
            "rng = jax.random.key(config.seed)",
            "train_rng, init_rng = jax.random.split(rng)",
            "rng, model_rng = jax.random.split(rng)",
            "model = config.model.create(model_rng)",
            "train_rng = jax.random.fold_in(rng, state.step)",
            "shuffle=True",
        ),
        "seed42 B0 trainer",
    )

    pi1_freeze = seed42["B1"]["freeze"]
    pi1_protocol_path = ROOT / "configs/simulation/s4_3_pi1_training_protocol.json"
    inputs.bytes(
        pi1_protocol_path,
        "seed42 B1/B2 frozen training protocol",
        str(pi1_freeze["config_sha256"]),
    )
    require(
        seed42["B1"]["base_gate"]["gates"]["pi05_base_not_B0_checkpoint"] == "PASS",
        "B1 base gate drift",
    )
    require(
        seed42["B2"]["calibration"]["gates"]["exact_pi05_base_initialization"] == "PASS",
        "B2 base gate drift",
    )
    require(
        seed42["B2"]["calibration"]["gates"]["seed42_deterministic_shuffle"] == "PASS",
        "B2 calibration shuffle gate drift",
    )

    bhva_freeze = seed42["B_HVA"]["freeze"]
    bhva_entrypoint_path = ROOT / "scripts/simulation/train_s4_3_pi2m_bhva.py"
    inputs.bytes(
        bhva_entrypoint_path,
        "seed42 B_HVA frozen entrypoint",
        str(bhva_freeze["code_sha256"]["scripts/simulation/train_s4_3_pi2m_bhva.py"]),
    )
    bhva_protocol_path = ROOT / "configs/simulation/s4_3_pi2m_bhva_protocol.json"
    inputs.bytes(
        bhva_protocol_path,
        "seed42 B_HVA frozen protocol",
        str(bhva_freeze["config_sha256"]),
    )

    dataset_identity = {
        "manifest_path": str(DATASET_MANIFEST.resolve()),
        "manifest_sha256": inputs.observed_sha256(DATASET_MANIFEST),
        "repository": dataset_manifest["repository"],
        "revision": dataset_manifest["revision"],
        "episodes": int(starting["dataset"]["episodes"]),
        "frames": int(starting["dataset"]["frames"]),
        "file_count": len(dataset_manifest["files"]),
        "evidence_kind": "FROZEN_FILE_MANIFEST_NO_LIVE_DATASET_RESCAN",
    }
    base_identity = {
        "manifest_path": str(BASE_MANIFEST.resolve()),
        "manifest_sha256": inputs.observed_sha256(BASE_MANIFEST),
        "repository": base_manifest["repository"],
        "revision": base_manifest["revision"],
        "file_count": len(base_manifest["files"]),
        "bytes": sum(int(row["bytes"]) for row in base_manifest["files"]),
        "evidence_kind": "FROZEN_FILE_MANIFEST_NO_LIVE_12GB_BASE_RESCAN",
    }
    sidecar_identities = {
        "contact": {
            "path": str(CONTACT_SIDECAR.resolve()),
            "sha256": contact_sha,
            "rows": int(starting["sidecars"]["contact"]["rows"]),
            "valid_rows": int(starting["sidecars"]["contact"]["valid_rows"]),
        },
        "va27": {
            "path": str(VA27_SIDECAR.resolve()),
            "sha256": va27_sha,
            "rows": int(starting["sidecars"]["va27"]["rows"]),
            "valid_rows": int(starting["sidecars"]["va27"]["valid_rows"]),
        },
    }

    track_rng_contract = {
        "evidence_kind": "STATIC_CONTROL_FLOW_IN_SHA256_BOUND_ACCEPTED_OPENPI_TRAINER",
        "source_path": str(official_train_path.resolve()),
        "source_sha256": inputs.observed_sha256(official_train_path),
        "statements": rng_statements,
        "interpretation": {
            "root_key": "jax.random.key(config.seed)",
            "root_split": "train_rng, init_rng = jax.random.split(root_key)",
            "model_initialization": "init_rng is split again and model_rng is passed to config.model.create",
            "per_optimizer_step": "train_rng is folded with persisted state.step before loss evaluation",
            "data_loader": (
                "shuffle=True is requested; exact permutation implementation/history is not "
                "source-bound here"
            ),
        },
        "not_runtime_evidence": True,
    }
    b0_rng_contract = {
        "evidence_kind": "STATIC_CONTROL_FLOW_IN_SHA256_BOUND_SEED42_B0_TRAINER",
        "source_path": str(b0_train_path.resolve()),
        "source_sha256": inputs.observed_sha256(b0_train_path),
        "statements": b0_rng_statements,
        "not_runtime_evidence": True,
    }

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            matrix_row = matrix_rows[(model, seed)]
            recipe = matrix_row["recipe"]
            if seed == 42:
                fields = _seed42_direct_fields(model, seed42[model], recipe)
                source = matrix_row["source_and_import_evidence"]
                if model == "B0":
                    loader = {
                        "class": "CheckpointWeightLoader",
                        "evidence_kind": "STATIC_SHA256_BOUND_OFFICIAL_CONFIG_SOURCE",
                    }
                    rng_contract = "SEED42_B0_STATIC_RNG_CONTROL_FLOW"
                    trainer_runtime_identity = "SHA256_BOUND_SOURCE_FILE"
                elif model == "B_VA27":
                    loader = {
                        "class": "TactileCheckpointWeightLoader",
                        "evidence_kind": "STATIC_SHA256_BOUND_PRELAUNCH_SOURCE_SNAPSHOT",
                    }
                    rng_contract = (
                        "UNVERIFIABLE_HISTORICAL_OFFICIAL_TRAINER_BYTES_NOT_IN_PRELAUNCH_SNAPSHOT"
                    )
                    trainer_runtime_identity = (
                        "UNVERIFIABLE_PER_PROCESS_IMPORT_SNAPSHOT_NOT_PERSISTED"
                    )
                elif model == "B_HVA":
                    loader = {
                        "class": "TactileCheckpointWeightLoader",
                        "evidence_kind": "STATIC_SHA256_BOUND_ENTRYPOINT",
                    }
                    rng_contract = (
                        "UNVERIFIABLE_HISTORICAL_OFFICIAL_TRAINER_BYTES_NOT_HASHED_BY_STAGE_FREEZE"
                    )
                    trainer_runtime_identity = (
                        "UNVERIFIABLE_PER_PROCESS_IMPORT_SNAPSHOT_NOT_PERSISTED"
                    )
                else:
                    loader = {
                        "class": "TactileCheckpointWeightLoader",
                        "evidence_kind": "INFERENCE_FROM_STAGE_ENTRYPOINT_AND_LOADED_BASE_GATES",
                    }
                    rng_contract = "UNVERIFIABLE_HISTORICAL_ENTRYPOINT_AND_OFFICIAL_TRAINER_BYTES_NOT_PERSISTED"
                    trainer_runtime_identity = (
                        "UNVERIFIABLE_PER_PROCESS_IMPORT_SNAPSHOT_NOT_PERSISTED"
                    )
                source_identity = {
                    "cohort": source["cohort"],
                    "stage": source["stage"],
                    "entrypoint": source["entrypoint"],
                    "source_commit": source["source_commit"],
                    "freeze_artifact": source["freeze_artifact"],
                    "per_process_import_origin": source["per_process_import_origin"],
                    "trainer_runtime_identity": trainer_runtime_identity,
                }
            else:
                run = run_rows[(model, seed)]
                fields = {
                    "numeric_training_seed": seed,
                    "mode": run["mode"],
                    "mode_evidence_kind": "DIRECT_TRACK_A_FROZEN_RUN_FIELD",
                    "global_batch_size": int(run["global_batch_size"]),
                    "optimizer_steps": int(run["optimizer_steps"]),
                    "lambda_phys": float(run["lambda_phys"]),
                    "initialization_declaration": protocol["initialization"],
                    "initialization_evidence_kind": "DIRECT_FROZEN_PROTOCOL_AND_RUN_BASE_FIELDS",
                    "resume": False,
                }
                require(fields["mode"] == recipe["mode"], f"Track A mode drift: {model}/seed{seed}")
                require(
                    fields["lambda_phys"] == recipe["lambda_phys"],
                    f"Track A lambda drift: {model}/seed{seed}",
                )
                loader = {
                    "class": (
                        "CheckpointWeightLoader"
                        if model == "B0"
                        else "TactileCheckpointWeightLoader"
                    ),
                    "evidence_kind": "STATIC_SHA256_BOUND_TRACK_A_CONFIG_BUILDER",
                }
                rng_contract = "TRACK_A_STATIC_RNG_CONTROL_FLOW"
                source_identity = {
                    "cohort": "TRACK_A_COMMON_SOURCE_SEEDS43_44",
                    "git_head": run["git_head"],
                    "entrypoint": run["entrypoint"],
                    "config_sha256": run["config_sha256"],
                    "immutable_snapshot_sha256": snapshot_manifest["sha256"],
                    "historical_import_origins": imports["modules"],
                    "trainer_runtime_identity": "BOUND_BY_PRETRAINING_IMPORT_AUDIT_AND_ACCEPTED_OPENPI_SHA256",
                }

            if model == "B0":
                sidecars: list[str] = []
            elif model == "B_VA27":
                sidecars = ["va27"]
            elif model == "B_HVA":
                sidecars = ["contact", "va27"]
            else:
                sidecars = ["contact"]

            rows.append(
                {
                    "checkpoint_id": matrix_row["checkpoint_id"],
                    "model": model,
                    "training_seed": seed,
                    "source_identity": source_identity,
                    "initialization": {
                        **fields,
                        "official_parent": recipe["official_parent"],
                        "base_identity_reference": "OFFICIAL_PI05_BASE_MANIFEST",
                        "loader": loader,
                        "post_loader_full_parameter_tree_sha256": "UNVERIFIABLE_NOT_PERSISTED_BEFORE_OPTIMIZER_STEP_0",
                        "new_adapter_auxiliary_initial_values_sha256": "UNVERIFIABLE_NOT_PERSISTED",
                    },
                    "data": {
                        "dataset_identity_reference": "OFFICIAL_PINCH_TONGS_MANIFEST",
                        "sidecar_identity_references": sidecars,
                        "shuffle_requested": (
                            True
                            if seed != 42 or model == "B0"
                            else "EXPECTED_NOT_PER_PROCESS_SOURCE_VERIFIED"
                        ),
                        "exact_batch_permutation_history": "UNVERIFIABLE_NOT_PERSISTED",
                        "worker_scheduling_history": "UNVERIFIABLE_NOT_PERSISTED",
                    },
                    "rng": {
                        "numeric_training_seed": seed,
                        "static_control_flow_reference": rng_contract,
                        "realized_root_key_data": "UNVERIFIABLE_NOT_PERSISTED",
                        "realized_init_rng_data": "UNVERIFIABLE_NOT_PERSISTED",
                        "realized_model_rng_data": "UNVERIFIABLE_NOT_PERSISTED",
                        "parameter_to_subkey_trace": "UNVERIFIABLE_NOT_PERSISTED",
                        "dropout_subkey_trace": "UNVERIFIABLE_NOT_PERSISTED",
                        "augmentation_subkey_trace": "UNVERIFIABLE_NOT_PERSISTED",
                        "flow_noise_subkey_trace": "UNVERIFIABLE_NOT_PERSISTED",
                        "per_step_loss_subkey_trace": "UNVERIFIABLE_NOT_PERSISTED",
                    },
                    "claim_boundary": (
                        "The numeric seed and static source describe intended key derivation only; they do not prove "
                        "byte-identical initialization across modes, runs, source cohorts, device "
                        "counts, or software stacks."
                    ),
                }
            )

    require(len(rows) == 15, "expected fifteen initialization/RNG rows")
    b2_calibration = seed42["B2"]["calibration"]
    recorded_calibration_keys = [
        {
            "iterator_batch_index": int(row["iterator_batch_index"]),
            "loss_rng_key_data": [int(value) for value in row["loss_rng_key_data"]],
            "batch_tree_sha256": str(row["batch_tree_sha256"]),
        }
        for row in b2_calibration["calibration_batches"]
    ]
    require(len(recorded_calibration_keys) == 4, "B2 calibration RNG record count drift")

    body = {
        "stage": "S4.3-PI2S-S1 Initialization/RNG/Source Audit",
        "status": "COMPLETE_WITH_DECLARED_UNVERIFIABLE_RNG_AND_INITIALIZATION_FIELDS",
        "analysis_kind": "CPU_ONLY_STATIC_AND_MANIFEST_AUDIT_NO_MODEL_LOAD_NO_GPU",
        "matrix_shape": {"models": 5, "training_seeds": 3, "checkpoints": 15},
        "evidence_vocabulary": {
            "DIRECT_ARTIFACT_FIELD": "A value explicitly persisted by a historical JSON artifact.",
            "STATIC_SHA256_BOUND_SOURCE": (
                "A control-flow or loader statement present in source bytes bound by SHA256."
            ),
            "INFERENCE": (
                "A reconstruction consistent with evidence but not directly persisted; never "
                "promoted to verified runtime fact."
            ),
            "UNVERIFIABLE": "The required historical value or trace was not persisted.",
        },
        "official_base_identity": base_identity,
        "official_dataset_identity": dataset_identity,
        "sidecar_identities": sidecar_identities,
        "track_a_source_and_rng_contract": {
            "immutable_snapshot": {
                "path": str(snapshot_path),
                "sha256": snapshot_manifest["sha256"],
                "git_head": snapshot_manifest["git_head"],
                "members": track_source_identities,
            },
            "historical_import_origins": imports,
            "config_builder_evidence": builder_statements,
            "rng_control_flow": track_rng_contract,
            "scope_note": "Applies to Track A seeds43/44 only; it is not retroactively assigned to seed42 processes.",
        },
        "seed42_source_contracts": {
            "B0": {
                "rng_control_flow": b0_rng_contract,
                "official_config_sha256": matrix_rows[("B0", 42)]["recipe"]["config_identity"][
                    "sha256"
                ],
                "recorded_source_file_sha256": b0_freeze["source_file_sha256"],
            },
            "B_VA27": {
                "prelaunch_snapshot_path": str(BVA_SOURCE_SNAPSHOT.resolve()),
                "prelaunch_snapshot_sha256": bva_freeze["source_snapshot_sha256"],
                "members": bva_source_identities,
                "official_trainer_runtime_bytes": "UNVERIFIABLE_NOT_INCLUDED_IN_PRELAUNCH_SNAPSHOT",
            },
            "B1": {
                "training_protocol_sha256": pi1_freeze["config_sha256"],
                "loaded_base_gate": seed42["B1"]["base_gate"],
                "historical_entrypoint_sha256": "UNVERIFIABLE_NOT_PERSISTED_IN_SELECTED_STAGE_ARTIFACTS",
            },
            "B_HVA": {
                "entrypoint_sha256": bhva_freeze["code_sha256"][
                    "scripts/simulation/train_s4_3_pi2m_bhva.py"
                ],
                "protocol_sha256": bhva_freeze["config_sha256"],
                "recorded_code_sha256": bhva_freeze["code_sha256"],
                "official_trainer_runtime_bytes": "UNVERIFIABLE_NOT_HASHED_BY_STAGE_FREEZE",
            },
            "B2": {
                "training_protocol_sha256": pi1_freeze["config_sha256"],
                "pretraining_calibration": {
                    "scope": "B2 lambda calibration only; not a training PRNG trace",
                    "recorded_rng_keys": recorded_calibration_keys,
                    "deterministic_shuffle_gate": b2_calibration["gates"][
                        "seed42_deterministic_shuffle"
                    ],
                },
                "historical_entrypoint_sha256": "UNVERIFIABLE_NOT_PERSISTED_IN_SELECTED_STAGE_ARTIFACTS",
            },
        },
        "per_checkpoint": rows,
        "cross_run_findings": {
            "seed43_vs_seed44_source_identity": "SAME_IMMUTABLE_TRACK_A_SOURCE_SNAPSHOT_VERIFIED",
            "seed42_vs_seed43_44_source_identity": "NOT_SOURCE_IDENTICAL",
            "seed42_source_structure": "FOUR_STAGE_SPECIFIC_ENTRYPOINTS_WITH_NONUNIFORM_SOURCE_AND_IMPORT_EVIDENCE",
            "same_numeric_seed_across_modes": "DOES_NOT_PROVE_BYTE_IDENTICAL_INITIALIZATION",
            "same_base_manifest": (
                "PROVES_FROZEN_BASE_FILE_IDENTITY_WHERE_REFERENCED_NOT_POST_LOADER_FULL_MODEL_IDENTITY"
            ),
            "track_a_seed_difference": (
                "SEEDS43_AND44_ARE_DISTINCT_NUMERIC_ROOT_SEEDS; "
                "REALIZED_SUBKEYS_WERE_NOT_PERSISTED"
            ),
            "causal_interpretation": (
                "No performance difference is attributed to initialization, RNG, source, or data "
                "order by this audit."
            ),
        },
        "unverifiable_from_persisted_history": [
            "pre-optimizer-step full parameter-tree hashes for all fifteen runs",
            "bytewise equality of initialized parameters across different modes sharing one numeric seed",
            "bytewise equality of randomly initialized adapter/auxiliary parameters across source cohorts",
            "realized root, init, model, dropout, augmentation, and flow-noise PRNG subkeys",
            "parameter-to-PRNG-subkey consumption order inside model construction",
            "complete per-step loss PRNG key history",
            "complete data-loader permutation, worker scheduling, and augmentation histories",
            "per-process import-origin snapshots for the five seed42 training processes",
            "historical official-trainer bytes for seed42 B_VA27/B1/B_HVA/B2 where stage freezes omitted them",
            "software/runtime determinism across different device counts and historical environments",
        ],
        "conclusion": {
            "supported": [
                "Track A seeds43/44 share one immutable source snapshot and accepted OpenPI source identity.",
                (
                    "Track A statically derives root RNG from config.seed, splits train/init RNG, "
                    "and folds train RNG with state.step."
                ),
                (
                    "All runs declare the official pi05_base parent and the same official dataset "
                    "identity; loader class and sidecars are reported with evidence scope."
                ),
                "Seed42 and Track A are not source-identical cohorts.",
            ],
            "not_supported": [
                "Numeric seed equality proves byte-identical initialization across modes.",
                (
                    "The absence of persisted PRNG traces proves identical dropout, augmentation, "
                    "shuffle, or flow-noise streams."
                ),
                "Base-manifest equality proves equality of the complete post-loader initialized model tree.",
                "Initialization or RNG differences explain any observed rollout outcome.",
            ],
        },
        "gates": {
            "checkpoint_recipe_matrix_semantic_sha_valid": "PASS",
            "fifteen_checkpoint_rows_present": "PASS",
            "track_a_immutable_snapshot_sha_and_member_hashes_match": "PASS",
            "track_a_historical_import_origins_bound": "PASS",
            "accepted_openpi_trainer_and_config_sha_match": "PASS",
            "official_base_and_dataset_manifests_bound": "PASS",
            "training_sidecar_sha_values_match": "PASS",
            "seed42_stage_manifests_and_available_code_identities_bound": "PASS",
            "seed42_vs_track_a_source_nonidentity_explicit": "PASS",
            "numeric_seed_not_promoted_to_byte_identity": "PASS",
            "runtime_rng_subkeys_and_missing_histories_declared_unverifiable": "PASS",
            "calibration_rng_keys_not_relabelled_as_training_trace": "PASS",
            "no_model_load_no_simulator_no_gpu": "PASS",
        },
    }
    inputs.verify_unchanged()
    return output_artifact(body, inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "read-only: recompute from frozen inputs and compare the existing output; "
            "never create, replace, or update it"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = build_artifact()
    result = execute_artifact(OUTPUT, artifact, check=args.check)
    output_sha = result.pop("output_sha256")
    print(
        json.dumps(
            {
                **result,
                "output": {"path": str(OUTPUT), "sha256": output_sha},
                "checkpoints": 15,
                "model_loaded": False,
                "simulator_used": False,
                "gpu_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except AuditError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from error
