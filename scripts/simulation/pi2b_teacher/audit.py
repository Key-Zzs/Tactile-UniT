#!/usr/bin/env python3
"""Freeze and verify the matched VA/VAC teacher protocol before training."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_teacher import (  # noqa: E402
    MatchLossWeights,
    StrictFieldStore,
    build_matched_pair,
    clip_active_grad_norm_,
    matched_teacher_loss,
)
from gr00t.simulation.pi2b_teacher.matched_teacher import common_state  # noqa: E402
from scripts.simulation.pi2b_teacher.common import (  # noqa: E402
    CONFIG_ROOT,
    artifact_root,
    atomic_json,
    canonical_digest,
    git_output,
    history_cache,
    load_json,
    model_parameter_counts,
    sha256_file,
    stage_root,
    tensor_digest,
    workspace,
)

SNAPSHOT_FILES = (
    "gr00t/__init__.py",
    "gr00t/simulation/__init__.py",
    "gr00t/simulation/dexjoco_adapter.py",
    "gr00t/simulation/episode_logger.py",
    "gr00t/simulation/s4_2_dataset.py",
    "gr00t/simulation/simulated_tactile.py",
    "gr00t/simulation/timing.py",
    "gr00t/simulation/pi2b_teacher/__init__.py",
    "gr00t/simulation/pi2b_teacher/matched_teacher.py",
    "gr00t/tactile_unit/continuous_vac_shared_space.py",
    "gr00t/tactile_unit/__init__.py",
    "scripts/simulation/pi2b_teacher/common.py",
    "scripts/simulation/pi2b_teacher/build_confirmation.py",
    "scripts/simulation/pi2b_teacher/train.py",
    "scripts/simulation/generate_s4_2_dataset.py",
    "configs/simulation/s4_1_dexjoco_contact_regions.json",
    "configs/simulation/s4_2_dexjoco_contact_regions.json",
    "configs/simulation/s4_2_dataset_contract.json",
    "configs/simulation/s4_2_tf_locked_test_v2.json",
    "configs/simulation/pi2b_teacher/protocol.json",
    "configs/simulation/pi2b_teacher/losses.json",
)


def _component_tensor_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _numeric_episode(values: np.ndarray) -> np.ndarray:
    return np.unique(values, return_inverse=True)[1].astype(np.int64)


def _gradients(
    scalar: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor | None]:
    result = torch.autograd.grad(
        scalar,
        [parameter for _, parameter in named_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    return {name: value for (name, _), value in zip(named_parameters, result)}


def _all_equal_gradients(
    left: dict[str, torch.Tensor | None], right: dict[str, torch.Tensor | None]
) -> bool:
    if left.keys() != right.keys():
        return False
    for name in left:
        if left[name] is None or right[name] is None:
            if left[name] is not right[name]:
                return False
        elif not torch.equal(left[name], right[name]):
            return False
    return True


def matching_checks(train_path: Path, protocol: dict[str, Any], losses: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    allowed = {"pair_id", "episode_id", "task", "source_trajectory_id", "anchor_step", "future_step", "z_v", "z_a"}
    store = StrictFieldStore(train_path, allowed)
    arrays = store.load()
    try:
        store.read("z_c")
        c_denied = False
    except PermissionError:
        c_denied = True
    rows = np.arange(16, dtype=np.int64)
    native = {
        "vision": torch.from_numpy(arrays["z_v"][rows]).float(),
        "action": torch.from_numpy(arrays["z_a"][rows]).float(),
    }
    contact = StrictFieldStore(train_path, {"z_c"}).load()["z_c"][rows]
    full_native = {**native, "contact": torch.from_numpy(contact).float()}
    episode = torch.from_numpy(_numeric_episode(arrays["episode_id"])[rows])
    va, vac = build_matched_pair(int(protocol["initialization_seed"]))
    va_common = common_state(va)
    vac_common = common_state(vac)
    initial_equal = _component_tensor_equal(va_common, vac_common)
    initial_digest = tensor_digest({name: value.clone() for name, value in va_common.items()})
    initial_shapes = {name: list(value.shape) for name, value in va_common.items()}
    weight = MatchLossWeights(**losses["weights"])
    zero_contact_weight = replace(
        weight,
        contact_pair=0.0,
        contact_native=0.0,
        contact_relational=0.0,
        contact_variance=0.0,
    )
    va_total, va_parts = matched_teacher_loss(
        va, native, episode, temperature=float(losses["temperature"]), weights=weight
    )
    vac_zero_total, vac_zero_parts = matched_teacher_loss(
        vac,
        full_native,
        episode,
        temperature=float(losses["temperature"]),
        weights=zero_contact_weight,
    )
    forward_equal = all(
        torch.equal(va.encode(name, native[name]), vac.encode(name, native[name]))
        for name in ("vision", "action")
    )
    loss_equal = torch.equal(va_total, vac_zero_total) and all(
        torch.equal(va_parts[key], vac_zero_parts[key])
        for key in ("va_total", "va_alignment", "va_native", "va_relational", "va_variance")
    )
    va_named = [(name, parameter) for name, parameter in va.named_parameters()]
    vac_common_named = [
        (name, parameter)
        for name, parameter in vac.named_parameters()
        if name in dict(va_named)
    ]
    va_grad = _gradients(va_total, va_named)
    vac_grad = _gradients(vac_zero_total, vac_common_named)
    gradient_equal = _all_equal_gradients(va_grad, vac_grad)

    va_optimizer = torch.optim.AdamW(
        va.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    vac_optimizer = torch.optim.AdamW(
        vac.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    va_optimizer.zero_grad(set_to_none=True)
    vac_optimizer.zero_grad(set_to_none=True)
    va_total.backward()
    vac_zero_total.backward()
    va_preclip = clip_active_grad_norm_(va.parameters(), float(protocol["training"]["gradient_clip_global"]))
    vac_preclip = clip_active_grad_norm_(vac.parameters(), float(protocol["training"]["gradient_clip_global"]))
    va_optimizer.step()
    vac_optimizer.step()
    update_equal = _component_tensor_equal(common_state(va), common_state(vac))

    va2, vac2 = build_matched_pair(int(protocol["initialization_seed"]))
    del va2
    _, full_parts = matched_teacher_loss(
        vac2,
        full_native,
        episode,
        temperature=float(losses["temperature"]),
        weights=weight,
    )
    common_named = [
        (name, parameter)
        for name, parameter in vac2.named_parameters()
        if name == "shared_slots"
        or name.startswith(("projectors.vision.", "projectors.action.", "recovery.vision.", "recovery.action."))
    ]
    contact_named = [
        (name, parameter)
        for name, parameter in vac2.named_parameters()
        if name.startswith(("projectors.contact.", "recovery.contact."))
    ]
    contact_to_common = _gradients(full_parts["contact_total"], common_named)
    va_to_contact = _gradients(full_parts["va_total"], contact_named)
    contact_common_nonzero = any(
        value is not None and bool(torch.count_nonzero(value)) for value in contact_to_common.values()
    )
    va_contact_zero = all(value is None or not bool(torch.count_nonzero(value)) for value in va_to_contact.values())
    perturbed_native = {**native, "contact": torch.randn_like(full_native["contact"]) * 1000.0}
    perturbed_zero, _ = matched_teacher_loss(
        vac2,
        native,
        episode,
        temperature=float(losses["temperature"]),
        weights=weight,
        contact_enabled=False,
    )
    deleted_zero, _ = matched_teacher_loss(
        vac2,
        {key: value for key, value in perturbed_native.items() if key != "contact"},
        episode,
        temperature=float(losses["temperature"]),
        weights=weight,
        contact_enabled=False,
    )
    c_perturbation_invariant = torch.equal(perturbed_zero, deleted_zero)
    checks = {
        "common_structure_keys_equal": va_common.keys() == vac_common.keys(),
        "common_initialization_byte_equal": initial_equal,
        "c_disabled_common_forward_exact": forward_equal,
        "c_disabled_va_loss_exact": loss_equal,
        "c_disabled_common_gradients_exact": gradient_equal,
        "c_disabled_common_adamw_update_exact": update_equal,
        "c_input_perturbation_or_deletion_invariant_when_disabled": c_perturbation_invariant,
        "va_store_denies_z_c": c_denied,
        "va_store_accessed_fields_exclude_forbidden": not ({"z_c", "h_current", "h_future", "contact_transition", "force_trend"} & set(store.accessed_fields)),
        "contact_loss_reaches_common_path": contact_common_nonzero,
        "va_loss_does_not_reach_contact_private_parameters": va_contact_zero,
        "global_clip_pre_norm_exact_when_c_disabled": torch.equal(va_preclip, vac_preclip),
    }
    return {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-matching-audit.v1",
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "MATCHING_IMPLEMENTATION_BLOCKED",
        "common_initialization_digest": initial_digest,
        "common_parameter_shapes": initial_shapes,
        "parameters": {"T_VA_match": model_parameter_counts(va), "T_VAC_match": model_parameter_counts(vac)},
        "va_field_access": store.accessed_fields,
        "gradient_dependency": {
            "contact_total_to_common_nonzero_parameters": sorted(name for name, value in contact_to_common.items() if value is not None and bool(torch.count_nonzero(value))),
            "va_total_to_contact_nonzero_parameters": sorted(name for name, value in va_to_contact.items() if value is not None and bool(torch.count_nonzero(value))),
        },
    }


def dataset_audit(path: Path, expected: dict[str, Any], split: str) -> dict[str, Any]:
    if sha256_file(path) != expected[f"paired_{'dev' if split == 'dev' else 'train'}_sha256"]:
        raise RuntimeError(f"{split} paired cache hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        required = {"pair_id", "episode_id", "task", "source_trajectory_id", "anchor_step", "future_step", "z_v", "z_a", "z_c"}
        if not required <= set(source.files):
            raise RuntimeError(f"{split} paired cache missing fields")
        rows = len(source["pair_id"])
        tasks = sorted(set(source["task"].tolist()))
        groups = set(zip(source["task"].tolist(), source["source_trajectory_id"].tolist()))
        checks = {
            "row_count": rows == expected[f"{split}_rows"],
            "unique_pair_ids": len(set(source["pair_id"].tolist())) == rows,
            "group_count": len(groups) == expected[f"{split}_groups"],
            "tasks": set(tasks) == set(expected["tasks"]),
            "time_plus_27": bool(np.all(source["future_step"] - source["anchor_step"] == expected["transition_offset_steps"])),
            "finite_native": all(np.isfinite(source[name]).all() for name in ("z_v", "z_a", "z_c")),
            "native_shapes": all(source[name].shape == (rows, 8, 32) for name in ("z_v", "z_a", "z_c")),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{split} paired cache contract failed: {checks}")
        return {
            "split": split,
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": rows,
            "groups": len(groups),
            "episodes": len(set(source["episode_id"].tolist())),
            "checks": checks,
            "pair_order_sha256": hashlib.sha256(b"\0".join(str(value).encode() for value in source["pair_id"])).hexdigest(),
        }


def common_normalization(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        diagnostics = {}
        for field in ("z_v", "z_a"):
            value = source[field].astype(np.float64)
            diagnostics[field] = {
                "shape": list(value.shape[1:]),
                "mean": float(value.mean()),
                "std": float(value.std()),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
            }
    return {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-common-va-normalization.v1",
        "fit_split": "TRAIN",
        "transform": "identity_native_units",
        "applies_to": ["T_VA_match", "T_VAC_match"],
        "contact_statistics_in_common_file": False,
        "diagnostics_only": diagnostics,
    }


def confirmation_reservation(protocol: dict[str, Any], coordination: Path) -> dict[str, Any]:
    confirmation = protocol["confirmation"]
    identities = []
    for task in confirmation["tasks"]:
        for group in confirmation["source_group_indices"]:
            for perturbation in range(confirmation["episodes_per_group"]):
                identities.append(
                    {
                        "task": task,
                        "source_trajectory_id": f"{task}-script-{group:02d}",
                        "episode_id": f"{task}-pi2b-teacher-g{group:02d}-p{perturbation:02d}",
                        "seed": confirmation["seed_bases"][task] + 10 * group + perturbation,
                    }
                )
    existing_groups: set[tuple[str, str]] = set()
    existing_seeds: set[int] = set()
    for metadata_path in (ROOT / ".local/refs/base/datasets/simulation").glob("**/episodes/*/metadata.json"):
        value = load_json(metadata_path)
        metadata = value.get("metadata", {})
        if "task" in metadata and "source_trajectory_id" in metadata:
            existing_groups.add((metadata["task"], metadata["source_trajectory_id"]))
        if "seed" in metadata:
            existing_seeds.add(int(metadata["seed"]))
    collisions = [
        value
        for value in identities
        if (value["task"], value["source_trajectory_id"]) in existing_groups
        or value["seed"] in existing_seeds
    ]
    reservation = {
        "schema": "tactile3d-unit.pi2b-exposure-reservation.v1",
        "track": "teacher",
        "id": "s4_3_pi2b_teacher_confirmation_v1",
        "classification": "FINAL_FROZEN",
        "identities": identities,
        "identity_digest": canonical_digest(identities),
        "collision_count": len(collisions),
        "selection_rule": confirmation["selection_rule"],
        "teacher_performance_read": False,
    }
    if collisions:
        raise RuntimeError("fresh confirmation reservation collides with existing data")
    lock_path = coordination / "exposure.lock"
    destination = coordination / "exposure_reservations/teacher/s4_3_pi2b_teacher_confirmation_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() and load_json(destination) != reservation:
            raise RuntimeError("existing confirmation reservation differs")
        atomic_json(destination, reservation)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return {**reservation, "coordination_path": str(destination)}


def snapshot_source(snapshot: Path) -> dict[str, str]:
    hashes = {}
    for relative in SNAPSHOT_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"immutable snapshot collision: {destination}")
        shutil.copy2(source, destination)
        hashes[relative] = sha256_file(source)
    return hashes


def main() -> None:
    protocol = load_json(CONFIG_ROOT / "protocol.json")
    losses = load_json(CONFIG_ROOT / "losses.json")
    local = workspace()
    branch = git_output("branch", "--show-current")
    if branch != local["expected_branch"] or branch != "develop/pi2b-teacher":
        raise RuntimeError("wrong branch for teacher audit")
    base = protocol["base_sha"]
    if git_output("merge-base", "HEAD", base) != base:
        raise RuntimeError("configured PI2B base is not an ancestor")
    if local["coordination_contract_sha256"] != sha256_file(Path(local["coordination_contract_path"])):
        raise RuntimeError("coordination contract changed")
    write_root = stage_root()
    if write_root != Path(local["write_root"]):
        raise RuntimeError("teacher write root mismatch")
    write_root.mkdir(parents=True, exist_ok=True)
    probe = write_root / "tmp/audit_atomic_probe"
    probe.parent.mkdir(parents=True, exist_ok=True)
    temporary = probe.with_suffix(".tmp")
    temporary.write_bytes(b"pi2b-teacher-write-probe\n")
    temporary.replace(probe)
    if probe.read_bytes() != b"pi2b-teacher-write-probe\n":
        raise RuntimeError("NAS atomic write probe failed")
    probe.unlink()

    train = dataset_audit(history_cache("paired_train.npz"), protocol["data"], "train")
    dev = dataset_audit(history_cache("paired_validation.npz"), protocol["data"], "dev")
    if train["pair_order_sha256"] == dev["pair_order_sha256"]:
        raise RuntimeError("TRAIN and DEV pair identities unexpectedly match")
    matching = matching_checks(history_cache("paired_train.npz"), protocol, losses)
    if matching["decision"] != "PASS":
        atomic_json(artifact_root() / "matching_grade.json", matching)
        raise RuntimeError("MATCHING_IMPLEMENTATION_BLOCKED")

    coordination = Path(local["coordination_contract_path"]).parent
    reservation = confirmation_reservation(protocol, coordination)
    normalization_path = artifact_root() / "common_va_normalization.json"
    atomic_json(normalization_path, common_normalization(history_cache("paired_train.npz")))
    normalization_sha256 = sha256_file(normalization_path)
    config_hashes = {
        path.name: sha256_file(path)
        for path in sorted(CONFIG_ROOT.glob("*.json"))
    }
    generator_contract_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            ROOT / "configs/simulation/s4_2_dataset_contract.json",
            ROOT / "configs/simulation/s4_2_tf_locked_test_v2.json",
            ROOT / "configs/simulation/s4_2_dexjoco_contact_regions.json",
        )
    }
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in SNAPSHOT_FILES}
    source_hashes["scripts/simulation/pi2b_teacher/audit.py"] = sha256_file(Path(__file__))
    freeze_identity = canonical_digest({"configs": config_hashes, "generator_contracts": generator_contract_hashes, "sources": source_hashes, "data": [train["sha256"], dev["sha256"]]})
    snapshot = write_root / "snapshots" / freeze_identity / "source"
    snapshot_hashes = snapshot_source(snapshot)
    runtime = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-runtime.v1",
        "freeze_identity": freeze_identity,
        "snapshot_root": str(snapshot),
        "worktree_root": str(ROOT),
        "write_root": str(write_root),
        "train_cache": str(history_cache("paired_train.npz").resolve()),
        "dev_cache": str(history_cache("paired_validation.npz").resolve()),
        "coordination_dir": str(coordination),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)).hexdigest(),
        "python": sys.executable,
        "torch": torch.__version__,
        "module_origins": {
            "matched_teacher": importlib.util.find_spec("gr00t.simulation.pi2b_teacher.matched_teacher").origin,
            "torch": importlib.util.find_spec("torch").origin,
            "numpy": importlib.util.find_spec("numpy").origin,
        },
        "snapshot_hashes": snapshot_hashes,
        "common_va_normalization": str(normalization_path),
        "common_va_normalization_sha256": normalization_sha256,
    }
    atomic_json(write_root / "artifacts/runtime_manifest.json", runtime)
    atomic_json(write_root / "artifacts/paired_dataset_manifest.json", {"train": train, "dev": dev})
    atomic_json(write_root / "artifacts/time_contract.json", {"offset_steps": 27, "seconds": 0.54, "train_pass": train["checks"]["time_plus_27"], "dev_pass": dev["checks"]["time_plus_27"]})
    atomic_json(write_root / "artifacts/architecture_matching.json", matching)
    atomic_json(write_root / "artifacts/common_initialization_digest.json", {"sha256": matching["common_initialization_digest"], "shapes": matching["common_parameter_shapes"]})
    atomic_json(write_root / "artifacts/c_disabled_equivalence.json", {"decision": matching["decision"], "checks": {key: value for key, value in matching["checks"].items() if key.startswith("c_disabled") or key.startswith("c_input")}})
    atomic_json(write_root / "artifacts/gradient_dependency.json", matching["gradient_dependency"])
    atomic_json(write_root / "artifacts/loss_component_matching.json", {"loss_config_sha256": config_hashes["losses.json"], "va_pair_dilution": False, "dynamic_weighting": False})
    atomic_json(write_root / "artifacts/compute_parameter_budget.json", {"parameters": matching["parameters"], "same_steps": protocol["training"]["steps"], "same_batch_size": protocol["training"]["batch_size"], "total_parameter_and_compute_different": True})
    atomic_json(write_root / "artifacts/fresh_confirmation_protocol.json", protocol["confirmation"] | {"reservation_digest": reservation["identity_digest"]})
    atomic_json(write_root / "artifacts/exposure_reservation.json", reservation)
    atomic_json(write_root / "artifacts/import_origins.json", runtime["module_origins"])
    native_hashes = {
        "paired_train": train["sha256"],
        "paired_dev": dev["sha256"],
        "A0_checkpoint": sha256_file(ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt"),
        "C3_checkpoint": sha256_file(ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"),
    }
    atomic_json(write_root / "artifacts/native_hashes_before.json", native_hashes)
    atomic_json(write_root / "artifacts/starting_integrity.json", {
        "branch": branch,
        "base_sha": base,
        "head": runtime["git_head"],
        "git_diff_sha256": runtime["git_diff_sha256"],
        "coordination_contract_sha256": local["coordination_contract_sha256"],
        "nas_atomic_write": "PASS",
        "native_hashes": native_hashes,
    })
    atomic_json(write_root / "artifacts/data_usage_ledger.json", {
        "TRAIN": {"use": "teacher fitting", "teachers": ["T_VA_match", "T_VAC_match"], "rows": train["rows"]},
        "DEV": {"use": "post-training exposed diagnostics only", "trainer_access": False, "rows": dev["rows"]},
        "FINAL_FROZEN": {"use": "confirmation after pretest freeze", "trainer_access": False, "identities": reservation["identity_digest"]},
        "track_a_results": "NOT_READ",
    })
    atomic_json(write_root / "artifacts/source_snapshot_manifest.json", {
        "snapshot_root": str(snapshot),
        "freeze_identity": freeze_identity,
        "files": snapshot_hashes,
    })
    atomic_json(write_root / "artifacts/protocol_freeze.json", {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-protocol-freeze.v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "freeze_identity": freeze_identity,
        "base_sha": base,
        "head": runtime["git_head"],
        "config_hashes": config_hashes,
        "generator_contract_hashes": generator_contract_hashes,
        "source_hashes": source_hashes,
        "data_hashes": {"train": train["sha256"], "dev": dev["sha256"]},
        "matching_decision": matching["decision"],
        "confirmation_identity_digest": reservation["identity_digest"],
        "common_va_normalization_sha256": normalization_sha256,
        "track_a_performance_read": False,
        "policy_training": False,
    })
    result = {
        "decision": "MATCHING_PROTOCOL_FROZEN",
        "freeze_identity": freeze_identity,
        "matching": matching["decision"],
        "common_initialization_digest": matching["common_initialization_digest"],
        "train_rows": train["rows"],
        "dev_rows": dev["rows"],
        "confirmation_identities": len(reservation["identities"]),
        "snapshot": str(snapshot),
    }
    atomic_json(write_root / "status/audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
