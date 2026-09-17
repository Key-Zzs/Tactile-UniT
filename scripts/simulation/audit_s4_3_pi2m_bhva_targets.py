#!/usr/bin/env python3
"""Audit corrected B_HVA VA targets, B2 mask parity, and protected history."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2m_bhva_target_protocol.json"
TARGET = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
B2_SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
HISTORICAL_BVA_TARGET = ROOT / ".local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz"
BRIDGE = ROOT / ".local/experiments/simulation/s4_3_pi2u/va_bridge/frozen.pt"
E_T = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"

EXPECTED_POLICY = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
    "BVA": "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d",
}
POLICY_PATHS = {
    "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
    "BVA": ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42/29999",
}
EXPECTED_S42 = {
    "s4_2": "7da07e8f5babe705d1fd643fc54dcce6db25e3636edf043aae2f7fec940a35ba",
    "s4_2_formal": "4404d9bd732999a6bec98bfcd9ea072d6215c642c60f66bfa0aa6671319035e9",
    "s4_2dr": "48bdf8a02773a898c25725d3afa46d083371e5113dee71a1f8ed34e596fa5e75",
    "s4_2ds": "528c2ecd55e529cba4604c5ecfd57466b56f9fcdeff900f86714ee9e44624d7e",
    "s4_2r": "90d801dfd541bc3d8065fb8e3a9d88d443377b51a653188290ea4fbd6a37b3eb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tree_hash(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "pi2w_hash", ROOT / "scripts/simulation/audit_s4_3_pi2w_preflight.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tree_hash(path)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    manifest_path = ARTIFACTS / "corrected_target_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "BUILT_PENDING_PARITY_AUDIT":
        raise RuntimeError("corrected target manifest is not awaiting its one parity audit")
    lengths = np.asarray(
        json.loads(
            (ROOT / ".local/artifacts/simulation/s4_3_pi1/official_dataset_alignment.json").read_text()
        )["episode_lengths"],
        dtype=np.int64,
    )
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    expected = np.zeros(int(lengths.sum()), dtype=np.bool_)
    for start, length in zip(starts, lengths, strict=True):
        expected[start : start + length - 27] = True

    with np.load(TARGET, allow_pickle=False) as target_source:
        target_fields = list(target_source.files)
        target_index = np.asarray(target_source["index"])
        target_values = np.asarray(target_source["va_shared_target"])
        target_valid = np.asarray(target_source["va_aux_valid"])
    with np.load(B2_SIDECAR, allow_pickle=False) as b2_source:
        b2_index = np.asarray(b2_source["index"])
        b2_valid = np.asarray(b2_source["physical_aux_valid"])
    with np.load(HISTORICAL_BVA_TARGET, allow_pickle=False) as old_source:
        old_fields = list(old_source.files)
        old_valid = np.asarray(old_source["va_aux_valid"])

    checkpoint = torch.load(BRIDGE, map_location="cpu", weights_only=False)
    contact_parameter_names = [name for name in checkpoint["state_dict"] if "contact" in name.lower()]
    policy_hashes = {name: tree_hash(path)["tree_sha256"] for name, path in POLICY_PATHS.items()}
    s42_hashes = {
        name: tree_hash(ROOT / ".local/experiments/simulation" / name)["tree_sha256"]
        for name in EXPECTED_S42
    }
    pi2u_artifacts = tree_hash(ROOT / ".local/artifacts/simulation/s4_3_pi2u")["tree_sha256"]

    per_episode_tail_27 = all(
        bool(np.all(target_valid[start : start + length - 27]))
        and not bool(np.any(target_valid[start + length - 27 : start + length]))
        for start, length in zip(starts, lengths, strict=True)
    )
    old_tail_16 = all(
        bool(np.all(old_valid[start : start + length - 16]))
        and not bool(np.any(old_valid[start + length - 16 : start + length]))
        for start, length in zip(starts, lengths, strict=True)
    )
    gates = {
        "target_fields_exact": target_fields == ["index", "va_shared_target", "va_aux_valid"],
        "target_shape": target_values.shape == (40065, 8, 32),
        "target_finite": bool(np.isfinite(target_values).all()),
        "invalid_targets_zero": bool(np.all(target_values[~target_valid] == 0)),
        "index_equal_B2": bool(np.array_equal(target_index, b2_index)),
        "valid_mask_equal_B2": bool(np.array_equal(target_valid, b2_valid)),
        "valid_mask_equal_expected_episode_local_plus_27": bool(np.array_equal(target_valid, expected)),
        "valid_rows_37365": int(target_valid.sum()) == 37365,
        "invalid_rows_2700": int((~target_valid).sum()) == 2700,
        "every_episode_tail_27_invalid": per_episode_tail_27,
        "no_cross_episode": per_episode_tail_27,
        "physical_horizon_0p54": manifest.get("physical_horizon_seconds") == 0.54,
        "exact_row_plus_27_no_interpolation": manifest.get("source_future_offset_rows") == 27
        and manifest.get("source_frame_selection") == "same episode exact row i+27; no interpolation",
        "clean_target_inputs": manifest.get("contact_tactile_fields_read") == []
        and manifest.get("target_inputs")
        == ["observation.images.front[current_row]", "observation.images.front[future_row_i_plus_27]"],
        "B3_VAC_not_used": manifest.get("B3_VAC_projector_used") is False,
        "frozen_clean_VA_bridge": checkpoint.get("modalities") == ["vision", "action"]
        and not contact_parameter_names
        and sha256_file(BRIDGE) == protocol["teacher"]["bridge_checkpoint_sha256"],
        "B2_sidecar_unchanged": sha256_file(B2_SIDECAR) == protocol["parity_reference"]["B2_sidecar_sha256"],
        "historical_BVA_target_unchanged": sha256_file(HISTORICAL_BVA_TARGET)
        == protocol["historical_BVA"]["target_sidecar_sha256"],
        "historical_BVA_fields_unchanged": old_fields == ["index", "va_shared_target", "va_aux_valid"],
        "historical_BVA_tail_16_preserved": old_tail_16 and int(old_valid.sum()) == 38465,
        "B0_B1_B2_BVA_checkpoint_hashes_unchanged": policy_hashes == EXPECTED_POLICY,
        "E_T_hash_unchanged": sha256_file(E_T)
        == "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19",
        "S4_2_checkpoint_trees_unchanged": s42_hashes == EXPECTED_S42,
        "PI2U_artifact_tree_unchanged": pi2u_artifacts
        == "8b795aaea7d44f2481afa03c400e6dcd15d0de4794ca4687bb133d28e807970a",
        "manifest_sidecar_hash": sha256_file(TARGET) == manifest.get("sidecar_sha256"),
    }
    result = {
        "schema": "tactile3d-unit.s4-3-pi2m-corrected-target-audit.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "target": {
            "sidecar_sha256": sha256_file(TARGET),
            "rows": len(target_valid),
            "valid_rows": int(target_valid.sum()),
            "invalid_rows": int((~target_valid).sum()),
            "mask_disagreements_with_B2": int(np.count_nonzero(target_valid != b2_valid)),
            "index_disagreements_with_B2": int(np.count_nonzero(target_index != b2_index)),
            "physical_horizon_seconds": 0.54,
            "row_offset": 27,
        },
        "teacher": {
            "bridge_sha256": sha256_file(BRIDGE),
            "modalities": checkpoint.get("modalities"),
            "contact_parameter_names": contact_parameter_names,
            "retrained": False,
        },
        "historical_BVA": {
            "target_sidecar_sha256": sha256_file(HISTORICAL_BVA_TARGET),
            "valid_rows": int(old_valid.sum()),
            "row_offset": 16,
            "actual_physical_horizon_seconds": 0.32,
            "matched_horizon_control_for_B2": False,
            "modified": False,
        },
        "protected_hashes": {
            "policy_checkpoints": policy_hashes,
            "S4_2": s42_hashes,
            "E_T": sha256_file(E_T),
            "PI2U_artifact_tree": pi2u_artifacts,
        },
    }
    atomic_json(ARTIFACTS / "corrected_target_audit.json", result)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit("PI2M_CORRECTED_TARGET_AUDIT_FAIL")


if __name__ == "__main__":
    main()
