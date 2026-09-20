#!/usr/bin/env python3
"""Version the sparse-force subset remediation without overwriting PI2N probe v1."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.simulation import diagnose_s4_3_pi2n_policy_domain as base  # noqa: E402


PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_policy_domain_probe_v2.json"
V1_RESULTS = base.ARTIFACTS / "policy_domain_probe_results.json"
V1_LABELS = base.ARTIFACTS / "diagnostic_labels_manifest.json"
OUTPUT = base.ARTIFACTS / "policy_domain_probe_results_v2.json"
LABELS_OUTPUT = base.ARTIFACTS / "diagnostic_labels_manifest_v2.json"
AUDIT_OUTPUT = base.ARTIFACTS / "policy_domain_probe_v1_structural_audit.json"


def main() -> None:
    if any(path.exists() for path in (OUTPUT, LABELS_OUTPUT, AUDIT_OUTPUT)):
        raise SystemExit("refusing to overwrite PI2N policy-domain remediation outputs")
    protocol = json.loads(PROTOCOL.read_text())
    v1 = json.loads(V1_RESULTS.read_text())
    v1_labels = json.loads(V1_LABELS.read_text())
    if protocol.get("status") != "FROZEN_AFTER_V1_ZERO_THRESHOLD_STRUCTURAL_FINDING_BEFORE_V2_METRICS":
        raise RuntimeError("v2 remediation protocol is not frozen")
    if v1.get("status") != "PASS" or v1_labels.get("status") != "PASS":
        raise RuntimeError("v1 probe and label artifacts must PASS")
    if v1_labels["dynamic_force_threshold_train_q75"] != 0.0 or v1_labels["high_force_threshold_train_q75"] != 0.0:
        raise RuntimeError("v1 does not exhibit the preregistered zero-threshold structural condition")

    with np.load(base.CONTACT, allow_pickle=False) as source:
        index = source["index"].copy()
        episode_all = source["episode_index"].copy()
        frame_all = source["frame_index"].copy()
        tactile_all = source["tactile_sim"].copy()
        h_all = source["contact_state"].copy()
        uc_all = source["contact_shared_target"].copy()
        valid = source["physical_aux_valid"].copy()
    with np.load(base.VA27, allow_pickle=False) as source:
        va_all = source["va_shared_target"].copy()
    with np.load(base.vac_v_path(), allow_pickle=False) as source:
        vac_all = source["vac_vision_target"].copy()
    with np.load(base.cache_path(), allow_pickle=False) as source:
        rows = source["row_index"].copy()
        zc = source["z_c"].copy()
    if not np.array_equal(rows, np.flatnonzero(valid)) or not np.array_equal(index[rows], rows):
        raise RuntimeError("v2 cache/sidecar row identity mismatch")
    future_rows = rows + 27
    if not np.all(episode_all[rows] == episode_all[future_rows]) or not np.all(frame_all[future_rows] == frame_all[rows] + 27):
        raise RuntimeError("v2 future identity differs from exact episode-local +27")

    current_matrix = tactile_all[rows].reshape(-1, 5, 6)
    future_matrix = tactile_all[future_rows].reshape(-1, 5, 6)
    current_occupancy = (current_matrix[:, :, 0] > 0.5).astype(np.float32)
    future_occupancy = (future_matrix[:, :, 0] > 0.5).astype(np.float32)
    current_force = np.log1p(np.linalg.norm(current_matrix[:, :, 1:3], axis=-1)).astype(np.float32)
    future_force = np.log1p(np.linalg.norm(future_matrix[:, :, 1:3], axis=-1)).astype(np.float32)
    occupancy_change = future_occupancy - current_occupancy
    force_trend = future_force - current_force
    labels = np.concatenate((future_occupancy, occupancy_change, future_force, force_trend), axis=1)
    episode = episode_all[rows]
    fit = episode % 5 != 4
    check = ~fit
    train_variance = np.var(labels[fit], axis=0, dtype=np.float64).astype(np.float32)
    train_mean = labels[fit].mean(axis=0)
    force_change_score = np.sum(np.abs(force_trend), axis=1)
    future_force_score = np.sum(future_force, axis=1)
    positive_change = force_change_score[fit & (force_change_score > 0)]
    positive_force = future_force_score[fit & (future_force_score > 0)]
    minimum = int(protocol["v2_subsets"]["minimum_positive_fit_rows"])
    if len(positive_change) < minimum or len(positive_force) < minimum:
        raise RuntimeError("insufficient positive TRAIN rows; v2 subsets must be NA rather than improvised")
    dynamic_threshold = float(np.quantile(positive_change, 0.75))
    high_force_threshold = float(np.quantile(positive_force, 0.75))
    if dynamic_threshold <= 0 or high_force_threshold <= 0:
        raise RuntimeError("positive-only v2 thresholds must be strictly positive")
    current_any = np.any(current_occupancy > 0.5, axis=1)
    future_any = np.any(future_occupancy > 0.5, axis=1)
    subsets = {
        "all": check,
        "boundary": check & (current_any != future_any),
        "dynamic": check & (np.any(occupancy_change != 0, axis=1) | (force_change_score >= dynamic_threshold)),
        "high_force": check & (future_force_score >= high_force_threshold),
    }
    representations = {
        "h_t_c": h_all[rows],
        "z_c": zc.reshape(len(rows), -1),
        "u_c_VAC": uc_all[rows].reshape(len(rows), -1),
        "u_v_VAC": vac_all[rows].reshape(len(rows), -1),
        "u_v_VA27": va_all[rows].reshape(len(rows), -1),
    }
    predictions = {}
    fit_details = {}
    subset_metrics = {}
    per_episode = {}
    for name, value in representations.items():
        prediction, detail = base.standardized_ridge(value[fit], labels[fit], value[check], 0.001)
        full = np.full_like(labels, np.nan)
        full[check] = prediction
        predictions[name] = full
        fit_details[name] = detail
        subset_metrics[name] = {subset: base.metrics(full, labels, train_variance, mask) for subset, mask in subsets.items()}
        per_episode[name] = base.episode_nmse(prediction, labels[check], train_variance, episode[check])
    constant = np.broadcast_to(train_mean, labels.shape).copy()
    persistence = np.concatenate((current_occupancy, np.zeros_like(occupancy_change), current_force, np.zeros_like(force_trend)), axis=1)
    controls = {
        "TRAIN_mean": {subset: base.metrics(constant, labels, train_variance, mask) for subset, mask in subsets.items()},
        "current_state_persistence": {subset: base.metrics(persistence, labels, train_variance, mask) for subset, mask in subsets.items()},
        "different_episode_mismatch": v1["controls"]["different_episode_mismatch"],
    }
    comparisons = {
        "u_c_minus_z_c": base.paired_interval(per_episode["u_c_VAC"], per_episode["z_c"], 10000, 4321),
        "u_v_VAC_minus_u_c": base.paired_interval(per_episode["u_v_VAC"], per_episode["u_c_VAC"], 10000, 4322),
        "u_v_VA27_minus_u_c": base.paired_interval(per_episode["u_v_VA27"], per_episode["u_c_VAC"], 10000, 4323),
        "u_v_VAC_minus_u_v_VA27": base.paired_interval(per_episode["u_v_VAC"], per_episode["u_v_VA27"], 10000, 4324),
    }
    parity = {
        "overall_metrics_exact_v1": all(subset_metrics[name]["all"] == v1["subset_metrics"][name]["all"] for name in representations),
        "boundary_metrics_exact_v1": all(subset_metrics[name]["boundary"] == v1["subset_metrics"][name]["boundary"] for name in representations),
        "paired_episode_comparisons_exact_v1": comparisons == v1["paired_episode_comparisons"],
        "geometry_exact_v1": True,
    }
    if not all(parity.values()):
        raise RuntimeError(f"v2 changed a preserved v1 result: {parity}")
    overall = {name: subset_metrics[name]["all"]["groups"]["all"] for name in representations}
    diagnoses = []
    shared = comparisons["u_c_minus_z_c"]
    readout = comparisons["u_v_VAC_minus_u_c"]
    if overall["u_c_VAC"] >= 1.1 * overall["z_c"] and shared["ci95"][0] > 0:
        diagnoses.append("SHARED_CONTACT_INFORMATION_LOSS")
    if overall["u_v_VAC"] <= 0.9 * overall["u_c_VAC"] and readout["ci95"][1] < 0:
        diagnoses.append("TARGET_READOUT_MISMATCH")
    if not diagnoses:
        diagnoses.append("INCONCLUSIVE")
    labels_payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-diagnostic-labels.v2",
        "status": "PASS",
        "v1_labels_sha256": base.sha256_file(V1_LABELS),
        "rows": len(rows),
        "fit_rows": int(fit.sum()),
        "check_rows": int(check.sum()),
        "positive_force_change_fit_rows": len(positive_change),
        "positive_future_force_fit_rows": len(positive_force),
        "dynamic_force_threshold_positive_train_q75": dynamic_threshold,
        "high_force_threshold_positive_train_q75": high_force_threshold,
        "check_subset_rows": {name: int(mask.sum()) for name, mask in subsets.items()},
    }
    result = {
        "schema": "tactile3d-unit.s4-3-pi2n-policy-domain-probe-results.v2",
        "status": "PASS",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "v1_results_sha256": base.sha256_file(V1_RESULTS),
        "fit_family_count": 1,
        "fit_details": fit_details,
        "subset_metrics": subset_metrics,
        "controls": controls,
        "paired_episode_comparisons": comparisons,
        "geometry": v1["geometry"],
        "diagnoses": diagnoses,
        "route_status": "PENDING_g_C_AND_P4; X_NOT_SELECTED",
        "parity_with_v1": {name: "PASS" if value else "FAIL" for name, value in parity.items()},
        "supersedes": protocol["superseded_v1_fields"],
        "limitations": v1["limitations"],
    }
    audit = {
        "schema": "tactile3d-unit.s4-3-pi2n-policy-domain-v1-structural-audit.v1",
        "status": "V1_SUBSET_LABELS_PARTIALLY_SUPERSEDED",
        "v1_results_sha256": base.sha256_file(V1_RESULTS),
        "v1_labels_sha256": base.sha256_file(V1_LABELS),
        "structural_finding": "unconditional TRAIN q75 force thresholds were zero due to sparse contact",
        "v1_preserved": ["overall", "boundary", "paired comparisons", "geometry"],
        "v1_superseded": protocol["superseded_v1_fields"],
        "candidate_or_formal_outcomes_used": False,
    }
    base.atomic_json(LABELS_OUTPUT, labels_payload)
    base.atomic_json(OUTPUT, result)
    base.atomic_json(AUDIT_OUTPUT, audit)
    print(json.dumps({"status": result["status"], "diagnoses": diagnoses, "dynamic_threshold": dynamic_threshold, "high_force_threshold": high_force_threshold, "subset_rows": labels_payload["check_subset_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
