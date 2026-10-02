#!/usr/bin/env python3
"""Run the single frozen PI2B teacher representation confirmation evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_teacher.evaluation import (  # noqa: E402
    classification_probe,
    continuity,
    deterministic_control_indices,
    geometry_metrics,
    paired_cosine,
    recovery_metrics,
    regression_probe,
    retrieval_metrics,
    source_relative_rank,
)
from gr00t.simulation.pi2b_teacher.matched_teacher import MatchedTeacher  # noqa: E402
from scripts.simulation.pi2b_teacher.common import (  # noqa: E402
    atomic_json,
    canonical_digest,
    load_json,
    sha256_file,
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def load_teacher(path: Path, expected_sha: str, device: torch.device) -> MatchedTeacher:
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"teacher checkpoint hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MatchedTeacher(tuple(payload["modalities"]), initialization_seed=int(payload["initialization_seed"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def encode(
    model: MatchedTeacher,
    modality: str,
    native: np.ndarray,
    device: torch.device,
    *,
    recover: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    shared = np.empty_like(native, dtype=np.float32)
    recovered = np.empty_like(native, dtype=np.float32) if recover else None
    for start in range(0, len(native), 1024):
        stop = min(start + 1024, len(native))
        value = model.encode(modality, torch.from_numpy(native[start:stop]).to(device))
        shared[start:stop] = value.float().cpu().numpy()
        if recovered is not None:
            recovered[start:stop] = model.recover(modality, value).float().cpu().numpy()
    return shared, recovered


def group_keys(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [f"{task}::{source}" for task, source in zip(arrays["task"], arrays["source_trajectory_id"])],
        dtype=str,
    )


def train_gap_rows(arrays: dict[str, np.ndarray]) -> np.ndarray:
    keys = group_keys(arrays)
    selected = []
    for task in sorted(set(arrays["task"].tolist())):
        groups = sorted(set(keys[arrays["task"] == task].tolist()))[:3]
        selected.extend(np.flatnonzero(np.isin(keys, groups)).tolist())
    result = np.asarray(sorted(selected), dtype=np.int64)
    if len(result) != 4860:
        raise RuntimeError(f"train-gap subset is not 4,860 rows: {len(result)}")
    return result


def masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {"overall": np.ones(len(arrays["pair_id"]), dtype=bool)}
    for task in sorted(set(arrays["task"].tolist())):
        result[f"task:{task}"] = arrays["task"] == task
    if "dynamic" in arrays:
        result["dynamic"] = arrays["dynamic"].astype(bool)
        result["non_dynamic"] = ~arrays["dynamic"].astype(bool)
    if "contact_boundary" in arrays:
        result["contact_boundary"] = arrays["contact_boundary"].astype(bool)
        result["non_boundary"] = ~arrays["contact_boundary"].astype(bool)
    return result


def alignment_report(left: np.ndarray, right: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    report = {}
    for name, mask in masks(arrays).items():
        if int(mask.sum()) < 2:
            report[name] = {"status": "N/A_TOO_FEW_ROWS", "rows": int(mask.sum())}
        else:
            report[name] = {
                "rows": int(mask.sum()),
                "forward": retrieval_metrics(left[mask], right[mask]),
                "reverse": retrieval_metrics(right[mask], left[mask]),
                "paired_cosine": float(paired_cosine(left[mask], right[mask]).mean()),
            }
    return report


def group_common_metrics(
    arrays: dict[str, np.ndarray],
    u_v: np.ndarray,
    u_a: np.ndarray,
    r_v: np.ndarray,
    r_a: np.ndarray,
    u_a_reversed: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    groups = group_keys(arrays)
    result = {}
    for group in sorted(set(groups.tolist())):
        rows = groups == group
        va = retrieval_metrics(u_v[rows], u_a[rows])
        av = retrieval_metrics(u_a[rows], u_v[rows])
        item = {
            "vision_native_recovery_mse": float(np.square(r_v[rows].astype(np.float64) - arrays["z_v"][rows]).mean()),
            "action_native_recovery_mse": float(np.square(r_a[rows].astype(np.float64) - arrays["z_a"][rows]).mean()),
            "mean_bidirectional_va_retrieval_r10": 0.5 * (va["recall_at_10"] + av["recall_at_10"]),
            "mean_bidirectional_va_mrr": 0.5 * (va["mrr"] + av["mrr"]),
        }
        if u_a_reversed is not None:
            item["action_reversal_margin"] = float((paired_cosine(u_v[rows], u_a[rows]) - paired_cosine(u_v[rows], u_a_reversed[rows])).mean())
        for stratum, field in (("dynamic", "dynamic"), ("boundary", "contact_boundary")):
            if field not in arrays:
                continue
            subset = rows & arrays[field].astype(bool)
            if int(subset.sum()) >= 2:
                subset_va = retrieval_metrics(u_v[subset], u_a[subset])
                subset_av = retrieval_metrics(u_a[subset], u_v[subset])
                item[f"{stratum}_vision_recovery_mse"] = float(np.square(r_v[subset].astype(np.float64) - arrays["z_v"][subset]).mean())
                item[f"{stratum}_action_recovery_mse"] = float(np.square(r_a[subset].astype(np.float64) - arrays["z_a"][subset]).mean())
                item[f"{stratum}_mean_va_retrieval_r10"] = 0.5 * (subset_va["recall_at_10"] + subset_av["recall_at_10"])
        result[group] = item
    return result


def recovery_strata(
    prediction: np.ndarray, target: np.ndarray, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    result = {}
    for name, mask in masks(arrays).items():
        result[name] = (
            recovery_metrics(prediction[mask], target[mask])
            if int(mask.sum()) >= 2
            else {"status": "N/A_TOO_FEW_ROWS", "rows": int(mask.sum())}
        )
    return result


def probe_family(
    train_arrays: dict[str, np.ndarray],
    fresh: dict[str, np.ndarray],
    train_features: dict[str, np.ndarray],
    fresh_features: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    report: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for feature_name in sorted(train_features):
        report[feature_name] = {}
        predictions[feature_name] = {}
        for target, classes in (("contact_transition", range(4)), ("force_trend", range(3))):
            metric, prediction = classification_probe(
                train_features[feature_name], fresh_features[feature_name],
                train_arrays[target], fresh[target], classes,
            )
            report[feature_name][target] = metric
            predictions[feature_name][target] = prediction
        metric, prediction = regression_probe(
            train_features[feature_name], fresh_features[feature_name],
            train_arrays["future_total_force"] - train_arrays["current_total_force"],
            fresh["force_delta"],
        )
        report[feature_name]["force_delta"] = metric
        predictions[feature_name]["force_delta"] = prediction
        if "region_transition" in fresh:
            regions = {}
            for region in range(fresh["region_transition"].shape[1]):
                # Historical TRAIN cache has no per-region transition labels; the
                # fresh-only regional class balance remains a descriptive limit.
                regions[str(region)] = {
                    "status": "N/A_TRAIN_REGION_LABEL_UNAVAILABLE",
                    "fresh_positive": int(fresh["region_transition"][:, region].sum()),
                }
            report[feature_name]["region_transition"] = regions
    groups = group_keys(fresh)
    group_metrics: dict[str, dict[str, float]] = {}
    for group in sorted(set(groups.tolist())):
        rows = groups == group
        item = {}
        for feature_name in sorted(predictions):
            prediction = predictions[feature_name]["contact_transition"]
            item[f"{feature_name}_contact_macro_f1"] = float(
                f1_score_local(fresh["contact_transition"][rows], prediction[rows], [0, 1, 2, 3])
            )
        group_metrics[group] = item
    return report, group_metrics


def f1_score_local(target: np.ndarray, prediction: np.ndarray, classes: list[int]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(target, prediction, labels=classes, average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    write_root = Path(runtime["write_root"])
    pretest = load_json(write_root / "artifacts/pretest_freeze.json")
    if pretest["status"] != "PASS" or pretest["fresh_performance_loaded"]:
        raise RuntimeError("fresh evaluation is not authorized")
    for relative, expected in pretest["implementation_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"frozen evaluator changed: {relative}")
    cache_manifest = load_json(write_root / "artifacts/confirmation_cache_manifest.json")
    fresh_path = Path(cache_manifest["path"])
    if cache_manifest["status"] != "COMPLETE_NOT_EVALUATED" or sha256_file(fresh_path) != cache_manifest["sha256"]:
        raise RuntimeError("fresh native cache is not frozen")
    statuses = {mode: load_json(write_root / "status" / f"{mode}.json") for mode in ("T_VA_match", "T_VAC_match")}
    for mode, value in statuses.items():
        if value["state"] != "COMPLETE" or value["final_step"] != 800:
            raise RuntimeError(f"{mode} is incomplete")
        if value["checkpoint_sha256"] != pretest["teacher_checkpoint_sha256"][mode]:
            raise RuntimeError(f"{mode} checkpoint identity changed")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("evaluation requires exactly one supervisor-assigned GPU")
    history_root = Path(runtime["worktree_root"])
    train = load_npz(history_root / ".local/refs/base/cache/simulation/s4_2_formal/paired_train.npz")
    dev = load_npz(history_root / ".local/refs/base/cache/simulation/s4_2_formal/paired_validation.npz")
    fresh = load_npz(fresh_path)
    if set(train["pair_id"].tolist()) & set(fresh["pair_id"].tolist()) or set(dev["pair_id"].tolist()) & set(fresh["pair_id"].tolist()):
        raise RuntimeError("fresh pair identity overlaps TRAIN/DEV")
    models = {
        mode: load_teacher(Path(statuses[mode]["checkpoint"]), statuses[mode]["checkpoint_sha256"], device)
        for mode in statuses
    }
    encoded: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    results: dict[str, Any] = {}
    all_group_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for mode, model in models.items():
        encoded[mode] = {}
        results[mode] = {"common": {}, "splits": {}}
        for split_name, arrays in (("train", train), ("dev", dev), ("fresh", fresh)):
            encoded[mode][split_name] = {}
            for modality, field in (("vision", "z_v"), ("action", "z_a")):
                shared, recovered = encode(model, modality, arrays[field], device, recover=True)
                encoded[mode][split_name][modality] = shared
                encoded[mode][split_name][f"recovered_{modality}"] = recovered
            rows = train_gap_rows(arrays) if split_name == "train" else np.arange(len(arrays["pair_id"]))
            split_arrays = {name: value[rows] for name, value in arrays.items() if len(value) == len(arrays["pair_id"])}
            u_v = encoded[mode][split_name]["vision"][rows]
            u_a = encoded[mode][split_name]["action"][rows]
            r_v = encoded[mode][split_name]["recovered_vision"][rows]
            r_a = encoded[mode][split_name]["recovered_action"][rows]
            results[mode]["splits"][split_name] = {
                "rows": len(rows),
                "vision_recovery": recovery_metrics(r_v, split_arrays["z_v"]),
                "action_recovery": recovery_metrics(r_a, split_arrays["z_a"]),
                "vision_recovery_strata": recovery_strata(r_v, split_arrays["z_v"], split_arrays),
                "action_recovery_strata": recovery_strata(r_a, split_arrays["z_a"], split_arrays),
                "va_alignment": alignment_report(u_v, u_a, split_arrays),
                "vision_geometry": geometry_metrics(u_v),
                "action_geometry": geometry_metrics(u_a),
                "vision_source_relative_rank": source_relative_rank(u_v, group_keys(split_arrays)),
                "action_source_relative_rank": source_relative_rank(u_a, group_keys(split_arrays)),
                "vision_continuity": continuity(u_v, split_arrays["episode_id"], split_arrays["anchor_step"]),
                "action_continuity": continuity(u_a, split_arrays["episode_id"], split_arrays["anchor_step"]),
            }
        u_v = encoded[mode]["fresh"]["vision"]
        u_a = encoded[mode]["fresh"]["action"]
        u_a_reversed, _ = encode(model, "action", fresh["z_a_reversed"], device)
        u_v_current, _ = encode(model, "vision", fresh["z_v_current"], device)
        groups = group_keys(fresh)
        different = deterministic_control_indices(fresh["task"], groups, fresh["episode_id"], same_task=False)
        same_task = deterministic_control_indices(fresh["task"], groups, fresh["episode_id"], same_task=True)
        correct = paired_cosine(u_v, u_a)
        results[mode]["common"]["controls"] = {
            "paired_cosine": float(correct.mean()),
            "different_group_cosine": float(paired_cosine(u_v, u_a[different]).mean()),
            "same_task_wrong_time_cosine": float(paired_cosine(u_v, u_a[same_task]).mean()),
            "reversed_action_cosine": float(paired_cosine(u_v, u_a_reversed).mean()),
            "current_vision_cosine": float(paired_cosine(u_v_current, u_a).mean()),
            "paired_minus_different_group": float((correct - paired_cosine(u_v, u_a[different])).mean()),
            "paired_minus_same_task_wrong_time": float((correct - paired_cosine(u_v, u_a[same_task])).mean()),
            "action_reversal_margin": float((correct - paired_cosine(u_v, u_a_reversed)).mean()),
            "vision_transition_margin": float((correct - paired_cosine(u_v_current, u_a)).mean()),
        }
        group_report = group_common_metrics(
            fresh, u_v, u_a,
            encoded[mode]["fresh"]["recovered_vision"],
            encoded[mode]["fresh"]["recovered_action"],
            u_a_reversed,
        )
        if mode == "T_VAC_match":
            for split_name, arrays in (("train", train), ("dev", dev), ("fresh", fresh)):
                u_c, r_c = encode(model, "contact", arrays["z_c"], device, recover=True)
                encoded[mode][split_name]["contact"] = u_c
                encoded[mode][split_name]["recovered_contact"] = r_c
            u_c = encoded[mode]["fresh"]["contact"]
            u_c_reversed, _ = encode(model, "contact", fresh["z_c_reversed"], device)
            vc_correct = paired_cosine(u_v, u_c)
            ac_correct = paired_cosine(u_a, u_c)
            results[mode]["contact"] = {
                "contact_recovery": recovery_metrics(encoded[mode]["fresh"]["recovered_contact"], fresh["z_c"]),
                "vc_alignment": alignment_report(u_v, u_c, fresh),
                "ac_alignment": alignment_report(u_a, u_c, fresh),
                "temporal_controls": {
                    "vision_contact_reversed_margin": float((vc_correct - paired_cosine(u_v, u_c_reversed)).mean()),
                    "action_contact_reversed_margin": float((ac_correct - paired_cosine(u_a, u_c_reversed)).mean()),
                },
                "contact_geometry": geometry_metrics(u_c),
                "contact_source_relative_rank": source_relative_rank(u_c, groups),
            }
            for group in group_report:
                rows = groups == group
                diff_index = different[rows]
                group_report[group]["vision_contact_paired_margin"] = float((paired_cosine(u_v[rows], u_c[rows]) - paired_cosine(u_v[rows], u_c[diff_index])).mean())
                group_report[group]["action_contact_paired_margin"] = float((paired_cosine(u_a[rows], u_c[rows]) - paired_cosine(u_a[rows], u_c[diff_index])).mean())
                group_report[group]["vision_contact_temporal_margin"] = float((paired_cosine(u_v[rows], u_c[rows]) - paired_cosine(u_v[rows], u_c_reversed[rows])).mean())
                group_report[group]["action_contact_temporal_margin"] = float((paired_cosine(u_a[rows], u_c[rows]) - paired_cosine(u_a[rows], u_c_reversed[rows])).mean())
        all_group_metrics[mode] = group_report
    train_features, fresh_features = {}, {}
    for mode in models:
        for modality in ("vision", "action"):
            key = f"{mode}:{modality}"
            train_features[key] = encoded[mode]["train"][modality]
            fresh_features[key] = encoded[mode]["fresh"][modality]
    for modality, field in (("native_vision", "z_v"), ("native_action", "z_a")):
        train_features[modality] = train[field]
        fresh_features[modality] = fresh[field]
    probes, probe_groups = probe_family(train, fresh, train_features, fresh_features)
    for mode in models:
        for group, values in all_group_metrics[mode].items():
            vision_key = f"{mode}:vision_contact_macro_f1"
            action_key = f"{mode}:action_contact_macro_f1"
            values["mean_common_va_contact_probe_macro_f1"] = 0.5 * (
                probe_groups[group][vision_key] + probe_groups[group][action_key]
            )
    result = {
        "schema": "tactile3d-unit.s4-3-pi2b-teacher-confirmation-evaluation.v1",
        "status": "COMPLETE",
        "fresh_rows": len(fresh["pair_id"]),
        "fresh_groups": len(set(group_keys(fresh).tolist())),
        "models": results,
        "probes": probes,
        "group_metrics": all_group_metrics,
        "group_tasks": {group: group.split("::", 1)[0] for group in sorted(set(group_keys(fresh).tolist()))},
        "checkpoint_sha256": pretest["teacher_checkpoint_sha256"],
        "cache_sha256": cache_manifest["sha256"],
        "policy_utility": "NOT_TESTED",
        "track_a_performance_read": False,
    }
    result["metric_digest"] = canonical_digest(result)
    atomic_json(write_root / "artifacts/common_va_metrics.json", {"models": {mode: {"splits": value["splits"], "common": value["common"]} for mode, value in results.items()}, "metric_digest": result["metric_digest"]})
    atomic_json(write_root / "artifacts/contact_capability_metrics.json", {"T_VAC_match": results["T_VAC_match"]["contact"], "common_path_probes": probes, "T_VA_match_contact_path": "N/A"})
    atomic_json(write_root / "artifacts/group_metrics.json", {"group_metrics": all_group_metrics, "group_tasks": result["group_tasks"]})
    atomic_json(write_root / "artifacts/evaluation_result.json", result)
    atomic_json(write_root / "status/evaluation.json", {"state": "COMPLETE", "metric_digest": result["metric_digest"], "fresh_rows": len(fresh["pair_id"]), "fresh_groups": result["fresh_groups"]})
    print(json.dumps({"status": "COMPLETE", "metric_digest": result["metric_digest"], "fresh_rows": len(fresh["pair_id"])}, indent=2))


if __name__ == "__main__":
    main()
