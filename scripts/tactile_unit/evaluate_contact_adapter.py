#!/usr/bin/env python3
"""Evaluate S3.2 adaptors on the frozen canonical S2 test split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_shared_rq_compatibility import (
    distribution_summary,
    probe_metric,
    query_metrics,
    sha256_file,
)
from contact_adapter_common import (
    DEFAULT_CACHE,
    DEFAULT_CODES,
    DEFAULT_S1,
    DEFAULT_S2,
    DEFAULT_SPEC,
    DEFAULT_T4,
    DEFAULT_TRANSITIONS,
    component_digests,
    decode_codes,
    evaluate_transformed,
    load_arrays,
    load_runtime,
    reconstruction_bundle,
    transform_codes,
    verify_gpu,
)
from gr00t.contact_dynamics.evaluation import different_episode_permutation
from gr00t.tactile_unit.compatibility import code_frequency, codebook_usage
from gr00t.tactile_unit.contact_adapter import ContactCodebookAdaptor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS = ROOT / ".local/experiments/tactile_unit/s3_2"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_2"
S3_0_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_0"
S3_0_CACHE = ROOT / ".local/cache/tactile_unit/s3_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--transition-cache", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--code-cache", type=Path, default=DEFAULT_CODES)
    parser.add_argument("--runtime-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--s1-checkpoint", type=Path, default=DEFAULT_S1)
    parser.add_argument("--s2-checkpoint", type=Path, default=DEFAULT_S2)
    parser.add_argument("--t4-dir", type=Path, default=DEFAULT_T4)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_adaptor(path: Path, expected_hash: str, device: torch.device) -> ContactCodebookAdaptor:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"adaptor checkpoint identity mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "tactile3d-unit.s3-2-contact-adaptor.v1":
        raise RuntimeError("adaptor checkpoint schema mismatch")
    adaptor = ContactCodebookAdaptor(str(payload["architecture"]))
    adaptor.load_state_dict(payload["state_dict"], strict=True)
    return adaptor.eval().to(device)


def semantic_probe_bundle(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    definitions = {
        "contact_transition": ("contact_transition", 4),
        "force_trend": ("force_trend_class", 3),
    }
    return {
        name: probe_metric(
            train_feature,
            test_feature,
            np.asarray(train_arrays[key]),
            np.asarray(test_arrays[key]),
            classes,
            device,
            batch_size,
            10.0,
        )
        for name, (key, classes) in definitions.items()
    }


def noncollapse_metrics(values: np.ndarray, seed: int) -> dict[str, Any]:
    array = np.asarray(values)
    variance = array.reshape(-1, array.shape[-1]).var(axis=0)
    result = distribution_summary(array, seed)
    result.update(query_metrics(array))
    result["per_dimension_variance"] = variance.tolist()
    result["near_zero_variance_fraction"] = float(np.mean(variance < 1e-8))
    result["per_query_variance"] = array.var(axis=(0, 2)).tolist()
    return result


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    device = verify_gpu()
    runtime = load_runtime(
        args.spec,
        args.transition_cache,
        args.code_cache,
        args.s1_checkpoint,
        args.s2_checkpoint,
        args.t4_dir,
        device,
    )
    spec = runtime["spec"]
    training_path = args.experiment_dir / "training_summary.json"
    training = json.loads(training_path.read_text())
    if training.get("status") != "PASS" or training.get("test_used_for_selection"):
        raise RuntimeError("S3.2 training/validation selection is not valid")
    if training["identity"] != runtime["identity"]:
        raise RuntimeError("training and evaluation frozen identities disagree")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_cache.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    seed = int(spec["seed"])
    before = component_digests(runtime)

    train_arrays = load_arrays(args.transition_cache, "train")
    test_arrays = load_arrays(args.transition_cache, "test")
    train_codes = np.load(args.code_cache / "train.npy", mmap_mode="r")
    test_codes = np.load(args.code_cache / "test.npy", mmap_mode="r")
    if tuple(test_codes.shape) != (17504, 8, 32):
        raise RuntimeError("canonical S3.2 test code geometry mismatch")

    adaptors = {
        architecture: load_adaptor(
            Path(training["checkpoint_paths"][architecture]),
            training["checkpoint_hashes"][architecture],
            device,
        )
        for architecture in ("affine", "mlp")
    }
    identity = ContactCodebookAdaptor("identity").eval().to(device)
    conditions: dict[str, dict[str, np.ndarray]] = {}
    identity_q = np.load(S3_0_CACHE / "contact_test_quantized.npy", mmap_mode="r")
    identity_i = np.load(S3_0_CACHE / "contact_test_indices.npy", mmap_mode="r")
    first_a, first_q, first_i = transform_codes(
        identity, runtime["rq"], test_codes[:64], device, args.batch_size
    )
    if not (
        np.array_equal(first_a, np.asarray(test_codes[:64]))
        and np.array_equal(first_q, np.asarray(identity_q[:64]))
        and np.array_equal(first_i, np.asarray(identity_i[:64]))
    ):
        raise RuntimeError("S3.0 identity frozen-RQ cache does not reproduce")
    conditions["identity"] = {
        "adapted": np.asarray(test_codes),
        "quantized": identity_q,
        "indices": identity_i,
    }
    for architecture, adaptor in adaptors.items():
        adapted, quantized, indices = transform_codes(
            adaptor,
            runtime["rq"],
            test_codes,
            device,
            args.batch_size,
            adapted_path=args.runtime_cache / f"{architecture}_test_adapted.npy",
            quantized_path=args.runtime_cache / f"{architecture}_test_quantized.npy",
            indices_path=args.runtime_cache / f"{architecture}_test_indices.npy",
        )
        if indices.min() < 0 or indices.max() >= int(runtime["rq_identity"]["codes_per_stage"]):
            raise RuntimeError(f"{architecture} produced invalid RQ indices")
        conditions[architecture] = {
            "adapted": adapted,
            "quantized": quantized,
            "indices": indices,
        }

    s3_0 = json.loads((S3_0_OUTPUT / "s3_0_summary.json").read_text())
    quantization_rows = [
        row for row in s3_0["quantization"]["rows"] if row["modality"] != "contact"
    ]
    reconstruction: dict[str, Any] = {}
    quantization: dict[str, Any] = {}
    usage: dict[str, list[dict[str, Any]]] = {}
    usage_rows: list[dict[str, Any]] = []
    query_diversity: dict[str, Any] = {}
    code_frequencies: dict[str, np.ndarray] = {}
    codebook_size = int(runtime["rq_identity"]["codes_per_stage"])
    for name, values in conditions.items():
        evaluation = evaluate_transformed(
            runtime["s2"].decoder,
            values["adapted"],
            values["quantized"],
            test_arrays,
            device,
            args.batch_size,
        )
        quantization[name] = evaluation["quantization"]
        quantization_rows.append({"modality": name, **evaluation["quantization"]})
        reconstruction[f"{name}_continuous"] = evaluation[
            "adapted_continuous_reconstruction"
        ]
        reconstruction[f"{name}_quantized"] = evaluation[
            "adapted_quantized_reconstruction"
        ]
        usage[name] = []
        frequencies = []
        for stage in range(values["indices"].shape[-1]):
            row = codebook_usage(values["indices"][:, :, stage], codebook_size)
            usage[name].append({"stage": stage, **row})
            usage_rows.append(
                {"row_type": "model_stage", "model": name, "query": "all", "stage": stage, **row}
            )
            frequencies.append(code_frequency(values["indices"][:, :, stage], codebook_size))
            for query in range(8):
                query_row = codebook_usage(values["indices"][:, query, stage], codebook_size)
                usage_rows.append(
                    {
                        "row_type": "model_query_stage",
                        "model": name,
                        "query": query,
                        "stage": stage,
                        **query_row,
                    }
                )
        code_frequencies[name] = np.stack(frequencies)
        query_diversity[f"{name}_continuous"] = noncollapse_metrics(
            values["adapted"], seed
        )
        query_diversity[f"{name}_quantized"] = noncollapse_metrics(
            values["quantized"], seed
        )

    current = np.asarray(test_arrays["current"])
    future = np.asarray(test_arrays["future"])
    dynamic = np.asarray(test_arrays["dynamic"], dtype=bool)
    continuous_prediction = decode_codes(
        runtime["s2"].decoder, test_codes, current, device, args.batch_size
    )
    reconstruction["continuous"] = reconstruction_bundle(
        current, future, continuous_prediction, dynamic
    )
    zero_prediction = decode_codes(
        runtime["s2"].decoder, np.zeros_like(test_codes), current, device, args.batch_size
    )
    reconstruction["zero"] = reconstruction_bundle(current, future, zero_prediction, dynamic)
    permutation = different_episode_permutation(test_arrays["episode_id"], seed=seed)
    shuffled_prediction = decode_codes(
        runtime["s2"].decoder,
        np.asarray(test_codes)[permutation],
        current,
        device,
        args.batch_size,
    )
    reconstruction["shuffled"] = reconstruction_bundle(
        current, future, shuffled_prediction, dynamic
    )

    identity_train_q = np.load(S3_0_CACHE / "contact_train_quantized.npy", mmap_mode="r")
    semantic: dict[str, dict[str, Any]] = {
        "continuous": semantic_probe_bundle(
            train_codes, test_codes, train_arrays, test_arrays, device, args.batch_size
        ),
        "identity_quantized": semantic_probe_bundle(
            identity_train_q, identity_q, train_arrays, test_arrays, device, args.batch_size
        ),
    }
    for architecture in ("affine", "mlp"):
        train_adapted = np.load(
            args.runtime_cache / f"{architecture}_train_adapted.npy", mmap_mode="r"
        )
        train_quantized = np.load(
            args.runtime_cache / f"{architecture}_train_quantized.npy", mmap_mode="r"
        )
        semantic[f"{architecture}_continuous"] = semantic_probe_bundle(
            train_adapted,
            conditions[architecture]["adapted"],
            train_arrays,
            test_arrays,
            device,
            args.batch_size,
        )
        semantic[f"{architecture}_quantized"] = semantic_probe_bundle(
            train_quantized,
            conditions[architecture]["quantized"],
            train_arrays,
            test_arrays,
            device,
            args.batch_size,
        )

    semantic_rows = [
        {"representation": representation, "probe": probe, **metrics}
        for representation, probes in semantic.items()
        for probe, metrics in probes.items()
    ]
    reconstruction_rows = [
        {"condition": condition, "scope": scope, **metrics}
        for condition, scopes in reconstruction.items()
        for scope, metrics in scopes.items()
    ]
    diversity_rows = []
    for condition, metrics in query_diversity.items():
        diversity_rows.append(
            {
                "condition": condition,
                "collapsed_query_fraction": metrics["collapsed_sample_fraction"],
                "mean_off_diagonal_query_cosine": metrics["mean_off_diagonal_cosine"],
                "mean_distance_from_token_mean": metrics[
                    "mean_distance_from_sample_token_mean"
                ],
                "flattened_effective_rank": metrics["flattened_effective_rank"],
                "pooled_effective_rank": metrics["pooled_effective_rank"],
                "near_zero_variance_fraction": metrics["near_zero_variance_fraction"],
                "token_norm_mean": metrics["token_norm"]["mean"],
                "token_norm_std": metrics["token_norm"]["std"],
                "pairwise_distance_mean": metrics["pooled_pairwise_distance"]["mean"],
            }
        )

    selected = str(training["selected_architecture"])
    identity_relative = quantization["identity"]["relative_distortion"]
    selected_relative = quantization[selected]["relative_distortion"]
    identity_dynamic = reconstruction["identity_quantized"]["dynamic"]["future_mse"]
    selected_dynamic = reconstruction[f"{selected}_quantized"]["dynamic"]["future_mse"]
    control_dynamic = min(
        reconstruction["zero"]["dynamic"]["future_mse"],
        reconstruction["shuffled"]["dynamic"]["future_mse"],
    )
    semantic_retention: dict[str, float] = {}
    semantic_not_worse_than_identity = True
    for probe in ("contact_transition", "force_trend"):
        continuous = semantic["continuous"][probe]["macro_f1"]
        majority = semantic["continuous"][probe]["majority"]["macro_f1"]
        adapted = semantic[f"{selected}_quantized"][probe]["macro_f1"]
        identity_score = semantic["identity_quantized"][probe]["macro_f1"]
        semantic_retention[probe] = float(
            (adapted - majority) / max(continuous - majority, 1e-12)
        )
        semantic_not_worse_than_identity &= adapted >= identity_score
    perplexity_improved = all(
        usage[selected][stage]["perplexity"] > usage["identity"][stage]["perplexity"]
        for stage in range(len(usage[selected]))
    )
    no_collapse = (
        query_diversity[f"{selected}_continuous"]["collapsed_sample_fraction"] == 0.0
        and query_diversity[f"{selected}_quantized"]["collapsed_sample_fraction"] == 0.0
        and all(row["top1_frequency"] < 0.9 and row["active_codes"] > 1 for row in usage[selected])
    )
    compatibility_solved = selected_relative < identity_relative and perplexity_improved and no_collapse
    semantic_preserved = semantic_not_worse_than_identity and all(
        value > 0 for value in semantic_retention.values()
    )
    decoder_meaningful = selected_dynamic < identity_dynamic and selected_dynamic < control_dynamic

    after = component_digests(runtime)
    integrity = {
        name: {"before": digest, "after": after[name], "unchanged": digest == after[name]}
        for name, digest in before.items()
    }
    structural = (
        all(item["unchanged"] for item in integrity.values())
        and training["gradient_integrity"]["status"] == "PASS"
        and all(np.isfinite(conditions[name]["adapted"]).all() for name in conditions)
    )
    if not structural:
        decision = "STRUCTURAL_FAIL"
    elif compatibility_solved and semantic_preserved and decoder_meaningful:
        decision = "ADAPTOR_READY"
    elif compatibility_solved and semantic_preserved:
        decision = "ADAPTOR_READY_WITH_DECODER_WARNING"
    else:
        decision = "ADAPTOR_INSUFFICIENT"

    write_csv(metrics_dir / "quantization.csv", quantization_rows)
    write_csv(metrics_dir / "reconstruction.csv", reconstruction_rows)
    write_csv(metrics_dir / "codebook_usage.csv", usage_rows)
    write_csv(metrics_dir / "semantic_retention.csv", semantic_rows)
    write_csv(metrics_dir / "query_diversity.csv", diversity_rows)
    architecture_rows = [
        {
            "model": name,
            "parameters": 0 if name == "identity" else adaptors[name].parameter_count,
            "relative_distortion": quantization[name]["relative_distortion"],
            "dynamic_future_mse": reconstruction[f"{name}_quantized"]["dynamic"][
                "future_mse"
            ],
            "stage0_perplexity": usage[name][0]["perplexity"],
            "stage1_perplexity": usage[name][1]["perplexity"],
            "selected": name == selected,
        }
        for name in ("identity", "affine", "mlp")
    ]
    write_csv(args.output_dir / "architecture_comparison.csv", architecture_rows)

    contact_manifest = json.loads((S3_0_OUTPUT / "contact_manifest.json").read_text())
    subset = np.asarray([row["source_index"] for row in contact_manifest["rows"]], dtype=np.int64)
    with np.load(args.t4_dir / "features/unit_representation_features.npz", allow_pickle=False) as data:
        reference_l2 = np.asarray(data["l2"], dtype=np.float32)
    visualization_arrays = {
        "reference_l2": reference_l2,
        "contact_identity": np.asarray(test_codes)[subset],
        "contact_identity_quantized": np.asarray(identity_q)[subset],
        "selected_adapted": np.asarray(conditions[selected]["adapted"])[subset],
        "selected_quantized": np.asarray(conditions[selected]["quantized"])[subset],
        "code_frequency": np.stack(
            [code_frequencies[name] for name in ("identity", "affine", "mlp")]
        ),
        "model": np.asarray(("identity", "affine", "mlp")),
    }
    np.savez_compressed(args.output_dir / "visualization_data.npz", **visualization_arrays)

    summary = {
        "schema": "tactile3d-unit.s3-2-contact-adaptor-evaluation.v1",
        "status": "COMPLETE" if decision != "STRUCTURAL_FAIL" else "INVALID",
        "decision": decision,
        "identity": runtime["identity"],
        "environment": training["environment"],
        "data": {
            "train_pairs": len(train_codes),
            "validation_pairs": int(spec["data"]["pairs"]["validation"]),
            "test_pairs": len(test_codes),
            "dynamic_test_pairs": int(dynamic.sum()),
            "test_used_for_selection": False,
            "cache_identity": "PASS",
        },
        "frozen_components": integrity,
        "gradient_integrity": training["gradient_integrity"],
        "adaptors": {
            "identity": {"parameters": 0},
            **{
                name: {
                    "parameters": adaptor.parameter_count,
                    "checkpoint": training["checkpoint_paths"][name],
                    "checkpoint_sha256": training["checkpoint_hashes"][name],
                }
                for name, adaptor in adaptors.items()
            },
        },
        "validation_selection": {
            "selected": selected,
            "reason": training["selection_reason"],
            "hyperparameters": training["selected_hyperparameters"],
            "candidate_metrics": {
                name: training["candidates"][name]["best_validation"]
                for name in ("affine", "mlp")
            },
        },
        "quantization": quantization,
        "original_unit_quantization_references": {
            row["modality"]: row for row in quantization_rows[:3]
        },
        "codebook_usage": usage,
        "reconstruction": reconstruction,
        "dynamic_semantic_retention": semantic,
        "semantic_advantage_retention": semantic_retention,
        "query_diversity_and_noncollapse": query_diversity,
        "decision_diagnostics": {
            "selected_distortion_over_identity": selected_relative / identity_relative,
            "selected_distortion_over_original_worst": selected_relative
            / max(row["relative_distortion"] for row in quantization_rows[:3]),
            "selected_dynamic_mse_over_identity": selected_dynamic / identity_dynamic,
            "selected_dynamic_mse_over_zero": selected_dynamic
            / reconstruction["zero"]["dynamic"]["future_mse"],
            "selected_dynamic_mse_over_shuffled": selected_dynamic
            / reconstruction["shuffled"]["dynamic"]["future_mse"],
            "perplexity_improved_both_stages": perplexity_improved,
            "no_collapse": no_collapse,
            "semantic_not_worse_than_identity": semantic_not_worse_than_identity,
            "compatibility_solved_comparatively": compatibility_solved,
            "semantic_preserved_comparatively": semantic_preserved,
            "frozen_decoder_meaningful": decoder_meaningful,
            "engineering_comparisons_not_scientific_absolute_thresholds": True,
        },
        "decoder_compatibility": (
            "YES" if decoder_meaningful else "PARTIAL" if compatibility_solved else "NO"
        ),
        "artifacts": {
            "metrics": sorted(str(path) for path in metrics_dir.glob("*.csv")),
            "architecture_comparison": str(args.output_dir / "architecture_comparison.csv"),
            "visualization_data": str(args.output_dir / "visualization_data.npz"),
        },
        "runtime_seconds": time.monotonic() - started,
        "s3_3_started": False,
    }
    write_json(args.output_dir / "s3_2_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": decision,
                "selected": selected,
                "diagnostics": summary["decision_diagnostics"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 2 if decision == "STRUCTURAL_FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
