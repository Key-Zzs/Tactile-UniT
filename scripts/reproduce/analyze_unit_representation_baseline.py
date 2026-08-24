#!/usr/bin/env python3
"""Analyze extracted canonical UniT representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from unit_representation_metrics import (
    PAIRS,
    ami_score,
    code_agreement,
    codebook_stats,
    linear_cka,
    mean_query_pool,
    mmd_rbf,
    nmi_score,
    paired_cosine_statistics,
    retrieval_metrics,
    shuffled_negative_cosine,
    sliced_wasserstein,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / ".local/artifacts/reproduction/t4"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def analyze_layer(features: np.ndarray, seed: int) -> dict[str, Any]:
    pooled = mean_query_pool(features)
    result: dict[str, Any] = {"pooling": "mean_query_then_l2_normalize", "pairs": {}, "pooled_shape": list(pooled.shape)}
    for left, right, pair_name in PAIRS:
        source = pooled[:, left]
        target = pooled[:, right]
        result["pairs"][pair_name] = {
            "paired_cosine": paired_cosine_statistics(source, target),
            "negative_cosine": shuffled_negative_cosine(source, target, seed=seed),
            "retrieval": {
                "source_to_target": retrieval_metrics(source, target),
                "target_to_source": retrieval_metrics(target, source),
            },
            "distribution": {
                "mmd": mmd_rbf(source, target),
                "sliced_wasserstein": sliced_wasserstein(source, target, projections=128, seed=seed),
                "linear_cka": linear_cka(source, target),
            },
        }
    return result


def collapse_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality in dict.fromkeys(str(row["modality"]) for row in rows):
        modality_rows = [row for row in rows if row["modality"] == modality]
        max_top1 = max(float(row["top1_frequency"]) for row in modality_rows)
        max_active_ratio = min(float(row["active_ratio"]) for row in modality_rows)
        if max_top1 >= 0.9 or max_active_ratio <= 0.02:
            classification = "SEVERE COLLAPSE"
        elif max_top1 >= 0.5 or max_active_ratio <= 0.1:
            classification = "WARNING"
        else:
            classification = "NORMAL"
        result[modality] = {
            "classification": classification,
            "max_top1_frequency": max_top1,
            "min_active_ratio": max_active_ratio,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    feature_path = args.output_dir / "features" / "unit_representation_features.npz"
    extraction_path = args.output_dir / "extraction_summary.json"
    if not feature_path.exists() or not extraction_path.exists():
        raise FileNotFoundError("Run extract_unit_representation_baseline.py first")
    extraction = json.loads(extraction_path.read_text())
    with np.load(feature_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    for key in ("l1", "l2", "l3", "l4"):
        if key not in arrays:
            raise KeyError(f"Missing feature layer {key}")
    if arrays["l4"].ndim != 4:
        raise ValueError(f"Expected L4 [N, modality, query, stage], got {arrays['l4'].shape}")

    output_metrics = args.output_dir / "metrics"
    output_metrics.mkdir(parents=True, exist_ok=True)
    seed = 42
    layer_results = {layer: analyze_layer(arrays[layer], seed) for layer in ("l1", "l2", "l3")}

    paired_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for layer, layer_result in layer_results.items():
        for pair_name, pair_result in layer_result["pairs"].items():
            paired = pair_result["paired_cosine"]
            negative = pair_result["negative_cosine"]
            paired_rows.append({
                "layer": layer,
                "pair": pair_name,
                **paired,
                "negative_mean": negative["negative_mean"],
                "positive_mean": negative["positive_mean"],
                "positive_minus_negative_margin": negative["margin"],
            })
            for direction, values in pair_result["retrieval"].items():
                retrieval_rows.append({"layer": layer, "pair": pair_name, "direction": direction, **values})
            distribution_rows.append({
                "layer": layer,
                "pair": pair_name,
                "mmd": pair_result["distribution"]["mmd"]["mmd"],
                "mmd_bandwidth": pair_result["distribution"]["mmd"]["bandwidth"],
                "swd": pair_result["distribution"]["sliced_wasserstein"]["swd"],
                "cka": pair_result["distribution"]["linear_cka"],
            })

    codebook_size = int(extraction["codebook_size"])
    modality_names = tuple(arrays["modality"].tolist())
    usage_rows = codebook_stats(arrays["l4"].astype(np.int64), codebook_size, modality_names=modality_names)
    agreement_rows = code_agreement(
        arrays["l4"].astype(np.int64), modality_names=modality_names, pair_specs=PAIRS
    )
    exact_rows = [row for row in agreement_rows if "stage_exact_match" in row]
    jaccard_rows = [row for row in agreement_rows if "active_set_jaccard" in row]
    nmi_rows = [row for row in agreement_rows if "nmi" in row]
    collapse = collapse_classification(usage_rows)
    sanity = {
        "paired_vs_shuffled": all(row["positive_minus_negative_margin"] > 0 for row in paired_rows),
        "retrieval_vs_chance": all(
            row[f"recall_at_{k}"] > row[f"chance_recall_at_{k}"]
            for row in retrieval_rows
            for k in (1, 5, 10)
        ),
        "interpretation": "A false sanity expectation is retained as a warning when implementation validity checks pass.",
    }
    representation_metrics = {
        "benchmark_layer_order": {"L1": "routed pre-VQ", "L2": "continuous VQ input", "L3": "quantized embedding"},
        "primary_layer": "L2",
        "pooling": "mean across Q then L2 normalize",
        "negative_seed": seed,
        "layers": layer_results,
        "sanity": sanity,
        "collapse": collapse,
    }
    (output_metrics / "representation_metrics.json").write_text(json.dumps(json_safe(representation_metrics), indent=2) + "\n")
    write_csv(output_metrics / "paired_alignment.csv", paired_rows)
    write_csv(output_metrics / "retrieval.csv", retrieval_rows)
    write_csv(output_metrics / "distribution_alignment.csv", distribution_rows)
    write_csv(output_metrics / "codebook_usage.csv", usage_rows)
    write_csv(output_metrics / "code_agreement.csv", exact_rows)
    write_csv(output_metrics / "active_set_overlap.csv", jaccard_rows)
    write_csv(output_metrics / "code_nmi_ami.csv", nmi_rows)

    analysis_summary = {
        "status": "PASS" if all(np.isfinite(arrays[key]).all() for key in ("l1", "l2", "l3")) else "FAIL",
        "sample_count": int(arrays["l1"].shape[0]),
        "layers_analyzed": ["L1", "L2", "L3", "L4"],
        "metric_files": sorted(str(path.relative_to(args.output_dir)) for path in output_metrics.glob("*.csv")) + ["metrics/representation_metrics.json"],
        "sanity": sanity,
        "collapse": collapse,
        "warning_if_sanity_false": True,
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(json_safe(analysis_summary), indent=2) + "\n")
    print(json.dumps(json_safe(analysis_summary), indent=2))
    return 0 if analysis_summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
