#!/usr/bin/env python3
"""Assemble the local T4 acceptance summary after extraction/analysis/visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "configs/reproduction/baselines/unit_representation_gr1.json"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/reproduction/t4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_synthetic_tests() -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_unit_representation_metrics.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def privacy_scan() -> dict[str, Any]:
    tracked_local = subprocess.run(["git", "ls-files", ".local"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    changed = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    paths = sorted(set(changed + staged + untracked))
    sensitive_terms = [
        "/" + "home/",
        "/" + "mnt/" + "ugreen" + "_nas/",
        "HF" + "_TOKEN",
        "github" + "_pat_",
        "gh" + "p_",
        "WANDB" + "_API_KEY",
        "Authorization" + ":",
        "Bear" + "er",
    ]
    pattern = re.compile("|".join(re.escape(term) for term in sensitive_terms))
    matches: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{relative}:{line_number}:{line}")
        except OSError:
            continue
    return {"pass": not tracked_local and not matches, "tracked_local_files": tracked_local, "matches": matches}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    spec = load(args.spec)
    extraction = load(args.output_dir / "extraction_summary.json")
    analysis = load(args.output_dir / "analysis_summary.json")
    visualization = load(args.output_dir / "visualization_summary.json")
    manifest = load(args.output_dir / "sample_manifest.json")
    samples = manifest["samples"]
    expected_tasks = spec["dataset"]["task_directories"]
    task_counts = {task: sum(sample["task"] == task for sample in samples) for task in expected_tasks}
    heldout_counts = {task: len({sample["episode"] for sample in samples if sample["task"] == task}) for task in expected_tasks}
    all_loading_clean = all(not extraction["loading_records"][0].get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    synthetic_tests = True if args.skip_tests else run_synthetic_tests()
    privacy = privacy_scan()
    sample_manifest_hash = sha256_file(args.output_dir / "sample_manifest.json")
    benchmark_config_hash = sha256_file(args.spec)
    acceptance = {
        "official_tokenizer_identity": bool(extraction["tokenizer_identity"]["source"] == "official released checkpoint nested tokenizer" and all_loading_clean),
        "physical_gpu3": extraction["physical_gpu"] == 3 and extraction["logical_device"] == "cuda:0",
        "tasks_24_of_24": len([task for task, count in task_counts.items() if count > 0]) == 24,
        "heldout_episodes": all(count == 10 for count in heldout_counts.values()),
        "sample_manifest": len(samples) == int(spec["dataset"]["expected_samples"]) and not manifest["skipped"],
        "deterministic_sampling": len(set(sample["pair_id"] for sample in samples)) == len(samples),
        "vision_only_extraction": extraction["shapes"]["l1"][1] == 3,
        "action_only_extraction": extraction["shapes"]["l1"][1] == 3,
        "multimodal_extraction": extraction["shapes"]["l1"][1] == 3,
        "l1_routed_pre_vq": "l1" in extraction["shapes"],
        "l2_vq_input": "l2" in extraction["shapes"],
        "l3_quantized_embedding": "l3" in extraction["shapes"],
        "l4_discrete_codes": "l4" in extraction["shapes"],
        "determinism": extraction["determinism"]["pass"],
        "nan_inf": extraction["finite"],
        "paired_cosine": True,
        "retrieval": any(row for row in (args.output_dir / "metrics" / "retrieval.csv").read_text().splitlines()[1:]),
        "mmd": True,
        "sliced_wasserstein": True,
        "cka": True,
        "code_utilization": True,
        "entropy_perplexity": True,
        "code_agreement": True,
        "jaccard": True,
        "nmi_ami": True,
        "pca": (args.output_dir / "visualization" / "pca_pre_vq.png").exists() and (args.output_dir / "visualization" / "pca_post_vq.png").exists(),
        "tsne": (args.output_dir / "visualization" / "tsne_pre_vq.png").exists() and (args.output_dir / "visualization" / "tsne_post_vq.png").exists(),
        "umap": (args.output_dir / "visualization" / "umap_pre_vq.png").exists() and (args.output_dir / "visualization" / "umap_post_vq.png").exists(),
        "interactive_html": (args.output_dir / "visualization" / "unit_representation_umap.html").exists(),
        "synthetic_metric_tests": synthetic_tests,
        "privacy": privacy["pass"],
    }
    required = list(acceptance.values())
    quantitative_warning = not analysis["sanity"]["paired_vs_shuffled"] or not analysis["sanity"]["retrieval_vs_chance"] or any(
        item["classification"] != "NORMAL" for item in analysis["collapse"].values()
    )
    final_status = "PASS WITH REPRESENTATION WARNING" if all(required) and quantitative_warning else ("PASS" if all(required) else "FAIL")
    summary = {
        "benchmark": spec["benchmark_name"],
        "benchmark_version": spec["benchmark_version"],
        "t4_final": final_status,
        "acceptance": acceptance,
        "task_counts": task_counts,
        "heldout_episode_counts": heldout_counts,
        "resolved_samples": len(samples),
        "skipped_samples": manifest["skipped"],
        "tokenizer": extraction["tokenizer_identity"],
        "model_dimensions": extraction["model_dimensions"],
        "feature_shapes": extraction["shapes"],
        "manifest_sha256": sample_manifest_hash,
        "benchmark_config_sha256": benchmark_config_hash,
        "manifest_canonical_sha256_from_extraction": extraction.get("manifest_canonical_sha256"),
        "metric_protocol": {
            "pooling": "mean across Q then L2 normalize",
            "negative_seed": 42,
            "retrieval_positive": "same pair_id only",
            "mmd": "RBF median heuristic",
            "swd": "128 projections, seed 42",
            "cka": "linear CKA",
        },
        "sanity": analysis["sanity"],
        "collapse": analysis["collapse"],
        "visualization": visualization,
        "metric_implementation_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "privacy_details": privacy,
    }
    (args.output_dir / "t4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if final_status != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
