#!/usr/bin/env python3
"""Create the minimal evidence plots for a complete PI2N FINAL analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
STATISTICS = ARTIFACTS / "paired_statistics.json"
PROCESS = ARTIFACTS / "contact_process_metrics.json"
CLAIM = ARTIFACTS / "claim_freeze.json"
PLOTS = ARTIFACTS / "plots"
MANIFEST = ARTIFACTS / "plots_manifest.json"
MODEL_ORDER = ("B0", "B_VA27", "B1", "B_HVA", "B2", "B_VAC_V")
MODEL_LABELS = ("B0", "B_VA27", "B1", "B_HVA", "B2", "VAC*")
PAIR_ORDER = (
    "VAC_STAR-B_HVA",
    "VAC_STAR-B2",
    "VAC_STAR-B0",
    "B_HVA-B_VA27",
    "B_VA27-B0",
    "B1-B0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_figure(figure: Any, path: Path, *, format_name: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(
        temporary,
        format=format_name,
        dpi=220 if format_name == "png" else None,
        bbox_inches="tight",
        metadata={"Creator": "Tactile-UniT S4.3-PI2N frozen analysis"},
    )
    temporary.replace(path)


def save_both(figure: Any, stem: str) -> list[Path]:
    paths = [PLOTS / f"{stem}.png", PLOTS / f"{stem}.pdf"]
    atomic_figure(figure, paths[0], format_name="png")
    atomic_figure(figure, paths[1], format_name="pdf")
    plt.close(figure)
    return paths


def success_plot(statistics: dict[str, Any]) -> list[Path]:
    rows = statistics["model_statistics"]
    rates = np.asarray([rows[model]["success_percent"] for model in MODEL_ORDER])
    intervals = np.asarray(
        [rows[model]["wilson_95ci_percent"] for model in MODEL_ORDER]
    )
    errors = np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates))
    colors = ["#6b7280", "#60a5fa", "#94a3b8", "#2563eb", "#f59e0b", "#dc2626"]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    positions = np.arange(len(MODEL_ORDER))
    axis.bar(positions, rates, color=colors, width=0.72)
    axis.errorbar(
        positions,
        rates,
        yerr=errors,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
    )
    axis.set_xticks(positions, MODEL_LABELS)
    axis.set_ylabel("Native success (%)")
    axis.set_title("PI2N FINAL: 200 matched resets per policy")
    axis.set_ylim(0, max(100.0, float(intervals[:, 1].max()) + 5))
    axis.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, rates, strict=True):
        axis.text(position, value + 1.5, f"{value:.1f}", ha="center", fontsize=9)
    figure.tight_layout()
    return save_both(figure, "formal_success_rates")


def paired_effect_plot(statistics: dict[str, Any]) -> list[Path]:
    rows = statistics["paired_comparisons"]
    effects = np.asarray(
        [rows[name]["risk_difference_percentage_points"] for name in PAIR_ORDER]
    )
    intervals = np.asarray(
        [rows[name]["paired_bootstrap_95ci_percentage_points"] for name in PAIR_ORDER]
    )
    errors = np.vstack((effects - intervals[:, 0], intervals[:, 1] - effects))
    positions = np.arange(len(PAIR_ORDER))[::-1]
    colors = [
        "#dc2626" if value > 0 else "#2563eb" if value < 0 else "#6b7280"
        for value in effects
    ]
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.errorbar(
        effects,
        positions,
        xerr=errors,
        fmt="none",
        ecolor="#374151",
        elinewidth=2,
        capsize=4,
    )
    axis.scatter(effects, positions, c=colors, s=45, zorder=3)
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(positions, PAIR_ORDER)
    axis.set_xlabel("Paired success-rate difference (percentage points)")
    axis.set_title("PI2N FINAL paired effects with bootstrap 95% CI")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return save_both(figure, "paired_effects")


def contact_plot(process: dict[str, Any]) -> list[Path]:
    metrics = (
        ("max_pinch_count", "Max pinch count"),
        ("max_normal_force_proxy", "Peak normal-force proxy"),
        ("active_contact_samples", "Active contact samples"),
        ("executed_action_mean_l2", "Executed-action mean L2"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    colors = ["#6b7280", "#60a5fa", "#94a3b8", "#2563eb", "#f59e0b", "#dc2626"]
    for axis, (field, label) in zip(axes.flat, metrics, strict=True):
        rows = [process["models"][model][field] for model in MODEL_ORDER]
        available = all(row.get("available") is True for row in rows)
        if not available:
            axis.text(0.5, 0.5, "NA in frozen runtime", ha="center", va="center")
            axis.set_axis_off()
            continue
        values = [row["mean"] for row in rows]
        axis.bar(np.arange(len(values)), values, color=colors, width=0.72)
        axis.set_xticks(np.arange(len(values)), MODEL_LABELS, rotation=25)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("PI2N FINAL contact/process summaries (descriptive only)")
    figure.tight_layout()
    return save_both(figure, "contact_process_summary")


def main() -> None:
    if MANIFEST.exists() or PLOTS.exists():
        raise SystemExit("refusing to overwrite PI2N FINAL plots")
    required = (STATISTICS, PROCESS, CLAIM)
    if any(not path.is_file() for path in required):
        raise SystemExit("PI2N FINAL plot prerequisite missing")
    statistics = json.loads(STATISTICS.read_text())
    process = json.loads(PROCESS.read_text())
    claim = json.loads(CLAIM.read_text())
    if (
        statistics.get("status") != "PASS"
        or process.get("status") != "PASS"
        or claim.get("ENGINEERING_STATUS") != "COMPLETE_VALID"
    ):
        raise SystemExit("PI2N FINAL plot integrity gate failed")
    PLOTS.mkdir(parents=True, exist_ok=False)
    paths = [
        *success_plot(statistics),
        *paired_effect_plot(statistics),
        *contact_plot(process),
    ]
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-plots-manifest.v1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plots": [
            {
                "path": "$REPO_ROOT/" + path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "source_sha256": {
            "paired_statistics": sha256_file(STATISTICS),
            "contact_process_metrics": sha256_file(PROCESS),
            "claim_freeze": sha256_file(CLAIM),
        },
        "decorative_plots_added": False,
        "performance_claims_derived_from_plots": False,
    }
    atomic_json(MANIFEST, payload)
    print(json.dumps({"status": "PASS", "plots": len(paths)}, sort_keys=True))


if __name__ == "__main__":
    main()
