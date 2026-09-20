#!/usr/bin/env python3
"""Generate the minimum frozen PI2M figures directly from local artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
PLOTS = ARTIFACTS / "plots"
STATS = ARTIFACTS / "paired_statistics.json"
CLAIMS = ARTIFACTS / "claim_freeze.json"
STAT_AUDIT = ARTIFACTS / "statistics_independent_audit.json"
TRAIN_LOG = ROOT / ".local/logs/simulation/s4_3_pi2m/bhva/train.log"
MANIFEST = ARTIFACTS / "plot_manifest.json"
COLORS = {"B1": "#6B7280", "B_HVA": "#2563EB", "B2": "#DC2626"}
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig: plt.Figure, name: str) -> Path:
    path = PLOTS / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def design_matrix() -> tuple[Path, list[dict[str, str]]]:
    rows = [
        {"Model": "B0", "Online H": "No", "Aux target": "None", "Horizon": "—", "Role": "Historical baseline"},
        {"Model": "BVA", "Online H": "No", "Aux target": "clean VA", "Horizon": "+16 = 0.32 s", "Role": "Historical context only"},
        {"Model": "B1", "Online H": "Yes", "Aux target": "None", "Horizon": "—", "Role": "Fresh seed7 cohort"},
        {"Model": "B_HVA", "Online H": "Yes", "Aux target": "clean VA", "Horizon": "+27 = 0.54 s", "Role": "Fresh matched-input"},
        {"Model": "B2", "Online H": "Yes", "Aux target": "clean VAC", "Horizon": "+27 = 0.54 s", "Role": "Fresh matched-input"},
    ]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    columns = list(rows[0])
    table = ax.table(
        cellText=[[row[col] for col in columns] for row in rows],
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.10, 0.12, 0.18, 0.17, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif row in (4, 5):
            cell.set_facecolor("#DBEAFE" if row == 4 else "#FEE2E2")
        elif row == 2:
            cell.set_facecolor("#FEF3C7")
    ax.set_title("PI2M design matrix: online input and teacher target are separate axes", weight="bold", pad=18)
    return save(fig, "design_matrix.png"), rows


def physical_alignment() -> tuple[Path, list[dict[str, Any]]]:
    rows = [
        {"name": "Historical BVA", "offset_rows": 16, "seconds": 0.32, "matched_to_B2": False},
        {"name": "B_HVA corrected clean VA", "offset_rows": 27, "seconds": 0.54, "matched_to_B2": True},
        {"name": "B2 frozen clean VAC", "offset_rows": 27, "seconds": 0.54, "matched_to_B2": True},
    ]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows, strict=True):
        color = "#D97706" if not row["matched_to_B2"] else "#2563EB"
        ax.annotate("", xy=(row["seconds"], yi), xytext=(0, yi), arrowprops={"arrowstyle": "->", "lw": 3, "color": color})
        ax.scatter([0, row["seconds"]], [yi, yi], s=45, color=color, zorder=3)
        ax.text(row["seconds"] + 0.015, yi, f"+{row['offset_rows']} rows = {row['seconds']:.2f} s", va="center")
    ax.set_yticks(y, [row["name"] for row in rows])
    ax.set_xlim(-0.03, 0.72)
    ax.set_xlabel("Physical control time from anchor t (50 Hz; 0.02 s/tick)")
    ax.set_title("Teacher target horizon: PI2M matches B_HVA to B2 at +27 control ticks", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save(fig, "physical_time_alignment.png"), rows


def training_losses() -> tuple[Path, list[dict[str, float]]]:
    rows = []
    for match in STEP_PATTERN.finditer(TRAIN_LOG.read_text(errors="replace")):
        metrics = {item.group("key"): float(item.group("value")) for item in METRIC_PATTERN.finditer(match.group("metrics"))}
        rows.append({"step": int(match.group("step")), **metrics})
    if len(rows) != 300:
        raise RuntimeError(f"expected 300 B_HVA metric rows, got {len(rows)}")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    steps = np.asarray([row["step"] for row in rows])
    for key, label, color in (
        ("loss", "Total train loss", "#111827"),
        ("official_loss", "Official flow loss", "#7C3AED"),
        ("physical_loss", "VA auxiliary loss", "#2563EB"),
    ):
        ax.plot(steps, [row[key] for row in rows], label=label, color=color, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Logged loss (log scale)")
    ax.set_title("B_HVA seed42 training losses (30k optimizer steps)", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, "training_losses.png"), rows


def success_intervals(stats: dict[str, Any]) -> Path:
    models = ["B1", "B_HVA", "B2"]
    values = np.asarray([stats["model_statistics"][model]["success_percent"] for model in models])
    intervals = np.asarray([stats["model_statistics"][model]["wilson_95ci_percent"] for model in models])
    errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(models))
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="#111827", capsize=6, lw=2)
    ax.scatter(x, values, s=130, color=[COLORS[model] for model in models], zorder=3)
    for xi, model, value, row in zip(x, models, values, intervals, strict=True):
        ax.text(xi, row[1] + 1.8, f"{value:.1f}%\n[{row[0]:.1f}, {row[1]:.1f}]", ha="center", fontsize=9)
    ax.set_xticks(x, models)
    ax.set_ylabel("Success rate (%) with Wilson 95% CI")
    ax.set_ylim(0, max(intervals[:, 1]) + 11)
    ax.set_title("Fresh matched-reset evaluation (n=200 per model, evaluator seed7)", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save(fig, "success_wilson_ci.png")


def paired_differences(stats: dict[str, Any]) -> Path:
    names = ["B2-B_HVA", "B_HVA-B1", "B2-B1"]
    rows = [stats["paired_comparisons"][name] for name in names]
    values = np.asarray([row["risk_difference_percentage_points"] for row in rows])
    intervals = np.asarray([row["paired_bootstrap_95ci_percentage_points"] for row in rows])
    errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axvline(0, color="#111827", lw=1.2)
    for yi, name, value, interval, error, row in zip(y, names, values, intervals, errors.T, rows, strict=True):
        color = "#059669" if row["classification"] == "POSITIVE_CONFIRMED" else "#DC2626" if row["classification"] == "NEGATIVE_CONFIRMED" else "#6B7280"
        ax.errorbar(value, yi, xerr=[[error[0]], [error[1]]], fmt="o", color=color, capsize=5, markersize=8)
        ax.text(interval[1] + 1, yi, row["classification"].replace("_", " "), va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.set_xlabel("Paired risk difference (percentage points), 100k bootstrap 95% CI")
    ax.set_title("Pre-registered paired comparisons (Holm family of three)", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save(fig, "paired_differences.png")


def paired_counts(stats: dict[str, Any]) -> Path:
    names = ["B2-B_HVA", "B_HVA-B1", "B2-B1"]
    labels = ["Both success", "Left only", "Right only", "Both failure"]
    arrays = []
    for name in names:
        table = stats["paired_comparisons"][name]["table"]
        arrays.append([table["both_success"], table["left_only_success"], table["right_only_success"], table["both_failure"]])
    values = np.asarray(arrays)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(values, cmap="Blues", aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", color="white" if values[i, j] > values.max() * 0.55 else "#111827", weight="bold")
    ax.set_xticks(np.arange(4), labels)
    ax.set_yticks(np.arange(3), names)
    ax.set_title("Paired outcome 2×2 counts (each row sums to 200)", weight="bold")
    fig.colorbar(image, ax=ax, label="Matched reset count")
    fig.tight_layout()
    return save(fig, "paired_outcome_counts.png")


def claim_table(claim: dict[str, Any]) -> tuple[Path, list[list[str]]]:
    rows = [
        ["Engineering", claim["ENGINEERING_STATUS"], "600 complete; recovery disclosed"],
        ["Primary target effect", claim["PRIMARY_TARGET_EFFECT"], "B2 vs B_HVA; fixed seed only"],
        ["VA auxiliary vs no aux", claim["VA_AUX_VS_NO_AUX"], "B_HVA vs B1"],
        ["Contact aux replication", claim["CONTACT_AUX_REPLICATION"], "B2 vs B1"],
        ["Alignment necessity", "NOT PROVEN", "No native/aligned target control"],
        ["Generalization", "NOT ESTABLISHED", "One task; one training seed"],
    ]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Axis", "Frozen status", "Scope / limitation"], cellLoc="left", loc="center", colWidths=[0.23, 0.30, 0.43])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif row == 2:
            cell.set_facecolor("#FEE2E2")
    ax.set_title("PI2M claim freeze", weight="bold", pad=18)
    return save(fig, "claim_status_table.png"), rows


def main() -> None:
    if MANIFEST.exists() or PLOTS.exists():
        raise SystemExit("refusing to overwrite PI2M plots")
    if read_json(STAT_AUDIT).get("status") != "PASS":
        raise SystemExit("statistics independent audit is not PASS")
    stats = read_json(STATS)
    claim = read_json(CLAIMS)
    PLOTS.mkdir(parents=True, exist_ok=False)
    design_path, design_rows = design_matrix()
    time_path, time_rows = physical_alignment()
    training_path, training_rows = training_losses()
    paths = [
        design_path,
        time_path,
        training_path,
        success_intervals(stats),
        paired_differences(stats),
        paired_counts(stats),
    ]
    claim_path, claim_rows = claim_table(claim)
    paths.append(claim_path)
    data = {
        "design_matrix": design_rows,
        "physical_time_alignment": time_rows,
        "training_metrics": training_rows,
        "claim_status": claim_rows,
    }
    data_path = PLOTS / "figure_data.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    paths.append(data_path)
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-plot-manifest.v1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plots": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in paths
        },
        "source_sha256": {
            "paired_statistics": sha256_file(STATS),
            "claim_freeze": sha256_file(CLAIMS),
            "statistics_independent_audit": sha256_file(STAT_AUDIT),
            "training_log": sha256_file(TRAIN_LOG),
        },
        "all_values_from_local_artifacts": True,
        "historical_BVA_labeled_0p32s_and_context_only": True,
        "PI2B_started": False,
    }
    atomic_json(MANIFEST, payload)
    print(json.dumps({"status": "PASS", "plots": sorted(payload["plots"])}, sort_keys=True))


if __name__ == "__main__":
    main()
