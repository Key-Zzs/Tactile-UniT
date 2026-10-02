#!/usr/bin/env python3
"""Render the preregistered Track-A result figures from audited artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
PER_SEED = ARTIFACTS / "per_seed_statistics.json"
CROSSED = ARTIFACTS / "crossed_seed_reset_analysis.json"
INDEPENDENT = ARTIFACTS / "statistics_independent_audit.json"
RECIPES = ROOT / "configs/simulation/pi2b_policy/model_recipes.json"
PLOTS = ARTIFACTS / "plots"
MANIFEST = ARTIFACTS / "plots_manifest.json"
MODELS = ("B0", "B_VA27", "B1", "B_HVA", "B2")
SEEDS = (42, 43, 44)
COLORS = {"B0": "#4c78a8", "B_VA27": "#72b7b2", "B1": "#f58518", "B_HVA": "#e45756", "B2": "#765d69"}


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(figure: plt.Figure, name: str) -> Path:
    path = PLOTS / name
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    if MANIFEST.exists():
        raise SystemExit("refusing to overwrite plots manifest")
    independent = read_json(INDEPENDENT)
    if independent.get("status") != "PASS":
        raise SystemExit("independent statistics audit is not PASS")
    per_seed = read_json(PER_SEED)
    crossed = read_json(CROSSED)
    recipes = read_json(RECIPES)["models"]
    PLOTS.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[Path, str]] = []

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    positions = np.arange(len(MODELS), dtype=float)
    offsets = (-0.22, 0.0, 0.22)
    for offset, seed in zip(offsets, SEEDS, strict=True):
        values = []
        lower = []
        upper = []
        for model in MODELS:
            row = per_seed["checkpoint_statistics"][model][str(seed)]
            value = row["success_percent"]
            values.append(value)
            lower.append(value - row["wilson_95ci_percent"][0])
            upper.append(row["wilson_95ci_percent"][1] - value)
        axis.errorbar(positions + offset, values, yerr=[lower, upper], fmt="o", capsize=3, label=f"seed {seed}")
    axis.set_xticks(positions, MODELS)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("PI2B policy success by training seed (200 shared resets each)")
    axis.set_ylim(-2, 60)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    outputs.append((save(figure, "success_rates_by_seed.png"), "success rate and Wilson 95% interval for all 15 checkpoints"))

    contrast_order = [
        "B_HVA-B_VA27",
        "B2-B_HVA",
        "B2-B1",
        "B_HVA-B0",
        "B2-B0",
        "B_VA27-B0",
        "B1-B0",
    ]
    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    y = np.arange(len(contrast_order))
    seed_offsets = (-0.18, 0.0, 0.18)
    for offset, seed in zip(seed_offsets, SEEDS, strict=True):
        values = [crossed["contrasts"][name]["per_seed_risk_difference_percentage_points"][str(seed)] for name in contrast_order]
        axis.scatter(values, y + offset, s=42, label=f"seed {seed}")
    for index, name in enumerate(contrast_order):
        row = crossed["contrasts"][name]
        mean = row["training_seed_summary_percentage_points"]["mean"]
        low, high = row["conditional_shared_reset_bootstrap_percentage_points"]["interval"]
        axis.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="D", color="black", capsize=3, markersize=5)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvline(-10, color="gray", linewidth=0.7, linestyle="--")
    axis.axvline(10, color="gray", linewidth=0.7, linestyle="--")
    axis.set_yticks(y, contrast_order)
    axis.invert_yaxis()
    axis.set_xlabel("Paired risk difference (percentage points)")
    axis.set_title("Per-seed effects and conditional shared-reset intervals")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(ncol=3, loc="lower right")
    outputs.append((save(figure, "paired_effects_by_seed.png"), "all declared paired effects with seed points and conditional intervals"))

    primary = ("B_HVA-B_VA27", "B2-B_HVA")
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True)
    for axis, name in zip(axes, primary, strict=True):
        row = crossed["contrasts"][name]
        points = [row["per_seed_risk_difference_percentage_points"][str(seed)] for seed in SEEDS]
        axis.scatter(points, [2.0, 2.0, 2.0], c=["#4c78a8", "#f58518", "#e45756"], s=55)
        conditional = row["conditional_shared_reset_bootstrap_percentage_points"]
        two_way = row["two_way_seed_by_reset_sensitivity_percentage_points"]
        for ypos, estimate, interval, label in (
            (1.0, conditional["estimate"], conditional["interval"], "conditional reset"),
            (0.0, two_way["estimate"], two_way["interval"], "two-way sensitivity"),
        ):
            axis.errorbar(estimate, ypos, xerr=[[estimate - interval[0]], [interval[1] - estimate]], fmt="D", capsize=4, color="black")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks((0, 1, 2), ("two-way sensitivity", "conditional reset", "seed points"))
        axis.set_title(name)
        axis.set_xlabel("Risk difference (pp)")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Primary family: crossed seed/reset uncertainty")
    outputs.append((save(figure, "primary_crossed_intervals.png"), "primary-family conditional and two-way intervals"))

    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), constrained_layout=True)
    matrix_models = (("B0", "B_VA27"), ("B1", "B_HVA"))
    matrices = []
    for seed in SEEDS:
        matrices.append(np.asarray([[per_seed["checkpoint_statistics"][model][str(seed)]["success_percent"] for model in row] for row in matrix_models]))
    vmin = min(float(matrix.min()) for matrix in matrices)
    vmax = max(float(matrix.max()) for matrix in matrices)
    for axis, seed, matrix in zip(axes, SEEDS, matrices, strict=True):
        image = axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_xticks((0, 1), ("no VA", "VA27"))
        axis.set_yticks((0, 1), ("no H", "H"))
        axis.set_title(f"seed {seed}")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:.1f}%", ha="center", va="center", color="white" if matrix[row, column] < (vmin + vmax) / 2 else "black")
    figure.colorbar(image, ax=axes, label="Success rate (%)", shrink=0.84)
    figure.suptitle("H × VA27 recipe matrix on identical resets")
    outputs.append((save(figure, "h_by_va_matrix.png"), "H by VA27 success matrix for each training seed"))

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    extra_parameters = []
    mean_steps = []
    failures = []
    for model in MODELS:
        recipe = recipes[model]
        extra_parameters.append(int(recipe.get("contact_adapter_parameters", 0)) + int(recipe.get("physical_auxiliary_parameters", 0)))
        seed_rows = [per_seed["checkpoint_statistics"][model][str(seed)] for seed in SEEDS]
        mean_steps.append(statistics_mean([row["steps"]["mean"] for row in seed_rows]))
        failures.append(sum(row["failures"] for row in seed_rows))
    axes[0].bar(MODELS, np.asarray(extra_parameters) / 1e3, color=[COLORS[model] for model in MODELS])
    axes[0].set_ylabel("Added parameters (thousands)")
    axes[0].set_title("Recipe additions")
    axes[1].bar(MODELS, mean_steps, color=[COLORS[model] for model in MODELS])
    axes[1].set_ylabel("Mean control steps / episode")
    axes[1].set_title("Observed episode length")
    axes[2].bar(MODELS, failures, color=[COLORS[model] for model in MODELS])
    axes[2].set_ylabel("Failures / 600")
    axes[2].set_title("Failure count (all max_steps)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Capacity and process diagnostics (not causal efficiency estimates)")
    outputs.append((save(figure, "capacity_process_failures.png"), "added recipe parameters, observed control steps, and native failures"))

    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2b-policy-plots.v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plots": [
            {"path": str(path), "sha256": sha256_file(path), "description": description}
            for path, description in outputs
        ],
        "inputs_sha256": {
            "per_seed_statistics.json": sha256_file(PER_SEED),
            "crossed_seed_reset_analysis.json": sha256_file(CROSSED),
            "statistics_independent_audit.json": sha256_file(INDEPENDENT),
            "model_recipes.json": sha256_file(RECIPES),
        },
        "selection": "all preregistered models, seeds, and contrasts; no performance-based example selection",
    }
    temporary = MANIFEST.with_suffix(MANIFEST.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST)
    print(json.dumps({"status": "PASS", "plots": len(outputs), "manifest": str(MANIFEST)}, indent=2))


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
