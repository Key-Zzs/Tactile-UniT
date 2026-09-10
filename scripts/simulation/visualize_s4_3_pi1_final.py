#!/usr/bin/env python3
"""Render the measured S4.3-PI1 training and paired-evaluation figures 8--15."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
PLOTS = ARTIFACTS / "plots"
LOGS = ROOT / ".local/logs/simulation/s4_3_pi1"
MODELS = ("R0", "B0", "B1", "B2")
STEP_PATTERN = re.compile(r"Step (?P<step>\d+): (?P<metrics>[^\r\n]+)")
METRIC_PATTERN = re.compile(r"(?P<key>[a-z0-9_]+)=(?P<value>[-+0-9.eE]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finish(fig: plt.Figure, filename: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_training_log(path: Path) -> list[dict[str, float]]:
    rows = []
    for match in STEP_PATTERN.finditer(path.read_text(errors="replace")):
        row: dict[str, float] = {"step": float(match.group("step"))}
        row.update(
            {
                item.group("key"): float(item.group("value"))
                for item in METRIC_PATTERN.finditer(match.group("metrics"))
            }
        )
        rows.append(row)
    if not rows or not all(np.isfinite(list(row.values())).all() for row in rows):
        raise RuntimeError(f"training metrics missing or non-finite: {path}")
    return rows


def plot_pi1b_loss(rows: list[dict[str, float]]) -> Path:
    steps = np.asarray([row["step"] for row in rows])
    loss = np.asarray([row["official_loss"] for row in rows])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(steps, loss, color="#2563eb", linewidth=1.35, label="official π0.5 loss")
    ax.scatter([steps[-1]], [loss[-1]], color="#1d4ed8", s=28, zorder=3)
    ax.annotate(f"step {int(steps[-1]):,}: {loss[-1]:.4f}", (steps[-1], loss[-1]), xytext=(-125, 12), textcoords="offset points")
    ax.set(title="PI1B measured training loss", xlabel="Optimizer step", ylabel="Official π0.5 loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return finish(fig, "08_pi1b_training_loss.png")


def plot_pi1c_losses(rows: list[dict[str, float]]) -> Path:
    steps = np.asarray([row["step"] for row in rows])
    official = np.asarray([row["official_loss"] for row in rows])
    physical = np.asarray([row["physical_loss"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.7), sharex=True)
    axes[0].plot(steps, official, color="#2563eb", linewidth=1.3)
    axes[0].set_ylabel("Official π0.5 loss")
    axes[0].set_title("PI1C measured official and physical losses")
    axes[1].plot(steps, physical, color="#7c3aed", linewidth=1.3)
    axes[1].set(xlabel="Optimizer step", ylabel="Physical auxiliary MSE")
    for ax, values in zip(axes, (official, physical), strict=True):
        ax.grid(alpha=0.25)
        ax.scatter([steps[-1]], [values[-1]], s=26, color=ax.lines[0].get_color(), zorder=3)
        ax.annotate(f"{values[-1]:.4f}", (steps[-1], values[-1]), xytext=(-48, 10), textcoords="offset points")
    return finish(fig, "09_pi1c_official_physical_losses.png")


def plot_paired_success(outcomes: dict[str, np.ndarray]) -> Path:
    matrix = np.stack([outcomes[model].astype(int) for model in MODELS])
    fig, ax = plt.subplots(figsize=(12, 3.4))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#fee2e2", "#16a34a"]), vmin=0, vmax=1)
    del image
    ax.set_xticks(np.arange(0, 50, 5), labels=[str(value + 1) for value in range(0, 50, 5)])
    ax.set_yticks(np.arange(4), labels=[f"{model}  ({int(outcomes[model].sum())}/50)" for model in MODELS])
    ax.set(xlabel="Shared reset index", ylabel="Frozen model", title="R0/B0/B1/B2 paired success on identical ordered resets")
    ax.set_xticks(np.arange(-0.5, 50, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False)
    return finish(fig, "10_r0_b0_b1_b2_paired_success.png")


def plot_wilson(stats: dict[str, Any]) -> Path:
    rates = np.asarray([stats["model_results"][model]["success_rate"] for model in MODELS])
    intervals = np.asarray([stats["model_results"][model]["wilson_95ci"] for model in MODELS])
    errors = np.stack((rates - intervals[:, 0], intervals[:, 1] - rates))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(4)
    ax.errorbar(positions, rates, yerr=errors, fmt="o", color="#1d4ed8", ecolor="#64748b", capsize=6, markersize=7)
    for x, rate in zip(positions, rates, strict=True):
        ax.text(x, min(1.02, rate + 0.035), f"{100 * rate:.0f}%", ha="center")
    ax.set_xticks(positions, labels=MODELS)
    ax.set_ylim(0, 1.05)
    ax.set(xlabel="Frozen model", ylabel="Success probability", title="Success rates with Wilson 95% intervals")
    ax.grid(axis="y", alpha=0.25)
    return finish(fig, "11_wilson_intervals.png")


def plot_outcome_matrix(contrast: dict[str, Any], filename: str, title: str) -> Path:
    table = contrast["paired_outcome_table"]
    values = np.asarray(
        [[table["both_fail"], table["second_only"]], [table["first_only"], table["both_success"]]],
        dtype=int,
    )
    first, second = contrast["first"], contrast["second"]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.imshow(values, cmap="Blues", vmin=0, vmax=max(1, int(values.max())))
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(values[row, column]), ha="center", va="center", fontsize=18, color="#111827")
    ax.set_xticks((0, 1), labels=(f"{second} fail", f"{second} success"))
    ax.set_yticks((0, 1), labels=(f"{first} fail", f"{first} success"))
    ax.set(xlabel=second, ylabel=first, title=title)
    subtitle = f"Δ={100 * contrast['success_difference']:+.0f} pp; exact McNemar p={contrast['exact_mcnemar_two_sided_p']:.4g}"
    fig.text(0.5, 0.01, subtitle, ha="center")
    return finish(fig, filename)


def plot_gate_matrix() -> Path:
    rows = [
        ("PI1A alignment", "official_dataset_alignment.json"),
        ("Official-field parity", "official_field_parity.json"),
        ("Augmented dataset", "augmented_dataset_quality.json"),
        ("Mode contract", "mode_contract.json"),
        ("PI1B validation", "pi1b_mode_validation.json"),
        ("PI1B checkpoint", "pi1b_checkpoint_manifest.json"),
        ("PI1C validation", "pi1c_mode_validation.json"),
        ("PI1C checkpoint", "pi1c_checkpoint_manifest.json"),
        ("Pre-PI1D freeze", "pre_pi1d_freeze.json"),
        ("R0 evaluation", "pi1d_r0_eval.json"),
        ("B0 evaluation", "pi1d_b0_eval.json"),
        ("B1 evaluation", "pi1d_b1_eval.json"),
        ("B2 evaluation", "pi1d_b2_eval.json"),
        ("Paired statistics", "pi1d_paired_statistics.json"),
        ("S4.2 immutability", "s4_2_immutability.json"),
        ("Environment integrity", "environment_integrity.json"),
        ("Final decision", "final_decision.json"),
    ]
    statuses = [json.loads((ARTIFACTS / filename).read_text())["status"] for _, filename in rows]
    valid = {"PASS", "FROZEN_BEFORE_TRAINING"}
    values = np.asarray([[1 if status in valid else 0] for status in statuses])
    fig, ax = plt.subplots(figsize=(7.8, 7.3))
    ax.imshow(values, aspect="auto", cmap=ListedColormap(["#dc2626", "#16a34a"]), vmin=0, vmax=1)
    ax.set_xticks([0], labels=["Gate"])
    ax.set_yticks(np.arange(len(rows)), labels=[name for name, _ in rows])
    for index, status in enumerate(statuses):
        ax.text(0, index, status, ha="center", va="center", color="#ffffff", fontweight="bold")
    ax.set_title("S4.3-PI1 final gate matrix from frozen artifacts")
    return finish(fig, "15_final_gate_matrix.png")


def main() -> None:
    stats = json.loads((ARTIFACTS / "pi1d_paired_statistics.json").read_text())
    final = json.loads((ARTIFACTS / "final_decision.json").read_text())
    if stats["status"] != "PASS" or final["status"] != "PASS":
        raise SystemExit("PI1D statistics/final decision are not PASS")
    outcomes = {
        model: np.asarray(
            [row["success"] for row in json.loads((ARTIFACTS / f"pi1d_{model.lower()}_eval.json").read_text())["episode_results"]],
            dtype=bool,
        )
        for model in MODELS
    }
    created = [
        plot_pi1b_loss(parse_training_log(LOGS / "pi1b/train.log")),
        plot_pi1c_losses(parse_training_log(LOGS / "pi1c/train.log")),
        plot_paired_success(outcomes),
        plot_wilson(stats),
        plot_outcome_matrix(stats["primary_contrasts"]["B1-B0"], "12_b1_b0_paired_outcome_matrix.png", "B1−B0 paired outcome matrix"),
        plot_outcome_matrix(stats["primary_contrasts"]["B2-B1"], "13_b2_b1_paired_outcome_matrix.png", "B2−B1 paired outcome matrix"),
        plot_outcome_matrix(stats["primary_contrasts"]["B2-B0"], "14_b2_b0_paired_outcome_matrix.png", "B2−B0 paired outcome matrix"),
        plot_gate_matrix(),
    ]
    all_plots = sorted(PLOTS.glob("*.png"))
    expected_numbers = {f"{value:02d}" for value in range(1, 16)}
    actual_numbers = {path.name.split("_", 1)[0] for path in all_plots}
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1-visualizations.v1",
        "status": "PASS" if expected_numbers <= actual_numbers else "FAIL",
        "measured_not_fabricated": True,
        "required_plot_numbers": sorted(expected_numbers),
        "plot_count": len(all_plots),
        "files": [
            {
                "path": "$REPO_ROOT/" + path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in all_plots
        ],
        "generated_now": [path.name for path in created],
    }
    target = ARTIFACTS / "visualization_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps({"status": payload["status"], "plot_count": len(all_plots), "decision": final["decision"]}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI1_VISUALIZATION_FAIL")


if __name__ == "__main__":
    main()
