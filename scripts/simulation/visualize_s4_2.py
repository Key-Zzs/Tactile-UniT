#!/usr/bin/env python3
"""Render only the S4.2 evidence available before the Contact-State stop."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2"
PLOTS = ARTIFACTS / "plots"
PAIRS = ROOT / ".local/cache/simulation/s4_2/pairs/validation.npz"
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2/contact_teacher/selected.pt"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str, paths: list[Path]) -> None:
    path = PLOTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)


def main() -> None:
    from gr00t.simulation.s4_2_normalization import TactileNormalization
    from gr00t.simulation.sim_contact_models import load_teacher_checkpoint

    quality = read_json("dataset_quality.json")
    evaluation = read_json("teacher_evaluation.json")
    normalization = TactileNormalization.from_json(read_json("normalization.json"))
    paths: list[Path] = []

    regimes = list(next(iter(quality["regimes_by_task"].values())).keys())
    tasks = list(quality["regimes_by_task"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(tasks))
    for regime in regimes:
        values = np.asarray([quality["regimes_by_task"][task][regime] for task in tasks])
        ax.bar(tasks, values, bottom=bottom, label=regime.replace("_", "→"))
        bottom += values
    ax.set_ylabel("valid anchors")
    ax.set_title("Formal dataset task and contact-regime distribution")
    ax.legend(ncols=2, fontsize=8)
    save(fig, "01_dataset_task_regime_distribution.png", paths)

    region_names = ["palm", "index", "middle", "ring", "thumb"]
    activity = np.asarray(quality["region_activity_steps"])
    mean_force = np.asarray([row["mean"] for row in quality["region_normal_force"]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(region_names, activity)
    axes[0].set_ylabel("active steps")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(region_names, mean_force)
    axes[1].set_ylabel("mean normal force (N)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Tactile region activity and force")
    save(fig, "02_tactile_region_force_distribution.png", paths)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(np.arange(-25, 1) * 0.02, np.zeros(26), "o-", label="history [26]")
    ax.plot(np.arange(1, 14) * 0.02, np.ones(13), "o-", label="teacher target [13]")
    ax.axvline(27 * 0.02, color="tab:red", linestyle="--", label="transition +0.54 s")
    ax.set_yticks([0, 1], ["input", "future"])
    ax.set_xlabel("physical time relative to anchor t (s)")
    ax.set_title("Frozen physical-time alignment")
    ax.legend(ncols=3, fontsize=8)
    save(fig, "03_physical_time_window_alignment.png", paths)

    with np.load(PAIRS, allow_pickle=False) as source:
        raw_history = source["current_history"][:1024]
        raw_future = source["teacher_future"][:1024]
    history = normalization.transform(raw_history).astype(np.float32)
    future = normalization.transform(raw_future).astype(np.float32)
    model, _ = load_teacher_checkpoint(CHECKPOINT)
    model.eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(history))
    prediction = output["future"].numpy()
    latent = output["latent"].numpy()
    force_indices = np.asarray([1, 7, 13, 19, 25])
    sample_error = np.square(prediction - future).mean(axis=(1, 2))
    sample = int(np.argsort(sample_error)[len(sample_error) // 2])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, 14) * 0.02, future[sample][:, force_indices].sum(1), "o-", label="target")
    ax.plot(np.arange(1, 14) * 0.02, prediction[sample][:, force_indices].sum(1), "o-", label="prediction")
    ax.set_xlabel("future time (s)")
    ax.set_ylabel("sum normalized normal-force channels")
    ax.set_title("Median-error validation teacher prediction")
    ax.legend()
    save(fig, "04_teacher_future_prediction.png", paths)

    controls = evaluation["temporal_controls"]
    names = ["correct"] + list(controls)
    values = [next(iter(controls.values()))["full_dynamic_mse"]] + [
        controls[name]["dynamic_mse"] for name in controls
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([name.replace("_", "\n") for name in names], values)
    ax.set_ylabel("validation dynamic MSE")
    ax.set_title("Teacher temporal controls")
    save(fig, "05_teacher_temporal_ablation.png", paths)

    robustness = evaluation["robustness"]
    labels = ["clean"]
    values = [robustness["clean"]]
    for corruption, row in robustness.items():
        if corruption == "clean":
            continue
        for strength in ("mild", "strong"):
            labels.append(f"{corruption}\n{strength}")
            values.append(row[strength])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("validation future MSE")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.set_title("Teacher robustness (diagnostic, validation only)")
    save(fig, "06_teacher_robustness.png", paths)

    singular = np.linalg.svd(latent - latent.mean(0), compute_uv=False)
    energy = singular**2 / np.square(singular).sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, 65), energy[:64], "o-", markersize=3)
    ax.axvline(16, color="tab:red", linestyle="--", label="frozen effective-rank gate")
    ax.set_yscale("log")
    ax.set_xlabel("singular component")
    ax.set_ylabel("variance fraction (log)")
    ax.set_title(f"Contact-state spectrum; effective rank={evaluation['collapse']['effective_rank']:.3f}")
    ax.legend()
    save(fig, "09_contact_state_rank_spectrum.png", paths)

    gates = evaluation["gates"]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3a9d5d" if passed else "#c43d3d" for passed in gates.values()]
    ax.barh([name.replace("_", " ") for name in gates], [1] * len(gates), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("S4.2 gate matrix at hard stop (green PASS, red FAIL)")
    save(fig, "22_final_gate_matrix.png", paths)

    unavailable = [7, 8] + list(range(10, 22))
    summary = {
        "schema": "tactile3d-unit.s4-2-visualization.v1",
        "status": "PARTIAL_EVIDENCE_CONTACT_STATE_STOP",
        "plots": [str(path.relative_to(ROOT)) for path in paths],
        "unavailable_required_plot_numbers": unavailable,
        "reason": "S4.2-3 and later were not run after S4_2_CONTACT_STATE_FAIL",
        "test_loaded": False,
    }
    (ARTIFACTS / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
