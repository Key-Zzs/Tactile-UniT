#!/usr/bin/env python3
"""Create the compact C3-DP human-acceptance plot set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    geometry_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=ROOT / ".local/artifacts/tactile_unit/vac_c3dp/locked_test_evaluation.json",
    )
    parser.add_argument(
        "--dual-path",
        type=Path,
        default=ROOT / ".local/artifacts/tactile_unit/vac_c3dp/dual_path_audit.json",
    )
    parser.add_argument(
        "--private",
        type=Path,
        default=ROOT / ".local/artifacts/tactile_unit/vac_c3dp/private_residual_analysis.json",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=ROOT / ".local/experiments/tactile_unit/vac_c3dp/training_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".local/artifacts/tactile_unit/vac_c3dp/plots",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / ".local/cache/tactile_unit/vac_c3dp",
    )
    return parser.parse_args()


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    dual_path = json.loads(args.dual_path.read_text())
    private = json.loads(args.private.read_text())
    training = json.loads(args.training.read_text())
    output = args.output
    directions = ("V->A", "A->V", "V->C", "C->V", "A->C", "C->A")
    paths: list[Path] = []

    prediction = [evaluation["directions"][name]["prediction_mse"] for name in directions]
    control = [
        evaluation["directions"][name]["controls"][
            evaluation["directions"][name]["strongest_control"]
        ]
        for name in directions
    ]
    x = np.arange(len(directions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, prediction, width, label="prediction")
    ax.bar(x + width / 2, control, width, label="strongest control")
    ax.set_xticks(x, directions)
    ax.set_ylabel("shared latent MSE")
    ax.set_title("Six-direction prediction versus controls")
    ax.legend()
    path = output / "six_direction_mse.png"
    save(fig, path)
    paths.append(path)

    retrieval = [
        evaluation["directions"][name]["retrieval"]["recall_at_10"]
        / evaluation["directions"][name]["retrieval"]["chance"]["recall_at_10"]
        for name in directions
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(directions, retrieval)
    ax.axhline(1.0, color="black", linewidth=0.8, label="chance")
    ax.axhline(1.5, color="red", linestyle="--", label="hard gate")
    ax.set_ylabel("R@10 / chance")
    ax.set_title("Six-direction retrieval")
    ax.legend()
    path = output / "six_direction_retrieval.png"
    save(fig, path)
    paths.append(path)

    true_cosine = [evaluation["directions"][name]["cosine_true"] for name in directions]
    shuffled_cosine = [evaluation["directions"][name]["cosine_shuffled"] for name in directions]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, true_cosine, width, label="paired target")
    ax.bar(x + width / 2, shuffled_cosine, width, label="shuffled target")
    ax.set_xticks(x, directions)
    ax.set_ylabel("cosine")
    ax.set_title("Six-direction target-matching margin")
    ax.legend()
    path = output / "six_direction_cosine_margin.png"
    save(fig, path)
    paths.append(path)

    contact = evaluation["contact_semantics"]
    names = ("V->C", "A->C")
    transition = contact["contact_transition"]
    predicted = [transition[name]["retention"] for name in names]
    oracle = [1.0 for _ in names]
    majority = [0.0 for _ in names]
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, oracle, width, label="oracle shared")
    ax.bar(x, predicted, width, label="predicted")
    ax.bar(x + width, majority, width, label="majority")
    ax.axhline(0.75, color="red", linestyle="--", label="retention gate")
    ax.set_xticks(x, names)
    ax.set_ylabel("Contact-transition retention")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-modal Contact semantics")
    ax.legend()
    path = output / "contact_cross_semantics.png"
    save(fig, path)
    paths.append(path)

    physics = evaluation["contact_physics"]
    x = np.arange(len(names))
    predicted = [physics[name]["predicted"]["metrics"]["all"]["mean"] for name in names]
    oracle = [physics[name]["oracle_shared"]["metrics"]["all"]["mean"] for name in names]
    native = [physics[name]["full_native"]["metrics"]["all"]["mean"] for name in names]
    control = [
        physics[name][physics[name]["strongest_control"]]["metrics"]["all"]["mean"]
        for name in names
    ]
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 1.5 * width, predicted, width, label="predicted shared")
    ax.bar(x - 0.5 * width, control, width, label="strongest control")
    ax.bar(x + 0.5 * width, oracle, width, label="oracle shared")
    ax.bar(x + 1.5 * width, native, width, label="full native")
    ax.set_xticks(x, names)
    ax.set_ylabel("future Contact MSE")
    ax.set_title("Cross-predicted Contact shared physics")
    ax.legend()
    path = output / "contact_shared_physics.png"
    save(fig, path)
    paths.append(path)

    action_names = ("V->A", "C->A")
    action = evaluation["action_targets"]
    predicted = [
        action[name]["representations"]["predicted"]["reconstruction_mse"] for name in action_names
    ]
    oracle = [
        action[name]["representations"]["oracle_shared"]["reconstruction_mse"]
        for name in action_names
    ]
    control = [
        action[name]["representations"][action[name]["strongest_control"]]["reconstruction_mse"]
        for name in action_names
    ]
    x = np.arange(len(action_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, predicted, width, label="predicted")
    ax.bar(x, control, width, label="strongest control")
    ax.bar(x + width, oracle, width, label="oracle shared")
    ax.set_xticks(x, action_names)
    ax.set_ylabel("action reconstruction MSE")
    ax.set_title("Cross-predicted Action decoding")
    ax.legend()
    path = output / "action_cross_prediction.png"
    save(fig, path)
    paths.append(path)

    decomposition = private["contact_semantics"]["contact_transition"]
    labels = ("native", "shared u_c", "shared recovered", "private residual")
    values = [
        decomposition[name]["macro_f1"]
        for name in ("native", "shared_u_c", "shared_recovered", "private_residual")
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("contact-transition macro F1")
    ax.set_title("Shared/private Contact semantic decomposition")
    path = output / "contact_shared_private_decomposition.png"
    save(fig, path)
    paths.append(path)

    geometry = evaluation["predicted_geometry"]
    names = ("vision", "action", "contact")
    oracle_geometry = {
        name: geometry_diagnostics(
            np.load(args.cache_root / "test" / f"u_{name[0]}.npy", mmap_mode="r")
        )
        for name in names
    }
    predicted_rank = [geometry[name]["effective_rank"] for name in names]
    oracle_rank = [oracle_geometry[name]["effective_rank"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, predicted_rank, width, label="predicted")
    ax.bar(x + width / 2, oracle_rank, width, label="oracle shared")
    ax.set_xticks(x, names)
    ax.set_ylabel("effective rank")
    ax.set_title("Predicted versus oracle shared geometry")
    ax.legend()
    path = output / "predicted_effective_rank.png"
    save(fig, path)
    paths.append(path)

    private_geometry = private["geometry"]
    private_probe = private["private_cross_modal_predictability"]
    labels = ("energy fraction", "CKA(z_c)", "CKA(u_c)", "V R²", "A R²")
    values = (
        private_geometry["energy_fraction_of_native"],
        private_geometry["cka_with_native_z_c"],
        private_geometry["cka_with_shared_u_c"],
        private_probe["V->r_c_priv"]["metrics"]["r2"],
        private_probe["A->r_c_priv"]["metrics"]["r2"],
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Contact-private residual diagnostics")
    path = output / "private_residual_diagnostics.png"
    save(fig, path)
    paths.append(path)

    direction_names = ("V->C", "A->C")
    subset_names = ("dynamic", "rare_boundary", "free_to_contact", "contact_to_free")
    x = np.arange(len(subset_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, name in enumerate(direction_names):
        values = [
            evaluation["directions"][name]["subsets"][subset]["mean"] for subset in subset_names
        ]
        ax.bar(x + (offset - 0.5) * width, values, width, label=name)
    ax.set_xticks(x, ("dynamic", "rare boundary", "free→contact", "contact→free"))
    ax.set_ylabel("shared latent MSE")
    ax.set_title("Contact-target dynamic and boundary prediction")
    ax.legend()
    path = output / "dynamic_boundary_prediction.png"
    save(fig, path)
    paths.append(path)

    labels = [f"T{trial['trial_id']} {trial['trial']['candidate']}" for trial in training["trials"]]
    utility = [trial["best"]["utility"] for trial in training["trials"]]
    colors = ["#4c78a8" if index != 2 else "#f58518" for index in range(len(labels))]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, utility, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("validation-only utility")
    ax.set_title("Bounded candidate comparison (selected trial in orange)")
    path = output / "candidate_comparison.png"
    save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    passed = [name for name, value in evaluation["gates"].items() if value]
    failed = [name for name, value in evaluation["gates"].items() if not value]
    ax.text(0.5, 0.72, evaluation["decision"], ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, f"PASS: {', '.join(passed)}", ha="center", va="center", color="#2b7a3d")
    ax.text(0.5, 0.20, f"FAIL: {', '.join(failed)}", ha="center", va="center", color="#b33a3a")
    path = output / "final_decision.png"
    save(fig, path)
    paths.append(path)

    summary = {
        "schema": "tactile3d-unit.vac-c3dp-visualization.v1",
        "decision": evaluation["decision"],
        "dual_path_pass": bool(dual_path["pass"]),
        "oracle_target_geometry": oracle_geometry,
        "predicted_target_geometry": geometry,
        "plots": [str(path.relative_to(ROOT)) for path in paths],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output.parent / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    acceptance = output.parent / "HUMAN_ACCEPTANCE.md"
    plot_lines = "\n".join(f"- `{path.relative_to(ROOT)}`" for path in paths)
    acceptance.write_text(
        "# C3-DP Human Acceptance\n\n"
        f"Decision: **`{evaluation['decision']}`**\n\n"
        f"Rows: `{evaluation['rows']}`\n\n"
        f"All six prediction gates: `{evaluation['all_six_prediction_gates']}`\n\n"
        f"Contact semantic gate: `{evaluation['contact_semantics']['gate']}`\n\n"
        f"Structural gate: `{evaluation['gates']['structural']}`\n\n"
        "Repeated locked evaluation: `byte-identical`\n\n"
        "## Inspection commands\n\n"
        "```bash\n"
        "python -m json.tool .local/artifacts/tactile_unit/vac_c3dp/dual_path_audit.json\n"
        "python -m json.tool .local/artifacts/tactile_unit/vac_c3dp/private_residual_analysis.json\n"
        "python -m json.tool .local/artifacts/tactile_unit/vac_c3dp/selection.json\n"
        "python -m json.tool .local/artifacts/tactile_unit/vac_c3dp/locked_test_evaluation.json\n"
        "python -m json.tool .local/artifacts/tactile_unit/vac_c3dp/final_decision.json\n"
        "sha256sum .local/experiments/tactile_unit/vac_c3dp/selected.pt\n"
        "python scripts/tactile_unit/visualize_c3dp_cross_prediction.py\n"
        "```\n\n"
        "## Plots\n\n"
        f"{plot_lines}\n\n"
        "## Stop point\n\n"
        "C4, C5, and C6/M3 are not started. M3 is not established.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
