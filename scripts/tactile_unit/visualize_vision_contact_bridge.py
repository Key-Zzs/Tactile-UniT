#!/usr/bin/env python3
"""Render C0 alignment plots and the concise human-acceptance artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / ".local/artifacts/tactile_unit/c0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_ROOT / "evaluation.json")
    parser.add_argument("--contract-audit", type=Path, default=DEFAULT_ROOT / "contract_audit.json")
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=ROOT / ".local/experiments/tactile_unit/c0/training_summary.json",
    )
    parser.add_argument(
        "--test-cache",
        type=Path,
        default=ROOT / ".local/cache/tactile_unit/c0/paired_test.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    audit = json.loads(args.contract_audit.read_text())
    training = json.loads(args.training_summary.read_text())
    test_cache = np.load(args.test_cache, allow_pickle=False)
    if evaluation.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise RuntimeError(
            "visualization requires completed C0 evaluation and passing contract audit"
        )
    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    names = ["B0", "B1", "B2"]
    paired = [evaluation["candidates"][name]["paired"]["paired_cosine"] for name in names]
    shuffled = [
        evaluation["candidates"][name]["paired"]["different_episode_cosine"] for name in names
    ]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.18, paired, width=0.36, label="paired")
    ax.bar(x + 0.18, shuffled, width=0.36, label="different episode")
    ax.set_xticks(x, ["B0", "B1", "B2*"])
    ax.set_ylabel("flattened cosine")
    ax.set_title("C0 paired Vision–Contact alignment")
    ax.legend()
    ax.text(
        0.99,
        0.02,
        "* B2 is pair-conditioned fusion, not independent retrieval",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(plots / "paired_vs_shuffled.png", dpi=180)
    plt.close(fig)

    retrieval_names = ["B0", "B1"]
    recalls = {
        name: [
            evaluation["candidates"][name]["retrieval"]["all"]["v_to_c"][metric]
            for metric in ("recall_at_1", "recall_at_5", "recall_at_10")
        ]
        for name in retrieval_names
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, name in enumerate(retrieval_names):
        ax.bar(np.arange(3) + (offset - 0.5) * 0.32, recalls[name], width=0.32, label=name)
    chance = evaluation["candidates"]["B0"]["retrieval"]["all"]["v_to_c"]
    ax.plot(
        np.arange(3),
        [chance["chance_recall_at_1"], chance["chance_recall_at_5"], chance["chance_recall_at_10"]],
        color="black",
        linestyle="--",
        marker="o",
        label="chance",
    )
    ax.set_xticks(np.arange(3), ["R@1", "R@5", "R@10"])
    ax.set_ylabel("V→C recall")
    ax.set_title("Independent-tower retrieval · B2 pair-conditioned fusion excluded")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "retrieval.png", dpi=180)
    plt.close(fig)

    retention = evaluation["semantic_retention"]
    values = [
        retention["contact_transition"]["advantage_retention"],
        retention["force_trend_class"]["advantage_retention"],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(["Contact transition", "Force trend"], values)
    ax.axhline(0.9, color="black", linestyle="--", label="C0 engineering gate")
    ax.set_ylim(min(0, min(values) - 0.1), max(1.05, max(values) + 0.1))
    ax.set_ylabel("advantage retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "semantic_retention.png", dpi=180)
    plt.close(fig)

    selected = evaluation["selected_bridge"]
    selected_row = evaluation["candidates"][selected]
    paired_row = selected_row["paired"]
    retrieval = selected_row["retrieval"]
    prediction = selected_row["linear_prediction"]["vision_to_contact"]
    boundary = retention["contact_transition"]["rare_boundary_recall"]
    gate = evaluation["causal_gate"]
    example_indices = []
    for class_id in (1, 3):
        matches = np.flatnonzero(test_cache["contact_transition"] == class_id)
        if len(matches):
            example_indices.append(int(matches[0]))
    dynamic_matches = np.flatnonzero(test_cache["dynamic"])
    if len(dynamic_matches):
        example_indices.append(int(dynamic_matches[0]))
    example_indices = list(dict.fromkeys(example_indices))[:3]
    examples = [
        f"- `{test_cache['pair_id'][index]}` — transition class {int(test_cache['contact_transition'][index])}, dynamic={bool(test_cache['dynamic'][index])}"
        for index in example_indices
    ]
    fig, axes = plt.subplots(len(example_indices), 2, figsize=(10, 2.6 * len(example_indices)))
    if len(example_indices) == 1:
        axes = np.asarray([axes])
    for row_index, source_index in enumerate(example_indices):
        pair_label = str(test_cache["pair_id"][source_index])
        for column, (field, title) in enumerate((("z_v", "Vision z_v"), ("z_c", "Contact z_c"))):
            image = axes[row_index, column].imshow(
                test_cache[field][source_index], aspect="auto", cmap="coolwarm"
            )
            axes[row_index, column].set_title(f"{title} · {pair_label}")
            axes[row_index, column].set_xlabel("channel")
            axes[row_index, column].set_ylabel("query")
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.025, pad=0.02)
    fig.suptitle("Canonical paired transition examples", y=1.002)
    fig.tight_layout()
    fig.savefig(plots / "paired_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    acceptance = f"""# Track C0 Human Acceptance

## 1. Paired V-C examples

The evaluation uses all 960 exact S3.1 T-Rex `head_left` pair IDs at the accepted k=16 horizon. No replacement test subset was created. Representative fixed examples include:

{chr(10).join(examples)}

## 2. Paired vs shuffled

Selected bridge: `{selected}`. Paired cosine: `{paired_row['paired_cosine']:.6f}`; different-episode cosine: `{paired_row['different_episode_cosine']:.6f}`; margin: `{paired_row['different_episode_margin']:.6f}`; bootstrap 95% CI: `{paired_row['margin_bootstrap_95_ci']}`. B2 is reported only as a pair-conditioned cross-attention interface diagnostic and is excluded from independent-tower retrieval claims.

## 3. Retrieval

All-pair V→C R@1/R@5/R@10: `{retrieval['all']['v_to_c']['recall_at_1']:.6f}` / `{retrieval['all']['v_to_c']['recall_at_5']:.6f}` / `{retrieval['all']['v_to_c']['recall_at_10']:.6f}`. Dynamic and rare-boundary breakdowns are recorded in `evaluation.json`.

## 4. V→C prediction

Test MSE: `{prediction['test']['mse']:.6f}`; train-mean control: `{prediction['train_mean_control']['mse']:.6f}`; different-episode target control: `{prediction['different_episode_target_control']['mse']:.6f}`; dynamic MSE: `{prediction['dynamic']['mse']:.6f}`.

## 5. Dynamic / boundary result

Free→contact recall: `{boundary['free_to_contact']:.6f}`; contact→free recall: `{boundary['contact_to_free']:.6f}`. Dynamic retrieval count: `{retrieval['dynamic']['count']}`; rare-boundary count: `{retrieval['rare_boundary']['count']}`.

## 6. Contact semantic retention

R_contact: `{retention['contact_transition']['advantage_retention']:.6f}`; R_force: `{retention['force_trend_class']['advantage_retention']:.6f}`; no-collapse gate: `{evaluation['gates']['no_collapse']}`.

## 7. Causal / no-leak contract

Contract audit: `{audit['status']}`. Offline z_v/z_c are transition teachers. Runtime accepts current I_≤t, robot state, T_[t-0.5:t], h_t^c, and predicted z_hat_c; it rejects I_t+16, h_t+16^c, and true z_c outside explicit oracle-evaluation mode.

## 8. Missing-contact fallback

Status: `{gate['status']}`; missing gate exactly zero: `{gate['missing_gate_exact_zero']}`; masked residual output exactly equals Vision baseline: `{gate['masked_fallback_exact_vision']}`; deterministic: `{gate['deterministic']}`.

## 9. Final decision

`{evaluation['decision']}`
"""
    (args.output_dir / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    summary = {
        "schema": "tactile3d-unit.c0-visualization.v1",
        "status": "PASS",
        "plots": [
            "plots/paired_examples.png",
            "plots/paired_vs_shuffled.png",
            "plots/retrieval.png",
            "plots/semantic_retention.png",
        ],
        "human_acceptance": "HUMAN_ACCEPTANCE.md",
        "selected_bridge": selected,
        "decision": evaluation["decision"],
        "checkpoint_sha256": training["checkpoint_sha256"],
    }
    (args.output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
