#!/usr/bin/env python3
"""Generate the decision-focused S3.2-Q human-acceptance package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/s3_2_q"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_variance_bands(q0: dict[str, Any], path: Path) -> None:
    names = ("high", "mid", "low")
    bands = q0["variance_bands"]["bands"]
    variance = [bands[name]["explained_variance_fraction"] for name in names]
    contact = [bands[name]["probe_information"]["contact_transition"]["macro_f1"] for name in names]
    force = [bands[name]["probe_information"]["force_trend"]["macro_f1"] for name in names]
    x = np.arange(3)
    fig, left = plt.subplots(figsize=(8.4, 4.8))
    left.bar(x - 0.18, contact, 0.36, label="Contact transition F1", color="#277da1")
    left.bar(x + 0.18, force, 0.36, label="Force trend F1", color="#43aa8b")
    left.set_ylabel("Train-probe test macro-F1")
    left.set_xticks(x, [name.title() for name in names])
    left.set_ylim(0, 0.7)
    right = left.twinx()
    right.plot(x, variance, color="#f94144", marker="o", label="Explained variance")
    right.set_ylabel("Train explained-variance fraction")
    right.set_ylim(0, 1.05)
    handles, labels = left.get_legend_handles_labels()
    extra_handles, extra_labels = right.get_legend_handles_labels()
    left.legend(handles + extra_handles, labels + extra_labels, fontsize=8)
    left.set_title("Q0 semantic information by preregistered flattened-PC band")
    finish(fig, path)


def plot_boundary_loss(q0: dict[str, Any], path: Path) -> None:
    rows = q0["dynamic_boundary"]["per_class"]
    labels = ("free→free", "free→contact", "contact→contact", "contact→free")
    values = [rows[str(index)]["mean"] for index in range(4)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(labels, values, color=("#90be6d", "#f9844a", "#577590", "#f94144"))
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("Ordinary RQ z_c quantization MSE")
    ax.set_title("Rare Contact boundaries absorb disproportionate quantization error")
    finish(fig, path)


def plot_q1(evaluation: dict[str, Any], path: Path) -> None:
    names = ("ordinary", "whitened", "predictive")
    labels = ("Ordinary", "Whitened", "Predictive")
    metrics = ("r_recon_raw", "r_contact_raw", "r_force_raw")
    metric_labels = ("R_recon", "R_contact", "R_force")
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for offset, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [evaluation["q1"][name]["retention"][metric] for name in names]
        ax.bar(x + (offset - 1) * width, values, width, label=label)
    ax.axhline(0.8, color="#f8961e", linestyle="--", linewidth=1, label="reconstruction gate")
    ax.axhline(0.9, color="#f94144", linestyle="--", linewidth=1, label="semantic gates")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Raw advantage retention")
    ax.set_title("Q1: whitening and predictive objectives do not close the semantic gap")
    ax.legend(fontsize=8, ncol=2)
    finish(fig, path)


def plot_q2(evaluation: dict[str, Any], path: Path) -> None:
    names = ("semantic_only", "private_only", "full")
    labels = ("Semantic only", "Private only", "Full")
    values = [evaluation["q2"][name]["reconstruction"]["dynamic"]["future_mse"] for name in names]
    contact = [evaluation["q2"][name]["probes"]["contact_transition"]["macro_f1"] for name in names]
    x = np.arange(3)
    fig, left = plt.subplots(figsize=(8.4, 4.8))
    left.bar(x - 0.18, values, 0.36, color="#f3722c", label="Dynamic future MSE")
    left.set_ylabel("Dynamic future MSE")
    left.set_xticks(x, labels)
    right = left.twinx()
    right.bar(x + 0.18, contact, 0.36, color="#277da1", label="Contact macro-F1")
    right.set_ylabel("Direct-token Contact macro-F1")
    right.set_ylim(0, 0.72)
    handles, legend_labels = left.get_legend_handles_labels()
    extra, extra_labels = right.get_legend_handles_labels()
    left.legend(handles + extra, legend_labels + extra_labels, fontsize=8)
    left.set_title("Q2: both streams contribute, but the semantic stream misses its gate")
    finish(fig, path)


def plot_rare_boundary(evaluation: dict[str, Any], path: Path) -> None:
    names = ("ordinary", "whitened", "predictive")
    labels = ("Ordinary", "Whitened", "Predictive")
    q2 = evaluation["q2"]["semantic_only"]["probes"]["contact_transition"]["per_class"]
    f2c = [
        evaluation["q1"][name]["probes"]["contact_transition"]["per_class"][
            "free_to_contact"
        ]["recall"]
        for name in names
    ]
    c2f = [
        evaluation["q1"][name]["probes"]["contact_transition"]["per_class"][
            "contact_to_free"
        ]["recall"]
        for name in names
    ]
    f2c.append(q2["free_to_contact"]["recall"])
    c2f.append(q2["contact_to_free"]["recall"])
    labels = labels + ("Q2 semantic",)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, f2c, 0.36, label="free→contact recall")
    ax.bar(x + 0.18, c2f, 0.36, label="contact→free recall")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("Recall")
    ax.set_title("Rare-boundary retention remains far below continuous z_c")
    ax.legend()
    finish(fig, path)


def plot_rate_distortion(evaluation: dict[str, Any], path: Path) -> None:
    rows = [
        ("Ordinary", evaluation["q1"]["ordinary"]),
        ("Whitened", evaluation["q1"]["whitened"]),
        ("Predictive", evaluation["q1"]["predictive"]),
        ("Q2 semantic", evaluation["q2"]["semantic_only"]),
        ("Q2 full", evaluation["q2"]["full"]),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for label, row in rows:
        bits = row["bits"]
        mse = row["reconstruction"]["dynamic"]["future_mse"]
        contact = row["probes"]["contact_transition"]["macro_f1"]
        ax.scatter(bits, mse, s=240 * contact, label=f"{label} (F1={contact:.3f})")
        ax.annotate(label, (bits, mse), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Nominal bits / transition")
    ax.set_ylabel("Dynamic future MSE")
    ax.set_title("Rate–distortion: marker area encodes Contact macro-F1")
    ax.legend(fontsize=7, loc="upper right")
    finish(fig, path)


def plot_anti_bypass(evaluation: dict[str, Any], path: Path) -> None:
    anti = evaluation["q2"]["anti_bypass"]
    rows = (
        ("Full", evaluation["q2"]["full"]["reconstruction"]["dynamic"]["future_mse"]),
        ("Semantic zero", anti["semantic_zero"]["dynamic"]["future_mse"]),
        ("Private zero", anti["private_zero"]["dynamic"]["future_mse"]),
        ("Shuffle semantic", anti["shuffled_semantic"]["dynamic"]["future_mse"]),
        ("Shuffle private", anti["shuffled_private"]["dynamic"]["future_mse"]),
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar([row[0] for row in rows], [row[1] for row in rows], color="#4d908e")
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("Dynamic future MSE")
    ax.set_title(f"Q2 anti-bypass ablation — bypass={anti['bypass']}")
    finish(fig, path)


def plot_final_decision(evaluation: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axis("off")
    decision = evaluation["final_decision"]
    ordinary = evaluation["q1"]["ordinary"]["retention"]
    semantic = evaluation["q2"]["semantic_only"]["retention"]
    full = evaluation["q2"]["full"]["retention"]
    text = (
        f"{decision}\n\n"
        f"Best single (ordinary): R_recon={ordinary['r_recon_raw']:.3f}, "
        f"R_contact={ordinary['r_contact_raw']:.3f}, R_force={ordinary['r_force_raw']:.3f}\n"
        f"Q2 semantic: R_contact={semantic['r_contact_raw']:.3f}, "
        f"R_force={semantic['r_force_raw']:.3f}\n"
        f"Q2 full: R_recon={full['r_recon_raw']:.3f}; bypass="
        f"{evaluation['q2']['anti_bypass']['bypass']}\n\n"
        "Track C contract: continuous z_c [B,8,32] with frozen S2 E_c/D_c"
    )
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=13,
        bbox={"boxstyle": "round,pad=1", "facecolor": "#edf6f9", "edgecolor": "#006d77"},
    )
    finish(fig, path)


def human_acceptance(evaluation: dict[str, Any]) -> str:
    decision = evaluation["final_decision"]
    return f"""# Track B S3.2-Q Human Acceptance

Run all commands from the Track B repository root. Runtime files below are ignored by Git.

## 1. Semantic error by variance band

```bash
python -m json.tool .local/artifacts/tactile_unit/s3_2_q/q0_diagnosis.json | less
```

Inspect `plots/01_semantic_error_by_variance_band.png`.

## 2. Transition-boundary quantization loss

Inspect `plots/02_transition_boundary_quantization_loss.png`.
Confirm boundary/non-boundary ratio in Q0 is 3.77x.

## 3. Ordinary vs whitened vs predictive tokenizer

```bash
jq '.q1 | with_entries(.value={{bits:.value.bits, retention:.value.retention,
rare:.value.probes.contact_transition.per_class,
temporal:.value.temporal_controls}})' \
  .local/artifacts/tactile_unit/s3_2_q/evaluation.json
```

Inspect `plots/03_q1_candidate_comparison.png`.

## 4. Semantic-only vs private-only vs full

```bash
jq '.q2 | {{semantic_only:.semantic_only.retention,
private_only:.private_only.retention, full:.full.retention}}' \
  .local/artifacts/tactile_unit/s3_2_q/evaluation.json
```

Inspect `plots/04_q2_stream_comparison.png`.

## 5. Rare boundary retention

Inspect `plots/05_rare_boundary_retention.png` and the per-class precision/recall in
`evaluation.json`.

## 6. Rate–distortion

Inspect `plots/06_rate_distortion.png`. All two-stage totals are explicitly 112 bits;
Q2 semantic-only is 56 bits and Q2 full is 112 bits.

## 7. Anti-bypass ablation

```bash
jq '.q2.anti_bypass' .local/artifacts/tactile_unit/s3_2_q/evaluation.json
```

Inspect `plots/07_anti_bypass.png`. Both streams contribute; bypass must remain `false`.

## 8. Final decision

```bash
python -m json.tool .local/artifacts/tactile_unit/s3_2_q/final_decision.json
jq -r '.track_c_contract.checkpoint_sha256' .local/artifacts/tactile_unit/s3_2_q/final_decision.json
```

Inspect `plots/08_final_decision.png`.

Expected decision: `{decision}`.

Track C must consume continuous `z_c [B,8,32]`, retain the frozen S2 checkpoint
identity recorded in `final_decision.json`, and must not start from this Track B worktree.
"""


def main() -> int:
    args = parse_args()
    q0 = load(args.artifact_dir / "q0_diagnosis.json")
    evaluation = load(args.artifact_dir / "evaluation.json")
    plots = args.artifact_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_variance_bands(q0, plots / "01_semantic_error_by_variance_band.png")
    plot_boundary_loss(q0, plots / "02_transition_boundary_quantization_loss.png")
    plot_q1(evaluation, plots / "03_q1_candidate_comparison.png")
    plot_q2(evaluation, plots / "04_q2_stream_comparison.png")
    plot_rare_boundary(evaluation, plots / "05_rare_boundary_retention.png")
    plot_rate_distortion(evaluation, plots / "06_rate_distortion.png")
    plot_anti_bypass(evaluation, plots / "07_anti_bypass.png")
    plot_final_decision(evaluation, plots / "08_final_decision.png")
    (args.artifact_dir / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance(evaluation))
    print(json.dumps({"status": "COMPLETE", "plots": 8, "decision": evaluation["final_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
