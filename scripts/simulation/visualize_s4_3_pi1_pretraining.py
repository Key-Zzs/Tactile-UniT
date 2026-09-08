#!/usr/bin/env python3
"""Render only measured/predeclared S4.3-PI1 visuals available before training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
PLOTS = ARTIFACTS / "plots"
SIDECAR = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"


def finish(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, width, height, text, color="#dbeafe", fontsize=9):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#334155")
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#475569"))


def dataflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.1))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, (0.2, 2.45), 2.0, 0.8, "Raw DexJoCo\n100 episodes", "#fef3c7")
    box(ax, (2.7, 2.45), 2.0, 0.8, "Exact converter trim\n+ row alignment", "#e0f2fe")
    box(ax, (5.2, 2.45), 2.0, 0.8, "Read-only replay\ncurrent contacts", "#dcfce7")
    box(ax, (7.7, 2.45), 1.8, 0.8, "30D tactile\n26-tick history", "#ede9fe")
    box(ax, (10.0, 2.45), 1.7, 0.8, "Frozen E_T\nContact[256]", "#fce7f3")
    for a, b in ((2.2, 2.7), (4.7, 5.2), (7.2, 7.7), (9.5, 10.0)):
        arrow(ax, (a, 2.85), (b, 2.85))
    box(ax, (2.7, 0.65), 3.0, 0.8, "Immutable official LeRobot rows\nRGB + state + action + prompt", "#f8fafc")
    box(ax, (7.0, 0.65), 3.3, 0.8, "Indexed sidecar strict superset\ntactile + Contact + t→t+27 target", "#f8fafc")
    arrow(ax, (4.2, 1.45), (4.2, 2.4)); arrow(ax, (8.65, 2.4), (8.65, 1.5))
    ax.text(6, 3.72, "Official-data-to-tactile augmentation", ha="center", fontsize=14, weight="bold")
    finish(fig, "01_augmentation_dataflow.png")


def replay_errors(replay: dict) -> None:
    episodes = replay["episodes"]
    x = np.arange(len(episodes))
    tcp = np.array([row["errors"]["tcp_position_max_abs_m"] for row in episodes]) * 1000
    quat = np.array([row["errors"]["tcp_quaternion_max_abs"] for row in episodes])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(x, tcp, color="#2563eb", linewidth=1.2)
    axes[0].axhline(5, color="#dc2626", linestyle="--", label="frozen 5 mm gate")
    axes[0].set_ylabel("TCP max error (mm)"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(x, quat, color="#7c3aed", linewidth=1.2)
    axes[1].axhline(0.02, color="#dc2626", linestyle="--", label="frozen 0.02 component gate")
    axes[1].set_ylabel("Quaternion max error"); axes[1].set_xlabel("Official episode index")
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle("Replay/state alignment errors — all 100 episodes passed")
    finish(fig, "02_replay_alignment_errors.png")


def region_activity(tactile: np.ndarray) -> None:
    names = ["palm", "index", "middle", "ring", "thumb"]
    active = (tactile.reshape(-1, 5, 6)[:, :, 0] > 0).sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    bars = ax.bar(names, active, color=["#2563eb", "#0891b2", "#059669", "#ca8a04", "#db2777"])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Frames with active contact")
    ax.set_title("Measured tactile activity by right-hand region")
    ax.grid(axis="y", alpha=.25)
    finish(fig, "03_tactile_region_activity.png")


def history_alignment(tactile: np.ndarray, histories: np.ndarray, bootstrap: np.ndarray, episode_index: np.ndarray) -> None:
    start = int(np.flatnonzero(episode_index == 0)[0])
    count = 36
    values = histories[start : start + count, :, 0]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})
    image = ax0.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(image, ax=ax0, label="palm contact flag")
    ax0.set_ylabel("Current frame"); ax0.set_xlabel("History offset (0 oldest, 25 current)")
    ax0.set_title("LEFT_REPEAT_FIRST causal 26-tick history — episode 0 start")
    ax1.step(np.arange(count), bootstrap[start : start + count], where="mid", color="#dc2626")
    ax1.set_ylabel("Repeated ticks"); ax1.set_xlabel("Current frame"); ax1.grid(alpha=.25)
    assert np.array_equal(histories[:, -1], tactile)
    finish(fig, "04_tactile_history_alignment.png")


def contact_distribution(contact: np.ndarray) -> None:
    dim_mean = contact.mean(axis=0)
    dim_std = contact.std(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(contact.ravel(), bins=100, color="#2563eb", alpha=.85)
    axes[0].set_xlabel("Contact-State value"); axes[0].set_ylabel("Element count")
    axes[0].set_title("All 40,065 × 256 encoded values"); axes[0].set_yscale("log")
    axes[1].scatter(dim_mean, dim_std, s=14, alpha=.7, color="#7c3aed")
    axes[1].set_xlabel("Per-dimension mean"); axes[1].set_ylabel("Per-dimension std")
    axes[1].set_title("Frozen E_T dimension statistics"); axes[1].grid(alpha=.25)
    fig.suptitle("Precomputed Contact-State distribution")
    finish(fig, "05_contact_state_distribution.png")


def pi1b_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, (.3, 2.7), 2.2, .75, "Official prefix\nimages + language", "#dcfce7")
    box(ax, (6.8, 2.7), 2.1, .75, "append 8 Contact\nprefix tokens", "#dbeafe")
    box(ax, (9.5, 2.7), 2.1, .75, "Official action suffix\n30 positions", "#fef3c7")
    arrow(ax, (2.5, 3.08), (6.8, 3.08)); arrow(ax, (8.9, 3.08), (9.5, 3.08))
    box(ax, (.3, .75), 1.7, .75, "Contact-State\n[256]", "#fce7f3")
    box(ax, (2.45, .75), 2.4, .75, "LN → 256 → GELU → 256\nreshape [8,32]", "#ede9fe")
    box(ax, (5.3, .75), 1.7, .75, "shared 32→2048\n8 tokens", "#dbeafe")
    arrow(ax, (2.0, 1.13), (2.45, 1.13)); arrow(ax, (4.85, 1.13), (5.3, 1.13)); arrow(ax, (6.15, 1.5), (7.45, 2.68))
    ax.text(8.2, 1.55, "8 valid, non-autoregressive observation-prefix tokens", ha="center", fontsize=10)
    ax.text(8.2, .35, "Image/language token identities unchanged • action shape [30,32] unchanged", ha="center", fontsize=10)
    ax.text(6, 3.72, "PI1B Contact-token insertion", ha="center", fontsize=14, weight="bold")
    finish(fig, "06_pi1b_contact_token_insertion.png")


def pi1c_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 5)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, (.3, 3.15), 2.2, .8, "Same PI1B prefix\n+ official flow path", "#dbeafe")
    box(ax, (3.0, 3.15), 2.3, .8, "Final action-expert hidden\nfirst 27 positions", "#dcfce7")
    box(ax, (5.8, 3.15), 2.3, .8, "mean → LN → 1024→512\nGELU → 512→256", "#ede9fe")
    box(ax, (8.6, 3.15), 1.5, .8, "prediction\n[8,32]", "#fce7f3")
    box(ax, (8.6, 1.15), 1.5, .8, "frozen target\n[8,32]", "#fef3c7")
    box(ax, (10.55, 2.15), 1.15, .8, "masked\nMSE", "#fee2e2")
    for a, b in ((2.5, 3.0), (5.3, 5.8), (8.1, 8.6), (10.1, 10.55)): arrow(ax, (a, 3.55), (b, 3.55))
    arrow(ax, (10.1, 1.55), (10.55, 2.45))
    ax.text(6, .45, "Training only: L = Lπ0.5 + λphys Lphys • target/validity forbidden at inference", ha="center", fontsize=10)
    ax.text(6, 4.55, "PI1C shared-physical auxiliary", ha="center", fontsize=14, weight="bold")
    finish(fig, "07_pi1c_physical_auxiliary.png")


def main() -> None:
    replay = json.loads((ARTIFACTS / "tactile_replay_contract.json").read_text())
    if replay["status"] != "PASS":
        raise RuntimeError("replay contract is not PASS")
    with np.load(SIDECAR, allow_pickle=False) as sidecar:
        tactile = sidecar["tactile_sim"]
        histories = sidecar["tactile_history"]
        bootstrap = sidecar["history_bootstrap_count"]
        contact = sidecar["contact_state"]
        episode_index = sidecar["episode_index"]
        dataflow(); replay_errors(replay); region_activity(tactile)
        history_alignment(tactile, histories, bootstrap, episode_index)
        contact_distribution(contact); pi1b_diagram(); pi1c_diagram()
    print(json.dumps({"status": "PASS", "plots": len(list(PLOTS.glob("*.png")))}, sort_keys=True))


if __name__ == "__main__":
    main()
