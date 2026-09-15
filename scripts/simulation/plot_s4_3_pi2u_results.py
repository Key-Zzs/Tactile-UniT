#!/usr/bin/env python3
"""Render the complete, source-backed S4.3-PI2U visual evidence set."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_pi2u_va import VAOnlyBridge  # noqa: E402


ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
PLOTS = ARTIFACTS / "plots"
PLOTS_TMP = ARTIFACTS / ".plots.tmp"
LOG = ROOT / ".local/logs/simulation/s4_3_pi2u/bva/train.log"
VA_CONFIG = ROOT / "configs/simulation/s4_3_pi2u_va_bridge.json"
VA_CACHE = ROOT / ".local/cache/simulation/s4_3_pi2u/va_bridge/validation.npz"
VA_CHECKPOINT = ROOT / ".local/experiments/simulation/s4_3_pi2u/va_bridge/frozen.pt"
SEED = 5


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(PLOTS_TMP / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def box(axis: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="#263238",
        linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9)


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#455a64") -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.2))


def diagram(title: str, figsize: tuple[float, float] = (10, 5)) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=figsize)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5)
    axis.axis("off")
    axis.set_title(title, fontsize=13, weight="bold")
    return figure, axis


def architecture_visuals() -> list[str]:
    created: list[str] = []
    architecture = load_json(ARTIFACTS / "official_unit_architecture.json")
    inventory = load_json(ARTIFACTS / "official_unit_checkpoint_inventory.json")
    embodiment = load_json(ARTIFACTS / "official_unit_embodiment_contract.json")
    adaptation = load_json(ARTIFACTS / "unit_adaptation_classification.json")

    figure, axis = diagram("Official UniT architecture traced from pinned source", (12, 6))
    box(axis, (.2, 3.5), 1.6, .8, "RGB t, t+16\nDINOv2", "#bbdefb")
    box(axis, (2.2, 3.5), 1.6, .8, "Vision M-Former\nz_v", "#90caf9")
    box(axis, (.2, .7), 1.6, .8, "state + action16\ncategory adapter", "#ffe0b2")
    box(axis, (2.2, .7), 1.6, .8, "Action M-Former\nz_a", "#ffcc80")
    box(axis, (4.3, 2.1), 1.5, .8, "Fusion\nz_m", "#c5e1a5")
    box(axis, (6.2, 2.1), 1.5, .8, "shared RVQ\n[8,2], 128/stage", "#ce93d8")
    box(axis, (8.1, 3.5), 1.6, .8, "Vision decoder\ngoal DINO", "#b2dfdb")
    box(axis, (8.1, .7), 1.6, .8, "Action decoder\ncategory-specific", "#ffccbc")
    for start, end in [((1.8, 3.9), (2.2, 3.9)), ((1.8, 1.1), (2.2, 1.1)), ((3.8, 3.9), (4.6, 2.9)), ((3.8, 1.1), (4.6, 2.1)), ((5.8, 2.5), (6.2, 2.5)), ((7.7, 2.7), (8.1, 3.7)), ((7.7, 2.3), (8.1, 1.3))]:
        arrow(axis, start, end)
    axis.text(5.0, 4.75, "vision-only / action-only / fused routes share the discrete codebook", ha="center", fontsize=9)
    axis.text(5.0, .15, "multi-scenario cross-reconstruction aligns V and A; action input/decoder banks are embodiment-specific", ha="center", fontsize=9)
    save(figure, "official_unit_architecture_map.png")
    created.append("official_unit_architecture_map.png")

    counts = inventory["tensor_inventory"]["tokenizer_top_level"]
    labels = list(counts)
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh(labels, values, color=plt.cm.viridis(np.linspace(.15, .85, len(labels))))
    for index, value in enumerate(values):
        axis.text(value + max(values) * .01, index, str(value), va="center", fontsize=8)
    axis.set(xlabel="released tokenizer tensors", title="Official checkpoint component inventory (metadata-only audit)")
    save(figure, "official_unit_checkpoint_component_map.png")
    created.append("official_unit_checkpoint_component_map.png")

    recipes = embodiment["released_data_recipes"]
    rows = [
        ["GR1 joints", str(recipes["gr1_joints"]["raw_selected_state_dim"]), str(recipes["gr1_joints"]["raw_selected_action_dim"]), "ego", "released id24"],
        ["GR1 EEF", "bilateral EEF", "bilateral EEF", "ego", "released id23"],
        ["DexJoCo", "23: TCP quat + Allegro16", "22: TCP rotvec + Allegro16", "front+wrist", "no released slot"],
    ]
    figure, axis = plt.subplots(figsize=(12, 3.3))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=["Embodiment", "state semantics", "action semantics", "cameras", "category"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8)
    for col in range(5): table[(0, col)].set_facecolor("#b3e5fc")
    for col in range(5): table[(3, col)].set_facecolor("#ffcdd2")
    axis.set_title("Released GR1 contract vs DexJoCo: shape padding does not provide semantic compatibility", weight="bold")
    save(figure, "official_unit_gr1_dexjoco_mismatch.png")
    created.append("official_unit_gr1_dexjoco_mismatch.png")

    options = ["DIRECT", "ADAPTER_ONLY", "TOKENIZER_ADAPTATION", "FULL_RETRAIN", "NOT_FEASIBLE"]
    selected = adaptation["classification"].removeprefix("UNIT_DEXJOCO_").removesuffix("_COMPATIBLE")
    colors = ["#66bb6a" if option == selected else "#eceff1" for option in options]
    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.axis("off")
    for index, (option, color) in enumerate(zip(options, colors)):
        x = .03 + index * .195
        patch = FancyBboxPatch((x, .38), .17, .28, transform=axis.transAxes, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#37474f")
        axis.add_patch(patch); axis.text(x + .085, .52, option.replace("_", "\n"), transform=axis.transAxes, ha="center", va="center", fontsize=9)
        if index < len(options) - 1:
            axis.annotate("", xy=(x + .195, .52), xytext=(x + .17, .52), xycoords=axis.transAxes, arrowprops={"arrowstyle": "->"})
    axis.text(.5, .18, "Selected: freeze vision/fusion/RVQ/shared M-Former; train a dedicated DexJoCo state/action encoder+decoder slice", transform=axis.transAxes, ha="center", fontsize=9)
    axis.set_title("Future BUniT adaptation options (design only; no BUniT training)", weight="bold")
    save(figure, "future_bunit_adaptation_options.png")
    created.append("future_bunit_adaptation_options.png")
    return created


def different_episode_permutation(episode: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episode))
    result = np.empty(len(episode), np.int64)
    pools = {int(key): order[episode[order] != key] for key in np.unique(episode)}
    offsets = {key: 0 for key in pools}
    for index, value in enumerate(episode):
        key = int(value)
        result[index] = pools[key][offsets[key] % len(pools[key])]
        offsets[key] += 1
    return result


def normalize(value: np.ndarray) -> np.ndarray:
    flat = value.astype(np.float64).reshape(len(value), -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


@torch.inference_mode()
def va_distributions() -> tuple[np.ndarray, np.ndarray]:
    config = load_json(VA_CONFIG)
    metrics = load_json(ARTIFACTS / "va_bridge_metrics.json")
    if sha256(VA_CHECKPOINT) != metrics["checkpoint_sha256"]:
        raise RuntimeError("VA bridge checkpoint identity changed")
    checkpoint = torch.load(VA_CHECKPOINT, map_location="cpu", weights_only=False)
    model = VAOnlyBridge(width=int(config["architecture"]["hidden_width"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with np.load(VA_CACHE, allow_pickle=False) as source:
        z_v = np.asarray(source["z_v"])
        z_a = np.asarray(source["z_a"])
        _, episode = np.unique(np.asarray(source["episode_id"]), return_inverse=True)
    shared_v, shared_a = [], []
    for start in range(0, len(z_v), 512):
        stop = min(len(z_v), start + 512)
        shared_v.append(model.encode("vision", torch.from_numpy(z_v[start:stop])).numpy())
        shared_a.append(model.encode("action", torch.from_numpy(z_a[start:stop])).numpy())
    vision, action = normalize(np.concatenate(shared_v)), normalize(np.concatenate(shared_a))
    permutation = different_episode_permutation(episode.astype(np.int64), int(config["seed"]) + 91)
    paired = np.sum(vision * action, axis=1)
    shuffled = np.sum(vision * action[permutation], axis=1)
    if not np.isclose(paired.mean(), metrics["paired_cosine"], atol=1e-6) or not np.isclose(shuffled.mean(), metrics["different_episode_shuffled_cosine"], atol=1e-6):
        raise RuntimeError("recomputed VA distribution does not reproduce frozen metrics")
    return paired, shuffled


def va_bridge_visuals() -> list[str]:
    created: list[str] = []
    metrics = load_json(ARTIFACTS / "va_bridge_metrics.json")

    figure, axis = diagram("Clean VA-only continuous bridge", (11, 4.5))
    box(axis, (.4, 3.0), 1.6, .8, "RGB t→t+27\nfrozen Vision", "#bbdefb")
    box(axis, (.4, 1.0), 1.6, .8, "action t→t+27\nfrozen Action", "#ffe0b2")
    box(axis, (2.7, 3.0), 1.6, .8, "z_v [8,32]\nP_v^VA", "#90caf9")
    box(axis, (2.7, 1.0), 1.6, .8, "z_a [8,32]\nP_a^VA", "#ffcc80")
    box(axis, (5.2, 2.0), 1.6, .8, "shared alignment\nInfoNCE + retention", "#c5e1a5")
    box(axis, (7.7, 3.0), 1.8, .8, "u_v^VA [8,32]\nBVA target", "#ce93d8")
    box(axis, (7.7, 1.0), 1.8, .8, "u_a^VA [8,32]\nvalidation only", "#d1c4e9")
    for start, end in [((2.0, 3.4), (2.7, 3.4)), ((2.0, 1.4), (2.7, 1.4)), ((4.3, 3.4), (5.3, 2.7)), ((4.3, 1.4), (5.3, 2.1)), ((6.8, 2.6), (7.7, 3.3)), ((6.8, 2.2), (7.7, 1.5))]: arrow(axis, start, end)
    axis.text(5.0, .25, "Contact/tactile/C parameters: 0; training and DEV use source-group-disjoint paired V+A only", ha="center", fontsize=9)
    save(figure, "va_only_bridge_architecture.png")
    created.append("va_only_bridge_architecture.png")

    paired, shuffled = va_distributions()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(min(paired.min(), shuffled.min()), max(paired.max(), shuffled.max()), 55)
    axis.hist(shuffled, bins=bins, density=True, alpha=.60, label="different-episode shuffled", color="#ef5350")
    axis.hist(paired, bins=bins, density=True, alpha=.60, label="paired V↔A", color="#42a5f5")
    axis.axvline(paired.mean(), color="#1565c0", linestyle="--")
    axis.axvline(shuffled.mean(), color="#c62828", linestyle="--")
    axis.set(xlabel="cosine similarity", ylabel="density", title="VA DEV paired vs shuffled distributions")
    axis.legend()
    save(figure, "va_paired_shuffled_distributions.png")
    created.append("va_paired_shuffled_distributions.png")

    retrieval = metrics["retrieval"]
    labels = ["R@1", "R@5", "R@10"]
    x = np.arange(3); width = .24
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for offset, (name, key, color) in enumerate((("V→A", "vision_to_action", "#42a5f5"), ("A→V", "action_to_vision", "#ffb74d"))):
        values = [retrieval[key][f"recall_at_{k}"] for k in (1, 5, 10)]
        axis.bar(x + (offset - .5) * width, values, width, label=name, color=color)
    chance = [retrieval["vision_to_action"]["chance"][f"recall_at_{k}"] for k in (1, 5, 10)]
    axis.plot(x, chance, "ko--", label="chance")
    axis.set(xticks=x, xticklabels=labels, ylabel="recall", title="Bidirectional VA retrieval on frozen DEV")
    axis.legend()
    save(figure, "va_bidirectional_retrieval.png")
    created.append("va_bidirectional_retrieval.png")

    names = ["Vision native", "Vision shared", "Action native", "Action shared"]
    values = [metrics["native_effective_rank"]["vision"], metrics["geometry"]["vision"]["effective_rank"], metrics["native_effective_rank"]["action"], metrics["geometry"]["action"]["effective_rank"]]
    collapse = [np.nan, metrics["geometry"]["vision"]["near_zero_variance_fraction"], np.nan, metrics["geometry"]["action"]["near_zero_variance_fraction"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(names, values, color=["#90caf9", "#42a5f5", "#ffcc80", "#ffa726"]); axes[0].tick_params(axis="x", rotation=25); axes[0].set(ylabel="effective rank", title="Rank retention")
    axes[1].bar(["Vision shared", "Action shared"], [collapse[1], collapse[3]], color=["#42a5f5", "#ffa726"]); axes[1].axhline(.5, color="red", linestyle="--", label="max gate"); axes[1].set(ylabel="near-zero variance fraction", title="No-collapse gate", ylim=(0, .55)); axes[1].legend()
    save(figure, "va_bridge_rank_no_collapse.png")
    created.append("va_bridge_rank_no_collapse.png")
    return created


def bva_visuals(stats: dict[str, Any]) -> list[str]:
    created: list[str] = []
    figure, axis = diagram("BVA training-only physical auxiliary dataflow", (12, 4.8))
    box(axis, (.2, 3.1), 1.8, .8, "demo RGB t,t+27\nTRAIN only", "#bbdefb")
    box(axis, (2.5, 3.1), 1.8, .8, "frozen Vision +\nfrozen P_v^VA", "#90caf9")
    box(axis, (4.8, 3.1), 1.5, .8, "stop-grad\nu_v [8,32]", "#ce93d8")
    box(axis, (.2, 1.0), 1.8, .8, "official pi0.5\nB0 observations", "#c8e6c9")
    box(axis, (2.5, 1.0), 1.8, .8, "first 27 action\nhidden positions", "#a5d6a7")
    box(axis, (4.8, 1.0), 1.5, .8, "B2-matched head\nû_VA [8,32]", "#80cbc4")
    box(axis, (7.0, 2.0), 1.4, .8, "MSE × λ_VA", "#fff59d")
    box(axis, (8.9, 2.0), .9, .8, "+ L_π0.5", "#ffcc80")
    for start, end in [((2.0, 3.5), (2.5, 3.5)), ((4.3, 3.5), (4.8, 3.5)), ((2.0, 1.4), (2.5, 1.4)), ((4.3, 1.4), (4.8, 1.4)), ((6.3, 3.3), (7.1, 2.8)), ((6.3, 1.5), (7.1, 2.0)), ((8.4, 2.4), (8.9, 2.4))]: arrow(axis, start, end)
    axis.text(5, .25, "Runtime: auxiliary target/head absent; no tactile, Contact-State, Contact target, or future Vision", ha="center", fontsize=9)
    save(figure, "bva_auxiliary_dataflow.png")
    created.append("bva_auxiliary_dataflow.png")

    records = []
    for line in LOG.read_text(errors="replace").splitlines():
        match = re.search(r"Step (\d+): .*?official_loss=([0-9.eE+-]+).*?physical_loss=([0-9.eE+-]+)", line)
        if match:
            records.append(tuple(float(value) for value in match.groups()))
    if not records:
        raise RuntimeError("no BVA loss records found")
    steps, official, auxiliary = np.asarray(records).T
    for values, label, name in ((official, "official π0.5 loss", "bva_training_official_loss.png"), (auxiliary, "VA physical auxiliary loss", "bva_auxiliary_loss.png")):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(steps, values, linewidth=1)
        axis.set(xlabel="optimizer step", ylabel=label, title=f"BVA seed42 {label}")
        save(figure, name); created.append(name)

    models = ["B0", "BVA", "B1", "B2"]
    summaries = stats["summaries"]
    rates = np.asarray([100 * summaries[model]["success_rate"] for model in models])
    intervals = np.asarray([summaries[model]["wilson_95ci"] for model in models]) * 100
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(models, rates, yerr=np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates)), fmt="o", capsize=5)
    axis.set(ylabel="success rate (%)", title=f"Fresh seed{SEED} formal evaluation (Wilson 95% CI)", ylim=(0, 100))
    save(figure, "fresh_success_rates_wilson.png")
    created.append("fresh_success_rates_wilson.png")

    for effect, row in stats["contrasts"].items():
        table = row["paired_outcome_table"]
        matrix = np.array([[table["both_fail"], table["second_only"]], [table["first_only"], table["both_success"]]])
        figure, axis = plt.subplots(figsize=(4.5, 4)); image = axis.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix): axis.text(j, i, str(value), ha="center", va="center")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=[f"{row['second']} fail", f"{row['second']} success"], yticklabels=[f"{row['first']} fail", f"{row['first']} success"], title=f"Paired matrix: {effect}")
        figure.colorbar(image, ax=axis)
        name = f"paired_{effect.lower().replace('-', '_')}.png"
        save(figure, name); created.append(name)

    matrix_rows = []
    for name in ("BVA-B0", "B2-BVA", "B2-B1", "B1-B0"):
        row = stats["contrasts"][name]
        low, high = row["paired_bootstrap_95ci_percentage_points"]
        matrix_rows.append([name, f"{row['delta_percentage_points']:+.1f}", f"[{low:+.1f}, {high:+.1f}]", f"{row['exact_mcnemar_two_sided_p']:.4g}", f"{row['holm_adjusted_p']:.4g}", row["classification"]])
    figure, axis = plt.subplots(figsize=(12, 3.4)); axis.axis("off")
    table = axis.table(cellText=matrix_rows, colLabels=["contrast", "Δ pp", "paired 95% CI", "McNemar p", "Holm p", "class"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.7)
    for col in range(6): table[(0, col)].set_facecolor("#b3e5fc")
    axis.set_title("Final preregistered mechanism-ablation matrix", weight="bold")
    save(figure, "final_mechanism_ablation_matrix.png")
    created.append("final_mechanism_ablation_matrix.png")

    readiness = stats["PI2B_readiness"]
    choices = ["PI2B_READY_B0_B1_B2", "PI2B_READY_B0_BVA_B1_B2", "PI2B_WAIT_FOR_BUNIT", "PI2B_MECHANISM_DIAGNOSIS_FIRST"]
    reasons = ["BVA no meaningful gain; B2 strongest", "BVA contributes and belongs in multi-seed", "low-cost BUniT gate first", "pattern is unresolved or adverse"]
    figure, axis = plt.subplots(figsize=(11, 4)); axis.axis("off")
    rows = [[choice, reason, "SELECTED" if choice == readiness else "—"] for choice, reason in zip(choices, reasons)]
    table = axis.table(cellText=rows, colLabels=["readiness state", "decision rule", "seed6 decision"], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.7)
    for col in range(3): table[(0, col)].set_facecolor("#b3e5fc")
    for row_index, choice in enumerate(choices, start=1):
        if choice == readiness:
            for col in range(3): table[(row_index, col)].set_facecolor("#c8e6c9")
    axis.set_title("PI2B readiness matrix — decision only; PI2B not started", weight="bold")
    save(figure, "pi2b_readiness_matrix.png")
    created.append("pi2b_readiness_matrix.png")
    return created


def main() -> None:
    if PLOTS.exists() or PLOTS_TMP.exists() or (ARTIFACTS / "visualization_manifest.json").exists():
        raise SystemExit("refusing to overwrite PI2U visual evidence")
    stats = load_json(ARTIFACTS / "paired_ablation_statistics.json")
    if stats.get("status") != "PASS" or stats.get("evaluator_seed") != SEED:
        raise SystemExit("formal seed6 statistics are not complete")
    PLOTS_TMP.mkdir(parents=True)
    try:
        created = architecture_visuals() + va_bridge_visuals() + bva_visuals(stats)
        required = {
            "official_unit_architecture_map.png", "official_unit_checkpoint_component_map.png",
            "official_unit_gr1_dexjoco_mismatch.png", "future_bunit_adaptation_options.png",
            "va_only_bridge_architecture.png", "va_paired_shuffled_distributions.png",
            "va_bidirectional_retrieval.png", "va_bridge_rank_no_collapse.png",
            "bva_auxiliary_dataflow.png", "bva_training_official_loss.png",
            "bva_auxiliary_loss.png", "fresh_success_rates_wilson.png",
            "paired_bva_b0.png", "paired_b2_bva.png", "paired_b2_b1.png",
            "final_mechanism_ablation_matrix.png", "pi2b_readiness_matrix.png",
        }
        if not required.issubset(created) or any(not (PLOTS_TMP / name).is_file() for name in required):
            raise RuntimeError("required PI2U visual set is incomplete")
        PLOTS_TMP.rename(PLOTS)
        manifest = {
            "schema": "tactile3d-unit.s4-3-pi2u-visualization-manifest.v1",
            "status": "PASS",
            "evaluator_seed": SEED,
            "required_visuals": sorted(required),
            "additional_visuals": sorted(set(created) - required),
            "files": {name: {"sha256": sha256(PLOTS / name), "bytes": (PLOTS / name).stat().st_size} for name in sorted(created)},
            "metric_sources_sha256": {
                "va_bridge_metrics.json": sha256(ARTIFACTS / "va_bridge_metrics.json"),
                "paired_ablation_statistics.json": sha256(ARTIFACTS / "paired_ablation_statistics.json"),
                "bva_training_completion.json": sha256(ARTIFACTS / "bva_training_completion.json"),
            },
            "fabricated_metrics": False,
        }
        target = ARTIFACTS / "visualization_manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)
        print(json.dumps({"status": "PASS", "visuals": len(created)}, sort_keys=True))
    except Exception:
        if PLOTS_TMP.exists():
            shutil.rmtree(PLOTS_TMP)
        raise


if __name__ == "__main__":
    main()
