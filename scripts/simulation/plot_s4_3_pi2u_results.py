#!/usr/bin/env python3
"""Render factual PI2U BVA training and evaluation figures from frozen artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
PLOTS = ARTIFACTS / "plots"
LOG = ROOT / ".local/logs/simulation/s4_3_pi2u/bva/train.log"


def main() -> None:
    stats = json.loads((ARTIFACTS / "paired_ablation_statistics.json").read_text())
    PLOTS.mkdir(parents=True, exist_ok=False)
    records = []
    for line in LOG.read_text(errors="replace").splitlines():
        match = re.search(r"Step (\d+): .*?loss=([0-9.eE+-]+).*?physical_loss=([0-9.eE+-]+)", line)
        if match:
            records.append(tuple(float(value) for value in match.groups()))
    steps, loss, auxiliary = np.asarray(records).T
    for values, label, name in ((loss, "training loss", "bva_training_loss.png"), (auxiliary, "VA physical auxiliary loss", "bva_auxiliary_loss.png")):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(steps, values, linewidth=1)
        axis.set(xlabel="optimizer step", ylabel=label, title=f"BVA seed42 {label}")
        figure.tight_layout(); figure.savefig(PLOTS / name, dpi=180); plt.close(figure)
    models = ["B0", "BVA", "B1", "B2"]
    summaries = stats["summaries"]
    rates = [100 * summaries[model]["success_rate"] for model in models]
    intervals = np.asarray([summaries[model]["wilson_95ci"] for model in models]) * 100
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(models, rates, yerr=np.vstack((np.asarray(rates)-intervals[:,0], intervals[:,1]-np.asarray(rates))), fmt="o", capsize=5)
    axis.set(ylabel="success rate (%)", title="Fresh seed3 evaluation (Wilson 95% CI)", ylim=(0, 100))
    figure.tight_layout(); figure.savefig(PLOTS / "fresh_success_rates_wilson.png", dpi=180); plt.close(figure)
    for effect, row in stats["contrasts"].items():
        table = row["paired_outcome_table"]
        matrix = np.array([[table["both_success"], table["first_only"]], [table["second_only"], table["both_fail"]]])
        figure, axis = plt.subplots(figsize=(4, 3.5)); image = axis.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix): axis.text(j, i, str(value), ha="center", va="center")
        axis.set(xticks=[0,1], yticks=[0,1], xticklabels=["both success", f"{row['first']} only"], yticklabels=[f"{row['second']} only", "both fail"], title=effect)
        figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(PLOTS / f"paired_{effect.lower().replace('-', '_')}.png", dpi=180); plt.close(figure)
    print(PLOTS)


if __name__ == "__main__":
    main()
