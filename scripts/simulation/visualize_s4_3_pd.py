#!/usr/bin/env python3
"""Render the frozen S4.3-PD final gate matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def plot_final_gate_matrix(statuses: Mapping[str, str], output: Path) -> None:
    """Write a compact PASS/FAIL/blocked matrix without inventing measurements."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(statuses)
    values = [statuses[label] for label in labels]
    encoding = {"FAIL": 0, "BLOCKED": 1, "PASS": 2}
    matrix = np.asarray([[encoding.get(value, 0)] for value in values])

    fig, axis = plt.subplots(figsize=(8.8, max(5.4, 0.45 * len(labels))))
    axis.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap=plt.matplotlib.colors.ListedColormap(["#b91c1c", "#d97706", "#15803d"]),
    )
    axis.set_xticks([0], ["Gate status"])
    axis.set_yticks(np.arange(len(labels)), labels)
    for row, value in enumerate(values):
        axis.text(0, row, value, ha="center", va="center", color="white", weight="bold")
    axis.set_title("S4.3-PD final gate matrix")
    axis.tick_params(axis="both", length=0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    plot_final_gate_matrix(decision["gate_matrix"], args.output)


if __name__ == "__main__":
    main()
