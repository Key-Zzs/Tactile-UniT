#!/usr/bin/env python3
"""Apply the frozen C3-MS-CC source-selection reducer to completed validation trials."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3mscc_contact_context import load_checkpoint, sha256_file  # noqa: E402
from scripts.tactile_unit.c3mscc_runtime import DEFAULT_CONFIG, atomic_json, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    experiment_root = ROOT / config["runtime"]["experiment_root"]
    summary = json.loads((artifact_root / "training_summary.json").read_text())
    if summary.get("test_loaded") is not False:
        raise RuntimeError("STRUCTURAL_FAIL: existing trials are not pretest validation trials")
    results = summary["trials"]
    if len(results) > 6 or any(
        row["best"]["validation"].get("test_loaded") is not False for row in results
    ):
        raise RuntimeError("STRUCTURAL_FAIL: invalid bounded validation trial set")
    best_utility = max(float(row["best"]["utility"]) for row in results)
    best_ah = max(
        (row for row in results if row["trial"]["source"] == "AH"),
        key=lambda row: float(row["best"]["utility"]),
    )
    tolerance = float(config["validation"]["simplicity_tolerance"])
    if (
        best_ah["best"]["validation"]["gates"]["all"]
        and float(best_ah["best"]["utility"]) >= best_utility - tolerance
    ):
        selected = best_ah
        rationale = "A+H passes all validation gates and is within 0.01 of best utility"
    else:
        selected = max(
            (row for row in results if row["trial"]["source"] == "VAH"),
            key=lambda row: float(row["best"]["utility"]),
        )
        rationale = "best validation-only V+A+H trial; A+H all-gates simplicity condition not met"
    selected_path = experiment_root / "selected.pt"
    shutil.copyfile(ROOT / selected["best"]["checkpoint"], selected_path)
    model, metadata = load_checkpoint(selected_path)
    if metadata.get("test_loaded") is not False:
        raise RuntimeError("STRUCTURAL_FAIL: selected checkpoint permits test")
    selection = {
        "schema": "tactile3d-unit.vac-c3mscc-selection.v1",
        "source": selected["trial"]["source"], "trial": selected["trial"]["id"],
        "architecture": config["architecture"],
        "loss_weights": config["training"]["loss_weights"],
        "epoch": selected["best"]["epoch"],
        "validation_metrics": selected["best"]["validation"],
        "checkpoint": str(selected_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(selected_path),
        "parameter_summary": model.parameter_summary(),
        "selection_rationale": rationale,
        "vision_incremental_comparison": {
            row["trial"]["id"]: row["best"]["validation"]["utility"] for row in results
        },
        "selected_via": "VALIDATION ONLY", "selection_split": "validation only",
        "test_loaded": False, "action_exact_ar_transform": False,
        "frozen_shared_state_sha256": config["accepted"]["shared_state_sha256"],
    }
    path = artifact_root / "selection.json"
    atomic_json(path, selection)
    digest = sha256_file(path)
    (artifact_root / "selection.sha256").write_text(digest + "  selection.json\n")
    print(path)


if __name__ == "__main__":
    main()
