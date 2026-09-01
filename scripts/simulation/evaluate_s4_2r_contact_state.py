#!/usr/bin/env python3
"""Formally re-evaluate the frozen S4.2 Contact-State teacher under S4.2-R."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.intrinsic_dimension import (  # noqa: E402
    dimension_metrics,
    sampled_pairwise_diversity,
)
from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402
from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.s4_2r_acceptance import (  # noqa: E402
    evaluate_no_collapse_contract,
)
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint  # noqa: E402

PROTOCOL = ROOT / "configs/simulation/s4_2r_contact_state_rank_remediation.json"
FREEZE = ROOT / ".local/artifacts/simulation/s4_2r/protocol_freeze.json"
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
LATENT_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_state"
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2/contact_teacher/selected.pt"
ORIGINAL_EVALUATION = ROOT / ".local/artifacts/simulation/s4_2/teacher_evaluation.json"
ORIGINAL_DECISION = ROOT / "configs/simulation/s4_2_final_decision.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2r"
ACCEPTED_ROOT = ROOT / ".local/experiments/simulation/s4_2r/contact_state"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def predict_error(
    model: torch.nn.Module,
    history: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    errors = []
    predictions = []
    model = model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(history), batch_size):
            inputs = torch.from_numpy(history[start : start + batch_size]).to(device)
            output = model(inputs)["future"].cpu().numpy()
            predictions.append(output)
            errors.append(
                np.mean(np.square(output - target[start : start + batch_size]), axis=(1, 2))
            )
    return np.concatenate(errors), np.concatenate(predictions)


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        stop = min(samples, start + 100)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).astype(float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_EVALUATION.read_text(encoding="utf-8"))
    original_decision = json.loads(ORIGINAL_DECISION.read_text(encoding="utf-8"))
    if freeze["r4_started"] is not False or freeze["training_started"] is not False:
        raise RuntimeError("protocol freeze does not prove R4 preceded training")
    if original_decision["decision"] != "S4_2_CONTACT_STATE_FAIL":
        raise RuntimeError("original S4.2 failure was modified")
    expected_checkpoint_sha = protocol["frozen_inputs"]["teacher_checkpoint_sha256"]
    if sha256_file(CHECKPOINT) != expected_checkpoint_sha:
        raise RuntimeError("existing checkpoint identity mismatch")
    train = load_npz(PAIR_ROOT / "train.npz")
    validation = load_npz(PAIR_ROOT / "validation.npz")
    cached = load_npz(LATENT_ROOT / "validation.npz")
    if not np.array_equal(cached["pair_id"], validation["pair_id"]):
        raise RuntimeError("R1 latent cache pair identity mismatch")
    latent_metrics = dimension_metrics(cached["h_current"], twonn=True)
    pairwise = sampled_pairwise_diversity(
        cached["h_current"],
        seed=protocol["new_structural_collapse_contract"]["gate_e_pairwise_diversity"][
            "sampling_seed"
        ],
        maximum_samples=protocol["new_structural_collapse_contract"][
            "gate_e_pairwise_diversity"
        ]["maximum_samples"],
    )
    model, checkpoint = load_teacher_checkpoint(CHECKPOINT, "cpu")
    normalizer = TactileNormalization.from_json(checkpoint["normalization"])
    val_history = normalizer.transform(validation["current_history"]).astype(np.float32)
    val_target = normalizer.transform(validation["teacher_future"]).astype(np.float32)
    device = torch.device(args.device)
    correct_error, correct_prediction = predict_error(
        model, val_history, val_target, device, args.batch_size
    )
    controls = {
        "last_frame_repeated": np.repeat(val_history[:, -1:, :], 26, axis=1),
        "shuffled_history": val_history[:, np.random.default_rng(4242).permutation(26)],
        "reversed_history": val_history[:, ::-1].copy(),
    }
    dynamic_threshold = float(np.quantile(train["force_delta_abs"], 0.70))
    dynamic = validation["force_delta_abs"] > dynamic_threshold
    perturbation = {}
    gate_g = protocol["new_structural_collapse_contract"][
        "gate_g_perturbation_sensitivity"
    ]
    for index, name in enumerate(gate_g["correct_history_must_beat"]):
        control_error, _ = predict_error(
            model, controls[name], val_target, device, args.batch_size
        )
        difference = control_error[dynamic] - correct_error[dynamic]
        perturbation[name] = {
            "correct_mse": float(correct_error[dynamic].mean()),
            "control_mse": float(control_error[dynamic].mean()),
            "relative_improvement": 1.0
            - float(correct_error[dynamic].mean() / control_error[dynamic].mean()),
            "improvement_ci95": bootstrap_mean_ci(
                difference,
                int(gate_g["bootstrap_samples"]),
                int(gate_g["bootstrap_seed_base"]) + index,
            ),
            "dynamic_samples": int(dynamic.sum()),
        }
    first_model, _ = load_teacher_checkpoint(CHECKPOINT, device)
    second_model, _ = load_teacher_checkpoint(CHECKPOINT, device)
    _, first_prediction = predict_error(
        first_model, val_history[:128], val_target[:128], device, 128
    )
    _, second_prediction = predict_error(
        second_model, val_history[:128], val_target[:128], device, 128
    )
    original_gates = dict(original["gates"])
    original_gates["deterministic_reload"] = bool(
        np.array_equal(first_prediction, second_prediction)
    )
    result = evaluate_no_collapse_contract(
        protocol, latent_metrics, pairwise, original_gates, perturbation
    )
    warnings = ["THUMB_ZERO_ACTIVITY"]
    if latent_metrics["effective_rank"] < 16.0:
        warnings.append("LOW_EFFECTIVE_RANK_WARNING")
    classification = (
        "LOW_INTRINSIC_DIMENSION_NOT_COLLAPSE"
        if result["overall_pass"]
        else "RANK_REMEDIATION_REQUIRED"
    )
    state = (
        "S4_2R_CONTACT_STATE_ACCEPTED_EXISTING"
        if result["overall_pass"]
        else "R4_EXISTING_CHECKPOINT_FAIL"
    )
    evaluation = {
        "schema": "tactile3d-unit.s4-2r-contact-state-re-evaluation.v1",
        "state": state,
        "classification": classification,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": expected_checkpoint_sha,
        "candidate": checkpoint["candidate"],
        "normalization": normalizer.to_json(),
        "original_s4_2_decision": original_decision["decision"],
        "original_effective_rank": original_decision["contact_state"]["effective_rank"],
        "original_threshold": original_decision["contact_state"]["effective_rank_min"],
        "r4_validation_effective_rank": latent_metrics["effective_rank"],
        "latent_metrics": latent_metrics,
        "pairwise_diversity": pairwise,
        "validation_future_mse": float(correct_error.mean()),
        "dynamic_q70_threshold_newton": dynamic_threshold,
        "dynamic_validation_samples": int(dynamic.sum()),
        "perturbation_controls": perturbation,
        "new_contract": result,
        "warnings": warnings,
        "new_training": False,
        "test_loaded": False,
        "selection_uses_test": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "contact_state_re_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["overall_pass"]:
        ACCEPTED_ROOT.mkdir(parents=True, exist_ok=True)
        accepted_checkpoint = ACCEPTED_ROOT / "accepted.pt"
        shutil.copyfile(CHECKPOINT, accepted_checkpoint)
        if sha256_file(accepted_checkpoint) != expected_checkpoint_sha:
            raise RuntimeError("promoted checkpoint hash mismatch")
        normalization_path = ACCEPTED_ROOT / "normalization.json"
        normalization_path.write_text(
            json.dumps(normalizer.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        acceptance = {
            "schema": "tactile3d-unit.s4-2r-contact-state-acceptance.v1",
            "final_state": "S4_2R_CONTACT_STATE_ACCEPTED_EXISTING",
            "classification": classification,
            "checkpoint": str(accepted_checkpoint.relative_to(ROOT)),
            "source_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": expected_checkpoint_sha,
            "normalization": str(normalization_path.relative_to(ROOT)),
            "normalization_sha256": hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
            "architecture": "B3 PredictiveContactTeacher",
            "parameters": original["parameters"],
            "original_effective_rank": original_decision["contact_state"][
                "effective_rank"
            ],
            "new_effective_rank": latent_metrics["effective_rank"],
            "source_intrinsic_dimension": protocol["r1_source_dimension_reference"],
            "all_gates": result["gates"],
            "warnings": warnings,
            "remediation_required": False,
            "new_trials": 0,
            "test_loaded": False,
            "selection_uses_test": False,
            "s4_2_3_allowed": True,
        }
        (ARTIFACTS / "contact_state_acceptance.json").write_text(
            json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Original report", "R4 existing checkpoint"]
    values = [
        original_decision["contact_state"]["effective_rank"],
        latent_metrics["effective_rank"],
    ]
    ax.bar(labels, values, color=["#c44e52", "#4c72b0"])
    ax.axhline(16.0, color="#c44e52", linestyle="--", label="original threshold 16")
    ax.axhline(8.0, color="#55a868", linestyle=":", label="new structural floor 8")
    ax.set(ylabel="Effective rank", title="Original failure preserved; existing checkpoint audited")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "original_vs_rank_audited.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "state": state,
                "classification": classification,
                "overall_pass": result["overall_pass"],
                "gates": {name: row["pass"] for name, row in result["gates"].items()},
                "new_trials": 0,
                "test_loaded": False,
            },
            indent=2,
        )
    )
    if not result["overall_pass"]:
        raise SystemExit("RANK_REMEDIATION_REQUIRED")


if __name__ == "__main__":
    main()
