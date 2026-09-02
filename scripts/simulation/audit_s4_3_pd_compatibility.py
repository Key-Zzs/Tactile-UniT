#!/usr/bin/env python3
"""Run shape/finite compatibility through the frozen S4.2 stack without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_formal import (  # noqa: E402
    ConditionalContactPredictor,
    FormalActionEncoder,
    FormalVACBridge,
)
from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint  # noqa: E402
from gr00t.simulation.s4_3_pd import TASKS, sha256_file  # noqa: E402
from scripts.simulation.audit_s4_3_pd_pilot import write_json  # noqa: E402
from scripts.simulation.s4_2dr_common import load_model, predict as predict_c3  # noqa: E402
from scripts.simulation.train_s4_2ds_action_pilot import (  # noqa: E402
    features as action_features,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pd"
DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_3_policy_expert"
CHECKPOINTS = {
    "contact_state": ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "contact_C3": ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "action_A0": ROOT / ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "bridge_B3": ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "conditional_A_plus_H": ROOT
    / ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_action(device: torch.device) -> tuple[FormalActionEncoder, dict[str, np.ndarray]]:
    payload = torch.load(CHECKPOINTS["action_A0"], map_location="cpu", weights_only=False)
    if payload["candidate"] != "A0":
        raise RuntimeError("frozen Action representation is not A0")
    model = FormalActionEncoder(str(payload["candidate"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    stats = {name: np.asarray(value, dtype=np.float32) for name, value in payload["stats"].items()}
    return model.eval().requires_grad_(False).to(device), stats


def load_bridge(device: torch.device) -> FormalVACBridge:
    payload = torch.load(CHECKPOINTS["bridge_B3"], map_location="cpu", weights_only=False)
    model = FormalVACBridge(str(payload["adapter"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def load_conditional(device: torch.device) -> ConditionalContactPredictor:
    payload = torch.load(
        CHECKPOINTS["conditional_A_plus_H"], map_location="cpu", weights_only=False
    )
    if tuple(payload["modalities"]) != ("action", "history"):
        raise RuntimeError("frozen conditional checkpoint is not causal A+H")
    model = ConditionalContactPredictor(tuple(payload["modalities"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def main() -> None:
    device = torch.device("cpu")
    starting = read_json(ARTIFACT_ROOT / "starting_integrity.json")
    expected_hashes = starting["immutable_checkpoint_hashes_before"]
    actual_hashes = {name: sha256_file(path) for name, path in CHECKPOINTS.items()}
    if actual_hashes != {name: expected_hashes[name] for name in CHECKPOINTS}:
        raise SystemExit("frozen S4.2 checkpoint hash mismatch before compatibility audit")

    dataset = read_json(ARTIFACT_ROOT / "policy_expert_dataset_manifest.json")
    teacher, teacher_payload = load_teacher_checkpoint(CHECKPOINTS["contact_state"], "cpu")
    teacher = teacher.eval().requires_grad_(False).to(device)
    normalizer = TactileNormalization.from_json(teacher_payload["normalization"])
    action_model, action_stats = load_action(device)
    c3_model, c3_payload = load_model(CHECKPOINTS["contact_C3"], device)
    c3_model = c3_model.eval().requires_grad_(False)
    if c3_payload["model"] != "C3":
        raise RuntimeError("frozen Contact dynamics checkpoint is not C3")
    bridge = load_bridge(device)
    conditional = load_conditional(device)

    results = {}
    for task in TASKS:
        sample = next(row for row in dataset["successful_train"] if row["task"] == task)
        attempt_dir = DATASET_ROOT / sample["attempt_relative_path"]
        with np.load(attempt_dir / "steps.npz", allow_pickle=False) as loaded:
            tactile = loaded["sim_tactile"].astype(np.float32)
            proprio = loaded["proprio"].astype(np.float32)
            action = loaded["policy_action"].astype(np.float32)
        anchor = 25
        current_history = tactile[anchor - 25 : anchor + 1][None]
        future_step = anchor + 27
        future_history = tactile[anchor + 2 : future_step + 1][None]
        action_chunk = action[anchor:future_step][None]
        current_state = proprio[anchor][None]
        if (
            current_history.shape != (1, 26, 30)
            or future_history.shape != (1, 26, 30)
            or action_chunk.shape != (1, 27, 22)
            or current_state.shape != (1, 22)
        ):
            raise RuntimeError(f"{task} compatibility sample violates the frozen input schema")

        current_normalized = normalizer.transform(current_history).astype(np.float32)
        future_normalized = normalizer.transform(future_history).astype(np.float32)
        h_current = teacher(torch.from_numpy(current_normalized).to(device))["latent"]
        h_future = teacher(torch.from_numpy(future_normalized).to(device))["latent"]
        h_current_np = h_current.float().cpu().numpy()
        h_future_np = h_future.float().cpu().numpy()
        _, c3_code = predict_c3(
            c3_model,
            "C3",
            h_current_np,
            h_future_np,
            device,
            batch_size=1,
        )
        assert c3_code is not None

        action_feature, normalized_state, _ = action_features(
            current_state, action_chunk, action_stats
        )
        action_output = action_model(
            torch.from_numpy(action_feature).to(device),
            torch.from_numpy(normalized_state).to(device),
        )
        action_code = action_output["code"]
        shared_action = bridge.encode("action", action_code)
        conditional_output = conditional(
            {
                "action": shared_action,
                "history": h_current.reshape(1, 8, 32),
            }
        )
        arrays = {
            "contact_state": h_current_np,
            "action_A0": action_code.float().cpu().numpy(),
            "contact_C3": c3_code,
            "bridge_B3_action": shared_action.float().cpu().numpy(),
            "conditional_A_plus_H": conditional_output.float().cpu().numpy(),
        }
        shapes = {name: list(value.shape[1:]) for name, value in arrays.items()}
        expected_shapes = {
            "contact_state": [256],
            "action_A0": [8, 32],
            "contact_C3": [8, 32],
            "bridge_B3_action": [8, 32],
            "conditional_A_plus_H": [8, 32],
        }
        finite = {name: bool(np.isfinite(value).all()) for name, value in arrays.items()}
        passed = shapes == expected_shapes and all(finite.values())
        results[task] = {
            "sample_attempt_id": sample["attempt_id"],
            "selection_role": "non-selection compatibility sample",
            "anchor_step": anchor,
            "future_step": future_step,
            "tactile_history_shape": [26, 30],
            "action_chunk_shape": [27, 22],
            "output_shapes": shapes,
            "finite": finite,
            "same_transition_contact": True,
            "status": "PASS" if passed else "FAIL",
        }

    hashes_after = {name: sha256_file(path) for name, path in CHECKPOINTS.items()}
    status = "PASS" if all(row["status"] == "PASS" for row in results.values()) and hashes_after == actual_hashes else "FAIL"
    artifact = {
        "schema": "tactile3d-unit.s4-3-pd-s4-2-compatibility.v1",
        "stage": "PD8",
        "tasks": results,
        "checkpoint_sha256_before": actual_hashes,
        "checkpoint_sha256_after": hashes_after,
        "checkpoints_unchanged_during_audit": hashes_after == actual_hashes,
        "s4_2_retraining": False,
        "sample_used_for_selection": False,
        "status": status,
    }
    write_json(ARTIFACT_ROOT / "s4_2_compatibility.json", artifact)

    import matplotlib.pyplot as plt

    labels = ["Contact-State", "A0 Action", "C3", "A+H"]
    matrix = np.ones((len(TASKS), len(labels))) if status == "PASS" else np.zeros((len(TASKS), len(labels)))
    fig, axis = plt.subplots(figsize=(8, 4))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(TASKS)), TASKS)
    for row in range(len(TASKS)):
        for column in range(len(labels)):
            axis.text(column, row, "PASS" if matrix[row, column] else "FAIL", ha="center", va="center")
    axis.set_title("Frozen S4.2 compatibility on successful policy demos")
    fig.colorbar(image, ax=axis, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(ARTIFACT_ROOT / "plots/18_s4_2_compatibility_matrix.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"stage": "PD8", "status": status}, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
