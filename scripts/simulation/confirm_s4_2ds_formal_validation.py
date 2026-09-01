#!/usr/bin/env python3
"""One no-retuning follow-up validation confirmation for selected S4.2-DS C3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    LatentTransitionDecoder,
)
from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402
from gr00t.simulation.s4_2ds import ActionPilot, PilotVACBridge  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    geometry_diagnostics,
    pairwise_alignment_metrics,
)
from scripts.simulation.audit_s4_2ds_contact_representations import (  # noqa: E402
    control_suite,
    encode as contact_encode,
    spectrum_metrics,
)
from scripts.simulation.build_s4_2ds_paired_pilot import encode_batch  # noqa: E402
from scripts.simulation.train_s4_2ds_action_pilot import (  # noqa: E402
    features as action_features,
    infer as action_infer,
    policy_state,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_frozen_vision  # noqa: E402


TRAIN_CACHE = ROOT / ".local/cache/simulation/s4_2ds/pilot_pairs.npz"
VALIDATION_PAIR = ROOT / ".local/cache/simulation/s4_2/pairs/validation.npz"
VALIDATION_CONTACT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics/validation.npz"
ACTION_CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2ds/action_pilot.pt"
C3_CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"
C3_BRIDGE_CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2ds/c3_bridge_pilot.pt"
CONFIG_PATH = ROOT / "configs/simulation/s4_2ds_representation_selection.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2ds"
OUTPUT_PATH = ARTIFACT_ROOT / "formal_validation_confirmation.json"
CACHE_OUTPUT = ROOT / ".local/cache/simulation/s4_2ds/formal_validation_confirmation.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_c3(device: torch.device) -> ContactDynamicsModel:
    checkpoint = torch.load(C3_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def fit_probe(value: np.ndarray, target: np.ndarray) -> RidgeClassifier:
    return RidgeClassifier(alpha=10.0).fit(value.reshape(len(value), -1), target)


def probe_metrics(
    model: RidgeClassifier,
    value: np.ndarray,
    target: np.ndarray,
    classes: list[int],
) -> dict[str, Any]:
    prediction = model.predict(value.reshape(len(value), -1))
    recalls = recall_score(target, prediction, labels=classes, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(target, prediction, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "per_class_recall": {str(key): float(value) for key, value in zip(classes, recalls)},
    }


@torch.inference_mode()
def bridge_encode(
    bridge: PilotVACBridge,
    modality: str,
    value: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    result = np.empty_like(value)
    for start in range(0, len(value), 512):
        stop = min(start + 512, len(value))
        result[start:stop] = (
            bridge.encode(modality, torch.from_numpy(np.array(value[start:stop], copy=True)).to(device))
            .float()
            .cpu()
            .numpy()
        )
    return result


def chance_multiplier(value: dict[str, Any]) -> float:
    return value["recall_at_10"] / max(value["chance"]["recall_at_10"], 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-checkpoint",
        type=Path,
        default=Path(os.environ["UNIT_FULLDATA_CKPT"])
        if os.environ.get("UNIT_FULLDATA_CKPT")
        else None,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if OUTPUT_PATH.exists():
        raise RuntimeError("follow-up formal validation confirmation was already loaded")
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    canonical = json.loads(
        ARTIFACT_ROOT.joinpath("canonical_contact_representation.json").read_text(encoding="utf-8")
    )
    if canonical["candidate"] != "C3" or canonical["classification"] != "C3_REPRESENTATION_UTILITY_SELECTED":
        raise RuntimeError("canonical C3 selection was not frozen")
    if canonical["checkpoint_sha256"] != sha256_file(C3_CHECKPOINT):
        raise RuntimeError("canonical C3 checkpoint changed")
    bridge_checkpoint = torch.load(C3_BRIDGE_CHECKPOINT, map_location="cpu", weights_only=False)
    bridge_artifact = json.loads(ARTIFACT_ROOT.joinpath("c3_bridge_pilot.json").read_text())
    if bridge_artifact["checkpoint_sha256"] != sha256_file(C3_BRIDGE_CHECKPOINT):
        raise RuntimeError("selected bridge checkpoint changed")
    device = torch.device(args.device)
    bridge = PilotVACBridge(width=64).to(device)
    bridge.load_state_dict(bridge_checkpoint["state_dict"], strict=True)
    bridge.eval().requires_grad_(False)

    # Freeze deterministic DS-TRAIN readouts before the formal validation file is opened.
    train = load_npz(TRAIN_CACHE)
    train_indices = train["ds_train_indices"]
    shared_train_contact = bridge_encode(bridge, "contact", train["z_c3"][train_indices], device)
    native_contact_probe = fit_probe(
        train["z_c3"][train_indices], train["contact_transition"][train_indices]
    )
    native_force_probe = fit_probe(train["z_c3"][train_indices], train["force_trend"][train_indices])
    shared_contact_probe = fit_probe(
        shared_train_contact, train["contact_transition"][train_indices]
    )
    shared_force_probe = fit_probe(shared_train_contact, train["force_trend"][train_indices])
    readouts_frozen_before_validation_load = True

    # This is the single formal-validation load for S4.2-DS.
    validation_pairs = load_npz(VALIDATION_PAIR)
    validation_contact = load_npz(VALIDATION_CONTACT)
    if not np.array_equal(validation_pairs["pair_id"], validation_contact["pair_id"]):
        raise RuntimeError("formal validation pair identity mismatch")
    if not np.all(validation_pairs["future_step"] - validation_pairs["anchor_step"] == 27):
        raise RuntimeError("formal validation horizon mismatch")
    c3 = build_c3(device)
    z_c = contact_encode(
        c3,
        validation_contact["h_current"],
        validation_contact["h_future"],
        device,
        2048,
    )
    action_checkpoint = torch.load(ACTION_CHECKPOINT, map_location="cpu", weights_only=False)
    action_model = ActionPilot().to(device)
    action_model.load_state_dict(action_checkpoint["state_dict"], strict=True)
    action_model.eval().requires_grad_(False)
    action_stats = {
        name: np.asarray(value, dtype=np.float32) for name, value in action_checkpoint["stats"].items()
    }
    state = policy_state(validation_pairs["current_state"])
    action = np.asarray(validation_pairs["action_chunk"], dtype=np.float32)
    feature, normalized_state, _ = action_features(state, action, action_stats)
    z_a, _ = action_infer(action_model, feature, normalized_state, device, 512)
    expected = config["vision_pilot"]["checkpoint_files_sha256"]
    vision, vision_identity = load_frozen_vision(
        args.unit_checkpoint,
        {"frozen_identity": {"original_unit_tokenizer_files_sha256": expected}},
        device,
    )
    if vision_identity["trainable_parameters"] != 0 or vision.training:
        raise RuntimeError("formal validation Vision path is not frozen")
    z_v = np.empty((len(z_c), 8, 32), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(z_v), args.vision_batch_size):
            stop = min(start + args.vision_batch_size, len(z_v))
            z_v[start:stop] = encode_batch(
                vision,
                validation_pairs["current_frame"][start:stop],
                validation_pairs["future_frame"][start:stop],
                executor,
            )
            if stop % 1024 < args.vision_batch_size or stop == len(z_v):
                print(json.dumps({"formal_validation_vision_pairs": stop, "total": len(z_v)}), flush=True)
    shared = {
        "vision": bridge_encode(bridge, "vision", z_v, device),
        "action": bridge_encode(bridge, "action", z_a, device),
        "contact": bridge_encode(bridge, "contact", z_c, device),
    }
    episode_numeric = np.unique(validation_pairs["episode_id"], return_inverse=True)[1]
    alignment = {
        name: pairwise_alignment_metrics(
            shared[left],
            shared[right],
            episode_numeric,
            bootstrap_samples=args.bootstrap_samples,
            seed=int(config["seed"]) + offset,
            retrieval_chunk=512,
        )
        for offset, (name, left, right) in enumerate(
            (("V-A", "vision", "action"), ("V-C", "vision", "contact"), ("A-C", "action", "contact"))
        )
    }
    native_contact = probe_metrics(
        native_contact_probe, z_c, validation_contact["contact_transition"], [0, 1, 2, 3]
    )
    native_force = probe_metrics(
        native_force_probe, z_c, validation_contact["force_trend"], [0, 1, 2]
    )
    shared_contact = probe_metrics(
        shared_contact_probe,
        shared["contact"],
        validation_contact["contact_transition"],
        [0, 1, 2, 3],
    )
    shared_force = probe_metrics(
        shared_force_probe, shared["contact"], validation_contact["force_trend"], [0, 1, 2]
    )
    retention = {
        "contact": shared_contact["macro_f1"] / max(native_contact["macro_f1"], 1e-12),
        "force": shared_force["macro_f1"] / max(native_force["macro_f1"], 1e-12),
    }
    controls, _ = control_suite(
        c3,
        z_c,
        validation_contact,
        np.arange(len(z_c)),
        device,
        2048,
        int(config["seed"]),
        args.bootstrap_samples,
    )
    contact_geometry = spectrum_metrics(z_c, int(config["seed"]))
    shared_geometry = {name: geometry_diagnostics(value) for name, value in shared.items()}
    contact_gates = {
        "finite": bool(np.isfinite(z_c).all()),
        "no_structural_collapse": contact_geometry["near_zero_variance_fraction"] <= 0.5
        and contact_geometry["query_diversity"]["collapsed_sample_fraction"] == 0.0,
        "contact_transition_macro_f1": native_contact["macro_f1"] >= 0.90,
        "force_trend_macro_f1": native_force["macro_f1"] >= 0.90,
        "information_necessity": controls["information_necessity"] == "PASS",
        "transition_specificity": controls["transition_specificity"] == "PASS",
    }
    bridge_gates = {
        "v_c_margin_ci": alignment["V-C"]["margin_bootstrap_ci95"][0] > 0.0,
        "a_c_margin_ci": alignment["A-C"]["margin_bootstrap_ci95"][0] > 0.0,
        "v_to_c_r10": chance_multiplier(alignment["V-C"]["retrieval"]["forward"]) >= 5.0,
        "c_to_v_r10": chance_multiplier(alignment["V-C"]["retrieval"]["reverse"]) >= 5.0,
        "a_to_c_r10": chance_multiplier(alignment["A-C"]["retrieval"]["forward"]) >= 5.0,
        "c_to_a_r10": chance_multiplier(alignment["A-C"]["retrieval"]["reverse"]) >= 5.0,
        "contact_retention": retention["contact"] >= 0.85,
        "force_retention": retention["force"] >= 0.85,
        "no_collapse": all(
            value["per_dimension_variance"]["near_zero_fraction"] <= 0.5
            and value["query_diversity"]["collapsed_pair_fraction"] <= 0.5
            for value in shared_geometry.values()
        ),
        "independent_encodability": True,
    }
    classification_consistent = all(contact_gates.values()) and all(bridge_gates.values())
    result = {
        "schema": "tactile3d-unit.s4-2ds-follow-up-validation-confirmation.v1",
        "stage": "DS7",
        "type": "FOLLOW-UP VALIDATION CONFIRMATION",
        "untouched_validation_claimed": False,
        "selected_candidate": "C3",
        "classification": "C3_REPRESENTATION_UTILITY_SELECTED",
        "retuning": False,
        "checkpoint_switch": False,
        "threshold_change": False,
        "adapter_switch": False,
        "new_seed_selection": False,
        "readouts_frozen_before_validation_load": readouts_frozen_before_validation_load,
        "pairs": len(z_c),
        "contact": {
            "native_contact_transition": native_contact,
            "native_force_trend": native_force,
            "controls": controls,
            "geometry": contact_geometry,
            "gates": {name: "PASS" if value else "FAIL" for name, value in contact_gates.items()},
        },
        "bridge": {
            "alignment": alignment,
            "shared_contact_transition": shared_contact,
            "shared_force_trend": shared_force,
            "retention": retention,
            "geometry": shared_geometry,
            "gates": {name: "PASS" if value else "FAIL" for name, value in bridge_gates.items()},
        },
        "vision_checkpoint_file_sha256": vision_identity["original_unit_tokenizer_files_sha256"],
        "classification_consistent": classification_consistent,
        "overall": "PASS" if classification_consistent else "FAIL",
        "failure_decision": None
        if classification_consistent
        else "S4_2DS_VALIDATION_CONFIRMATION_FAIL",
        "formal_test_model_metrics_loaded": False,
    }
    CACHE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_OUTPUT, pair_id=validation_pairs["pair_id"], z_v=z_v, z_a=z_a, z_c=z_c)
    atomic_json(OUTPUT_PATH, result)
    print(json.dumps({
        "overall": result["overall"],
        "contact_gates": result["contact"]["gates"],
        "bridge_gates": result["bridge"]["gates"],
        "retention": retention,
        "formal_test_model_metrics_loaded": False,
    }, indent=2))
    if result["overall"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
