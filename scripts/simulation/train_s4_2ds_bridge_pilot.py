#!/usr/bin/env python3
"""Train equal-budget C2/C3 S4.2-DS VAC bridge pilots on DS-TRAIN."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)
from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402
from gr00t.simulation.s4_2ds import PilotVACBridge  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    different_episode_info_nce,
    geometry_diagnostics,
    pairwise_alignment_metrics,
    relational_preservation,
    variance_floor,
)
from scripts.simulation.train_s4_2ds_action_pilot import (  # noqa: E402
    features as action_features,
    infer as action_infer,
    policy_state,
)


CACHE_PATH = ROOT / ".local/cache/simulation/s4_2ds/pilot_pairs.npz"
RAW_PAIR_PATH = ROOT / ".local/cache/simulation/s4_2/pairs/train.npz"
ACTION_CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2ds/action_pilot.pt"
C2_CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/C2/best.pt"
C3_CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"
CONFIG_PATH = ROOT / "configs/simulation/s4_2ds_representation_selection.json"
PROTOCOL_PATH = ROOT / ".local/artifacts/simulation/s4_2ds/protocol_freeze.json"
CONTACT_UTILITY_PATH = ROOT / ".local/artifacts/simulation/s4_2ds/contact_representation_utility.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2ds"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2ds"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_ids(task: np.ndarray, source: np.ndarray) -> np.ndarray:
    groups = sorted({(str(t), str(s)) for t, s in zip(task, source)})
    mapping = {group: index for index, group in enumerate(groups)}
    return np.asarray([mapping[(str(t), str(s))] for t, s in zip(task, source)], dtype=np.int64)


def covariance_penalty(shared: torch.Tensor) -> torch.Tensor:
    flat = shared.flatten(1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(flat) - 1, 1)
    mask = ~torch.eye(covariance.shape[0], dtype=torch.bool, device=covariance.device)
    return covariance[mask].square().mean()


def bridge_loss(
    model: PilotVACBridge,
    native: dict[str, torch.Tensor],
    source_group: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    shared = {name: model.encode(name, native[name]) for name in model.modalities}
    temperature = float(config["training"]["temperature"])
    alignment_terms = []
    for left, right in (("vision", "action"), ("vision", "contact"), ("action", "contact")):
        alignment_terms.append(
            different_episode_info_nce(shared[left], shared[right], source_group, temperature=temperature)
        )
        alignment_terms.append(
            different_episode_info_nce(shared[right], shared[left], source_group, temperature=temperature)
        )
    alignment = torch.stack(alignment_terms).mean()
    relational = torch.stack(
        [relational_preservation(native[name], shared[name]) for name in model.modalities]
    ).mean()
    recovery = torch.stack(
        [F.mse_loss(model.recover(name, shared[name]), native[name]) for name in model.modalities]
    ).mean()
    variance_covariance = torch.stack(
        [variance_floor(shared[name]) + 0.01 * covariance_penalty(shared[name]) for name in model.modalities]
    ).mean()
    weights = config["loss_weights"]
    total = (
        alignment
        + float(weights["relational_preservation"]) * relational
        + float(weights["variance_covariance_no_collapse"]) * variance_covariance
        + float(weights["modality_recovery"]) * recovery
    )
    return total, {
        "total": total.detach(),
        "alignment": alignment.detach(),
        "relational": relational.detach(),
        "variance_covariance": variance_covariance.detach(),
        "recovery": recovery.detach(),
    }


@torch.inference_mode()
def encode_bridge(
    model: PilotVACBridge,
    arrays: dict[str, np.ndarray],
    contact_key: str,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    source = {"vision": "z_v", "action": "z_a", "contact": contact_key}
    shared = {name: np.empty((len(indices), 8, 32), dtype=np.float32) for name in model.modalities}
    recovered = {name: np.empty_like(shared[name]) for name in model.modalities}
    model.eval()
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        rows = indices[start:stop]
        for name in model.modalities:
            native = torch.from_numpy(np.array(arrays[source[name]][rows], copy=True)).to(device)
            value = model.encode(name, native)
            shared[name][start:stop] = value.float().cpu().numpy()
            recovered[name][start:stop] = model.recover(name, value).float().cpu().numpy()
    return shared, recovered


def classification_probe(
    train_x: np.ndarray,
    dev_x: np.ndarray,
    train_y: np.ndarray,
    dev_y: np.ndarray,
    classes: list[int],
) -> dict[str, Any]:
    model = RidgeClassifier(alpha=10.0).fit(train_x.reshape(len(train_x), -1), train_y)
    prediction = model.predict(dev_x.reshape(len(dev_x), -1))
    recalls = recall_score(dev_y, prediction, labels=classes, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(dev_y, prediction, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(dev_y, prediction)),
        "per_class_recall": {str(key): float(value) for key, value in zip(classes, recalls)},
    }


def load_dynamics(name: str, device: torch.device) -> ContactDynamicsModel:
    path = C2_CHECKPOINT if name == "C2" else C3_CHECKPOINT
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = DeltaMLPEncoder() if name == "C2" else ContactDynamicsEncoder()
    model = ContactDynamicsModel(encoder, LatentTransitionDecoder())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def contact_recovery(
    model: ContactDynamicsModel,
    native: np.ndarray,
    recovered: np.ndarray,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    predictions = {"native": np.empty_like(future), "bridge_recovered": np.empty_like(future)}
    for start in range(0, len(current), 512):
        stop = min(start + 512, len(current))
        current_t = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
        for name, value in (("native", native), ("bridge_recovered", recovered)):
            code_t = torch.from_numpy(np.array(value[start:stop], copy=True)).to(device)
            predictions[name][start:stop] = model.decoder(code_t, current_t).float().cpu().numpy()
    errors = {
        name: np.square(value.astype(np.float64) - future.astype(np.float64)).mean(axis=1)
        for name, value in predictions.items()
    }
    return {
        name: {"future_mse": float(value.mean()), "per_sample": value}
        for name, value in errors.items()
    }


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        stop = min(samples, start + 100)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[index].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).astype(float).tolist()


def action_temporal_retention(
    bridge: PilotVACBridge,
    arrays: dict[str, np.ndarray],
    dev_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    from gr00t.contact_dynamics.evaluation import different_episode_permutation
    from gr00t.simulation.s4_2ds import ActionPilot

    raw = load_npz(RAW_PAIR_PATH)
    checkpoint = torch.load(ACTION_CHECKPOINT, map_location="cpu", weights_only=False)
    action_model = ActionPilot().to(device)
    action_model.load_state_dict(checkpoint["state_dict"], strict=True)
    action_model.eval().requires_grad_(False)
    stats = {name: np.asarray(value, dtype=np.float32) for name, value in checkpoint["stats"].items()}
    state = policy_state(raw["current_state"])[dev_indices]
    action = np.asarray(raw["action_chunk"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(dev_indices)
    different = dev_indices[
        different_episode_permutation(arrays["episode_id"][dev_indices], seed=seed)
    ]
    controls = {
        "correct": action[dev_indices],
        "reversed": action[dev_indices, ::-1].copy(),
        "shuffled": action[shuffled],
        "different_episode": action[different],
    }
    shared = {}
    with torch.inference_mode():
        for name, value in controls.items():
            feature, normalized_state, _ = action_features(state, value, stats)
            code, _ = action_infer(action_model, feature, normalized_state, device, 512)
            output = []
            for start in range(0, len(code), 512):
                output.append(
                    bridge.encode("action", torch.from_numpy(code[start : start + 512]).to(device))
                    .float()
                    .cpu()
                    .numpy()
                )
            shared[name] = np.concatenate(output)
    correct = shared["correct"].reshape(len(dev_indices), -1).astype(np.float64)
    result = {}
    for name, value in shared.items():
        flat = value.reshape(len(dev_indices), -1).astype(np.float64)
        cosine = np.sum(correct * flat, axis=1) / np.maximum(
            np.linalg.norm(correct, axis=1) * np.linalg.norm(flat, axis=1), 1e-12
        )
        result[name] = {
            "cosine_to_correct": float(cosine.mean()),
            "l2_to_correct": float(np.linalg.norm(correct - flat, axis=1).mean()),
        }
    return result


def chance_multiplier(retrieval: dict[str, Any]) -> float:
    return retrieval["recall_at_10"] / max(retrieval["chance"]["recall_at_10"], 1e-12)


def train_candidate(
    candidate: str,
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    dev_indices: np.ndarray,
    source_group: np.ndarray,
    config: dict[str, Any],
    contact_utility: dict[str, Any],
    device: torch.device,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seed = int(config["seed"])
    seed_everything(seed)
    model = PilotVACBridge(width=64).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    bridge_config = config["bridge_pilot"]
    if parameters > int(bridge_config["architecture"]["total_trainable_parameters_max"]):
        raise RuntimeError("VAC pilot bridge exceeds parameter budget")
    training = bridge_config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    contact_key = "z_c2" if candidate == "C2" else "z_c3"
    source_names = {"vision": "z_v", "action": "z_a", "contact": contact_key}
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    for step in range(1, int(training["steps"]) + 1):
        rows = rng.choice(train_indices, size=int(training["batch_size"]), replace=False)
        native = {
            name: torch.from_numpy(np.array(arrays[key][rows], copy=True)).to(device)
            for name, key in source_names.items()
        }
        group_t = torch.from_numpy(source_group[rows]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, breakdown = bridge_loss(model, native, group_t, bridge_config)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite bridge loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(training["steps"]):
            row = {"step": step, **{name: float(value) for name, value in breakdown.items()}}
            history.append(row)
            print(json.dumps({"candidate": candidate, **row}), flush=True)
    checkpoint_path = EXPERIMENT_ROOT / f"{candidate.lower()}_bridge_pilot.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2ds-vac-bridge-pilot.v1",
            "contact_candidate": candidate,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "parameters": parameters,
            "seed": seed,
            "steps": int(training["steps"]),
            "formal_test_loaded": False,
        },
        checkpoint_path,
    )
    shared_train, _ = encode_bridge(model, arrays, contact_key, train_indices, device)
    shared_dev, recovered_dev = encode_bridge(model, arrays, contact_key, dev_indices, device)
    episode_numeric = np.unique(arrays["episode_id"], return_inverse=True)[1][dev_indices]
    alignment = {
        name: pairwise_alignment_metrics(
            shared_dev[left],
            shared_dev[right],
            episode_numeric,
            bootstrap_samples=bootstrap_samples,
            seed=seed + offset,
            retrieval_chunk=512,
        )
        for offset, (name, left, right) in enumerate(
            (("V-A", "vision", "action"), ("V-C", "vision", "contact"), ("A-C", "action", "contact"))
        )
    }
    native_contact = arrays[contact_key]
    contact_native_metrics = contact_utility["candidates"][candidate]["probes"]
    shared_contact_metrics = {
        "contact_transition": classification_probe(
            shared_train["contact"],
            shared_dev["contact"],
            arrays["contact_transition"][train_indices],
            arrays["contact_transition"][dev_indices],
            [0, 1, 2, 3],
        ),
        "force_trend": classification_probe(
            shared_train["contact"],
            shared_dev["contact"],
            arrays["force_trend"][train_indices],
            arrays["force_trend"][dev_indices],
            [0, 1, 2],
        ),
    }
    retention = {
        "contact": shared_contact_metrics["contact_transition"]["macro_f1"]
        / contact_native_metrics["contact_transition"]["macro_f1"],
        "force": shared_contact_metrics["force_trend"]["macro_f1"]
        / contact_native_metrics["force_trend"]["macro_f1"],
    }
    dynamics = load_dynamics(candidate, device)
    recovery = contact_recovery(
        dynamics,
        native_contact[dev_indices],
        recovered_dev["contact"],
        arrays["h_current"][dev_indices],
        arrays["h_future"][dev_indices],
        device,
    )
    geometry = {name: geometry_diagnostics(value) for name, value in shared_dev.items()}
    action_temporal = action_temporal_retention(model, arrays, dev_indices, device, seed)
    gates = {
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
            for value in geometry.values()
        ),
        "independent_encodability": True,
    }
    result = {
        "schema": "tactile3d-unit.s4-2ds-vac-bridge-pilot-evaluation.v1",
        "stage": "DS5",
        "contact_candidate": candidate,
        "role": "bridge pilot; not formal S4.2-5",
        "parameters": parameters,
        "candidate_specific_parameter_difference": 0,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training": {"seed": seed, "steps": int(training["steps"]), "history": history},
        "alignment": alignment,
        "geometry": geometry,
        "contact_semantics": {
            "native": {
                "contact_transition": contact_native_metrics["contact_transition"],
                "force_trend": contact_native_metrics["force_trend"],
            },
            "shared": shared_contact_metrics,
            "retention": retention,
        },
        "contact_recovery": {
            name: {"future_mse": value["future_mse"]} for name, value in recovery.items()
        },
        "action_temporal_retention": action_temporal,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "independent_encoder_api": "u_m=f_m(z_m); no counterpart argument",
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / f"{candidate.lower()}_bridge_pilot.json", result)
    auxiliary = {
        "contact_recovery_error": recovery["bridge_recovered"]["per_sample"],
        "shared_contact": shared_dev["contact"],
    }
    return result, auxiliary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("S4.2-DS protocol changed after freeze")
    if json.loads(ARTIFACT_ROOT.joinpath("vision_pilot.json").read_text())["overall"] != "PASS":
        raise RuntimeError("Vision pilot did not pass")
    contact_utility = json.loads(CONTACT_UTILITY_PATH.read_text(encoding="utf-8"))
    arrays = load_npz(CACHE_PATH)
    train_indices = arrays["ds_train_indices"]
    dev_indices = arrays["ds_dev_indices"]
    source_group = group_ids(arrays["task"], arrays["source_trajectory_id"])
    device = torch.device(args.device)
    results = {}
    auxiliary = {}
    for candidate in ("C2", "C3"):
        results[candidate], auxiliary[candidate] = train_candidate(
            candidate,
            arrays,
            train_indices,
            dev_indices,
            source_group,
            config,
            contact_utility,
            device,
            args.bootstrap_samples,
        )
    def contact_retrieval_means(result: dict[str, Any]) -> tuple[float, float]:
        retrievals = [
            result[pair]["retrieval"][direction]
            for pair in ("V-C", "A-C")
            for direction in ("forward", "reverse")
        ]
        return float(np.mean([value["recall_at_10"] for value in retrievals])), float(
            np.mean([value["mrr"] for value in retrievals])
        )
    c2_r10, c2_mrr = contact_retrieval_means(results["C2"]["alignment"])
    c3_r10, c3_mrr = contact_retrieval_means(results["C3"]["alignment"])
    utility_comparison = contact_utility["paired_comparison"]
    semantic_advantage = bool(
        (
            utility_comparison["contact_macro_f1_c3_minus_c2"] >= 0.01
            and utility_comparison["contact_macro_f1_difference_ci95"][0] > 0.0
        )
        or (
            utility_comparison["force_macro_f1_c3_minus_c2"] >= 0.01
            and utility_comparison["force_macro_f1_difference_ci95"][0] > 0.0
        )
    )
    bridge_advantage = bool(
        c3_r10 >= 1.20 * c2_r10 or c3_mrr >= 1.20 * c2_mrr
    )
    reconstruction_advantage = bool(
        utility_comparison["future_recovery_mse_c3_relative_improvement_over_c2"] >= 0.05
        and utility_comparison["future_recovery_error_c2_minus_c3_ci95"][0] > 0.0
    )
    comparison = {
        "schema": "tactile3d-unit.s4-2ds-candidate-comparison.v1",
        "same_architecture": True,
        "same_parameter_budget": True,
        "same_train_steps": True,
        "same_seed": True,
        "same_samples": True,
        "contact_semantic_advantage_C3": semantic_advantage,
        "contact_bridge_mean_r10": {"C2": c2_r10, "C3": c3_r10, "C3_over_C2": c3_r10 / max(c2_r10, 1e-12)},
        "contact_bridge_mean_mrr": {"C2": c2_mrr, "C3": c3_mrr, "C3_over_C2": c3_mrr / max(c2_mrr, 1e-12)},
        "bridge_advantage_C3": bridge_advantage,
        "frozen_future_reconstruction_advantage_C3": reconstruction_advantage,
        "bridge_recovered_contact_mse_C2_minus_C3_ci95": bootstrap_ci(
            auxiliary["C2"]["contact_recovery_error"] - auxiliary["C3"]["contact_recovery_error"],
            args.bootstrap_samples,
            int(config["seed"]),
        ),
        "candidate_status": {name: value["overall"] for name, value in results.items()},
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "candidate_comparison.json", comparison)
    print(json.dumps({
        "C2": {"overall": results["C2"]["overall"], "gates": results["C2"]["gates"]},
        "C3": {"overall": results["C3"]["overall"], "gates": results["C3"]["gates"]},
        "comparison": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
