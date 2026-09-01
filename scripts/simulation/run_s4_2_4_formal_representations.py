#!/usr/bin/env python3
"""Train formal Action candidates and build the exact paired VAC contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.evaluation import different_episode_permutation  # noqa: E402
from gr00t.simulation.s4_2_formal import FormalActionEncoder  # noqa: E402
from scripts.simulation.train_s4_2ds_action_pilot import (  # noqa: E402
    features as action_features,
)
from scripts.simulation.train_s4_2ds_action_pilot import (  # noqa: E402
    fit_stats,
    policy_state,
)

CONFIG_PATH = ROOT / "configs/simulation/s4_2_formal_downstream.json"
PROTOCOL_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/protocol_freeze.json"
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
CONTACT_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics"
CONTACT_CODE_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics_codes"
DS_CACHE = ROOT / ".local/cache/simulation/s4_2ds/pilot_pairs.npz"
DS_VALIDATION = ROOT / ".local/cache/simulation/s4_2ds/formal_validation_confirmation.npz"
DS_VISION_ARTIFACT = ROOT / ".local/artifacts/simulation/s4_2ds/vision_pilot.json"
DS_CONFIRMATION = ROOT / ".local/artifacts/simulation/s4_2ds/formal_validation_confirmation.json"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2_formal/action"
CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2_formal"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


@torch.inference_mode()
def infer(
    model: FormalActionEncoder,
    feature: np.ndarray,
    state: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    code = np.empty((len(feature), 8, 32), dtype=np.float32)
    prediction = np.empty((len(feature), 27, 22), dtype=np.float32)
    model.eval()
    for start in range(0, len(feature), batch_size):
        stop = min(start + batch_size, len(feature))
        output = model(
            torch.from_numpy(np.array(feature[start:stop], copy=True)).to(device),
            torch.from_numpy(np.array(state[start:stop], copy=True)).to(device),
        )
        code[start:stop] = output["code"].float().cpu().numpy()
        prediction[start:stop] = output["action"].float().cpu().numpy()
    return code, prediction


def effective_rank(value: np.ndarray) -> float:
    flat = value.reshape(len(value), -1).astype(np.float64)
    singular = np.linalg.svd(flat - flat.mean(axis=0), compute_uv=False)
    probability = np.square(singular) / max(float(np.square(singular).sum()), 1e-12)
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-15)))))


def geometry(value: np.ndarray) -> dict[str, float]:
    variance = value.reshape(-1, value.shape[-1]).astype(np.float64).var(axis=0)
    return {
        "effective_rank": effective_rank(value),
        "near_zero_variance_fraction": float(np.mean(variance < 1e-8)),
        "mean_dimension_variance": float(variance.mean()),
    }


def train_candidate(
    candidate: str,
    train_pairs: dict[str, np.ndarray],
    validation_pairs: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    seed = int(config["seed"])
    seed_everything(seed)
    training = config["s4_2_4"]["training"]
    state_train = policy_state(train_pairs["current_state"])
    state_validation = policy_state(validation_pairs["current_state"])
    action_train = np.asarray(train_pairs["action_chunk"], dtype=np.float32)
    action_validation = np.asarray(validation_pairs["action_chunk"], dtype=np.float32)
    stats = fit_stats(state_train, action_train, np.arange(len(action_train)))
    feature_train, normalized_state_train, normalized_action_train = action_features(
        state_train, action_train, stats
    )
    feature_validation, normalized_state_validation, normalized_action_validation = action_features(
        state_validation, action_validation, stats
    )
    model = FormalActionEncoder(candidate).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters > int(config["s4_2_4"]["action_parameter_max"]):
        raise RuntimeError(f"{candidate} exceeds the Action parameter budget")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    for step in range(1, int(training["steps"]) + 1):
        rows = rng.choice(len(feature_train), size=int(training["batch_size"]), replace=False)
        output = model(
            torch.from_numpy(feature_train[rows]).to(device),
            torch.from_numpy(normalized_state_train[rows]).to(device),
        )
        loss = F.mse_loss(
            output["action"], torch.from_numpy(normalized_action_train[rows]).to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(training["steps"]):
            row = {"step": step, "train_mse": float(loss.detach())}
            history.append(row)
            print(json.dumps({"candidate": candidate, **row}), flush=True)
    z_train, _ = infer(model, feature_train, normalized_state_train, device, 1024)
    z_validation, prediction = infer(
        model, feature_validation, normalized_state_validation, device, 1024
    )
    correct_error = np.square(
        prediction.astype(np.float64) - normalized_action_validation.astype(np.float64)
    ).mean(axis=(1, 2))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(action_validation))
    different = different_episode_permutation(validation_pairs["episode_id"], seed=seed)
    controls = {
        "reversed": action_validation[:, ::-1].copy(),
        "shuffled": action_validation[shuffled],
        "different_episode": action_validation[different],
    }
    ratios = {}
    control_mse = {}
    for name, action in controls.items():
        feature, normalized_state, _ = action_features(state_validation, action, stats)
        _, control_prediction = infer(model, feature, normalized_state, device, 1024)
        error = np.square(
            control_prediction.astype(np.float64) - normalized_action_validation.astype(np.float64)
        ).mean(axis=(1, 2))
        control_mse[name] = float(error.mean())
        ratios[name] = float(error.mean() / max(correct_error.mean(), 1e-12))
    gates_config = config["s4_2_4"]["gates"]
    code_geometry = geometry(z_validation)
    gates = {
        "reversed": ratios["reversed"] >= float(gates_config["reversed_over_correct_mse_min"]),
        "shuffled": ratios["shuffled"] >= float(gates_config["shuffled_over_correct_mse_min"]),
        "different_episode": ratios["different_episode"]
        >= float(gates_config["different_episode_over_correct_mse_min"]),
        "effective_rank": code_geometry["effective_rank"]
        >= float(gates_config["effective_rank_min"]),
        "variance": code_geometry["near_zero_variance_fraction"]
        <= float(gates_config["near_zero_variance_fraction_max"]),
        "finite": bool(np.isfinite(z_train).all() and np.isfinite(z_validation).all()),
    }
    checkpoint_path = EXPERIMENT_ROOT / candidate / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2-4-action-checkpoint.v1",
            "candidate": candidate,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "stats": {name: value.tolist() for name, value in stats.items()},
            "parameters": parameters,
            "seed": seed,
            "formal_test_loaded": False,
        },
        checkpoint_path,
    )
    result = {
        "schema": "tactile3d-unit.s4-2-4-action-evaluation.v1",
        "candidate": candidate,
        "parameters": parameters,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training": {"split": "train", "pairs": len(z_train), "history": history},
        "selection_split": "validation",
        "validation_pairs": len(z_validation),
        "validation_reconstruction_mse": float(correct_error.mean()),
        "control_mse": control_mse,
        "control_over_correct_ratio": ratios,
        "minimum_control_ratio": float(min(ratios.values())),
        "geometry": code_geometry,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / f"action_{candidate.lower()}.json", result)
    return result, z_train, z_validation


def exact_groups(arrays: dict[str, np.ndarray]) -> set[tuple[str, str]]:
    return set(zip(arrays["task"].tolist(), arrays["source_trajectory_id"].tolist()))


def save_paired(
    path: Path,
    pairs: dict[str, np.ndarray],
    contact: dict[str, np.ndarray],
    z_v: np.ndarray,
    z_a: np.ndarray,
    z_c: np.ndarray,
) -> None:
    if not np.array_equal(pairs["pair_id"], contact["pair_id"]):
        raise RuntimeError("raw/Contact pair identity mismatch")
    if not (len(z_v) == len(z_a) == len(z_c) == len(pairs["pair_id"])):
        raise RuntimeError("paired VAC array length mismatch")
    np.savez_compressed(
        path,
        pair_id=pairs["pair_id"],
        episode_id=pairs["episode_id"],
        task=pairs["task"],
        source_trajectory_id=pairs["source_trajectory_id"],
        anchor_step=pairs["anchor_step"],
        future_step=pairs["future_step"],
        z_v=z_v,
        z_a=z_a,
        z_c=z_c,
        h_current=contact["h_current"],
        h_future=contact["h_future"],
        contact_transition=contact["contact_transition"],
        force_trend=contact["force_trend"],
        dynamic=contact["dynamic"],
        current_total_force=contact["current_total_force"],
        future_total_force=contact["future_total_force"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    protocol = load_json(PROTOCOL_PATH)
    if protocol["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("formal protocol changed after freeze")
    if protocol["formal_test_loaded"]:
        raise RuntimeError("formal test access precedes S4.2-4")
    train_pairs = load_npz(PAIR_ROOT / "train.npz")
    validation_pairs = load_npz(PAIR_ROOT / "validation.npz")
    train_contact = load_npz(CONTACT_ROOT / "train.npz")
    validation_contact = load_npz(CONTACT_ROOT / "validation.npz")
    train_codes = load_npz(CONTACT_CODE_ROOT / "train.npz")
    validation_codes = load_npz(CONTACT_CODE_ROOT / "validation.npz")
    pilot = load_npz(DS_CACHE)
    confirmation = load_npz(DS_VALIDATION)
    if not np.array_equal(train_pairs["pair_id"], pilot["pair_id"]):
        raise RuntimeError("formal TRAIN Vision pair identity mismatch")
    if not np.array_equal(validation_pairs["pair_id"], confirmation["pair_id"]):
        raise RuntimeError("formal validation Vision pair identity mismatch")
    if not np.array_equal(train_pairs["pair_id"], train_codes["pair_id"]):
        raise RuntimeError("formal TRAIN Contact pair identity mismatch")
    if not np.array_equal(validation_pairs["pair_id"], validation_codes["pair_id"]):
        raise RuntimeError("formal validation Contact pair identity mismatch")
    if exact_groups(train_pairs) & exact_groups(validation_pairs):
        raise RuntimeError("formal train/validation source-group leakage")
    device = torch.device(args.device)
    results, codes = {}, {}
    for candidate in config["s4_2_4"]["action_candidates"]:
        results[candidate], train_code, validation_code = train_candidate(
            candidate, train_pairs, validation_pairs, config, device
        )
        codes[candidate] = {"train": train_code, "validation": validation_code}
    passing = [name for name, value in results.items() if value["overall"] == "PASS"]
    if not passing:
        decision = {
            "schema": "tactile3d-unit.s4-2-4-action-selection.v1",
            "decision": "S4_2_4_ACTION_FAIL",
            "selected": None,
            "formal_test_loaded": False,
        }
        atomic_json(ARTIFACT_ROOT / "action_selection.json", decision)
        raise SystemExit("S4_2_4_ACTION_FAIL")
    order = {name: index for index, name in enumerate(config["s4_2_4"]["action_candidates"])}
    selected = sorted(
        passing,
        key=lambda name: (
            -results[name]["minimum_control_ratio"],
            results[name]["validation_reconstruction_mse"],
            results[name]["parameters"],
            order[name],
        ),
    )[0]
    selected_checkpoint = ROOT / results[selected]["checkpoint"]
    selected_path = EXPERIMENT_ROOT / "selected.pt"
    shutil.copy2(selected_checkpoint, selected_path)
    selection = {
        "schema": "tactile3d-unit.s4-2-4-action-selection.v1",
        "decision": "S4_2_4_ACTION_ACCEPTED",
        "selected": selected,
        "selection_rule": config["s4_2_4"]["selection"],
        "passing": passing,
        "checkpoint": str(selected_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(selected_path),
        "validation_only_selection": True,
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "action_selection.json", selection)
    vision_pilot, vision_confirmation = load_json(DS_VISION_ARTIFACT), load_json(DS_CONFIRMATION)
    if vision_pilot["overall"] != "PASS" or vision_confirmation["overall"] != "PASS":
        raise RuntimeError("frozen UniT Vision evidence failed")
    if (
        vision_pilot["checkpoint_file_sha256"]
        != vision_confirmation["vision_checkpoint_file_sha256"]
    ):
        raise RuntimeError("frozen UniT Vision identity mismatch")
    vision = {
        "schema": "tactile3d-unit.s4-2-4-vision-identity.v1",
        "decision": "S4_2_4_VISION_ACCEPTED_FROZEN_UNIT",
        "frames": ["I_t", "I_t+27"],
        "output_shape": [8, 32],
        "checkpoint_file_sha256": vision_pilot["checkpoint_file_sha256"],
        "train_pairs": len(pilot["z_v"]),
        "validation_pairs": len(confirmation["z_v"]),
        "trainable_parameters": 0,
        "eval_mode": True,
        "parameter_unchanged": True,
        "deterministic": True,
        "reuse": "exact frozen offline teacher tensors extracted and stability-audited during S4.2-DS",
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "vision_identity.json", vision)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    save_paired(
        CACHE_ROOT / "paired_train.npz",
        train_pairs,
        train_contact,
        pilot["z_v"],
        codes[selected]["train"],
        train_codes["z_c"],
    )
    save_paired(
        CACHE_ROOT / "paired_validation.npz",
        validation_pairs,
        validation_contact,
        confirmation["z_v"],
        codes[selected]["validation"],
        validation_codes["z_c"],
    )
    paired = {
        "schema": "tactile3d-unit.s4-2-4-exact-paired-vac.v1",
        "join": [
            "pair_id",
            "episode_id",
            "task",
            "source_trajectory_id",
            "anchor t",
            "future t+27",
        ],
        "timing": {"offset_steps": 27, "actual_horizon_sec": 0.54},
        "shapes": {"vision": [8, 32], "action": [8, 32], "contact": [8, 32]},
        "contact": {
            "candidate": "C3",
            "checkpoint_sha256": config["scientific_boundary"][
                "canonical_contact_checkpoint_sha256"
            ],
        },
        "action": {"candidate": selected, "checkpoint_sha256": sha256_file(selected_path)},
        "vision": {"checkpoint_file_sha256": vision["checkpoint_file_sha256"]},
        "splits": {
            "train": {
                "pairs": len(train_pairs["pair_id"]),
                "groups": len(exact_groups(train_pairs)),
                "cache": ".local/cache/simulation/s4_2_formal/paired_train.npz",
                "sha256": sha256_file(CACHE_ROOT / "paired_train.npz"),
            },
            "validation": {
                "pairs": len(validation_pairs["pair_id"]),
                "groups": len(exact_groups(validation_pairs)),
                "cache": ".local/cache/simulation/s4_2_formal/paired_validation.npz",
                "sha256": sha256_file(CACHE_ROOT / "paired_validation.npz"),
            },
        },
        "source_group_overlap": 0,
        "test_cached": False,
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "paired_vac_contract.json", paired)
    final = {
        "schema": "tactile3d-unit.s4-2-4-final.v1",
        "decision": "S4_2_4_FORMAL_ACTION_VISION_ACCEPTED",
        "action": selected,
        "vision": "Original-UniT-frozen",
        "paired_contract": "PASS",
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "canonical_contact": "C3",
        "formal_test_loaded": False,
        "S4_2_5_readiness": "READY",
    }
    atomic_json(ARTIFACT_ROOT / "s4_2_4_final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
