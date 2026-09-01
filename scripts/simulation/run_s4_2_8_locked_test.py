#!/usr/bin/env python3
"""Open TEST once after freeze, evaluate, and run one deterministic repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    LatentTransitionDecoder,
)
from gr00t.simulation.s4_2_dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    build_pair_arrays,
)
from gr00t.simulation.s4_2_formal import (  # noqa: E402
    ConditionalContactPredictor,
    FormalActionEncoder,
    FormalVACBridge,
    ScalarLogVarianceHead,
    SharedPrivateDecomposer,
)
from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    pairwise_alignment_metrics,
)
from scripts.simulation.build_s4_2ds_paired_pilot import (  # noqa: E402
    encode_batch as encode_vision_batch,
)
from scripts.simulation.build_s4_2r_contact_dynamics_cache import (  # noqa: E402
    encode as encode_contact_state,
)
from scripts.simulation.run_s4_2_4_formal_representations import (  # noqa: E402
    infer as infer_action,
)
from scripts.simulation.run_s4_2_6_bottleneck_shared_private import (  # noqa: E402
    decompose,
)
from scripts.simulation.run_s4_2_7_conditional_uncertainty import (  # noqa: E402
    PREDICTORS,
    conditioning,
    gaussian_nll,
    infer_log_variance,
    predict as predict_contact,
)
from scripts.simulation.train_s4_2ds_action_pilot import (  # noqa: E402
    features as action_features,
)
from scripts.simulation.train_s4_2ds_action_pilot import policy_state  # noqa: E402
from scripts.simulation.train_s4_2ds_bridge_pilot import (  # noqa: E402
    classification_probe,
)
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_frozen_vision,
)
from gr00t.contact_dynamics.evaluation import different_episode_permutation  # noqa: E402

CONFIG_PATH = ROOT / "configs/simulation/s4_2_formal_downstream.json"
PRETEST_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/pretest_freeze.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"
CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2_formal"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2_formal"
TEST_CACHE = CACHE_ROOT / "paired_test.npz"
FIRST_RESULT = ARTIFACT_ROOT / "locked_test.json"
REPEAT_RESULT = ARTIFACT_ROOT / "locked_test_repeat.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_pretest(*, repeat: bool) -> dict[str, Any]:
    pretest = load_json(PRETEST_PATH)
    if pretest["status"] != "PASS" or pretest["test_loaded"]:
        raise RuntimeError("valid pretest freeze is required before TEST access")
    for path, expected in pretest["implementation_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise RuntimeError(f"implementation changed after pretest freeze: {path}")
    checkpoint_paths = {
        "contact_state": ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
        "contact_C3": ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
        "action": ".local/experiments/simulation/s4_2_formal/action/selected.pt",
        "bridge": ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
        "shared_private": ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
        "rejected_contact_rq": ".local/experiments/simulation/s4_2_formal/s4_2_6/contact_rq_selected.pt",
        "conditional_A_plus_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
        "conditional_V_plus_A_plus_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_plus_H.pt",
        "conditional_V_plus_A_missing_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/V_plus_A_missing_H.pt",
        "conditional_A_only_missing_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
        "uncertainty_full": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
        "uncertainty_missing_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
    }
    for name, path in checkpoint_paths.items():
        if sha256_file(ROOT / path) != pretest["checkpoint_sha256"][name]:
            raise RuntimeError(f"checkpoint changed after pretest freeze: {name}")
    if repeat:
        if not FIRST_RESULT.is_file() or not TEST_CACHE.is_file() or REPEAT_RESULT.exists():
            raise RuntimeError("deterministic repeat is not in the authorized state")
    elif FIRST_RESULT.exists() or TEST_CACHE.exists():
        raise RuntimeError("first locked TEST access has already occurred")
    return pretest


def load_action(device: torch.device) -> tuple[FormalActionEncoder, dict[str, np.ndarray]]:
    payload = torch.load(
        EXPERIMENT_ROOT / "action/selected.pt", map_location="cpu", weights_only=False
    )
    model = FormalActionEncoder(str(payload["candidate"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    stats = {name: np.asarray(value, dtype=np.float32) for name, value in payload["stats"].items()}
    return model.eval().requires_grad_(False).to(device), stats


def load_bridge(device: torch.device) -> FormalVACBridge:
    payload = torch.load(
        EXPERIMENT_ROOT / "bridge/selected.pt", map_location="cpu", weights_only=False
    )
    model = FormalVACBridge(str(payload["adapter"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def encode_dynamics(current: np.ndarray, future: np.ndarray, device: torch.device) -> np.ndarray:
    payload = torch.load(
        ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.eval().requires_grad_(False).to(device)
    result = np.empty((len(current), 8, 32), dtype=np.float32)
    for start in range(0, len(current), 2048):
        stop = min(start + 2048, len(current))
        output = model(
            torch.from_numpy(current[start:stop]).to(device),
            torch.from_numpy(future[start:stop]).to(device),
        )
        result[start:stop] = output["code"].float().cpu().numpy()
    return result


@torch.inference_mode()
def encode_shared(
    bridge: FormalVACBridge, native: np.ndarray, modality: str, device: torch.device
) -> np.ndarray:
    result = np.empty_like(native)
    for start in range(0, len(native), 2048):
        stop = min(start + 2048, len(native))
        result[start:stop] = (
            bridge.encode(modality, torch.from_numpy(native[start:stop]).to(device))
            .float()
            .cpu()
            .numpy()
        )
    return result


def build_locked_cache(
    unit_checkpoint: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    pretest_path: Path = PRETEST_PATH,
    test_cache: Path = TEST_CACHE,
    access_path: Path | None = None,
    prior_cache_paths: tuple[Path, ...] | None = None,
    access_schema: str = "tactile3d-unit.s4-2-8-locked-test-access.v1",
    test_name: str = "TEST_V1",
) -> dict[str, Any]:
    pairs = build_pair_arrays(
        "test",
        dataset_root,
        purpose="locked_test",
        pretest_freeze=pretest_path,
    )
    if len(pairs["pair_id"]) != 4860:
        raise RuntimeError(f"locked {test_name} pair count mismatch")
    if prior_cache_paths is None:
        prior_cache_paths = (
            CACHE_ROOT / "paired_train.npz",
            CACHE_ROOT / "paired_validation.npz",
        )
    test_groups = set(zip(pairs["task"].tolist(), pairs["source_trajectory_id"].tolist()))
    prior_groups: set[tuple[str, str]] = set()
    for cache_path in prior_cache_paths:
        prior = load_npz(cache_path)
        prior_groups |= set(zip(prior["task"].tolist(), prior["source_trajectory_id"].tolist()))
    if test_groups & prior_groups or len(test_groups) != 9:
        raise RuntimeError(f"locked {test_name} source-group leakage")
    tactile_current = pairs["current_history"][:, -1].reshape(-1, 5, 6)
    tactile_future = pairs["future_history"][:, -1].reshape(-1, 5, 6)
    current_contact = tactile_current[:, :, 0].sum(1) > 0
    future_contact = tactile_future[:, :, 0].sum(1) > 0
    contact_transition = current_contact.astype(np.int8) * 2 + future_contact.astype(np.int8)
    current_force = tactile_current[:, :, 1].sum(1).astype(np.float32)
    future_force = tactile_future[:, :, 1].sum(1).astype(np.float32)
    force_delta = future_force - current_force
    cache_manifest = load_json(
        ROOT / ".local/artifacts/simulation/s4_2r/contact_dynamics_cache_manifest.json"
    )
    dynamic = np.abs(force_delta) > float(cache_manifest["dynamic_q70_threshold_newton"])
    force_trend = np.ones(len(force_delta), dtype=np.int8)
    deadband = float(cache_manifest["force_trend_deadband_newton"])
    force_trend[force_delta < -deadband] = 0
    force_trend[force_delta > deadband] = 2
    teacher_path = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
    teacher, teacher_payload = load_teacher_checkpoint(teacher_path, "cpu")
    teacher = teacher.eval().requires_grad_(False).to(device)
    normalizer = TactileNormalization.from_json(teacher_payload["normalization"])
    h_current = encode_contact_state(teacher, pairs["current_history"], normalizer, device, 1024)
    h_future = encode_contact_state(teacher, pairs["future_history"], normalizer, device, 1024)
    z_c = encode_dynamics(h_current, h_future, device)
    action_model, stats = load_action(device)
    state = policy_state(pairs["current_state"])
    action_feature, normalized_state, _ = action_features(state, pairs["action_chunk"], stats)
    z_a, _ = infer_action(action_model, action_feature, normalized_state, device, 1024)
    vision_identity = load_json(ARTIFACT_ROOT / "vision_identity.json")
    specification = {
        "frozen_identity": {
            "original_unit_tokenizer_files_sha256": vision_identity["checkpoint_file_sha256"]
        }
    }
    vision, identity = load_frozen_vision(unit_checkpoint, specification, device)
    if identity["trainable_parameters"] != 0 or vision.training:
        raise RuntimeError("Original UniT Vision is not frozen for locked TEST")
    z_v = np.empty((len(pairs["pair_id"]), 8, 32), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(z_v), batch_size):
            stop = min(start + batch_size, len(z_v))
            z_v[start:stop] = encode_vision_batch(
                vision,
                pairs["current_frame"][start:stop],
                pairs["future_frame"][start:stop],
                executor,
            )
            if stop % 512 < batch_size or stop == len(z_v):
                print(json.dumps({"locked_vision_pairs": stop, "total": len(z_v)}), flush=True)
    del vision
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bridge = load_bridge(device)
    u_v = encode_shared(bridge, z_v, "vision", device)
    u_a = encode_shared(bridge, z_a, "action", device)
    u_c = encode_shared(bridge, z_c, "contact", device)
    test_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        test_cache,
        pair_id=pairs["pair_id"],
        episode_id=pairs["episode_id"],
        task=pairs["task"],
        source_trajectory_id=pairs["source_trajectory_id"],
        anchor_step=pairs["anchor_step"],
        future_step=pairs["future_step"],
        current_state=pairs["current_state"],
        action_chunk=pairs["action_chunk"],
        z_v=z_v,
        z_a=z_a,
        z_c=z_c,
        u_v=u_v,
        u_a=u_a,
        u_c=u_c,
        h_current=h_current,
        h_future=h_future,
        contact_transition=contact_transition,
        force_trend=force_trend,
        dynamic=dynamic,
        current_total_force=current_force,
        future_total_force=future_force,
    )
    access = {
        "schema": access_schema,
        "test_name": test_name,
        "test_loaded": True,
        "first_access": True,
        "pairs": len(pairs["pair_id"]),
        "source_groups": len(test_groups),
        "source_group_overlap": 0,
        "transition_offset_exact": bool(np.all(pairs["future_step"] - pairs["anchor_step"] == 27)),
        "vision_checkpoint_file_sha256": identity["original_unit_tokenizer_files_sha256"],
        "cache": str(test_cache.relative_to(ROOT)),
        "cache_sha256": sha256_file(test_cache),
    }
    atomic_json(access_path or ARTIFACT_ROOT / "locked_test_access.json", access)
    return access


def chance_multiplier(retrieval: dict[str, Any]) -> float:
    return float(retrieval["recall_at_10"] / max(retrieval["chance"]["recall_at_10"], 1e-12))


@torch.inference_mode()
def bridge_recover(
    bridge: FormalVACBridge, modality: str, value: np.ndarray, device: torch.device
) -> np.ndarray:
    result = np.empty_like(value)
    for start in range(0, len(value), 2048):
        stop = min(start + 2048, len(value))
        result[start:stop] = (
            bridge.recover(modality, torch.from_numpy(value[start:stop]).to(device))
            .float()
            .cpu()
            .numpy()
        )
    return result


def load_conditional(name: str, device: torch.device) -> ConditionalContactPredictor:
    payload = torch.load(
        EXPERIMENT_ROOT / f"s4_2_7/{name}.pt", map_location="cpu", weights_only=False
    )
    model = ConditionalContactPredictor(tuple(payload["modalities"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def validate_evaluator_schema(
    train: dict[str, np.ndarray],
    shared_train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Validate the native/shared join before any model-performance calculation."""

    native_required = {
        "pair_id", "episode_id", "task", "source_trajectory_id", "z_v", "z_a", "z_c",
        "h_current", "contact_transition", "force_trend",
    }
    shared_required = {"pair_id", "u_v", "u_a", "u_c"}
    test_required = native_required | shared_required | {"current_state", "action_chunk"}
    missing = {
        "native_train": sorted(native_required - set(train)),
        "shared_train": sorted(shared_required - set(shared_train)),
        "test": sorted(test_required - set(test)),
    }
    if any(missing.values()):
        raise RuntimeError(f"locked evaluator schema missing fields: {missing}")
    if "u_c" in train:
        raise RuntimeError("native TRAIN unexpectedly owns shared Contact u_c")
    if not np.array_equal(train["pair_id"], shared_train["pair_id"]):
        raise RuntimeError("native/shared TRAIN pair_id join mismatch")
    rows = len(train["pair_id"])
    test_rows = len(test["pair_id"])
    if any(len(shared_train[name]) != rows for name in shared_required):
        raise RuntimeError("shared TRAIN row count mismatch")
    if any(len(test[name]) != test_rows for name in test_required):
        raise RuntimeError("locked TEST row count mismatch")
    expected_latent = (8, 32)
    latent_shapes = {
        "shared_train_u_c": tuple(shared_train["u_c"].shape[1:]),
        "test_u_c": tuple(test["u_c"].shape[1:]),
    }
    if any(shape != expected_latent for shape in latent_shapes.values()):
        raise RuntimeError(f"locked evaluator Contact latent shape mismatch: {latent_shapes}")
    return {
        "status": "PASS",
        "native_train_rows": rows,
        "shared_train_rows": len(shared_train["pair_id"]),
        "test_rows": test_rows,
        "shared_contact_owner": "shared_train",
        "native_shared_pair_id_equal": True,
        "contact_latent_shape": list(expected_latent),
    }


def evaluate_locked(
    device: torch.device,
    bootstrap_samples: int,
    *,
    test_cache: Path = TEST_CACHE,
    result_schema: str = "tactile3d-unit.s4-2-8-locked-test.v1",
    test_name: str = "TEST_V1",
) -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    train, test = load_npz(CACHE_ROOT / "paired_train.npz"), load_npz(test_cache)
    shared_train = load_npz(CACHE_ROOT / "shared_train.npz")
    schema_audit = validate_evaluator_schema(train, shared_train, test)
    action_model, stats = load_action(device)
    state = policy_state(test["current_state"])
    action = np.asarray(test["action_chunk"], dtype=np.float32)
    feature, normalized_state, normalized_action = action_features(state, action, stats)
    _, correct_prediction = infer_action(action_model, feature, normalized_state, device, 1024)
    correct_error = np.square(correct_prediction.astype(np.float64) - normalized_action).mean(
        axis=(1, 2)
    )
    rng = np.random.default_rng(int(config["seed"]))
    controls = {
        "reversed": action[:, ::-1].copy(),
        "shuffled": action[rng.permutation(len(action))],
        "different_episode": action[
            different_episode_permutation(test["episode_id"], int(config["seed"]))
        ],
    }
    action_ratios = {}
    for name, value in controls.items():
        control_feature, control_state, _ = action_features(state, value, stats)
        _, prediction = infer_action(action_model, control_feature, control_state, device, 1024)
        error = np.square(prediction.astype(np.float64) - normalized_action).mean(axis=(1, 2))
        action_ratios[name] = float(error.mean() / max(correct_error.mean(), 1e-12))
    action_config = config["s4_2_4"]["gates"]
    action_gates = {
        "reversed": action_ratios["reversed"] >= action_config["reversed_over_correct_mse_min"],
        "shuffled": action_ratios["shuffled"] >= action_config["shuffled_over_correct_mse_min"],
        "different_episode": action_ratios["different_episode"]
        >= action_config["different_episode_over_correct_mse_min"],
    }
    episode = np.unique(test["episode_id"], return_inverse=True)[1]
    alignment = {
        name: pairwise_alignment_metrics(
            test[left],
            test[right],
            episode,
            bootstrap_samples=bootstrap_samples,
            seed=int(config["seed"]) + offset,
            retrieval_chunk=512,
        )
        for offset, (name, left, right) in enumerate(
            (("V-A", "u_v", "u_a"), ("V-C", "u_v", "u_c"), ("A-C", "u_a", "u_c"))
        )
    }
    native_contact = classification_probe(
        train["z_c"],
        test["z_c"],
        train["contact_transition"],
        test["contact_transition"],
        [0, 1, 2, 3],
    )
    shared_contact = classification_probe(
        shared_train["u_c"],
        test["u_c"],
        train["contact_transition"],
        test["contact_transition"],
        [0, 1, 2, 3],
    )
    native_force = classification_probe(
        train["z_c"], test["z_c"], train["force_trend"], test["force_trend"], [0, 1, 2]
    )
    shared_force = classification_probe(
        shared_train["u_c"], test["u_c"], train["force_trend"], test["force_trend"], [0, 1, 2]
    )
    retention = {
        "contact": shared_contact["macro_f1"] / max(native_contact["macro_f1"], 1e-12),
        "force": shared_force["macro_f1"] / max(native_force["macro_f1"], 1e-12),
    }
    retrievals = [
        alignment[pair]["retrieval"][direction]
        for pair in ("V-C", "A-C")
        for direction in ("forward", "reverse")
    ]
    bridge_config = config["s4_2_5"]["gates"]
    bridge_gates = {
        "v_c_margin": alignment["V-C"]["margin_bootstrap_ci95"][0]
        > bridge_config["paired_margin_ci_lower_min"],
        "a_c_margin": alignment["A-C"]["margin_bootstrap_ci95"][0]
        > bridge_config["paired_margin_ci_lower_min"],
        "retrieval": min(chance_multiplier(value) for value in retrievals)
        >= bridge_config["contact_retrieval_r10_chance_multiplier_min"],
        "contact_semantics": retention["contact"]
        >= bridge_config["contact_semantic_retention_min"],
        "force_semantics": retention["force"] >= bridge_config["force_semantic_retention_min"],
    }
    bridge = load_bridge(device)
    sp_payload = torch.load(
        EXPERIMENT_ROOT / "s4_2_6/shared_private.pt", map_location="cpu", weights_only=False
    )
    sp_model = SharedPrivateDecomposer()
    sp_model.load_state_dict(sp_payload["state_dict"], strict=True)
    sp_model = sp_model.eval().requires_grad_(False).to(device)
    native_test = {"vision": test["z_v"], "action": test["z_a"], "contact": test["z_c"]}
    shared_test = {"vision": test["u_v"], "action": test["u_a"], "contact": test["u_c"]}
    _, reconstruction, cross = decompose(sp_model, native_test, shared_test, device)
    permutation = different_episode_permutation(test["episode_id"], int(config["seed"]))
    private_improvements, cross_margins = {}, {}
    for name in native_test:
        shared_prediction = bridge_recover(bridge, name, shared_test[name], device)
        shared_mse = float(
            np.square(shared_prediction.astype(np.float64) - native_test[name]).mean()
        )
        private_mse = float(
            np.square(reconstruction[name].astype(np.float64) - native_test[name]).mean()
        )
        private_improvements[name] = (shared_mse - private_mse) / max(shared_mse, 1e-12)
        correct = np.square(cross[name].astype(np.float64) - native_test[name]).mean(axis=(1, 2))
        shuffled = np.square(cross[name].astype(np.float64) - native_test[name][permutation]).mean(
            axis=(1, 2)
        )
        cross_margins[name] = float((shuffled - correct).mean())
    s6_gates = config["s4_2_6"]["gates"]
    shared_private_gates = {
        "private": min(private_improvements.values())
        >= s6_gates["private_incremental_recovery_min"],
        "cross": min(cross_margins.values()) > s6_gates["shared_cross_modal_margin_min"],
    }
    train_values = conditioning(train, shared_train)
    test_values = conditioning(test, test)
    predictions_train, predictions_test = {}, {}
    for name in PREDICTORS:
        model = load_conditional(name, device)
        predictions_train[name] = predict_contact(model, train_values, device)
        predictions_test[name] = predict_contact(model, test_values, device)
    conditional_mse = {
        name: float(np.square(value.astype(np.float64) - test["u_c"]).mean())
        for name, value in predictions_test.items()
    }
    full_improvement = (conditional_mse["A_plus_H"] - conditional_mse["V_plus_A_plus_H"]) / max(
        conditional_mse["A_plus_H"], 1e-12
    )
    missing_improvement = (
        conditional_mse["A_only_missing_H"] - conditional_mse["V_plus_A_missing_H"]
    ) / max(conditional_mse["A_only_missing_H"], 1e-12)
    missing_semantic = classification_probe(
        predictions_train["V_plus_A_missing_H"],
        predictions_test["V_plus_A_missing_H"],
        train["contact_transition"],
        test["contact_transition"],
        [0, 1, 2, 3],
    )
    semantic_retention = missing_semantic["macro_f1"] / max(shared_contact["macro_f1"], 1e-12)
    s7_gates = config["s4_2_7"]["gates"]
    conditional_gates = {
        "full": full_improvement >= s7_gates["full_over_A_plus_H_relative_mse_improvement_min"],
        "missing_vision": missing_improvement
        >= s7_gates["missing_VA_over_A_relative_mse_improvement_min"],
        "missing_semantics": semantic_retention
        >= s7_gates["missing_contact_macro_f1_retention_min"],
    }
    uncertainty = {}
    for mode, mean_name in (("full", "V_plus_A_plus_H"), ("missing_H", "V_plus_A_missing_H")):
        payload = torch.load(
            EXPERIMENT_ROOT / f"s4_2_7/uncertainty_{mode}.pt",
            map_location="cpu",
            weights_only=False,
        )
        head = ScalarLogVarianceHead(tuple(payload["modalities"]))
        head.load_state_dict(payload["state_dict"], strict=True)
        head = head.eval().requires_grad_(False).to(device)
        test_error = np.square(predictions_test[mean_name].astype(np.float64) - test["u_c"]).mean(
            axis=(1, 2)
        )
        train_error = np.square(
            predictions_train[mean_name].astype(np.float64) - shared_train["u_c"]
        ).mean(axis=(1, 2))
        log_variance = infer_log_variance(head, test_values, device).astype(np.float64) + np.log(
            float(payload["variance_scale"])
        )
        nll = float(gaussian_nll(test_error, log_variance).mean())
        constant = np.log(max(float(train_error.mean()), 1e-12))
        baseline = float(gaussian_nll(test_error, np.full_like(test_error, constant)).mean())
        width = 1.6448536269514722 * np.sqrt(np.exp(log_variance))
        coverage = float(np.mean(np.sqrt(test_error) <= width))
        gates = {
            "nll": baseline - nll > s7_gates["uncertainty_nll_improvement_over_constant_min"],
            "coverage": s7_gates["interval_90_coverage_min"]
            <= coverage
            <= s7_gates["interval_90_coverage_max"],
        }
        uncertainty[mode] = {
            "nll": nll,
            "constant_baseline_nll": baseline,
            "nll_improvement": baseline - nll,
            "coverage_90": coverage,
            "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
            "overall": "PASS" if all(gates.values()) else "FAIL",
        }
    sections = {
        "action": {
            "reconstruction_mse": float(correct_error.mean()),
            "control_ratios": action_ratios,
            "gates": {name: "PASS" if value else "FAIL" for name, value in action_gates.items()},
            "overall": "PASS" if all(action_gates.values()) else "FAIL",
        },
        "bridge": {
            "alignment": alignment,
            "semantic_retention": retention,
            "minimum_contact_retrieval_chance_multiplier": min(
                chance_multiplier(value) for value in retrievals
            ),
            "gates": {name: "PASS" if value else "FAIL" for name, value in bridge_gates.items()},
            "overall": "PASS" if all(bridge_gates.values()) else "FAIL",
        },
        "shared_private": {
            "private_relative_improvement": private_improvements,
            "cross_prediction_margin": cross_margins,
            "gates": {
                name: "PASS" if value else "FAIL" for name, value in shared_private_gates.items()
            },
            "overall": "PASS" if all(shared_private_gates.values()) else "FAIL",
        },
        "conditional": {
            "mse": conditional_mse,
            "full_relative_improvement": full_improvement,
            "missing_VA_relative_improvement": missing_improvement,
            "missing_semantic_retention": semantic_retention,
            "gates": {
                name: "PASS" if value else "FAIL" for name, value in conditional_gates.items()
            },
            "overall": "PASS" if all(conditional_gates.values()) else "FAIL",
        },
        "uncertainty": {
            "modes": uncertainty,
            "overall": (
                "PASS"
                if all(value["overall"] == "PASS" for value in uncertainty.values())
                else "FAIL"
            ),
        },
    }
    result = {
        "schema": result_schema,
        "test_name": test_name,
        "split": "test",
        "pairs": len(test["pair_id"]),
        "source_groups": 9,
        "sections": sections,
        "overall": (
            "PASS" if all(value["overall"] == "PASS" for value in sections.values()) else "FAIL"
        ),
        "training_performed": False,
        "selection_performed": False,
        "threshold_change": False,
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "canonical_contact": "C3",
        "evaluator_schema_audit": schema_audit,
    }
    result["metric_digest"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("first", "repeat"), required=True)
    parser.add_argument("--unit-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    repeat = args.mode == "repeat"
    verify_pretest(repeat=repeat)
    torch.manual_seed(4242)
    np.random.seed(4242)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if not repeat:
        if args.unit_checkpoint is None:
            raise RuntimeError("--unit-checkpoint is required for first locked TEST access")
        build_locked_cache(args.unit_checkpoint, device, args.vision_batch_size, args.workers)
    result = evaluate_locked(device, args.bootstrap_samples)
    if repeat:
        first = load_json(FIRST_RESULT)
        result["deterministic_repeat_of"] = first["metric_digest"]
        result["deterministic_equal"] = result["metric_digest"] == first["metric_digest"]
        atomic_json(REPEAT_RESULT, result)
        if not result["deterministic_equal"]:
            raise SystemExit("S4_2_8_DETERMINISTIC_REPEAT_FAIL")
    else:
        atomic_json(FIRST_RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
