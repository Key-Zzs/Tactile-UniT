#!/usr/bin/env python3
"""Freeze the no-oracle C5 runtime router after mean/uncertainty selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c5_causal_visual import VisualSupport  # noqa: E402
from gr00t.tactile_unit.c3msccr_exact_action_closure import (  # noqa: E402
    canonical_action_from_raw, raw_action_from_canonical, same_split_different_indices,
)
from gr00t.tactile_unit.c5_planned_action import (  # noqa: E402
    ActionRepresentation, PlannedActionChunk, PlannedActionSource, encode_planned_action,
)
from gr00t.tactile_unit.c5_runtime_router import C5Availability, C5RuntimeRouter, route_c5_availability  # noqa: E402
from gr00t.tactile_unit.c5_uncertainty import CalibratedC5Uncertainty, C5RuntimeMode  # noqa: E402
from gr00t.tactile_unit.trex_action_bootstrap import ReleasedTokenizerSource  # noqa: E402
from gr00t.tactile_unit.trex_action_transition import load_shared_transition_checkpoint  # noqa: E402
from scripts.tactile_unit.c3mscc_runtime import load_frozen_shared_space  # noqa: E402
from scripts.tactile_unit.c5_runtime import (  # noqa: E402
    DEFAULT_CONFIG, atomic_json, identity_snapshot, load_config, load_full,
    load_selected_causal, load_selected_uncertainty, load_split, predict_causal,
)
from scripts.tactile_unit.train_c3mscc_contact_prediction import predict_numpy  # noqa: E402
from scripts.tactile_unit.train_c5_uncertainty import (  # noqa: E402
    causal_tokens, plan_ood_score, uncertainty_numpy,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def make_plan(actions: torch.Tensor, source: PlannedActionSource) -> PlannedActionChunk:
    return PlannedActionChunk(
        actions=actions, source=source, start_time=torch.zeros(len(actions), device=actions.device),
        representation=ActionRepresentation.NORMALIZED_PADDED_128,
        normalization_state="TRAIN_ONLY_STANDARDIZED_PADDED_128",
        validity_mask=torch.ones(len(actions), 16, dtype=torch.bool, device=actions.device),
        horizon=16, embodiment=31, planner_policy_id="contract-equivalence-only",
    )


@torch.inference_mode()
def encode_canonical_actions(actions, state, action_model, shared, device, batch_size):
    output = np.empty((len(actions), 8, 32), dtype=np.float32)
    for start in range(0, len(actions), batch_size):
        stop = min(start + batch_size, len(actions))
        action = torch.from_numpy(np.array(actions[start:stop], copy=True)).to(device)
        current = torch.from_numpy(np.array(state[start:stop], copy=True)).to(device)
        value = encode_planned_action(
            make_plan(action, PlannedActionSource.ORACLE_EVAL), current, action_model,
            lambda z: shared.encode("action", z), runtime=False, oracle_eval=True,
        )
        output[start:stop] = value.u_a.float().cpu().numpy()
    return output


def raw_plan_variants(validation, feature_stats, seed):
    raw = raw_action_from_canonical(np.asarray(validation["action"]), feature_stats)
    action_std = np.asarray(feature_stats["action_std"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    previous = np.concatenate((raw[:, :1], raw[:, :-1]), axis=1)
    different = same_split_different_indices(np.asarray(validation["episode_id"]), seed + 1)
    return {
        "oracle_demonstration_surrogate": raw,
        "mild_raw_noise": raw + rng.normal(0, 0.05, raw.shape).astype(np.float32) * action_std,
        "strong_raw_noise": raw + rng.normal(0, 0.20, raw.shape).astype(np.float32) * action_std,
        "temporal_smoothing": (0.75 * raw + 0.25 * previous).astype(np.float32),
        "one_step_lag": previous.astype(np.float32),
        "different_episode_plan": raw[different].copy(),
    }


def planned_action_domain_diagnostic(
    config, validation, visual, causal, uncertainty, support, action_model, shared,
    feature_stats, uncertainty_selection, artifacts, device, batch_size,
    reproduction_tolerance,
):
    training = json.loads((artifacts / "uncertainty_training.json").read_text())
    ood_mean = np.asarray(training["plan_ood_mean"], dtype=np.float32)
    ood_std = np.asarray(training["plan_ood_std"], dtype=np.float32)
    variants = raw_plan_variants(validation, feature_stats, int(config["seed"]) + 14100)
    c_v = causal_tokens(visual, validation, support, device, batch_size)
    target = np.asarray(validation["u_c"])
    scale = float(uncertainty_selection["calibration_scale"])
    metrics, oracle_u_a = {}, None
    for name, raw in variants.items():
        canonical = canonical_action_from_raw(raw, feature_stats)
        u_a = encode_canonical_actions(canonical, validation["state"], action_model, shared, device, batch_size)
        if oracle_u_a is None:
            oracle_u_a = u_a
        prediction = predict_causal(visual, causal, validation, support, device, batch_size, u_a=u_a)
        ood = plan_ood_score(u_a, ood_mean, ood_std)
        source = np.concatenate((c_v, u_a), axis=1)
        log_variance = uncertainty_numpy(
            uncertainty, C5RuntimeMode.FALLBACK_CAUSAL_VA, prediction, source, ood,
            device, batch_size,
        )
        metrics[name] = {
            "raw_rms_from_oracle": float(np.sqrt(np.square(raw - variants["oracle_demonstration_surrogate"]).mean())),
            "representation_rms_from_oracle": float(np.sqrt(np.square(u_a - oracle_u_a).mean())),
            "train_only_plan_ood_score_mean": float(ood.mean()),
            "prediction_mse": float(np.square(prediction - target).mean()),
            "mean_calibrated_uncertainty": float((np.exp(log_variance) * scale).mean()),
        }
    oracle = metrics["oracle_demonstration_surrogate"]
    for name, value in metrics.items():
        value["prediction_mse_delta_from_oracle"] = value["prediction_mse"] - oracle["prediction_mse"]
        value["uncertainty_delta_from_oracle"] = value["mean_calibrated_uncertainty"] - oracle["mean_calibrated_uncertainty"]
    monotonic_noise_uncertainty = (
        metrics["oracle_demonstration_surrogate"]["mean_calibrated_uncertainty"]
        <= metrics["mild_raw_noise"]["mean_calibrated_uncertainty"]
        <= metrics["strong_raw_noise"]["mean_calibrated_uncertainty"]
    )
    accepted_oracle_max_abs = float(np.max(np.abs(oracle_u_a - np.asarray(validation["u_a"]))))
    if accepted_oracle_max_abs > reproduction_tolerance:
        raise RuntimeError("C5_PLANNED_ACTION_INTERFACE_FAIL: full oracle plan encoding changed")
    result = {
        "schema": "tactile3d-unit.vac-c5-planned-action-domain-diagnostic.v1",
        "source": "ORACLE_EVAL demonstration surrogate plus controlled raw-58 perturbations",
        "perturbation_space": "raw 58-D Action before accepted train-only normalization and frozen A-R/C2-R encoding",
        "action_ordering": ["left arm 7", "left hand 22", "right arm 7", "right hand 22"],
        "actual_policy_available": False, "actual_policy_domain_validated": False,
        "accepted_oracle_u_a_reproduction_max_abs": accepted_oracle_max_abs,
        "accepted_c3msccr_reproduction_tolerance": reproduction_tolerance,
        "warning": "POLICY_PLAN_DOMAIN_WARNING", "metrics": metrics,
        "oracle_mild_strong_uncertainty_monotonic": bool(monotonic_noise_uncertainty),
        "monotonicity_is_diagnostic_not_policy_calibration": True, "test_loaded": False,
    }
    atomic_json(artifacts / "planned_action_domain_diagnostic.json", result)
    return result


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    if args.unit_checkpoint is None: raise RuntimeError("UNIT_FULLDATA_CKPT or --unit-checkpoint is required")
    artifacts = ROOT / config["runtime"]["artifact_root"]
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]: raise RuntimeError("frozen identities changed before router audit")
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("0", "1", "2", "3"))
    try:
        set_seed(int(config["seed"]) + 14000)
        validation = load_split(config, "validation")
        full, _ = load_full(config, device)
        visual, causal, mean_selection, mean_sha = load_selected_causal(config, device)
        uncertainty, uncertainty_selection, uncertainty_sha = load_selected_uncertainty(config, device)
        c3_config = json.loads((ROOT / "configs/tactile_unit/c3mscc_contact_context_prediction.json").read_text())
        shared, _, shared_digest = load_frozen_shared_space(c3_config, device)
        released = ReleasedTokenizerSource.open(args.unit_checkpoint / "tokenizer")
        action_path = ROOT / config["runtime"]["action_checkpoint"]
        exact_action_config = json.loads((ROOT / "configs/tactile_unit/c3msccr_exact_action_closure.json").read_text())
        reproduction_tolerance = float(exact_action_config["cache"]["correct_reproduction_atol"])
        action_payload = torch.load(action_path, map_location="cpu", weights_only=False)
        feature_stats = action_payload["feature_stats"]
        action_model, _ = load_shared_transition_checkpoint(action_path, released, map_location=device)
        action_model.eval().requires_grad_(False).to(device)
        rows = min(64, len(validation["u_a"]))
        state = torch.from_numpy(np.array(validation["state"][:rows], copy=True)).to(device)
        actions = torch.from_numpy(np.array(validation["action"][:rows], copy=True)).to(device)
        encoded = {}
        for source, flags in (
            (PlannedActionSource.POLICY_GENERATED, {"runtime": True}),
            (PlannedActionSource.DEMONSTRATION_TEACHER, {"runtime": False, "offline_training": True}),
            (PlannedActionSource.ORACLE_EVAL, {"runtime": False, "oracle_eval": True}),
        ):
            value = encode_planned_action(make_plan(actions.clone(), source), state, action_model, lambda z: shared.encode("action", z), **flags)
            encoded[source.value] = value.u_a.detach().cpu().numpy()
        policy = encoded[PlannedActionSource.POLICY_GENERATED.value]
        equivalence = {name: float(np.max(np.abs(value - policy))) for name, value in encoded.items()}
        accepted = np.asarray(validation["u_a"][:rows])
        reproduction_error = float(np.max(np.abs(policy - accepted)))
        if max(equivalence.values()) != 0.0 or reproduction_error > reproduction_tolerance:
            raise RuntimeError("C5_PLANNED_ACTION_INTERFACE_FAIL: numeric encoding changed")
        with torch.no_grad():
            try:
                make_plan(actions, PlannedActionSource.DEMONSTRATION_TEACHER).assert_legal(runtime=True)
                demo_rejected = False
            except PermissionError:
                demo_rejected = True
        full_prediction = predict_numpy(full, validation, device, args.batch_size)
        accepted_full = np.load(ROOT / ".local/cache/tactile_unit/vac_c4/validation/prediction_FULL_AH.npy", mmap_mode="r")
        full_max_error = float(np.max(np.abs(full_prediction - accepted_full)))
        if full_max_error != 0.0:
            raise RuntimeError("C5_FULL_PATH_REGRESSION")
        support = VisualSupport(mean_selection["visual_support"])
        plan_diagnostic = planned_action_domain_diagnostic(
            config, validation, visual, causal, uncertainty, support, action_model, shared,
            feature_stats, uncertainty_selection, artifacts, device, args.batch_size,
            reproduction_tolerance,
        )
        truth_table = []
        for action in (True, False):
            for contact in (True, False):
                for vision_available in (True, False):
                    availability = C5Availability(vision_available, action, contact)
                    truth_table.append({"vision_available": vision_available, "action_available": action, "contact_context_available": contact, "mode": route_c5_availability(availability).value})
        runtime_uncertainty = CalibratedC5Uncertainty(
            uncertainty, float(uncertainty_selection["calibration_scale"]),
        )
        router = C5RuntimeRouter(full, causal, lambda u_a: u_a, visual_support=support, uncertainty=runtime_uncertainty, policy_plan_domain_validated=False)
        try:
            router.predict_offline_oracle_va(runtime=True)
            offline_rejected = False
        except PermissionError:
            offline_rejected = True
        contract = {
            "schema": "tactile3d-unit.vac-c5-runtime-router-contract.v1",
            "truth_table": truth_table, "deterministic": True, "neural_missingness_router": False,
            "modes": [mode.value for mode in C5RuntimeMode],
            "full": {"predictor": "frozen F_AH", "planned_action": True, "contact_context": True, "vision_required": False},
            "causal_fallback": {"predictor": mean_selection["candidate"], "visual_support": support.value, "planned_action": True, "contact_context": False},
            "a_only": {"predictor": "frozen C4 F_A", "planned_action": True, "vision_required": False, "contact_context": False},
            "no_action": {"mode": "ABSTAIN_NO_ACTION", "prediction_available": False},
            "offline_oracle_va_runtime_routable": False,
            "offline_oracle_runtime_rejection": offline_rejected,
            "demo_action_runtime_rejection": demo_rejected,
            "planned_action_numeric_equivalence_max_abs": equivalence,
            "accepted_u_a_reproduction_max_abs": reproduction_error,
            "accepted_u_a_reproduction_tolerance": reproduction_tolerance,
            "full_validation_prediction_max_abs_vs_accepted": full_max_error,
            "full_validation_reproduction_batch_size": args.batch_size,
            "full_path_nonregression": full_max_error == 0.0,
            "policy_plan_domain_validated": False, "warning": "POLICY_PLAN_DOMAIN_WARNING",
            "uncertainty_output": "calibrated shared-error variance",
            "planned_action_domain_diagnostic": "planned_action_domain_diagnostic.json",
            "planned_action_domain_variants": list(plan_diagnostic["metrics"]),
            "causal_visual_selection_sha256": mean_sha, "uncertainty_selection_sha256": uncertainty_sha,
            "shared_state_sha256": shared_digest, "identity_before": identities_before,
            "identity_after": identity_snapshot(config), "test_loaded": False,
            "pass": bool(offline_rejected and demo_rejected and full_max_error == 0.0 and max(equivalence.values()) == 0.0 and identity_snapshot(config)["pass"]),
            "gpu": {**gpu, "preferred_physical": 1, "fallback": gpu.get("actual_physical") != 1 if gpu.get("actual_physical") is not None else True},
        }
        path = artifacts / "runtime_router_contract.json"; atomic_json(path, contract)
        digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest(); (artifacts / "runtime_router_contract.sha256").write_text(digest + "  runtime_router_contract.json\n")
        print(json.dumps({"router": "PASS" if contract["pass"] else "FAIL", "sha256": digest, "full_nonregression": contract["full_path_nonregression"], "planned_action_equivalence": equivalence}, indent=2))
    finally:
        if lock_handle is not None: lock_handle.close()


if __name__ == "__main__":
    main()
