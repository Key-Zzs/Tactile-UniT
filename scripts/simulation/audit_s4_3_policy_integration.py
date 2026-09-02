#!/usr/bin/env python3
"""Audit R4/R5/R7 ACT causality, frozen paths, fairness, and P3 gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.s4_3_act import (  # noqa: E402
    CausalACTPolicy,
    FrozenS42PolicyStack,
    PolicyNormalization,
)
from scripts.simulation.train_s4_2ds_action_pilot import (  # noqa: E402
    features as numpy_action_features,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
MANIFEST = ROOT / ".local/artifacts/simulation/s4_3_pd/policy_expert_dataset_manifest.json"
DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_3_policy_expert"
POLICY_CONFIG = ROOT / "configs/simulation/s4_3_restart_policy_protocol.json"
ACT_CONFIG = ROOT / "configs/simulation/s4_3_restart_act_protocol.json"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")
VARIANTS = ("P0", "P1", "P2", "P3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
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


def load_episode(row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = DATASET_ROOT / row["attempt_relative_path"] / "steps.npz"
    with np.load(path, allow_pickle=False) as source:
        return {
            name: source[name].astype(np.float32)
            for name in ("proprio", "policy_action", "sim_tactile")
        }


def fit_task_normalization(manifest: dict[str, Any], task: str) -> PolicyNormalization:
    episodes = [load_episode(row) for row in manifest["successful_train"] if row["task"] == task]
    return PolicyNormalization.fit(
        np.concatenate([row["proprio"] for row in episodes]),
        np.concatenate([row["policy_action"] for row in episodes]),
        np.concatenate([row["sim_tactile"] for row in episodes]),
    )


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    policy_config = read_json(POLICY_CONFIG)
    act_config = read_json(ACT_CONFIG)
    manifest = read_json(MANIFEST)
    frozen_hashes_before = {
        name: sha256_file(path) for name, path in FrozenS42PolicyStack.CHECKPOINTS.items()
    }
    expected = {name: policy_config["s4_2_immutable"][name] for name in frozen_hashes_before}
    if frozen_hashes_before != expected:
        raise RuntimeError("frozen S4.2 identity mismatch before policy integration audit")

    stack = FrozenS42PolicyStack().to(device)
    stack_digest_before = parameter_digest(stack)
    task_results = {}
    parameter_counts = {}
    gradient_results = {}
    normalization_results = {}
    for task in TASKS:
        normalization = fit_task_normalization(manifest, task)
        models = {}
        for variant in VARIANTS:
            seed_everything(0)
            models[variant] = CausalACTPolicy(variant, normalization).to(device)
        parameter_counts[task] = {
            variant: models[variant].trainable_parameter_count for variant in VARIANTS
        }
        p2_digest = parameter_digest(models["P2"])
        p3_digest = parameter_digest(models["P3"])
        raw_specific = models["P1"].raw_tactile.parameter_count
        fairness = {
            "P2_P3_parameter_count_equal": parameter_counts[task]["P2"]
            == parameter_counts[task]["P3"],
            "P2_P3_matched_initialization_equal": p2_digest == p3_digest,
            "P1_tactile_specific_parameters": raw_specific,
            "P1_tactile_specific_under_100k": raw_specific <= 100000,
        }

        row = next(item for item in manifest["successful_train"] if item["task"] == task)
        episode = load_episode(row)
        anchor = 25
        current_state_np = episode["proprio"][anchor : anchor + 1]
        current_history_np = episode["sim_tactile"][anchor - 25 : anchor + 1][None]
        future_history_np = episode["sim_tactile"][anchor + 2 : anchor + 28][None]
        target_action_np = episode["policy_action"][anchor : anchor + 27][None]
        current_state = torch.from_numpy(current_state_np).to(device)
        current_history = torch.from_numpy(current_history_np).to(device)
        future_history = torch.from_numpy(future_history_np).to(device)
        target_action = torch.from_numpy(target_action_np).to(device)
        vision = torch.zeros(1, 8, 32, device=device)

        with torch.no_grad():
            h_current = stack.encode_contact_state(current_history)
            target_shared = stack.target_shared_contact(current_history, future_history)
        p3 = models["P3"]
        p3.train()
        output = p3(
            vision,
            current_state,
            contact_state=h_current,
            target_action=target_action,
            sample_posterior=False,
        )
        output["physical_action"].retain_grad()
        contact_loss, predicted_shared = stack.contact_auxiliary_loss(
            current_state,
            output["physical_action"],
            current_history,
            target_shared,
        )
        contact_loss.backward()
        action_gradient = output["physical_action"].grad
        trainable_gradient = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in p3.parameters()
            if parameter.grad is not None
        )
        frozen_gradients = [
            name for name, parameter in stack.named_parameters() if parameter.grad is not None
        ]

        action_payload = torch.load(
            FrozenS42PolicyStack.CHECKPOINTS["action_A0"],
            map_location="cpu",
            weights_only=False,
        )
        action_stats = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in action_payload["stats"].items()
        }
        numpy_feature, numpy_state, _ = numpy_action_features(
            current_state_np, target_action_np, action_stats
        )
        torch_feature, torch_state = stack.action_features(current_state, target_action)
        feature_match = np.array_equal(numpy_feature, torch_feature.detach().cpu().numpy())
        state_match = np.array_equal(numpy_state, torch_state.detach().cpu().numpy())

        teacher_payload = torch.load(
            FrozenS42PolicyStack.CHECKPOINTS["contact_state"],
            map_location="cpu",
            weights_only=False,
        )
        numpy_normalizer = TactileNormalization.from_json(teacher_payload["normalization"])
        expected_tactile = numpy_normalizer.transform(current_history_np)
        actual_tactile = stack.tactile_normalization(current_history).detach().cpu().numpy()
        tactile_match = np.array_equal(expected_tactile, actual_tactile)
        normalization_results[task] = {
            "proprio_fit_split": "POLICY_EXPERT_TRAIN",
            "action_fit_split": "POLICY_EXPERT_TRAIN",
            "raw_tactile_fit_split": "POLICY_EXPERT_TRAIN",
            "s4_2_tactile_normalization_exact": tactile_match,
            "A0_action_features_exact": feature_match,
            "A0_state_normalization_exact": state_match,
        }
        gradient_results[task] = {
            "contact_loss": float(contact_loss.detach()),
            "predicted_shared_shape": list(predicted_shared.shape),
            "target_shared_shape": list(target_shared.shape),
            "predicted_action_requires_grad": output["physical_action"].requires_grad,
            "predicted_action_gradient_l1": float(action_gradient.abs().sum()),
            "ACT_trainable_parameter_gradient_l1": trainable_gradient,
            "S4_2_parameters_with_gradient": frozen_gradients,
            "S4_2_gradient_none": not frozen_gradients,
            "status": (
                "PASS"
                if action_gradient is not None
                and float(action_gradient.abs().sum()) > 0
                and trainable_gradient > 0
                and not frozen_gradients
                else "FAIL"
            ),
        }
        task_results[task] = {
            "fairness": fairness,
            "normalization": normalization_results[task],
            "gradient": gradient_results[task],
            "status": (
                "PASS"
                if all(
                    (
                        fairness["P2_P3_parameter_count_equal"],
                        fairness["P2_P3_matched_initialization_equal"],
                        fairness["P1_tactile_specific_under_100k"],
                        tactile_match,
                        feature_match,
                        state_match,
                        gradient_results[task]["status"] == "PASS",
                    )
                )
                else "FAIL"
            ),
        }

    stack_digest_after = parameter_digest(stack)
    frozen_hashes_after = {
        name: sha256_file(path) for name, path in FrozenS42PolicyStack.CHECKPOINTS.items()
    }
    overall = all(row["status"] == "PASS" for row in task_results.values())
    overall = (
        overall
        and stack_digest_before == stack_digest_after
        and frozen_hashes_before == frozen_hashes_after
    )
    observation = {
        "schema": "tactile3d-unit.s4-3-causal-observation-contract.v1",
        "stage": "R4",
        "vision": {"source": "I_t only", "shape": [8, 32], "frozen": True},
        "proprio": {"source": "s_t", "shape": [22]},
        "P1": {"source": "T[t-25:t]", "shape": [26, 30]},
        "P2_P3": {"source": "frozen E_T^sim(T[t-25:t])", "shape": [256]},
        "future_vision": False,
        "future_tactile_observation": False,
        "future_transition_latent": False,
        "warmup_samples": 26,
        "status": "PASS" if overall else "FAIL",
    }
    action_contract = {
        "schema": "tactile3d-unit.s4-3-causal-action-contract.v1",
        "stage": "R4",
        "policy_action_shape": [27, 22],
        "replan_stride": 5,
        "central_adapter": "gr00t.simulation.dexjoco_adapter.policy_action_to_env_action",
        "environment_action_dim": 23,
        "temporal_ensemble": False,
        "alternative_queue_policy": False,
        "differentiable_bound": act_config["act"]["differentiable_action_bound"],
        "status": "PASS" if overall else "FAIL",
    }
    auxiliary = {
        "schema": "tactile3d-unit.s4-3-causal-contact-auxiliary.v1",
        "stage": "R5",
        "predictor": "frozen causal A+H",
        "A0": "frozen",
        "B3": "frozen",
        "target_path": "frozen E_T future -> frozen C3 -> frozen B3 Contact",
        "target_training_only": True,
        "lambda_contact": 0.1,
        "action_detached": False,
        "future_vision_used": False,
        "uncertainty_runtime_status": "UNAVAILABLE_CAUSAL_INPUT_MISMATCH",
        "uncertainty_reason": "Frozen full uncertainty requires the illegal S4.2 future-derived Vision transition representation; no A+H-only uncertainty checkpoint exists.",
        "task_results": task_results,
        "checkpoint_sha256_before": frozen_hashes_before,
        "checkpoint_sha256_after": frozen_hashes_after,
        "status": "PASS" if overall else "FAIL",
    }
    gradient = {
        "schema": "tactile3d-unit.s4-3-p3-gradient-audit.v1",
        "stage": "R6",
        "tasks": gradient_results,
        "S4_2_parameter_digest_before": stack_digest_before,
        "S4_2_parameter_digest_after": stack_digest_after,
        "S4_2_parameter_unchanged": stack_digest_before == stack_digest_after,
        "status": "PASS" if overall else "FAIL",
    }
    implementation = {
        "schema": "tactile3d-unit.s4-3-act-implementation-audit.v1",
        "stage": "R7",
        "provenance": act_config["provenance"],
        "architecture": act_config["act"],
        "parameter_counts": parameter_counts,
        "P1_tactile_specific_parameter_count": next(iter(task_results.values()))["fairness"][
            "P1_tactile_specific_parameters"
        ],
        "P2_P3_inference_identical": all(
            row["fairness"]["P2_P3_parameter_count_equal"]
            and row["fairness"]["P2_P3_matched_initialization_equal"]
            for row in task_results.values()
        ),
        "fairness": "PASS" if overall else "FAIL",
        "network_code_fetched": False,
        "status": "PASS" if overall else "FAIL",
    }
    write_json(ARTIFACT_ROOT / "causal_observation_contract.json", observation)
    write_json(ARTIFACT_ROOT / "causal_action_contract.json", action_contract)
    write_json(ARTIFACT_ROOT / "causal_contact_auxiliary.json", auxiliary)
    write_json(ARTIFACT_ROOT / "p3_gradient_audit.json", gradient)
    write_json(ARTIFACT_ROOT / "act_implementation_audit.json", implementation)
    print(
        json.dumps(
            {
                "R4": observation["status"],
                "R5": auxiliary["status"],
                "R6_gradient": gradient["status"],
                "R7": implementation["status"],
            },
            sort_keys=True,
        )
    )
    if not overall:
        raise SystemExit("S4_3_1_CAUSAL_POLICY_INTERFACE_FAIL")


if __name__ == "__main__":
    main()
