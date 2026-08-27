#!/usr/bin/env python3
"""Evaluate raw and learned continuous Vision/Contact C0 bridges."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error, r2_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/tactile_unit"))

from continuous_contact_bridge_common import (  # noqa: E402
    DEFAULT_ARTIFACTS,
    DEFAULT_CACHE,
    DEFAULT_EXPERIMENTS,
    DEFAULT_SPEC,
    bootstrap_mean_ci,
    cosine_rows,
    different_episode_permutation,
    distribution_metrics,
    flatten_normalized,
    load_cache,
    retrieval_metrics,
    same_episode_wrong_time_permutation,
    set_seed,
    verify_gpu,
)
from gr00t.tactile_unit.continuous_contact_bridge import (  # noqa: E402
    CausalContactGate,
    TokenSetCrossAttentionBridge,
    TwoTowerContinuousProjector,
    parameter_count,
)
from gr00t.tactile_unit.paired_contract import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--training-summary", type=Path, default=DEFAULT_EXPERIMENTS / "training_summary.json"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_EXPERIMENTS / "continuous_bridge_candidates.pt"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--contract-audit", type=Path, default=DEFAULT_ARTIFACTS / "contract_audit.json"
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args()


def project(
    model: torch.nn.Module | None, data: dict[str, np.ndarray], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    if model is None:
        return np.asarray(data["z_v"]), np.asarray(data["z_c"])
    with torch.inference_mode():
        vision, contact = model(
            torch.from_numpy(data["z_v"]).to(device),
            torch.from_numpy(data["z_c"]).to(device),
        )
    return vision.float().cpu().numpy(), contact.float().cpu().numpy()


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(denominator, 1e-12))


def diagnostic_mmd(left: np.ndarray, right: np.ndarray) -> float:
    x = flatten_normalized(left)
    y = flatten_normalized(right)
    combined = np.concatenate([x, y])
    sample = combined[: min(len(combined), 512)]
    distances = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-np.sum((a[:, None] - b[None, :]) ** 2, axis=-1) / max(2 * bandwidth, 1e-12))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean())


def sliced_wasserstein(
    left: np.ndarray, right: np.ndarray, seed: int, projections: int = 64
) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(x.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    return float(np.mean(np.abs(np.sort(x @ directions, axis=0) - np.sort(y @ directions, axis=0))))


def paired_bundle(
    vision: np.ndarray,
    contact: np.ndarray,
    reversed_contact: np.ndarray,
    data: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    different = different_episode_permutation(data["episode_id"], seed)
    wrong = same_episode_wrong_time_permutation(data["episode_id"], data["anchor_frame"])
    valid_wrong = wrong >= 0
    paired = cosine_rows(vision, contact)
    shuffled = cosine_rows(vision, contact[different])
    reversed_values = cosine_rows(vision, reversed_contact)
    fixed = np.random.default_rng(seed).permutation(len(contact))
    fixed_values = cosine_rows(vision, contact[fixed])
    margin = paired - shuffled
    return {
        "paired_cosine": float(paired.mean()),
        "different_episode_cosine": float(shuffled.mean()),
        "different_episode_margin": float(margin.mean()),
        "margin_bootstrap_95_ci": bootstrap_mean_ci(margin, seed),
        "same_episode_wrong_time_cosine": (
            None
            if not valid_wrong.any()
            else float(cosine_rows(vision[valid_wrong], contact[wrong[valid_wrong]]).mean())
        ),
        "same_episode_wrong_time_coverage": float(valid_wrong.mean()),
        "reversed_contact_cosine": float(reversed_values.mean()),
        "fixed_seed_shuffle_cosine": float(fixed_values.mean()),
    }


def subset_bundle(
    vision: np.ndarray, contact: np.ndarray, data: dict[str, np.ndarray]
) -> dict[str, Any]:
    masks = {
        "all": np.ones(len(vision), dtype=bool),
        "dynamic": np.asarray(data["dynamic"], dtype=bool),
        "rare_boundary": np.isin(data["contact_transition"], [1, 3]),
    }
    result = {}
    for name, mask in masks.items():
        if int(mask.sum()) < 2:
            result[name] = {"count": int(mask.sum()), "v_to_c": None, "c_to_v": None}
            continue
        result[name] = {
            "count": int(mask.sum()),
            "v_to_c": retrieval_metrics(vision[mask], contact[mask]),
            "c_to_v": retrieval_metrics(contact[mask], vision[mask]),
        }
    return result


def fit_ridge_direction(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    dynamic: np.ndarray,
    shuffled_train_y: np.ndarray,
) -> dict[str, Any]:
    shapes = test_y.shape
    x_train = train_x.reshape(len(train_x), -1)
    y_train = train_y.reshape(len(train_y), -1)
    x_val = validation_x.reshape(len(validation_x), -1)
    y_val = validation_y.reshape(len(validation_y), -1)
    x_test = test_x.reshape(len(test_x), -1)
    y_test = test_y.reshape(len(test_y), -1)
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        candidates.append((mean_squared_error(y_val, model.predict(x_val)), alpha))
    alpha = min(candidates)[1]
    model = Ridge(alpha=alpha).fit(x_train, y_train)
    prediction = model.predict(x_test)
    mean_prediction = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    shuffled_control = Ridge(alpha=alpha).fit(
        x_train, shuffled_train_y.reshape(len(shuffled_train_y), -1)
    )
    shuffled_prediction = shuffled_control.predict(x_test)

    def metrics(
        target: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None
    ) -> dict[str, float]:
        if mask is not None:
            target, value = target[mask], value[mask]
        return {
            "mse": float(mean_squared_error(target, value)),
            "r2": float(r2_score(target, value, multioutput="variance_weighted")),
            "cosine": float(
                cosine_rows(
                    value.reshape((-1,) + shapes[1:]), target.reshape((-1,) + shapes[1:])
                ).mean()
            ),
        }

    return {
        "selected_alpha": alpha,
        "validation_candidates": [{"alpha": item[1], "mse": float(item[0])} for item in candidates],
        "test": metrics(y_test, prediction),
        "dynamic": metrics(y_test, prediction, np.asarray(dynamic, dtype=bool)),
        "train_mean_control": metrics(y_test, mean_prediction),
        "train_mean_control_dynamic": metrics(
            y_test, mean_prediction, np.asarray(dynamic, dtype=bool)
        ),
        "different_episode_target_control": metrics(y_test, shuffled_prediction),
        "different_episode_target_control_dynamic": metrics(
            y_test, shuffled_prediction, np.asarray(dynamic, dtype=bool)
        ),
    }


def linear_prediction_bundle(
    train_pair: tuple[np.ndarray, np.ndarray],
    validation_pair: tuple[np.ndarray, np.ndarray],
    test_pair: tuple[np.ndarray, np.ndarray],
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    train_permutation = different_episode_permutation(train["episode_id"], seed)
    train_v, train_c = train_pair
    val_v, val_c = validation_pair
    test_v, test_c = test_pair
    return {
        "vision_to_contact": fit_ridge_direction(
            train_v,
            train_c,
            val_v,
            val_c,
            test_v,
            test_c,
            test["dynamic"],
            train_c[train_permutation],
        ),
        "contact_to_vision": fit_ridge_direction(
            train_c,
            train_v,
            val_c,
            val_v,
            test_c,
            test_v,
            test["dynamic"],
            train_v[train_permutation],
        ),
    }


def choose_logistic_c(
    train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray
) -> float:
    rows = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced").fit(
            train_x, train_y
        )
        rows.append(
            (
                f1_score(
                    validation_y, model.predict(validation_x), average="macro", zero_division=0
                ),
                c_value,
            )
        )
    return max(rows)[1]


def probe(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    test_values: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    train_x = train_values.reshape(len(train_values), -1)
    val_x = validation_values.reshape(len(validation_values), -1)
    test_x = test_values.reshape(len(test_values), -1)
    c_value = choose_logistic_c(train_x, train_labels, val_x, validation_labels)
    model = LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced").fit(
        train_x, train_labels
    )
    prediction = model.predict(test_x)
    return {
        "selected_c": c_value,
        "macro_f1": float(f1_score(test_labels, prediction, average="macro", zero_division=0)),
        "per_class_recall": recall_score(
            test_labels, prediction, average=None, labels=np.unique(train_labels), zero_division=0
        ).tolist(),
    }, prediction


def semantic_retention(
    native: tuple[np.ndarray, np.ndarray, np.ndarray],
    projected: tuple[np.ndarray, np.ndarray, np.ndarray],
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, Any]:
    results = {}
    for label_name in ("contact_transition", "force_trend_class"):
        labels = (train[label_name], validation[label_name], test[label_name])
        native_probe, _ = probe(*native, *labels)
        projected_probe, prediction = probe(*projected, *labels)
        dummy = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((len(labels[0]), 1)), labels[0]
        )
        majority = f1_score(
            labels[2],
            dummy.predict(np.zeros((len(labels[2]), 1))),
            average="macro",
            zero_division=0,
        )
        denominator = native_probe["macro_f1"] - majority
        retention = (
            (projected_probe["macro_f1"] - majority) / denominator
            if denominator > 0
            else float("nan")
        )
        results[label_name] = {
            "native": native_probe,
            "projected": projected_probe,
            "majority_macro_f1": float(majority),
            "advantage_retention": float(retention),
        }
        if label_name == "contact_transition":
            recalls = recall_score(
                labels[2], prediction, labels=[1, 3], average=None, zero_division=0
            )
            results[label_name]["rare_boundary_recall"] = {
                "free_to_contact": float(recalls[0]),
                "contact_to_free": float(recalls[1]),
            }
    return results


@torch.inference_mode()
def gate_evaluation(
    gate: CausalContactGate, test: dict[str, np.ndarray], device: torch.device
) -> dict[str, Any]:
    vision = torch.from_numpy(test["z_v_current"]).to(device)
    current = torch.from_numpy(test["h_current"]).to(device)
    transition = torch.from_numpy(test["contact_transition"]).to(device)
    contact = (transition == 2) | (transition == 3)
    score = gate(vision, current).flatten()
    missing = gate(vision, None).flatten()
    zero_current = gate(vision, torch.zeros_like(current)).flatten()
    residual = torch.from_numpy(test["z_c"]).to(device)
    fallback = gate.residual_fuse(
        vision, residual, current, torch.zeros(len(vision), device=device)
    )
    free_score = score[~contact]
    contact_score = score[contact]
    return {
        "available_finite": bool(torch.isfinite(score).all()),
        "contact_free_accuracy": float(((score >= 0.5) == contact).float().mean()),
        "free_space_gate_mean": float(free_score.mean()),
        "contact_state_gate_mean": float(contact_score.mean()),
        "missing_gate_exact_zero": bool(torch.equal(missing, torch.zeros_like(missing))),
        "zero_current_tactile_finite": bool(torch.isfinite(zero_current).all()),
        "masked_fallback_exact_vision": bool(torch.equal(fallback, vision)),
        "deterministic": bool(torch.equal(score, gate(vision, current).flatten())),
        "shape": list(score.shape),
        "status": (
            "PASS"
            if bool(
                torch.isfinite(score).all()
                and torch.isfinite(zero_current).all()
                and torch.equal(missing, torch.zeros_like(missing))
                and torch.equal(fallback, vision)
            )
            else "FAIL"
        ),
    }


def main() -> int:
    args = parse_args()
    if args.device == "gpu":
        device, physical_gpu = verify_gpu()
        resource_status = "GPU_ACQUIRED"
    else:
        device, physical_gpu = torch.device("cpu"), None
        resource_status = "GPU_RESOURCE_BUSY_CPU_FALLBACK"
    spec = json.loads(args.spec.read_text())
    contract_audit = json.loads(args.contract_audit.read_text())
    if contract_audit.get("status") != "PASS":
        raise RuntimeError("C0 contract/provenance audit must pass before evaluation")
    training = json.loads(args.training_summary.read_text())
    if training.get("status") != "COMPLETE" or training.get("test_used_for_selection"):
        raise RuntimeError("invalid C0 training summary")
    if sha256_file(args.checkpoint) != training["checkpoint_sha256"]:
        raise RuntimeError("C0 bridge checkpoint identity mismatch")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "tactile3d-unit.c0-continuous-bridge-checkpoint.v1":
        raise RuntimeError("invalid C0 checkpoint schema")
    set_seed(int(spec["seed"]))
    data = {
        split: load_cache(args.cache_root / f"paired_{split}.npz")
        for split in ("train", "validation", "test")
    }
    if len(data["test"]["pair_id"]) != 960 or len(set(data["test"]["pair_id"].tolist())) != 960:
        raise RuntimeError("evaluation is not the exact canonical 960-pair set")
    models: dict[str, torch.nn.Module | None] = {"B0": None}
    b1 = TwoTowerContinuousProjector("residual_mlp")
    b1.load_state_dict(payload["state_dicts"]["B1"], strict=True)
    b2 = TokenSetCrossAttentionBridge(heads=4)
    b2.load_state_dict(payload["state_dicts"]["B2"], strict=True)
    models.update({"B1": b1.eval().to(device), "B2": b2.eval().to(device)})
    gate = CausalContactGate().eval().to(device)
    gate.load_state_dict(payload["state_dicts"]["B3"], strict=True)

    reload_b1 = TwoTowerContinuousProjector("residual_mlp").eval().to(device)
    reload_b1.load_state_dict(payload["state_dicts"]["B1"], strict=True)
    reload_b2 = TokenSetCrossAttentionBridge(heads=4).eval().to(device)
    reload_b2.load_state_dict(payload["state_dicts"]["B2"], strict=True)
    reload_gate = CausalContactGate().eval().to(device)
    reload_gate.load_state_dict(payload["state_dicts"]["B3"], strict=True)
    with torch.inference_mode():
        sample_v = torch.from_numpy(data["test"]["z_v"][:64]).to(device)
        sample_c = torch.from_numpy(data["test"]["z_c"][:64]).to(device)
        sample_current = torch.from_numpy(data["test"]["h_current"][:64]).to(device)
        deterministic_reload = {
            "B1": all(
                torch.equal(left, right)
                for left, right in zip(b1(sample_v, sample_c), reload_b1(sample_v, sample_c))
            ),
            "B2": all(
                torch.equal(left, right)
                for left, right in zip(b2(sample_v, sample_c), reload_b2(sample_v, sample_c))
            ),
            "B3": torch.equal(
                gate(sample_v, sample_current), reload_gate(sample_v, sample_current)
            ),
        }

    projected = {
        name: {split: project(model, values, device) for split, values in data.items()}
        for name, model in models.items()
    }
    # B2 is pair-conditioned by design: its Vision output already attends Contact and
    # vice versa. It is a fusion-interface preflight, not a valid independent-tower
    # retrieval selector. B1 is therefore the preregistered selectable alignment bridge.
    selected = "B1"
    candidate_results = {}
    for name in ("B0", "B1", "B2"):
        test_v, test_c = projected[name]["test"]
        if models[name] is None:
            reversed_contact = data["test"]["z_c_reversed"]
        else:
            with torch.inference_mode():
                _, reversed_tensor = models[name](
                    torch.from_numpy(data["test"]["z_v"]).to(device),
                    torch.from_numpy(data["test"]["z_c_reversed"]).to(device),
                )
            reversed_contact = reversed_tensor.float().cpu().numpy()
        candidate_results[name] = {
            "architecture": "identity/raw" if name == "B0" else type(models[name]).__name__,
            "evaluation_role": (
                "raw baseline"
                if name == "B0"
                else (
                    "independent-tower alignment candidate"
                    if name == "B1"
                    else "pair-conditioned cross-attention interface diagnostic"
                )
            ),
            "parameters": 0 if name == "B0" else parameter_count(models[name]),
            "distribution": {
                "vision": distribution_metrics(test_v),
                "contact": distribution_metrics(test_c),
                "mmd_diagnostic": diagnostic_mmd(test_v, test_c),
                "swd_diagnostic": sliced_wasserstein(test_v, test_c, int(spec["seed"])),
            },
            "paired": paired_bundle(
                test_v, test_c, reversed_contact, data["test"], int(spec["seed"])
            ),
            "retrieval": subset_bundle(test_v, test_c, data["test"]),
            "cka": linear_cka(test_v, test_c),
            "linear_prediction": linear_prediction_bundle(
                projected[name]["train"],
                projected[name]["validation"],
                projected[name]["test"],
                data["train"],
                data["validation"],
                data["test"],
                int(spec["seed"]),
            ),
        }
    native_contact = tuple(projected["B0"][split][1] for split in ("train", "validation", "test"))
    selected_contact = tuple(
        projected[selected][split][1] for split in ("train", "validation", "test")
    )
    retention = semantic_retention(
        native_contact, selected_contact, data["train"], data["validation"], data["test"]
    )
    gate_result = gate_evaluation(gate, data["test"], device)
    selected_result = candidate_results[selected]
    paired_evidence = selected_result["paired"]["margin_bootstrap_95_ci"][0] > 0
    retrieval_evidence = (
        selected_result["retrieval"]["all"]["v_to_c"]["recall_at_5"]
        > selected_result["retrieval"]["all"]["v_to_c"]["chance_recall_at_5"]
    )
    retrieval_strong = (
        selected_result["retrieval"]["all"]["v_to_c"]["recall_at_1"]
        > selected_result["retrieval"]["all"]["v_to_c"]["chance_recall_at_1"]
    )
    predictor = selected_result["linear_prediction"]["vision_to_contact"]
    predictor_evidence = predictor["test"]["mse"] < min(
        predictor["train_mean_control"]["mse"], predictor["different_episode_target_control"]["mse"]
    ) and predictor["dynamic"]["mse"] < min(
        predictor["train_mean_control_dynamic"]["mse"],
        predictor["different_episode_target_control_dynamic"]["mse"],
    )
    retention_pass = (
        retention["contact_transition"]["advantage_retention"] >= 0.9
        and retention["force_trend_class"]["advantage_retention"] >= 0.9
    )
    no_collapse = selected_result["distribution"]["contact"]["global_std"] > 1e-6
    structural = (
        gate_result["status"] == "PASS"
        and payload["frozen_integrity"]["unchanged"]
        and no_collapse
        and contract_audit["status"] == "PASS"
        and all(deterministic_reload.values())
    )
    if not structural:
        decision = "STRUCTURAL_FAIL"
    elif (
        paired_evidence
        and retrieval_evidence
        and retrieval_strong
        and predictor_evidence
        and retention_pass
    ):
        decision = "C0_READY"
    elif retention_pass and any((paired_evidence, retrieval_evidence, predictor_evidence)):
        decision = "C0_READY_WITH_ALIGNMENT_WARNING"
    else:
        decision = "C0_INSUFFICIENT"
    output = {
        "schema": "tactile3d-unit.c0-continuous-bridge-evaluation.v1",
        "status": "COMPLETE",
        "decision": decision,
        "physical_gpu": physical_gpu,
        "logical_device": str(device),
        "resource_status": resource_status,
        "canonical_test_pairs": 960,
        "test_pair_ids_sha256": __import__("hashlib")
        .sha256("\n".join(data["test"]["pair_id"].tolist()).encode())
        .hexdigest(),
        "test_used_once_after_validation_selection": True,
        "selected_bridge": selected,
        "selection_basis": "B1 is the preregistered independent-tower alignment candidate; B2 is pair-conditioned fusion and cannot select itself through retrieval",
        "candidates": candidate_results,
        "semantic_retention": retention,
        "causal_gate": gate_result,
        "deterministic_reload": deterministic_reload,
        "gates": {
            "paired_margin_ci_above_zero": paired_evidence,
            "retrieval_above_chance": retrieval_evidence,
            "retrieval_strong_r1_above_chance": retrieval_strong,
            "vision_to_contact_beats_controls": predictor_evidence,
            "contact_semantics_retained": retention_pass,
            "no_collapse": no_collapse,
            "missing_contact_fallback": gate_result["status"] == "PASS",
            "frozen_components_unchanged": payload["frozen_integrity"]["unchanged"],
            "contract_and_causal_audit": contract_audit["status"] == "PASS",
            "deterministic_reload": all(deterministic_reload.values()),
        },
        "checkpoint_sha256": training["checkpoint_sha256"],
        "contract_audit_sha256": sha256_file(args.contract_audit),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "evaluation.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if decision != "STRUCTURAL_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
