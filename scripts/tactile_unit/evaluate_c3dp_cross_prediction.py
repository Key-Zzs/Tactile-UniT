#!/usr/bin/env python3
"""Locked benchmark re-evaluation of the frozen C3-DP predictor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_teacher.evaluation import classification_metrics  # noqa: E402
from gr00t.tactile_unit.c3dp_shared_private import (  # noqa: E402
    ORDERED_DIRECTIONS,
    SHORT_DIRECTION,
    load_predictor_checkpoint,
    output_geometry_gate,
    private_geometry,
    sha256_file,
)
from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    bootstrap_mean_ci,
    different_episode_permutation,
    linear_cka,
    numpy_flatten_normalize,
    retrieval_metrics,
)
from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
)
from gr00t.tactile_unit.trex_action_data import (  # noqa: E402
    RAW_ACTION_DIM,
    action_activity,
)
from gr00t.tactile_unit.trex_action_transition import (  # noqa: E402
    load_shared_transition_checkpoint,
)
from gr00t.tactile_unit.vac_latent_dataset import load_split  # noqa: E402
from scripts.tactile_unit.c3dp_runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    atomic_json,
    build_split,
    ensure_cache_identities,
    load_config,
    load_derived_split,
    load_frozen_shared_space,
    shared_digest,
    validate_selection_lock,
)
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_s2_model,
)
from scripts.tactile_unit.evaluate_continuous_vac_shared_space import (  # noqa: E402
    same_episode_wrong_time,
)
from scripts.tactile_unit.vac_runtime_common import (  # noqa: E402
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument(
        "--unit-checkpoint",
        type=Path,
        default=(
            Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None
        ),
    )
    parser.add_argument(
        "--action-checkpoint",
        type=Path,
        default=ROOT / ".local/experiments/tactile_unit/s3_3_r/selected.pt",
    )
    parser.add_argument(
        "--s2-checkpoint",
        type=Path,
        default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt",
    )
    parser.add_argument(
        "--s1-checkpoint",
        type=Path,
        default=ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt",
    )
    return parser.parse_args()


def predict_numpy(predictor, source, source_name, target_name, device, batch_size):
    result = np.empty((len(source), 8, 32), dtype=np.float32)
    predictor.eval()
    with torch.inference_mode():
        for start in range(0, len(source), batch_size):
            stop = min(start + batch_size, len(source))
            value = torch.from_numpy(np.array(source[start:stop], copy=True)).to(device)
            result[start:stop] = predictor(value, source_name, target_name).float().cpu().numpy()
    return result


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(numpy_flatten_normalize(left) * numpy_flatten_normalize(right), axis=1)


def per_sample_mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        np.square(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
        .reshape(len(left), -1)
        .mean(1)
    )


def subset_means(value: np.ndarray, dynamic: np.ndarray, transition: np.ndarray) -> dict[str, Any]:
    masks = {
        "all": np.ones(len(value), dtype=bool),
        "dynamic": np.asarray(dynamic, dtype=bool),
        "rare_boundary": np.isin(transition, [1, 2]),
        "free_to_contact": np.asarray(transition) == 1,
        "contact_to_free": np.asarray(transition) == 2,
    }
    return {
        name: {
            "count": int(mask.sum()),
            "mean": None if not mask.any() else float(value[mask].mean()),
        }
        for name, mask in masks.items()
    }


def direction_evaluation(
    predictor,
    source,
    target,
    train_target,
    source_name,
    target_name,
    episode,
    t,
    dynamic,
    transition,
    config,
    device,
    batch_size,
    seed,
):
    prediction = predict_numpy(predictor, source, source_name, target_name, device, batch_size)
    random_order = np.random.default_rng(seed).permutation(len(source))
    different = different_episode_permutation(episode, seed + 1)
    wrong = same_episode_wrong_time(episode, t)
    shuffled_prediction = predict_numpy(
        predictor, source[random_order], source_name, target_name, device, batch_size
    )
    different_prediction = predict_numpy(
        predictor, source[different], source_name, target_name, device, batch_size
    )
    wrong_prediction = predict_numpy(
        predictor, source[wrong], source_name, target_name, device, batch_size
    )
    target_mean = np.asarray(train_target, dtype=np.float64).mean(0).astype(np.float32)
    mean_prediction = np.broadcast_to(target_mean[None], target.shape)
    error = per_sample_mse(prediction, target)
    controls = {
        "mean": per_sample_mse(mean_prediction, target),
        "shuffled_source": per_sample_mse(shuffled_prediction, target),
        "different_episode": per_sample_mse(different_prediction, target),
        "same_episode_wrong_time": per_sample_mse(wrong_prediction, target),
    }
    canonical_control_names = ("mean", "shuffled_source", "different_episode")
    strongest = min(canonical_control_names, key=lambda name: controls[name].mean())
    improvement = controls[strongest] - error
    target_shuffle = np.random.default_rng(seed + 2).permutation(len(target))
    cosine_true = row_cosine(prediction, target)
    cosine_shuffled = row_cosine(prediction, target[target_shuffle])
    margin = cosine_true - cosine_shuffled
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    improvement_ci = bootstrap_mean_ci(improvement, samples=bootstrap_samples, seed=seed + 3)
    margin_ci = bootstrap_mean_ci(margin, samples=bootstrap_samples, seed=seed + 4)
    retrieval = retrieval_metrics(
        prediction, target, chunk=int(config["evaluation"]["retrieval_chunk"])
    )
    multiplier = retrieval["recall_at_10"] / retrieval["chance"]["recall_at_10"]
    gates = {
        "mse": bool(error.mean() < controls[strongest].mean() and improvement_ci[0] > 0),
        "cosine": bool(margin.mean() > 0 and margin_ci[0] > 0),
        "retrieval": bool(
            multiplier >= float(config["evaluation"]["retrieval_r10_chance_multiplier_min"])
        ),
    }
    gates["pass"] = all(gates.values())
    return {
        "prediction": prediction,
        "control_predictions": {
            "mean": mean_prediction,
            "shuffled_source": shuffled_prediction,
            "different_episode": different_prediction,
            "same_episode_wrong_time": wrong_prediction,
        },
        "metrics": {
            "prediction_mse": float(error.mean()),
            "controls": {name: float(value.mean()) for name, value in controls.items()},
            "strongest_control": strongest,
            "improvement": float(improvement.mean()),
            "improvement_ci95": improvement_ci,
            "cosine_true": float(cosine_true.mean()),
            "cosine_shuffled": float(cosine_shuffled.mean()),
            "cosine_margin": float(margin.mean()),
            "cosine_margin_ci95": margin_ci,
            "retrieval": retrieval,
            "r10_chance_multiplier": float(multiplier),
            "subsets": subset_means(error, dynamic, transition),
            "control_subsets": {
                name: subset_means(value, dynamic, transition) for name, value in controls.items()
            },
            "gates": gates,
        },
    }


def fit_classifier(train_x, train_y):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced"))
    model.fit(np.asarray(train_x).reshape(len(train_x), -1), np.asarray(train_y, dtype=np.int64))
    return model


def per_class_recall(target, prediction, classes):
    return {
        str(label): (
            None
            if not np.any(target == label)
            else float(np.mean(prediction[target == label] == label))
        )
        for label in range(classes)
    }


def classifier_evaluation(model, value, target, majority_class, classes):
    prediction = model.predict(np.asarray(value).reshape(len(value), -1))
    majority = np.full(len(target), majority_class, dtype=np.int64)
    return {
        **classification_metrics(target, prediction),
        "majority": classification_metrics(target, majority),
        "per_class_recall": per_class_recall(target, prediction, classes),
        "_prediction": prediction,
    }


def bootstrap_f1_difference(target, prediction, control, samples, seed):
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(target), size=len(target))
        values[index] = (
            classification_metrics(target[selected], prediction[selected])["macro_f1"]
            - classification_metrics(target[selected], control[selected])["macro_f1"]
        )
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def contact_semantics(predictor, train, test, direction_results, config, device, batch_size, seed):
    result = {}
    overall = True
    train_representations = {}
    for direction_offset, (key, source_key, source_name) in enumerate(
        (("V->C", "u_v", "vision"), ("A->C", "u_a", "action"))
    ):
        source = np.asarray(train[source_key])
        shuffled = np.random.default_rng(seed + 1000 + direction_offset).permutation(len(source))
        different = different_episode_permutation(
            np.asarray(train["episode_id"]), seed + 1100 + direction_offset
        )
        mean_contact = np.asarray(train["u_c"], dtype=np.float64).mean(0).astype(np.float32)
        train_representations[key] = {
            "predicted": predict_numpy(
                predictor, source, source_name, "contact", device, batch_size
            ),
            "mean": np.broadcast_to(mean_contact[None], np.asarray(train["u_c"]).shape),
            "shuffled_source": predict_numpy(
                predictor, source[shuffled], source_name, "contact", device, batch_size
            ),
            "different_episode": predict_numpy(
                predictor, source[different], source_name, "contact", device, batch_size
            ),
        }
    for metric_offset, (metric, label, classes) in enumerate(
        (("contact_transition", "contact_transition", 4), ("force_trend", "force_trend_class", 3))
    ):
        target = np.asarray(test[label], dtype=np.int64)
        majority_class = int(
            np.bincount(np.asarray(train[label], dtype=np.int64), minlength=classes).argmax()
        )
        oracle_probe = fit_classifier(train["u_c"], train[label])
        oracle = classifier_evaluation(oracle_probe, test["u_c"], target, majority_class, classes)
        metric_result = {
            "oracle_shared": {
                key: value for key, value in oracle.items() if not key.startswith("_")
            }
        }
        for direction_offset, key in enumerate(("V->C", "A->C")):
            value = direction_results[key]
            predicted_probe = fit_classifier(train_representations[key]["predicted"], train[label])
            predicted = classifier_evaluation(
                predicted_probe, value["prediction"], target, majority_class, classes
            )
            control_metrics = {}
            for control_name in ("mean", "shuffled_source", "different_episode"):
                control_probe = fit_classifier(
                    train_representations[key][control_name], train[label]
                )
                control = classifier_evaluation(
                    control_probe,
                    value["control_predictions"][control_name],
                    target,
                    majority_class,
                    classes,
                )
                control_metrics[control_name] = control
            strongest_name = max(
                control_metrics,
                key=lambda name: float(control_metrics[name]["macro_f1"]),
            )
            strongest = control_metrics[strongest_name]
            majority_f1 = float(oracle["majority"]["macro_f1"])
            denominator = float(oracle["macro_f1"]) - majority_f1
            retention = (float(predicted["macro_f1"]) - majority_f1) / max(denominator, 1e-12)
            ci = bootstrap_f1_difference(
                target,
                predicted["_prediction"],
                strongest["_prediction"],
                int(config["evaluation"]["bootstrap_samples"]),
                seed + metric_offset * 100 + direction_offset,
            )
            if metric == "contact_transition":
                gate = retention >= float(config["evaluation"]["contact_cross_retention_min"])
                denominator_issue = False
            else:
                denominator_issue = abs(denominator) < 0.02
                gate = (
                    retention >= float(config["evaluation"]["contact_cross_retention_min"])
                    if not denominator_issue
                    else ci[0] > 0
                )
            overall = overall and gate
            metric_result[key] = {
                "predicted": {
                    name: item for name, item in predicted.items() if not name.startswith("_")
                },
                "controls": {
                    name: {
                        field: item for field, item in control.items() if not field.startswith("_")
                    }
                    for name, control in control_metrics.items()
                },
                "strongest_control": strongest_name,
                "predicted_minus_control_f1_ci95": ci,
                "retention": float(retention),
                "oracle_advantage_denominator": float(denominator),
                "denominator_issue": denominator_issue,
                "gate": bool(gate),
            }
        result[metric] = metric_result
    result["gate"] = bool(overall)
    return result


def contact_physics(decoder, split, representations, device, batch_size):
    result = {}
    for name, value in representations.items():
        error_actual = np.empty(len(split), dtype=np.float64)
        predicted_future = np.empty((len(split), 256), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(split), batch_size):
                stop = min(start + batch_size, len(split))
                z = torch.from_numpy(np.array(value[start:stop], copy=True)).to(device)
                current = torch.from_numpy(
                    np.array(split.arrays["h_current"][start:stop], copy=True)
                ).to(device)
                target = torch.from_numpy(
                    np.array(split.arrays["h_future"][start:stop], copy=True)
                ).to(device)
                prediction = decoder(z, current)
                predicted_future[start:stop] = prediction.float().cpu().numpy()
                error_actual[start:stop] = (
                    torch.square(prediction - target).mean(1).double().cpu().numpy()
                )
        result[name] = {
            "future": predicted_future,
            "actual_error": error_actual,
            "metrics": subset_means(
                error_actual,
                np.asarray(split.arrays["dynamic"]),
                np.asarray(split.arrays["contact_transition"]),
            ),
        }
    oracle = result["oracle_shared"]["future"]
    for name, value in result.items():
        value["shared_oracle_mse"] = float(per_sample_mse(value["future"], oracle).mean())
        del value["future"]
        del value["actual_error"]
    return result


def recover_numpy(shared_space, modality, value, device, batch_size):
    result = np.empty_like(value, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(value), batch_size):
            stop = min(start + batch_size, len(value))
            tensor = torch.from_numpy(np.array(value[start:stop], copy=True)).to(device)
            result[start:stop] = shared_space.recover(modality, tensor).float().cpu().numpy()
    return result


def r2_score(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(
        1.0
        - np.square(target - prediction).sum()
        / max(float(np.square(target - target.mean()).sum()), 1e-12)
    )


def decode_actions(action_model, z_values, state, device, batch_size):
    result = np.empty((len(state), 16, 128), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(state), batch_size):
            stop = min(start + batch_size, len(state))
            state_t = torch.from_numpy(np.array(state[start:stop], copy=True)).to(device)
            zero_action = torch.zeros((stop - start, 16, 128), dtype=state_t.dtype, device=device)
            embodiment = torch.full(
                (stop - start,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device
            )
            # Only current state and a fixed zero placeholder define decoder context.
            _, state_features, _ = action_model.encode(state_t, zero_action, embodiment)
            z = torch.from_numpy(np.array(z_values[start:stop], copy=True)).to(device)
            result[start:stop] = (
                action_model.decode(z, state_features, embodiment).float().cpu().numpy()
            )
    return result


def action_metrics(prediction, target, dynamic):
    from sklearn.metrics import balanced_accuracy_score

    predicted_activity = action_activity(prediction[..., :RAW_ACTION_DIM])
    target_activity = action_activity(target[..., :RAW_ACTION_DIM])
    error = np.square(
        prediction[..., :RAW_ACTION_DIM].astype(np.float64)
        - target[..., :RAW_ACTION_DIM].astype(np.float64)
    ).mean(axis=(1, 2))
    return {
        "reconstruction_mse": float(error.mean()),
        "dynamic_mse": float(error[np.asarray(dynamic, dtype=bool)].mean()),
        "magnitude_r2": r2_score(target_activity["magnitude"], predicted_activity["magnitude"]),
        "trend_r2": r2_score(target_activity["trend"], predicted_activity["trend"]),
        "active_side_balanced_accuracy": float(
            balanced_accuracy_score(
                target_activity["active_side"], predicted_activity["active_side"]
            )
        ),
        "arm_vs_hand_balanced_accuracy": float(
            balanced_accuracy_score(
                target_activity["arm_vs_hand"], predicted_activity["arm_vs_hand"]
            )
        ),
        "per_sample_mse": error,
    }


def action_evaluation(
    shared_space, action_model, split, oracle_u_a, directions, device, batch_size
):
    result = {}
    target = np.asarray(split.arrays["action"])
    state = np.asarray(split.arrays["state"])
    dynamic = np.asarray(split.arrays["dynamic"])
    for key in ("V->A", "C->A"):
        direction = directions[key]
        representations = {"predicted": direction["prediction"], **direction["control_predictions"]}
        representations["oracle_shared"] = oracle_u_a
        values = {}
        for name, shared in representations.items():
            z = recover_numpy(shared_space, "action", shared, device, batch_size)
            decoded = decode_actions(action_model, z, state, device, batch_size)
            metrics = action_metrics(decoded, target, dynamic)
            values[name] = {
                field: value for field, value in metrics.items() if field != "per_sample_mse"
            }
            values[name]["_per_sample_mse"] = metrics["per_sample_mse"]
        strongest = min(
            ("mean", "shuffled_source", "different_episode"),
            key=lambda name: values[name]["reconstruction_mse"],
        )
        improvement = values[strongest]["_per_sample_mse"] - values["predicted"]["_per_sample_mse"]
        gate = values["predicted"]["reconstruction_mse"] < values[strongest]["reconstruction_mse"]
        result[key] = {
            "representations": {
                name: {field: item for field, item in value.items() if not field.startswith("_")}
                for name, value in values.items()
            },
            "strongest_control": strongest,
            "improvement_ci95": bootstrap_mean_ci(
                improvement, samples=5000, seed=920 + len(result)
            ),
            "decoder_context": "current state plus fixed zero action placeholder; no target action chunk",
            "gate": bool(
                gate and bootstrap_mean_ci(improvement, samples=5000, seed=920 + len(result))[0] > 0
            ),
        }
    result["gate"] = all(result[key]["gate"] for key in ("V->A", "C->A"))
    return result


def vision_evaluation(shared_space, split, directions, device, batch_size):
    result = {}
    target = np.asarray(split.arrays["z_v"])
    for key in ("A->V", "C->V"):
        direction = directions[key]
        representations = {"predicted": direction["prediction"], **direction["control_predictions"]}
        values = {}
        for name, shared in representations.items():
            recovered = recover_numpy(shared_space, "vision", shared, device, batch_size)
            values[name] = {
                "mse": float(per_sample_mse(recovered, target).mean()),
                "cosine": float(row_cosine(recovered, target).mean()),
                "r2": r2_score(target, recovered),
                "cka": linear_cka(recovered, target),
            }
        strongest = min(
            ("mean", "shuffled_source", "different_episode"), key=lambda name: values[name]["mse"]
        )
        result[key] = {
            "representations": values,
            "strongest_control": strongest,
            "gate": values["predicted"]["mse"] < values[strongest]["mse"],
        }
    result["gate"] = all(result[key]["gate"] for key in ("A->V", "C->V"))
    return result


def load_ridge(path):
    value = np.load(path, allow_pickle=False)
    return {name: np.asarray(value[name]) for name in value.files}


def apply_ridge(model, source):
    x = np.asarray(source, dtype=np.float64).reshape(len(source), -1)
    prediction = (x - model["x_mean"]) @ model["coefficient"] + model["y_mean"]
    return prediction.reshape(len(source), 8, 32).astype(np.float32)


def regression_metrics(prediction, target):
    return {
        "mse": float(per_sample_mse(prediction, target).mean()),
        "r2": r2_score(target, prediction),
        "cosine": float(row_cosine(prediction, target).mean()),
    }


def private_test_diagnostic(train, test, episode, experiment_root, seed):
    result = {}
    for offset, (name, source_key, model_name) in enumerate(
        (
            ("V->r_c_priv", "u_v", "private_ridge_vision.npz"),
            ("A->r_c_priv", "u_a", "private_ridge_action.npz"),
        )
    ):
        model_path = experiment_root / model_name
        model = load_ridge(model_path)
        prediction = apply_ridge(model, test[source_key])
        shuffled = np.random.default_rng(seed + offset).permutation(len(prediction))
        different = different_episode_permutation(episode, seed + 10 + offset)
        mean = np.broadcast_to(model["y_mean"].reshape(1, 8, 32), prediction.shape)
        result[name] = {
            "selected_alpha": float(model["alpha"]),
            "model_sha256": sha256_file(model_path),
            "prediction": regression_metrics(prediction, test["r_c_priv"]),
            "mean_control": regression_metrics(mean, test["r_c_priv"]),
            "shuffled_source": regression_metrics(
                apply_ridge(model, test[source_key][shuffled]), test["r_c_priv"]
            ),
            "different_episode_source": regression_metrics(
                apply_ridge(model, test[source_key][different]), test["r_c_priv"]
            ),
        }
    return result


def source_perturbations(
    predictor,
    shared_space,
    action_model,
    s2,
    split,
    derived,
    directions,
    device,
    batch_size,
):
    count = len(split)
    reversed_action_shared = np.empty((count, 8, 32), dtype=np.float32)
    reversed_contact_shared = np.empty((count, 8, 32), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            state = torch.from_numpy(np.array(split.arrays["state"][start:stop], copy=True)).to(
                device
            )
            action = torch.from_numpy(np.array(split.arrays["action"][start:stop], copy=True)).to(
                device
            )
            embodiment = torch.full(
                (stop - start,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device
            )
            reversed_z, _, _ = action_model.encode(state, action.flip(1), embodiment)
            reversed_action_shared[start:stop] = (
                shared_space.encode("action", reversed_z).float().cpu().numpy()
            )
            current = torch.from_numpy(
                np.array(split.arrays["h_current"][start:stop], copy=True)
            ).to(device)
            future = torch.from_numpy(np.array(split.arrays["h_future"][start:stop], copy=True)).to(
                device
            )
            reversed_z_c = s2.encoder(future, current)
            reversed_contact_shared[start:stop] = (
                shared_space.encode("contact", reversed_z_c).float().cpu().numpy()
            )
    result = {}
    names = {"vision": "u_v", "action": "u_a", "contact": "u_c"}
    for key, source_name, target_name, reversed_source in (
        ("A->V", "action", "vision", reversed_action_shared),
        ("A->C", "action", "contact", reversed_action_shared),
        ("C->V", "contact", "vision", reversed_contact_shared),
        ("C->A", "contact", "action", reversed_contact_shared),
    ):
        reversed_prediction = predict_numpy(
            predictor, reversed_source, source_name, target_name, device, batch_size
        )
        target = np.asarray(derived[names[target_name]])
        result[key] = {
            "correct_mse": directions[key]["metrics"]["prediction_mse"],
            "reversed_mse": float(per_sample_mse(reversed_prediction, target).mean()),
            "shuffled_mse": directions[key]["metrics"]["controls"]["shuffled_source"],
            "different_episode_mse": directions[key]["metrics"]["controls"]["different_episode"],
        }
        result[key]["reversed_degrades"] = result[key]["reversed_mse"] > result[key]["correct_mse"]
    return result


def native_identity(args, config, shared_space, action_model, s2):
    c1_config = json.loads((ROOT / "configs/tactile_unit/c1_vac_latent_dataset.json").read_text())
    expected = c1_config["frozen_identity"]
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    tokenizer = args.unit_checkpoint / "tokenizer"
    source = ReleasedTokenizerSource.open(tokenizer)
    c1_manifest = ROOT / config["runtime"]["c1_cache_root"] / "manifest.json"
    actual = {
        "s1_teacher_file": sha256_file(args.s1_checkpoint),
        "s2_checkpoint_file": sha256_file(args.s2_checkpoint),
        "action_checkpoint_file": sha256_file(args.action_checkpoint),
        "old_action_rows": source.old_rows_digest(),
        "s2_encoder": parameter_digest(s2.encoder),
        "s2_decoder": parameter_digest(s2.decoder),
        "action_model": parameter_digest(action_model),
        "c1_cache_manifest": sha256_file(c1_manifest),
        **{
            f"vision_checkpoint/{name}": sha256_file(tokenizer / name)
            for name in expected["original_unit_tokenizer_files_sha256"]
        },
    }
    expected_values = {
        "s1_teacher_file": expected["s1_teacher_checkpoint_sha256"],
        "s2_checkpoint_file": expected["s2_checkpoint_sha256"],
        "action_checkpoint_file": expected["action_checkpoint_sha256"],
        "old_action_rows": expected["old_action_rows_digest"],
        "s2_encoder": expected["s2_encoder_parameter_digest"],
        "s2_decoder": expected["s2_decoder_parameter_digest"],
        "c1_cache_manifest": sha256_file(c1_manifest),
        **{
            f"vision_checkpoint/{name}": digest
            for name, digest in expected["original_unit_tokenizer_files_sha256"].items()
        },
    }
    equality = {name: actual[name] == value for name, value in expected_values.items()}
    return {
        "actual": actual,
        "expected": expected_values,
        "equality": equality,
        "pass": all(equality.values()),
        "shared_state_sha256": shared_digest(shared_space),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.bootstrap_samples is not None:
        config["evaluation"]["bootstrap_samples"] = int(args.bootstrap_samples)
    runtime = config["runtime"]
    artifact_root = ROOT / runtime["artifact_root"]
    experiment_root = ROOT / runtime["experiment_root"]

    # No test split is loaded before every selection/checkpoint lock passes.
    selection = validate_selection_lock(config)
    dual_path = json.loads((artifact_root / "dual_path_audit.json").read_text())
    private_validation = json.loads((artifact_root / "private_residual_analysis.json").read_text())
    if not dual_path.get("pass") or dual_path.get("test_loaded") is not False:
        raise RuntimeError("STRUCTURAL_FAIL: invalid pretest dual-path audit")
    if private_validation.get("test_loaded") is not False:
        raise RuntimeError("STRUCTURAL_FAIL: private diagnostic used test during selection")

    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("2", "3"))
    try:
        set_seed(int(config["seed"]))
        cache_root = ROOT / runtime["cache_root"]
        if not (cache_root / "test/manifest.json").is_file():
            build_split(config, "test", device, args.batch_size)
        train, train_manifest = load_derived_split(cache_root, "train")
        test, test_manifest = load_derived_split(cache_root, "test")
        ensure_cache_identities(config, train_manifest, test_manifest)
        c1_root = ROOT / runtime["c1_cache_root"]
        train_native = load_split(c1_root, "train", verify_hashes=True)
        test_native = load_split(c1_root, "test", verify_hashes=True)
        if len(test_native) != int(config["evaluation"]["rows"]):
            raise RuntimeError("STRUCTURAL_FAIL: locked C3-DP test row count changed")
        shared_space, shared_metadata, c2r_sha = load_frozen_shared_space(config, device)
        shared_before = shared_digest(shared_space)
        predictor, predictor_metadata = load_predictor_checkpoint(
            ROOT / selection["checkpoint"], device
        )
        predictor.eval().requires_grad_(False).to(device)
        if predictor_metadata.get("test_loaded") is not False:
            raise RuntimeError("STRUCTURAL_FAIL: predictor checkpoint saw test")

        names = {"vision": "u_v", "action": "u_a", "contact": "u_c"}
        episode = np.asarray(test["episode_id"])
        t = np.asarray(test["t"])
        dynamic = np.asarray(test["dynamic"])
        transition = np.asarray(test["contact_transition"])
        directions = {}
        for offset, (source_name, target_name) in enumerate(ORDERED_DIRECTIONS):
            key = SHORT_DIRECTION[(source_name, target_name)]
            directions[key] = direction_evaluation(
                predictor,
                np.asarray(test[names[source_name]]),
                np.asarray(test[names[target_name]]),
                np.asarray(train[names[target_name]]),
                source_name,
                target_name,
                episode,
                t,
                dynamic,
                transition,
                config,
                device,
                args.batch_size,
                int(config["seed"]) + offset * 100,
            )
        direction_metrics = {key: value["metrics"] for key, value in directions.items()}
        all_prediction_gates = all(value["gates"]["pass"] for value in direction_metrics.values())

        semantic_inputs_train = {
            **train,
            "force_trend_class": np.asarray(train_native.arrays["force_trend_class"]),
        }
        semantic_inputs_test = {
            **test,
            "force_trend_class": np.asarray(test_native.arrays["force_trend_class"]),
        }
        semantics = contact_semantics(
            predictor,
            semantic_inputs_train,
            semantic_inputs_test,
            directions,
            config,
            device,
            args.batch_size,
            int(config["seed"]) + 1000,
        )
        s2 = load_s2_model(args.s2_checkpoint, device).eval().requires_grad_(False)
        contact_physics_result = {}
        for key in ("V->C", "A->C"):
            direction = directions[key]
            shared_representations = {
                "predicted": direction["prediction"],
                **direction["control_predictions"],
                "oracle_shared": np.asarray(test["u_c"]),
            }
            native_representations = {
                name: recover_numpy(shared_space, "contact", value, device, args.batch_size)
                for name, value in shared_representations.items()
            }
            native_representations["full_native"] = np.asarray(test_native.arrays["z_c"])
            values = contact_physics(
                s2.decoder, test_native, native_representations, device, args.batch_size
            )
            strongest = min(
                ("mean", "shuffled_source", "different_episode"),
                key=lambda name: values[name]["metrics"]["all"]["mean"],
            )
            values["strongest_control"] = strongest
            values["gate"] = (
                values["predicted"]["metrics"]["all"]["mean"]
                < values[strongest]["metrics"]["all"]["mean"]
            )
            contact_physics_result[key] = values
        contact_physics_result["gate"] = all(
            contact_physics_result[key]["gate"] for key in ("V->C", "A->C")
        )

        if args.unit_checkpoint is None:
            raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
        source = ReleasedTokenizerSource.open(args.unit_checkpoint / "tokenizer")
        action_model, action_metadata = load_shared_transition_checkpoint(
            args.action_checkpoint, source, device
        )
        action_model.eval().requires_grad_(False).to(device)
        actions = action_evaluation(
            shared_space,
            action_model,
            test_native,
            np.asarray(test["u_a"]),
            directions,
            device,
            args.batch_size,
        )
        vision = vision_evaluation(shared_space, test_native, directions, device, args.batch_size)
        perturbations = source_perturbations(
            predictor,
            shared_space,
            action_model,
            s2,
            test_native,
            test,
            directions,
            device,
            args.batch_size,
        )
        private_test = private_test_diagnostic(
            train,
            test,
            episode,
            experiment_root,
            int(config["seed"]) + 2000,
        )
        private_test["classification"] = private_validation["classification"]
        private_test["diagnostic_only"] = True

        predicted_geometry = {}
        geometry_gate = True
        for target_name in ("vision", "action", "contact"):
            values = [
                directions[SHORT_DIRECTION[(source, target_name)]]["prediction"]
                for source, target in ORDERED_DIRECTIONS
                if target == target_name
            ]
            combined = np.concatenate(values, axis=0)
            geometry, passed = output_geometry_gate(combined)
            predicted_geometry[target_name] = geometry
            geometry_gate = geometry_gate and passed
        target_mean_collapse = any(
            value["per_dimension_variance"]["near_zero_fraction"] >= 0.5
            for value in predicted_geometry.values()
        )
        pair_losses = {key: value["prediction_mse"] for key, value in direction_metrics.items()}
        pair_ratio = max(pair_losses.values()) / max(min(pair_losses.values()), 1e-12)
        pair_imbalance_warning = pair_ratio > 10.0

        native_before = native_identity(args, config, shared_space, action_model, s2)
        shared_after = shared_digest(shared_space)
        native_after = native_identity(args, config, shared_space, action_model, s2)
        shared_integrity = shared_before == shared_after == selection["frozen_shared_state_sha256"]
        native_integrity = native_before == native_after and native_before["pass"]
        selection_hash = sha256_file(artifact_root / "selection.json")
        structural = bool(
            selection_hash == (artifact_root / "selection.sha256").read_text().split()[0]
            and c2r_sha == config["accepted_c2r_checkpoint_sha256"]
            and shared_integrity
            and native_integrity
            and geometry_gate
            and not target_mean_collapse
        )
        semantic_gate = bool(
            semantics["gate"]
            and actions["gate"]
            and vision["gate"]
            and contact_physics_result["gate"]
        )
        if not structural:
            decision = "STRUCTURAL_FAIL"
        elif not all_prediction_gates:
            decision = "C3DP_CROSS_PREDICTION_INSUFFICIENT"
        elif not semantic_gate:
            decision = "C3DP_SHARED_SEMANTIC_LOSS"
        elif (
            private_validation["classification"] == "PRIVATE_RESIDUAL_CONTAINS_STRONG_SHARED_SIGNAL"
        ):
            decision = "C3DP_SHARED_CROSS_PREDICTION_READY_WITH_PRIVATE_WARNING"
        else:
            decision = "C3DP_SHARED_CROSS_PREDICTION_READY"

        private_contact_geometry = private_geometry(
            test_native.arrays["z_c"], test["u_c"], test["z_c_shared"], test["r_c_priv"]
        )
        result = {
            "schema": "tactile3d-unit.vac-c3dp-locked-evaluation.v1",
            "decision": decision,
            "evaluation": "LOCKED BENCHMARK RE-EVALUATION",
            "first_look_untouched": False,
            "rows": len(test_native),
            "selection_frozen": True,
            "selection_artifact_sha256": selection_hash,
            "test_loaded_before_freeze": False,
            "selection": selection,
            "predictor_metadata": predictor_metadata,
            "gpu": gpu,
            "directions": direction_metrics,
            "all_six_prediction_gates": bool(all_prediction_gates),
            "contact_semantics": semantics,
            "contact_physics": contact_physics_result,
            "action_targets": actions,
            "vision_targets": vision,
            "source_temporal_perturbation": perturbations,
            "dynamic_boundary": {
                key: value["subsets"]
                for key, value in direction_metrics.items()
                if key in ("V->C", "A->C")
            },
            "private_residual": {
                "validation_classification": private_validation["classification"],
                "test_diagnostic": private_test,
                "geometry": private_contact_geometry,
                "diagnostic_only": True,
            },
            "predicted_geometry": predicted_geometry,
            "target_mean_collapse": target_mean_collapse,
            "pair_balance": {
                "prediction_mse": pair_losses,
                "max_min_ratio": float(pair_ratio),
                "PAIR_IMBALANCE_WARNING": pair_imbalance_warning,
            },
            "integrity": {
                "shared_state_before": shared_before,
                "shared_state_after": shared_after,
                "shared_space_unchanged": shared_integrity,
                "native_before": native_before,
                "native_after": native_after,
                "native_unchanged": native_integrity,
                "c2r_checkpoint_sha256": c2r_sha,
            },
            "gates": {
                "structural": structural,
                "all_six_prediction": bool(all_prediction_gates),
                "contact_semantics": bool(semantics["gate"]),
                "contact_physics": bool(contact_physics_result["gate"]),
                "action_semantics": bool(actions["gate"]),
                "vision_semantics": bool(vision["gate"]),
                "noncollapse": bool(geometry_gate and not target_mean_collapse),
            },
            "frozen_shared_metadata": shared_metadata,
            "action_checkpoint_metadata": action_metadata,
            "causal_boundary": {
                "stage": "offline representation learning",
                "offline_teachers": ["z_v", "z_a", "z_c", "u_v", "u_a", "u_c"],
                "online_legal_current_tactile": "h_t^c",
                "future_teacher_exposed_as_runtime_observation": False,
                "C5_started": False,
            },
            "scope": {
                "C4": "NOT STARTED",
                "C5": "NOT STARTED",
                "C6_M3": "NOT STARTED",
                "M3": "NOT ESTABLISHED",
            },
        }
        atomic_json(artifact_root / "locked_test_evaluation.json", result)
        atomic_json(
            artifact_root / "final_decision.json",
            {
                "schema": "tactile3d-unit.vac-c3dp-decision.v1",
                "decision": decision,
                "reasons": {
                    "structural": structural,
                    "all_six_prediction_gates": bool(all_prediction_gates),
                    "semantic_gate": semantic_gate,
                    "private_classification": private_validation["classification"],
                },
                "C4": "NOT STARTED",
                "C5": "NOT STARTED",
                "C6_M3": "NOT STARTED",
                "M3": "NOT ESTABLISHED",
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
