#!/usr/bin/env python3
"""Run the frozen PI2N common-reference gradient and action-order probes."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_optimization_probe.json"
DECISION_PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_refinement_decision_protocol.json"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
CONTACT = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
POLICY_RESULTS = ARTIFACTS / "policy_domain_probe_results_v2.json"
GRADIENT_OUTPUT = ARTIFACTS / "auxiliary_gradient_diagnostics.json"
P4_OUTPUT = ARTIFACTS / "action_hidden_order_probe.json"
DECISION_OUTPUT = ARTIFACTS / "refinement_decision.json"


def run_root() -> Path:
    path = Path(os.environ.get("PI2N_RUN_ROOT", ROOT / ".local/experiments/simulation/s4_3_pi2n")).resolve()
    expected = Path("/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n")
    if path != expected:
        raise RuntimeError(f"PI2N run root differs from audited NAS root: {path}")
    return path


def vac_target_path() -> Path:
    return run_root() / "caches/pinch_tongs_vac_v_t27/sidecar.npz"


def hidden_cache_path() -> Path:
    return run_root() / "caches/action_hidden_order_probe/hidden_first27.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def batch_sha256(tree: Any) -> str:
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for index, value in enumerate(jax.tree.leaves(jax.device_get(tree))):
        array = np.asarray(value)
        digest.update(f"{index}\0{array.dtype}\0{array.shape}\0".encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def standardized_ridge(x_fit, y_fit, x_check, alpha: float):
    import numpy as np

    x_mean = x_fit.mean(axis=0, dtype=np.float64)
    x_std = x_fit.std(axis=0, dtype=np.float64)
    x_std = np.where(x_std > 1e-6, x_std, 1.0)
    y_mean = y_fit.mean(axis=0, dtype=np.float64)
    y_std = y_fit.std(axis=0, dtype=np.float64)
    y_std = np.where(y_std > 1e-6, y_std, 1.0)
    train_x = (x_fit.astype(np.float64) - x_mean) / x_std
    train_y = (y_fit.astype(np.float64) - y_mean) / y_std
    gram = train_x.T @ train_x / len(train_x)
    rhs = train_x.T @ train_y / len(train_x)
    weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    prediction = ((x_check.astype(np.float64) - x_mean) / x_std) @ weights
    prediction = prediction * y_std + y_mean
    return prediction.astype(np.float32), {
        "feature_constant_dimensions": int(np.count_nonzero(x_fit.std(axis=0) <= 1e-6)),
        "label_constant_dimensions": int(np.count_nonzero(y_fit.std(axis=0) <= 1e-6)),
        "weight_parameters": int(weights.size),
        "intercept_parameters": int(y_fit.shape[1]),
        "trainable_parameters": int(weights.size + y_fit.shape[1]),
        "weight_frobenius_norm": float(np.linalg.norm(weights)),
    }


def normalized_mse(prediction, target, variance) -> float:
    import numpy as np

    return float(np.mean(np.square(prediction - target) / np.maximum(variance, 1e-8)))


def paired_interval(values, samples: int, seed: int) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "mean_mean_probe_minus_order_probe": float(values.mean()),
        "ci95": [float(low), float(high)],
        "episodes": int(len(values)),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def load_model_and_dataset(protocol):
    import flax.traverse_util
    from flax import nnx
    import jax
    from gr00t.simulation.pi05_tactile_unit import TactilePi0
    from openpi.shared import array_typing as at
    from openpi.training import data_loader
    from scripts.simulation.train_s4_3_pi1 import build_config

    config = dataclasses.replace(
        build_config("CONTACT_STATE_TOKENS_PHYSICAL_AUX", "s43_pi2n_common_reference", protocol["gradient_metrics"]["lambda_C"]),
        batch_size=32,
        num_workers=0,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.transform_dataset(dataset, data_config)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    _, model_rng = jax.random.split(init_rng)
    model = config.model.create(model_rng)
    graphdef, state = nnx.split(model)
    reference = state.to_pure_dict()
    loaded = config.weight_loader.load(reference)
    at.check_pytree_equality(expected=reference, got=loaded, check_shapes=True, check_dtypes=True)
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    model.train()
    if type(model) is not TactilePi0:
        raise RuntimeError("common reference did not instantiate TactilePi0")
    flat = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    lora_names = sorted(name for name in flat if "lora" in name)
    return config, data_config, dataset, model, lora_names


def make_batch(dataset, rows):
    import jax
    import jax.numpy as jnp
    from gr00t.simulation.pi05_tactile_unit import TactileObservation
    from openpi.training import data_loader

    values = data_loader._collate_fn([dataset[int(row)] for row in rows])
    values = jax.tree.map(jnp.asarray, values)
    return TactileObservation.from_dict(values), values["actions"]


def gradient_probe(protocol, dataset, model, arrays, vac_values, lora_names):
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from openpi.models.pi0 import Pi0
    from openpi.shared import nnx_utils

    batch_spec = protocol["gradient_batches"]
    rng = np.random.default_rng(int(batch_spec["row_selection_seed"]))
    valid = arrays["valid"]
    active = arrays["active"]
    active_rows = np.flatnonzero(valid & active)
    free_rows = np.flatnonzero(valid & ~active)
    rng.shuffle(active_rows)
    rng.shuffle(free_rows)
    selected_batches = []
    for batch_index in range(int(batch_spec["count"])):
        rows = np.concatenate((active_rows[16 * batch_index : 16 * (batch_index + 1)], free_rows[16 * batch_index : 16 * (batch_index + 1)]))
        rng.shuffle(rows)
        selected_batches.append(rows)

    gradient_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora.*"))

    def loss_fn(kind):
        def calculate(candidate, loss_rng, observation, actions, target, mask):
            chunked, hidden = Pi0.compute_loss_with_hidden(candidate, loss_rng, observation, actions, train=True)
            if kind == "pi":
                per_sample = jnp.mean(chunked, axis=-1)
            else:
                prediction = candidate.physical_auxiliary(hidden)
                per_sample = jnp.mean(jnp.square(prediction - jax.lax.stop_gradient(target)), axis=(-2, -1))
            weight = mask.astype(per_sample.dtype)
            return jnp.sum(per_sample * weight) / jnp.maximum(jnp.sum(weight), 1.0)

        transformed = nnx.value_and_grad(calculate, argnums=nnx.DiffState(0, gradient_filter))
        return nnx.jit(transformed)

    gradient_functions = {kind: loss_fn(kind) for kind in ("pi", "V", "C")}

    def flatten(grads):
        leaves = jax.tree.leaves(grads)
        return leaves, optax.global_norm(grads)

    def cosine(left, right):
        left_leaves, left_norm = flatten(left)
        right_leaves, right_norm = flatten(right)
        dot = sum((jnp.vdot(a, b) for a, b in zip(left_leaves, right_leaves, strict=True)), jnp.asarray(0.0))
        return dot / jnp.maximum(left_norm * right_norm, jnp.asarray(1e-12))

    rows_output = []
    for batch_index, rows in enumerate(selected_batches):
        observation, actions = make_batch(dataset, rows)
        target_c = observation.contact_shared_target
        target_v = jnp.asarray(vac_values[rows])
        loss_rng = jax.random.fold_in(jax.random.key(42), 9100 + batch_index)
        _, _, time_rng = jax.random.split(loss_rng, 3)
        times = np.asarray(jax.device_get(jax.random.beta(time_rng, 1.5, 1, (32,)) * 0.999 + 0.001))
        contact = active[rows]
        groups = {
            "overall_valid": np.ones(32, dtype=bool),
            "contact_active_valid": contact,
            "contact_free_valid": ~contact,
        }
        edges = (0.001, 0.25, 0.5, 0.75, 1.000001)
        for index in range(4):
            groups[f"flow_time_{index}"] = (times >= edges[index]) & (times < edges[index + 1])
        group_output = {}
        for group_name, mask_value in groups.items():
            sample_count = int(mask_value.sum())
            if sample_count == 0:
                group_output[group_name] = {"samples": 0, "status": "NA_EMPTY"}
                continue
            mask = jnp.asarray(mask_value)
            values = {}
            gradients = {}
            for kind, target in (("pi", target_c), ("V", target_v), ("C", target_c)):
                value, grads = gradient_functions[kind](model, loss_rng, observation, actions, target, mask)
                values[kind] = float(jax.device_get(value))
                gradients[kind] = grads
            norms = {kind: float(jax.device_get(flatten(grads)[1])) for kind, grads in gradients.items()}
            cos_v = float(jax.device_get(cosine(gradients["pi"], gradients["V"])))
            cos_c = float(jax.device_get(cosine(gradients["pi"], gradients["C"])))
            group_output[group_name] = {
                "status": "PASS",
                "samples": sample_count,
                "loss": values,
                "norm": norms,
                "cos_g_pi_g_V": cos_v,
                "cos_g_pi_g_C": cos_c,
                "weighted_norm_ratio_V": float(protocol["gradient_metrics"]["lambda_V"] * norms["V"] / max(norms["pi"], 1e-12)),
                "weighted_norm_ratio_C": float(protocol["gradient_metrics"]["lambda_C"] * norms["C"] / max(norms["pi"], 1e-12)),
            }
        rows_output.append({
            "batch_index": batch_index,
            "row_indices": rows.astype(int).tolist(),
            "row_indices_sha256": hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(),
            "batch_tree_sha256": batch_sha256((observation, actions)),
            "loss_rng_key_data": np.asarray(jax.random.key_data(loss_rng)).astype(int).tolist(),
            "flow_times": times.tolist(),
            "current_contact_active_samples": int(contact.sum()),
            "groups": group_output,
        })

    summaries = {}
    qualifying = []
    for group_name in batch_spec["allowed_reference_groups"]:
        records = [row["groups"][group_name] for row in rows_output if row["groups"][group_name]["status"] == "PASS"]
        negative = sum(record["cos_g_pi_g_C"] < 0 for record in records)
        ratios = [record["weighted_norm_ratio_C"] for record in records]
        qualifies = bool(records and negative > len(records) / 2 and float(np.median(ratios)) > 0.5)
        if qualifies:
            qualifying.append(group_name)
        summaries[group_name] = {
            "nonempty_batches": len(records),
            "negative_cos_g_pi_g_C_batches": negative,
            "more_than_half_negative": bool(records and negative > len(records) / 2),
            "median_weighted_norm_ratio_C": None if not ratios else float(np.median(ratios)),
            "qualifies_for_C_REBALANCE_group": qualifies,
        }
    gates = {
        "four_batches": len(rows_output) == 4,
        "batch32": all(len(row["row_indices"]) == 32 for row in rows_output),
        "balanced_contact": all(row["current_contact_active_samples"] == 16 for row in rows_output),
        "all_gradient_values_finite": all(
            math.isfinite(value)
            for row in rows_output
            for record in row["groups"].values()
            if record["status"] == "PASS"
            for key in ("cos_g_pi_g_V", "cos_g_pi_g_C", "weighted_norm_ratio_V", "weighted_norm_ratio_C")
            for value in (record[key],)
        ),
        "shared_lora_paths_nonempty": len(lora_names) > 0,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2n-auxiliary-gradient-diagnostics.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "reference": "exact pi05_base, B2 structure, optimizer step zero",
        "shared_lora_leaf_paths": lora_names,
        "batches": rows_output,
        "group_summary": summaries,
        "C_REBALANCE_qualifying_groups": qualifying,
        "C_REBALANCE_threshold_pass": len(qualifying) >= 2,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "candidate_policy_checkpoints_read": False,
        "rollout_outcomes_read": False,
        "interpretation": "common-initialization TRAIN gradient evidence; negative cosine is not closed-loop causality",
    }
    return payload


def p4_probe(protocol, dataset, model, arrays):
    import jax
    import numpy as np
    from openpi.models.pi0 import Pi0

    p4 = protocol["p4"]
    selected = []
    for episode_value in range(100):
        episode_rows = np.flatnonzero((arrays["episode"] == episode_value) & arrays["valid"])
        positions = np.rint(np.linspace(0, len(episode_rows) - 1, 5)).astype(int)
        rows = episode_rows[positions]
        if len(np.unique(rows)) != 5:
            raise RuntimeError(f"episode {episode_value} lacks five unique valid diagnostic rows")
        selected.extend(rows.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    hidden_values = []
    batch_hashes = []
    batch_size = 32
    for batch_index, start in enumerate(range(0, len(selected), batch_size)):
        rows = selected[start : start + batch_size]
        actual = len(rows)
        if actual < batch_size:
            rows = np.pad(rows, (0, batch_size - actual), mode="edge")
        observation, actions = make_batch(dataset, rows)
        loss_rng = jax.random.fold_in(jax.random.key(42), 12000 + batch_index)
        _, hidden = Pi0.compute_loss_with_hidden(model, loss_rng, observation, actions, train=True)
        hidden_values.append(np.asarray(jax.device_get(hidden[:actual, :27]), dtype=np.float32))
        batch_hashes.append({
            "batch_index": batch_index,
            "actual_rows": actual,
            "batch_tree_sha256": batch_sha256((observation, actions)),
            "loss_rng_key_data": np.asarray(jax.random.key_data(loss_rng)).astype(int).tolist(),
        })
    hidden = np.concatenate(hidden_values, axis=0)
    target = arrays["contact_target"][selected].reshape(len(selected), -1).astype(np.float32)
    episode = arrays["episode"][selected]
    fit = episode % 5 != 4
    check = ~fit
    projection_rng = np.random.default_rng(int(p4["projection_seed"]))
    mean_projection = projection_rng.normal(size=(1024, 648)).astype(np.float32) / np.sqrt(1024.0)
    order_projection = projection_rng.normal(size=(1024, 24)).astype(np.float32) / np.sqrt(1024.0)
    mean_features = hidden.mean(axis=1) @ mean_projection
    order_features = (hidden @ order_projection).reshape(len(hidden), -1)
    mean_prediction, mean_detail = standardized_ridge(mean_features[fit], target[fit], mean_features[check], 0.001)
    order_prediction, order_detail = standardized_ridge(order_features[fit], target[fit], order_features[check], 0.001)
    variance = np.var(target[fit], axis=0, dtype=np.float64).astype(np.float32)
    mean_error = normalized_mse(mean_prediction, target[check], variance)
    order_error = normalized_mse(order_prediction, target[check], variance)
    relative_improvement = (mean_error - order_error) / max(mean_error, 1e-12)
    episode_values = []
    per_episode = {}
    check_episode = episode[check]
    check_target = target[check]
    for episode_value in sorted(np.unique(check_episode)):
        mask = check_episode == episode_value
        left = normalized_mse(mean_prediction[mask], check_target[mask], variance)
        right = normalized_mse(order_prediction[mask], check_target[mask], variance)
        episode_values.append(left - right)
        per_episode[str(int(episode_value))] = {"mean_probe": left, "order_probe": right, "mean_minus_order": left - right}
    interval = paired_interval(episode_values, int(p4["bootstrap_samples"]), int(p4["bootstrap_seed"]))
    route_pass = relative_improvement >= 0.10 and interval["ci95"][0] > 0
    cache = hidden_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, row_index=selected, episode_index=episode, hidden_first27=hidden)
    temporary.replace(cache)
    gates = {
        "rows_500": len(selected) == 500,
        "five_rows_each_episode": all(int(np.sum(episode == value)) == 5 for value in range(100)),
        "fit_episodes_80": len(np.unique(episode[fit])) == 80,
        "check_episodes_20": len(np.unique(episode[check])) == 20,
        "feature_width_equal_648": mean_features.shape[1] == order_features.shape[1] == 648,
        "trainable_parameters_equal_166144": mean_detail["trainable_parameters"] == order_detail["trainable_parameters"] == 166144,
        "finite_errors": math.isfinite(mean_error) and math.isfinite(order_error),
    }
    return {
        "schema": "tactile3d-unit.s4-3-pi2n-action-hidden-order-probe.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "rows": 500,
        "fit_rows": int(fit.sum()),
        "check_rows": int(check.sum()),
        "hidden_shape": list(hidden.shape),
        "hidden_cache": "$PI2N_RUN_ROOT/caches/action_hidden_order_probe/hidden_first27.npz",
        "hidden_cache_sha256": sha256_file(cache),
        "batch_provenance": batch_hashes,
        "mean_probe": {"normalized_mse": mean_error, **mean_detail},
        "order_probe": {"normalized_mse": order_error, **order_detail},
        "relative_normalized_mse_improvement": relative_improvement,
        "paired_episode_interval": interval,
        "per_check_episode": per_episode,
        "C_ORDER_threshold_pass": route_pass,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "interpretation": "frozen-feature training-domain routing probe; not a policy or closed-loop result",
        "candidate_policy_checkpoints_read": False,
        "rollout_outcomes_read": False,
    }


def route_decision(protocol, gradient, p4):
    policy = json.loads(POLICY_RESULTS.read_text())
    prerequisites = {
        "teacher_policy_domain": json.loads((ARTIFACTS / "teacher_policy_domain_audit.json").read_text())["status"] == "PASS",
        "vision_contact_provenance": json.loads((ARTIFACTS / "vision_contact_gradient_provenance.json").read_text())["status"] == "PASS",
        "teacher_comparability": (
            json.loads((ARTIFACTS / "teacher_comparability.json").read_text())["status"] == "PASS_WITH_LIMITATION"
            and json.loads((ARTIFACTS / "teacher_comparability.json").read_text())["decision"] == "PACKAGE_LEVEL_TEACHER_COMPARISON_ONLY"
        ),
        "B2_target": json.loads((ARTIFACTS / "b_va27_target_audit.json").read_text())["status"] == "PASS",
        "VAC_V_target": json.loads((ARTIFACTS / "b_vac_v_target_audit.json").read_text())["status"] == "PASS",
        "gradient_diagnostic": gradient["status"] == "PASS",
        "P4": p4["status"] == "PASS",
    }
    blocked = not all(prerequisites.values())
    rebalanced = gradient["C_REBALANCE_threshold_pass"]
    ordered = p4["C_ORDER_threshold_pass"]
    metrics = policy["subset_metrics"]
    controls = policy["controls"]
    c_overall = metrics["u_c_VAC"]["all"]["groups"]["all"]
    v_overall = metrics["u_v_VAC"]["all"]["groups"]["all"]
    mean_overall = controls["TRAIN_mean"]["all"]["groups"]["all"]
    both_informative = c_overall <= 0.9 * mean_overall and v_overall <= 0.9 * mean_overall
    event_metrics = []
    for subset in ("boundary", "dynamic"):
        for metric in ("region_occupancy_change", "log_force_magnitude_trend"):
            c_value = metrics["u_c_VAC"][subset]["groups"][metric]
            v_value = metrics["u_v_VAC"][subset]["groups"][metric]
            event_metrics.append({
                "subset": subset,
                "metric": metric,
                "u_c": c_value,
                "u_v": v_value,
                "contact_relative_improvement": (v_value - c_value) / max(v_value, 1e-12),
            })
    complement = any(row["contact_relative_improvement"] >= 0.05 for row in event_metrics)
    mismatch = controls["different_episode_mismatch"]
    no_identity = mismatch["u_c_VAC"] >= 1.1 * c_overall and mismatch["u_v_VAC"] >= 1.1 * v_overall
    joint = both_informative and complement and no_identity
    if blocked:
        route = "BLOCKED_TEACHER"
    elif rebalanced:
        route = "C_REBALANCE"
    elif ordered:
        route = "C_ORDER"
    elif joint:
        route = "C_JOINT"
    else:
        route = "NOT_RUN_NOT_JUSTIFIED"
    return {
        "schema": "tactile3d-unit.s4-3-pi2n-refinement-decision.v1",
        "status": "COMPLETE_VALID" if route != "BLOCKED_TEACHER" else "BLOCKED_DEPENDENCY",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "decision_protocol_sha256": sha256_file(DECISION_PROTOCOL),
        "route": route,
        "X_policy_authorized": route in {"C_REBALANCE", "C_ORDER", "C_JOINT"},
        "priority_applied_once": protocol["route"]["priority"],
        "prerequisites": {name: "PASS" if value else "FAIL" for name, value in prerequisites.items()},
        "C_REBALANCE": {
            "pass": rebalanced,
            "qualifying_groups": gradient["C_REBALANCE_qualifying_groups"],
            "required_groups": 2,
        },
        "C_ORDER": {
            "pass": ordered,
            "relative_normalized_mse_improvement": p4["relative_normalized_mse_improvement"],
            "paired_ci95": p4["paired_episode_interval"]["ci95"],
        },
        "C_JOINT": {
            "pass": joint,
            "both_targets_beat_mean_by_10pct": both_informative,
            "contact_event_or_trend_complement_5pct": complement,
            "different_episode_mismatch_10pct": no_identity,
            "event_metrics": event_metrics,
        },
        "candidate_policy_checkpoints_read": False,
        "rollout_outcomes_read": False,
        "interpretation": "one-shot preregistered route decision; NOT_RUN_NOT_JUSTIFIED is a valid informative outcome",
    }


def main() -> None:
    for path in (GRADIENT_OUTPUT, P4_OUTPUT, DECISION_OUTPUT, hidden_cache_path()):
        if path.exists():
            raise SystemExit(f"refusing to overwrite N2-O output: {path}")
    protocol = json.loads(PROTOCOL.read_text())
    decision_protocol = json.loads(DECISION_PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_NUMERICAL_DIAGNOSTIC":
        raise SystemExit("optimization probe protocol is not frozen")
    if decision_protocol.get("status") != "FROZEN_BEFORE_READING_N2_O_RESULTS":
        raise SystemExit("refinement decision protocol is not frozen")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("set CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise SystemExit("N2-O requires exactly one explicitly visible GPU")

    os.chdir(OPENPI)
    sys.path[:0] = [str(ROOT), str(OPENPI / "src")]
    import jax
    import numpy as np

    if jax.device_count() != 1 or jax.devices()[0].platform != "gpu":
        raise SystemExit("N2-O did not resolve exactly one GPU")
    with np.load(CONTACT, allow_pickle=False) as source:
        arrays = {
            "index": source["index"].copy(),
            "episode": source["episode_index"].copy(),
            "valid": source["physical_aux_valid"].copy(),
            "active": np.any(source["tactile_sim"].reshape(-1, 5, 6)[:, :, 0] > 0.5, axis=1),
            "contact_target": source["contact_shared_target"].copy(),
        }
    with np.load(vac_target_path(), allow_pickle=False) as source:
        if not np.array_equal(arrays["index"], source["index"]) or not np.array_equal(arrays["valid"], source["vac_aux_valid"]):
            raise RuntimeError("VAC-V target row or validity identity differs from B2")
        vac_values = source["vac_vision_target"].copy()

    config, _, dataset, model, lora_names = load_model_and_dataset(protocol)
    gradient = gradient_probe(protocol, dataset, model, arrays, vac_values, lora_names)
    if gradient["status"] != "PASS":
        atomic_json(GRADIENT_OUTPUT, gradient)
        raise SystemExit("common-reference gradient diagnostic failed")
    p4 = p4_probe(protocol, dataset, model, arrays)
    atomic_json(GRADIENT_OUTPUT, gradient)
    atomic_json(P4_OUTPUT, p4)
    if p4["status"] != "PASS":
        raise SystemExit("P4 action-order diagnostic failed")
    decision = route_decision(protocol, gradient, p4)
    decision.update({
        "device": str(jax.devices()[0]),
        "base_params": str(config.weight_loader.params_path),
        "contact_sidecar_sha256": sha256_file(CONTACT),
        "vac_v_target_sha256": sha256_file(vac_target_path()),
    })
    atomic_json(DECISION_OUTPUT, decision)
    print(json.dumps({
        "status": "PASS",
        "route": decision["route"],
        "C_REBALANCE": gradient["C_REBALANCE_threshold_pass"],
        "C_ORDER": p4["C_ORDER_threshold_pass"],
        "P4_relative_improvement": p4["relative_normalized_mse_improvement"],
        "P4_ci95": p4["paired_episode_interval"]["ci95"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
