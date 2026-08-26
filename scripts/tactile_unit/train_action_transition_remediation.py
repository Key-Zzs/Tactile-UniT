#!/usr/bin/env python3
"""Train R1-P, then escalate to R1-N only when validation gates require it."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
    effective_rank,
    latent_noncollapse_losses,
)
from gr00t.tactile_unit.trex_action_data import (  # noqa: E402
    RAW_ACTION_DIM,
    TReXActionCache,
    action_activity,
    different_episode_indices,
)
from gr00t.tactile_unit.trex_action_transition import (  # noqa: E402
    NativeTransitionActionModel,
    TransitionFeatureStats,
    build_shared_candidate,
    fit_transition_feature_stats,
    load_shared_transition_checkpoint,
    load_transition_checkpoint,
    save_shared_transition_checkpoint,
    save_transition_checkpoint,
)


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/s3_3_r_action_transition_remediation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=None)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_isolated_gpu(config: Mapping[str, Any]) -> tuple[torch.device, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A-R training requires exactly one visible CUDA device")
    physical = int(os.environ.get("TACTILE_PHYSICAL_GPU", "-1"))
    if physical not in set(map(int, config["gpu"]["allowed_physical"])):
        raise RuntimeError("A-R training received a forbidden physical GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("physical/logical GPU isolation mismatch")
    return torch.device("cuda:0"), physical


def selected_indices(length: int, count: int) -> np.ndarray:
    count = min(length, int(count))
    return np.floor((np.arange(count) + 0.5) * length / count).astype(np.int64)


def make_tensors(cache: TReXActionCache, indices: np.ndarray, device: torch.device):
    batch = cache.batch(indices)
    state = torch.from_numpy(batch["state"]).to(device)
    action = torch.from_numpy(batch["action"]).to(device)
    embodiment = torch.full((len(indices),), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
    return batch, state, action, embodiment


def reconstruction_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(
        prediction[..., :RAW_ACTION_DIM].float(), target[..., :RAW_ACTION_DIM], reduction="none"
    ).mean(dim=(1, 2))


def transition_losses(model: torch.nn.Module, state: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    relative_prediction = model.features.relative_target(state[:, :RAW_ACTION_DIM], prediction[..., :RAW_ACTION_DIM])
    relative_target = model.features.relative_target(state[:, :RAW_ACTION_DIM], target[..., :RAW_ACTION_DIM])
    velocity_prediction = model.features.velocity_target(prediction[..., :RAW_ACTION_DIM])
    velocity_target = model.features.velocity_target(target[..., :RAW_ACTION_DIM])
    return (
        F.smooth_l1_loss(relative_prediction.float(), relative_target.float(), reduction="none").mean(dim=(1, 2)),
        F.smooth_l1_loss(velocity_prediction.float(), velocity_target.float(), reduction="none").mean(dim=(1, 2)),
    )


@torch.no_grad()
def validation_metrics(
    model: torch.nn.Module,
    cache: TReXActionCache,
    *,
    count: int,
    temporal_count: int,
    batch_size: int,
    device: torch.device,
    dynamic_threshold: float,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    indices = selected_indices(len(cache), count)
    temporal_limit = min(len(indices), int(temporal_count))
    rng = np.random.default_rng(seed)
    records: dict[str, list[np.ndarray]] = {
        name: [] for name in ("correct", "reversed", "shuffled", "different_episode", "zero")
    }
    dynamic_records: dict[str, list[np.ndarray]] = {name: [] for name in records}
    reconstruction_errors: list[np.ndarray] = []
    latents: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    embodiments: list[torch.Tensor] = []
    full_predictions: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        batch, state, action, embodiment = make_tensors(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(state, action, embodiment)
        correct_error = reconstruction_per_sample(output["prediction"], action).cpu().numpy()
        reconstruction_errors.append(correct_error)
        activity = action_activity(batch["action"][..., :RAW_ACTION_DIM])
        dynamic = activity["magnitude"] > dynamic_threshold
        full_predictions.append(output["prediction"][..., :RAW_ACTION_DIM].float().cpu().numpy())
        latents.append(output["z_action"].float().cpu())
        states.append(output["state_features"].detach().cpu())
        targets.append(action.detach().cpu())
        embodiments.append(embodiment.detach().cpu())
        if start >= temporal_limit:
            continue
        allowed = min(len(current), temporal_limit - start)
        state_t = state[:allowed]
        action_t = action[:allowed]
        embodiment_t = embodiment[:allowed]
        different = cache.batch(different_episode_indices(cache, current[:allowed]))
        controls = {
            "reversed": action_t.flip(1),
            "shuffled": action_t[:, torch.from_numpy(rng.permutation(16)).to(device)],
            "different_episode": torch.from_numpy(different["action"]).to(device),
        }
        dynamic_t = dynamic[:allowed]
        records["correct"].append(correct_error[:allowed])
        dynamic_records["correct"].append(correct_error[:allowed][dynamic_t])
        for name, candidate in controls.items():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(state_t, candidate, embodiment_t)["prediction"]
            error = reconstruction_per_sample(prediction, action_t).cpu().numpy()
            records[name].append(error)
            dynamic_records[name].append(error[dynamic_t])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            zero_prediction = model.decode(
                torch.zeros_like(output["z_action"][:allowed]),
                output["state_features"][:allowed],
                embodiment_t,
            )
        zero_error = reconstruction_per_sample(zero_prediction, action_t).cpu().numpy()
        records["zero"].append(zero_error)
        dynamic_records["zero"].append(zero_error[dynamic_t])
    all_z = torch.cat(latents)
    mean_z = all_z.mean(dim=0)
    mean_errors = []
    mean_dynamic_errors = []
    cursor = 0
    for state_features, target, embodiment in zip(states, targets, embodiments):
        take = min(len(target), max(0, temporal_limit - cursor))
        if take == 0:
            break
        target = target[:take].to(device)
        state_features = state_features[:take].to(device)
        embodiment = embodiment[:take].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model.decode(mean_z.to(device).unsqueeze(0).expand(take, -1, -1), state_features, embodiment)
        error = reconstruction_per_sample(prediction, target).cpu().numpy()
        mean_errors.append(error)
        target_np = target[..., :RAW_ACTION_DIM].cpu().numpy()
        dynamic = action_activity(target_np)["magnitude"] > dynamic_threshold
        mean_dynamic_errors.append(error[dynamic])
        cursor += take
    records["mean"] = mean_errors
    dynamic_records["mean"] = mean_dynamic_errors
    means = {name: float(np.concatenate(values).mean()) for name, values in records.items()}
    dynamic_means = {
        name: float(np.concatenate(values).mean()) if sum(len(item) for item in values) else float("nan")
        for name, values in dynamic_records.items()
    }
    result = {
        "selection_split": "validation",
        "windows": len(indices),
        "temporal_windows": temporal_limit,
        "normalized_mse": float(np.concatenate(reconstruction_errors).mean()),
        "finite": bool(np.isfinite(np.concatenate(full_predictions)).all() and torch.isfinite(all_z).all()),
        "z_action_shape_without_batch": list(all_z.shape[1:]),
        "all": means,
        "dynamic": dynamic_means,
        "effective_rank": effective_rank(all_z.flatten(1)),
        "collapsed_query_fraction": float(
            (all_z.numpy().var(axis=(0, 2)) < 1e-8).mean()
        ),
    }
    for subset, values in (("all", means), ("dynamic", dynamic_means)):
        for name in ("reversed", "shuffled", "different_episode", "zero", "mean"):
            result[f"{subset}_{name}_ratio"] = values[name] / max(values["correct"], 1e-12)
    return result


def validation_selection_key(metrics: Mapping[str, Any], acceptance: Mapping[str, Any]) -> tuple[tuple[float, ...], bool, float]:
    limits = {
        "normalized_mse": float(acceptance["normalized_mse_max"]),
        "dynamic_reversed_ratio": float(acceptance["dynamic_reversed_ratio_min"]),
        "dynamic_shuffled_ratio": float(acceptance["dynamic_shuffled_ratio_min"]),
        "all_different_episode_ratio": float(acceptance["dynamic_reversed_ratio_min"]),
        "all_zero_ratio": float(acceptance["zero_ratio_min"]),
        "all_mean_ratio": float(acceptance["mean_ratio_min"]),
    }
    shortfall = max(
        0.0, float(metrics["normalized_mse"]) / limits.pop("normalized_mse") - 1.0
    ) + sum(
        max(0.0, limit - float(metrics[name])) / limit for name, limit in limits.items()
    )
    noncollapse = (
        float(metrics["effective_rank"]) >= float(acceptance["effective_rank_min"])
        and float(metrics["collapsed_query_fraction"])
        <= float(acceptance["collapsed_query_fraction_max"])
    )
    if not noncollapse:
        shortfall += 1.0
    passed = (
        shortfall == 0.0
        and bool(metrics["finite"])
        and metrics["z_action_shape_without_batch"] == [8, 32]
    )
    return (0.0 if passed else 1.0, shortfall, float(metrics["normalized_mse"])), passed, shortfall


def save_candidate(path: Path, model: torch.nn.Module, metadata: Mapping[str, Any]) -> str:
    if isinstance(model, NativeTransitionActionModel):
        return save_transition_checkpoint(path, model, metadata)
    return save_shared_transition_checkpoint(path, model, metadata)


def load_candidate(path: Path, candidate: str, source: ReleasedTokenizerSource):
    if candidate == "R1-N":
        return load_transition_checkpoint(path)
    return load_shared_transition_checkpoint(path, source)


def train_candidate(
    model: torch.nn.Module,
    *,
    candidate: str,
    options: Mapping[str, Any],
    config: Mapping[str, Any],
    train_cache: TReXActionCache,
    val_cache: TReXActionCache,
    dynamic_threshold: float,
    device: torch.device,
    steps: int,
    checkpoint: Path,
) -> dict[str, Any]:
    model = model.to(device)
    model.train()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(options["learning_rate"]),
        weight_decay=float(options.get("weight_decay", 0.0)),
    )
    rng = np.random.default_rng(int(config["seed"]) + (0 if candidate == "R1-P" else 1))
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 19)
    batch_size = int(options["batch_size"])
    validation_interval = min(int(options["validation_interval"]), steps)
    history = []
    best_key = None
    best_step = 0
    best_metrics = None
    start_time = time.monotonic()
    loss_weights = options.get("loss_weights", {
        "absolute": 1.0, "relative": 0.5, "velocity": 0.5,
        "temporal": 0.5, "token_necessity": 0.25, "variance": 0.01, "query_diversity": 0.01,
    })
    negative_names = ("reversed", "shuffled", "different_episode")
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(train_cache), size=batch_size, endpoint=False)
        batch, state, action, embodiment = make_tensors(train_cache, indices, device)
        dynamic = torch.from_numpy(
            (action_activity(batch["action"][..., :RAW_ACTION_DIM])["magnitude"] > dynamic_threshold).astype(np.float32)
        ).to(device)
        sample_weight = 1.0 + (float(options["dynamic_weight"]) - 1.0) * dynamic
        negative_name = negative_names[(step - 1) % len(negative_names)]
        if negative_name == "reversed":
            negative = action.flip(1)
        elif negative_name == "shuffled":
            negative = action[:, torch.randperm(16, generator=generator).to(device)]
        else:
            other = train_cache.batch(different_episode_indices(train_cache, indices))
            negative = torch.from_numpy(other["action"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(state, action, embodiment)
            negative_output = model(state, negative, embodiment)
            absolute = reconstruction_per_sample(output["prediction"], action)
            relative, velocity = transition_losses(model, state, output["prediction"], action)
            negative_error = reconstruction_per_sample(negative_output["prediction"], action)
            temporal = torch.relu(float(options["temporal_margin"]) + absolute - negative_error)
            zero_prediction = model.decode(torch.zeros_like(output["z_action"]), output["state_features"], embodiment)
            zero_error = reconstruction_per_sample(zero_prediction, action)
            token_necessity = torch.relu(float(options["temporal_margin"]) + absolute - zero_error)
            variance, diversity = latent_noncollapse_losses(output["z_action"].float())
            weighted = lambda value: (value * sample_weight).mean() / sample_weight.mean()
            loss = (
                float(loss_weights["absolute"]) * weighted(absolute)
                + float(loss_weights["relative"]) * weighted(relative)
                + float(loss_weights["velocity"]) * weighted(velocity)
                + float(loss_weights["temporal"]) * weighted(temporal)
                + float(loss_weights["token_necessity"]) * weighted(token_necessity)
                + float(loss_weights["variance"]) * variance
                + float(loss_weights["query_diversity"]) * diversity
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {candidate} loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % validation_interval == 0 or step == steps:
            metrics = validation_metrics(
                model,
                val_cache,
                count=int(options["validation_windows"]),
                temporal_count=int(options["temporal_windows"]),
                batch_size=batch_size,
                device=device,
                dynamic_threshold=dynamic_threshold,
                seed=int(config["seed"]) + 73,
            )
            key, passed, shortfall = validation_selection_key(metrics, config["evaluation"]["acceptance"])
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "negative": negative_name,
                "validation": metrics,
                "gate_passed": passed,
                "shortfall": shortfall,
            }
            history.append(record)
            print(json.dumps({"candidate": candidate, **record}), flush=True)
            if best_key is None or key < best_key:
                best_key = key
                best_step = step
                best_metrics = copy.deepcopy(metrics)
                save_candidate(checkpoint, model, {
                    "selection_split": "validation",
                    "best_step": best_step,
                    "validation": best_metrics,
                })
            model.train()
    runtime = time.monotonic() - start_time
    return {
        "candidate": candidate,
        "executed": True,
        "steps": steps,
        "best_step": best_step,
        "best_validation": best_metrics,
        "gate_passed": bool(best_key is not None and best_key[0] == 0.0),
        "history": history,
        "runtime_seconds": runtime,
        "parameter_summary": model.parameter_summary(),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    seed_everything(int(config["seed"]))
    device, physical = require_isolated_gpu(config)
    cache_root = resolve(config["data"]["cache_root"])
    normalization = resolve(config["data"]["normalization"])
    train_cache = TReXActionCache(cache_root, "train", normalization)
    val_cache = TReXActionCache(cache_root, "val", normalization)
    training = json.loads(resolve(config["paths"]["bootstrap_training"]).read_text())
    dynamic_threshold = float(training["distribution"]["dynamic_threshold_normalized_rms_delta"])
    artifact_root = resolve(config["paths"]["artifact_root"])
    experiment_root = resolve(config["paths"]["experiment_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)
    stats_path = artifact_root / "transition_feature_stats.json"
    if stats_path.is_file():
        stats = TransitionFeatureStats.from_dict(json.loads(stats_path.read_text()))
    else:
        stats = fit_transition_feature_stats(train_cache)
        atomic_json(stats_path, stats.to_dict())
    source = ReleasedTokenizerSource.open(args.tokenizer_root)
    p_model = build_shared_candidate(
        source, resolve(config["paths"]["bootstrap_checkpoint"]), stats
    )
    p_steps = int(args.smoke_steps or config["r1_p"]["steps"])
    p_path = experiment_root / "r1_p_best.pt"
    r1_p = train_candidate(
        p_model,
        candidate="R1-P",
        options=config["r1_p"],
        config=config,
        train_cache=train_cache,
        val_cache=val_cache,
        dynamic_threshold=dynamic_threshold,
        device=device,
        steps=p_steps,
        checkpoint=p_path,
    )
    del p_model
    torch.cuda.empty_cache()
    smoke = args.smoke_steps is not None
    r1_n: dict[str, Any] = {"executed": False, "reason": "R1-P passed validation gates"}
    selected_candidate = "R1-P"
    selected_path = p_path
    if not smoke and not r1_p["gate_passed"]:
        n_model = NativeTransitionActionModel(stats, hidden_size=int(config["r1_n"]["hidden_size"]))
        n_path = experiment_root / "r1_n_best.pt"
        r1_n = train_candidate(
            n_model,
            candidate="R1-N",
            options=config["r1_n"],
            config=config,
            train_cache=train_cache,
            val_cache=val_cache,
            dynamic_threshold=dynamic_threshold,
            device=device,
            steps=int(config["r1_n"]["steps"]),
            checkpoint=n_path,
        )
        del n_model
        torch.cuda.empty_cache()
        if r1_n["gate_passed"] or not r1_p["gate_passed"]:
            selected_candidate = "R1-N"
            selected_path = n_path
    selected_model, selected_metadata = load_candidate(selected_path, selected_candidate, source)
    selected_model = selected_model.to(device).eval()
    indices = np.arange(2, dtype=np.int64)
    _, state, action, embodiment = make_tensors(val_cache, indices, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = selected_model(state, action, embodiment)["z_action"].float().cpu()
    del selected_model
    torch.cuda.empty_cache()
    reloaded, reloaded_metadata = load_candidate(selected_path, selected_candidate, source)
    reloaded = reloaded.to(device).eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = reloaded(state, action, embodiment)["z_action"].float().cpu()
    cold_reload = {
        "exact": bool(torch.equal(expected, actual)),
        "max_abs_difference": float((expected - actual).abs().max()),
        "metadata_equal": selected_metadata == reloaded_metadata,
    }
    selected_deploy = experiment_root / "selected.pt"
    checkpoint_hash = save_candidate(selected_deploy, reloaded, {**reloaded_metadata, "cold_reload": cold_reload})
    summary = {
        "schema": "tactile3d-unit.s3-3-r-training-summary.v1",
        "status": "SMOKE_COMPLETE" if smoke else "COMPLETE",
        "gpu": {
            "preferred_physical": int(config["gpu"]["preferred_physical"]),
            "actual_physical": physical,
            "fallback": physical != int(config["gpu"]["preferred_physical"]),
            "logical": "cuda:0",
            "visible_device_count": torch.cuda.device_count(),
            "isolation": "PASS",
        },
        "data": {
            "splits": train_cache.manifest["splits"],
            "leakage": train_cache.manifest["leakage"],
            "action_interval": train_cache.manifest["action_interval"],
            "transition_stats": str(stats_path.relative_to(ROOT)),
        },
        "r1_p": r1_p,
        "r1_n": r1_n,
        "r1_s": {"executed": False, "reason": "R1-S is unnecessary unless both R1-P and R1-N are insufficient"},
        "selected_candidate": selected_candidate,
        "selected_encoder_type": reloaded.encoder_type,
        "selected_parameter_summary": reloaded.parameter_summary(),
        "checkpoint": {"relative_path": str(selected_deploy.relative_to(ROOT)), "sha256": checkpoint_hash},
        "cold_reload": cold_reload,
        "original_unit_preservation": {
            "old_rows_digest_before": training["old_rows_digest_before"],
            "old_rows_digest_after": training["old_rows_digest_after"],
            "old_rows_bit_identical": training["old_rows_bit_identical"],
            "gr1_action_l2": "exactly preserved; no shared parameter is present in the native candidate" if selected_candidate == "R1-N" else "shared tensors frozen; only T-Rex row/adapter trained",
            "t4_non_regression_required": False,
        },
    }
    atomic_json(artifact_root / "training_summary.json", summary)
    print(json.dumps({"status": summary["status"], "selected_candidate": selected_candidate, "checkpoint": summary["checkpoint"], "cold_reload": cold_reload}, indent=2))


if __name__ == "__main__":
    main()
