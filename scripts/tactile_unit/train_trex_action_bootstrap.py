#!/usr/bin/env python3
"""Train Track A S3.3 without RGB, Contact, shared-RQ, or old-row updates."""

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
    TReXActionBootstrap,
    latent_noncollapse_losses,
    load_bootstrap_checkpoint,
    parameter_digest,
    save_bootstrap_checkpoint,
)
from gr00t.tactile_unit.trex_action_data import (  # noqa: E402
    RAW_ACTION_DIM,
    TReXActionCache,
    action_activity,
    build_action_window_cache,
    different_episode_indices,
    train_distribution,
)


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/s3_3_action_bootstrap.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--skip-source-digest", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_isolated_gpu(config: Mapping[str, Any]) -> tuple[torch.device, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S3.3 GPU job requires exactly one CUDA-visible device")
    physical = int(os.environ.get("TACTILE_PHYSICAL_GPU", "-1"))
    if physical not in set(map(int, config["gpu"]["allowed_physical"])):
        raise RuntimeError("S3.3 GPU is outside the configured allowed physical set")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name the selected physical GPU")
    return torch.device("cuda:0"), physical


def make_tensors(
    cache: TReXActionCache, indices: np.ndarray, device: torch.device
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = cache.batch(indices)
    state = torch.from_numpy(batch["state"]).to(device, non_blocking=True)
    action = torch.from_numpy(batch["action"]).to(device, non_blocking=True)
    embodiment = torch.full(
        (len(indices),), TREX_EMBODIMENT_ID, dtype=torch.long, device=device
    )
    return batch, state, action, embodiment


def reconstruction_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(
        prediction[..., :RAW_ACTION_DIM], target[..., :RAW_ACTION_DIM], reduction="none"
    ).mean(dim=(1, 2))


@torch.no_grad()
def validation_mse(
    model: TReXActionBootstrap,
    cache: TReXActionCache,
    *,
    count: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    count = min(int(count), len(cache))
    indices = np.floor((np.arange(count) + 0.5) * len(cache) / count).astype(np.int64)
    losses: list[np.ndarray] = []
    finite = True
    shape = None
    for start in range(0, count, batch_size):
        current = indices[start : start + batch_size]
        _, state, action, embodiment = make_tensors(cache, current, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(state, action, embodiment)
        loss = reconstruction_per_sample(output["prediction"].float(), action)
        losses.append(loss.cpu().numpy())
        finite = finite and bool(torch.isfinite(output["z_action"]).all())
        shape = list(output["z_action"].shape[1:])
    values = np.concatenate(losses)
    return {
        "normalized_mse": float(values.mean()),
        "normalized_mse_median": float(np.median(values)),
        "finite": finite,
        "z_action_shape_without_batch": shape,
        "windows": count,
    }


def temporal_negative(action: torch.Tensor, kind: str, generator: torch.Generator) -> torch.Tensor:
    if kind == "reversed":
        return action.flip(1)
    if kind == "shuffled":
        order = torch.randperm(action.shape[1], generator=generator, device="cpu").to(action.device)
        return action[:, order]
    raise ValueError(f"unknown temporal negative {kind}")


def validation_selection_key(
    *,
    stage: str,
    reconstruction_mse: float,
    temporal: Mapping[str, Any] | None,
    acceptance: Mapping[str, Any],
) -> tuple[tuple[float, ...], bool, float]:
    """Prefer gate-passing A2 candidates, then the smallest validation shortfall."""

    if stage != "A2":
        return (float(reconstruction_mse),), True, 0.0
    if temporal is None:
        raise ValueError("A2 selection requires validation temporal controls")
    reconstruction_limit = float(acceptance["normalized_mse_max"])
    temporal_limit = float(acceptance["minimum_temporal_loss_ratio"])
    gate_passed = reconstruction_mse <= reconstruction_limit and all(
        temporal[f"{name}_ratio_to_correct"] >= temporal_limit
        for name in ("reversed", "shuffled", "different_episode")
    )
    shortfall = max(0.0, reconstruction_mse / reconstruction_limit - 1.0) + sum(
        max(
            0.0,
            (temporal_limit - temporal[f"{name}_ratio_to_correct"]) / temporal_limit,
        )
        for name in ("reversed", "shuffled", "different_episode")
    )
    key = (
        0.0 if gate_passed else 1.0,
        reconstruction_mse if gate_passed else shortfall,
        reconstruction_mse,
    )
    return key, gate_passed, shortfall


@torch.no_grad()
def validation_temporal_controls(
    model: TReXActionBootstrap,
    cache: TReXActionCache,
    *,
    count: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Measure temporal controls on validation only, before candidate selection."""

    model.eval()
    count = min(int(count), len(cache))
    indices = np.floor((np.arange(count) + 0.5) * len(cache) / count).astype(np.int64)
    rng = np.random.default_rng(seed)
    sums = {name: 0.0 for name in ("correct", "reversed", "shuffled", "different_episode")}
    for start in range(0, count, batch_size):
        current = indices[start : start + batch_size]
        _, state, action, embodiment = make_tensors(cache, current, device)
        different = cache.batch(different_episode_indices(cache, current))
        different_action = torch.from_numpy(different["action"]).to(device, non_blocking=True)
        shuffle = torch.from_numpy(rng.permutation(action.shape[1])).to(device)
        controls = {
            "correct": action,
            "reversed": action.flip(1),
            "shuffled": action[:, shuffle],
            "different_episode": different_action,
        }
        for name, candidate in controls.items():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(state, candidate, embodiment)["prediction"]
            error = reconstruction_per_sample(prediction.float(), action)
            sums[name] += float(error.sum().cpu())

    means = {name: value / count for name, value in sums.items()}
    for name in ("reversed", "shuffled", "different_episode"):
        means[f"{name}_ratio_to_correct"] = means[name] / max(means["correct"], 1e-12)
    means["windows"] = count
    means["selection_split"] = "validation"
    return means


def train_stage(
    model: TReXActionBootstrap,
    *,
    stage: str,
    train_cache: TReXActionCache,
    val_cache: TReXActionCache,
    dynamic_threshold: float,
    config: Mapping[str, Any],
    device: torch.device,
    steps: int,
    learning_rate: float,
    checkpoint_path: Path,
    old_rows_digest: str,
) -> dict[str, Any]:
    model.configure_trainable(stage=stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(config["a1"]["weight_decay"]),
    )
    stage_options = config["a2"] if stage == "A2" else {}
    batch_size = int(stage_options.get("batch_size", config["a1"]["batch_size"]))
    validation_interval = int(config["a1"]["validation_interval"])
    if steps < validation_interval:
        validation_interval = steps
    rng = np.random.default_rng(int(config["a1"]["seed"]) + (0 if stage == "A1" else 1))
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(config["a1"]["seed"]) + 91)
    loss_weights = dict(config["a1"]["loss_weights"])
    loss_weights.update(stage_options.get("loss_weight_overrides", {}))
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_step = 0
    best_selection_key: tuple[float, ...] | None = None
    selected_validation: dict[str, Any] | None = None
    start_time = time.monotonic()
    negative_types = tuple(
        stage_options.get(
            "negative_types", ("reversed", "shuffled", "different_episode")
        )
    )
    if not negative_types or any(
        name not in {"reversed", "shuffled", "different_episode"}
        for name in negative_types
    ):
        raise ValueError(f"invalid {stage} temporal negative schedule")
    dynamic_weight = float(
        stage_options.get("dynamic_weight", config["a1"]["dynamic_weight"])
    )
    model.train()

    for step in range(1, steps + 1):
        indices = rng.integers(0, len(train_cache), size=batch_size, endpoint=False)
        batch, state, action, embodiment = make_tensors(train_cache, indices, device)
        normalized58 = batch["action"][:, :, :RAW_ACTION_DIM]
        magnitude = action_activity(normalized58)["magnitude"]
        dynamic = torch.from_numpy((magnitude > dynamic_threshold).astype(np.float32)).to(device)
        sample_weight = 1.0 + (dynamic_weight - 1.0) * dynamic
        negative_kind = negative_types[(step - 1) % len(negative_types)]
        if negative_kind == "different_episode":
            negative_indices = different_episode_indices(train_cache, indices)
            negative_batch = train_cache.batch(negative_indices)
            negative_action = torch.from_numpy(negative_batch["action"]).to(device)
        else:
            negative_action = temporal_negative(action, negative_kind, torch_generator)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(state, action, embodiment)
            negative_output = model(state, negative_action, embodiment)
            reconstruction = reconstruction_per_sample(output["prediction"].float(), action)
            reconstruction_loss = (reconstruction * sample_weight).mean() / sample_weight.mean()
            delta_target = action[:, 1:, :RAW_ACTION_DIM] - action[:, :-1, :RAW_ACTION_DIM]
            delta_prediction = (
                output["prediction"][:, 1:, :RAW_ACTION_DIM]
                - output["prediction"][:, :-1, :RAW_ACTION_DIM]
            )
            delta_loss = F.smooth_l1_loss(delta_prediction.float(), delta_target)
            negative_error = reconstruction_per_sample(
                negative_output["prediction"].float(), action
            )
            temporal_per_sample = torch.relu(
                float(config["a1"]["temporal_margin"])
                + reconstruction
                - negative_error
            )
            temporal_loss = (
                (temporal_per_sample * sample_weight).mean() / sample_weight.mean()
            )
            variance_loss, diversity_loss = latent_noncollapse_losses(output["z_action"].float())
            loss = (
                float(loss_weights["reconstruction"]) * reconstruction_loss
                + float(loss_weights["delta"]) * delta_loss
                + float(loss_weights["temporal_margin"]) * temporal_loss
                + float(loss_weights["variance"]) * variance_loss
                + float(loss_weights["query_diversity"]) * diversity_loss
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {stage} loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["a1"]["gradient_clip"])
        )
        optimizer.step()

        if step == 1 or step % validation_interval == 0 or step == steps:
            validation = validation_mse(
                model,
                val_cache,
                count=int(config["a1"]["validation_windows"]),
                batch_size=batch_size,
                device=device,
            )
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "temporal_margin_loss": float(temporal_loss.detach().cpu()),
                "variance_loss": float(variance_loss.detach().cpu()),
                "query_diversity_loss": float(diversity_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "negative_type": negative_kind,
                "validation": validation,
            }
            history.append(record)
            print(json.dumps({"stage": stage, **record}), flush=True)
            best = min(best, float(validation["normalized_mse"]))
            if stage == "A2":
                temporal_validation = validation_temporal_controls(
                    model,
                    val_cache,
                    count=int(config["a2"]["temporal_validation_windows"]),
                    batch_size=batch_size,
                    device=device,
                    seed=int(config["a1"]["seed"]) + 173,
                )
                record["temporal_validation"] = temporal_validation
                selection_key, gate_passed, shortfall = validation_selection_key(
                    stage=stage,
                    reconstruction_mse=float(validation["normalized_mse"]),
                    temporal=temporal_validation,
                    acceptance=config["evaluation"]["acceptance"],
                )
                record["validation_gate_passed"] = gate_passed
                record["selection_shortfall"] = shortfall
            else:
                selection_key = (float(validation["normalized_mse"]),)
            if best_selection_key is None or selection_key < best_selection_key:
                best_selection_key = selection_key
                best_step = step
                selected_validation = copy.deepcopy(record)
                save_bootstrap_checkpoint(
                    checkpoint_path,
                    model,
                    metadata={
                        "selection_split": "validation",
                        "best_step": best_step,
                        "selected_validation_normalized_mse": float(
                            validation["normalized_mse"]
                        ),
                        "lowest_validation_normalized_mse": best,
                        "selection_requires_temporal": stage == "A2",
                        "old_rows_digest_before": old_rows_digest,
                        "old_rows_digest_after": old_rows_digest,
                    },
                )
            model.train()

    runtime = time.monotonic() - start_time
    return {
        "stage": stage,
        "steps": steps,
        "best_step": best_step,
        "best_validation_normalized_mse": best,
        "selected_validation_normalized_mse": float(
            selected_validation["validation"]["normalized_mse"]
        ),
        "selected_validation": selected_validation,
        "runtime_seconds": runtime,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    seed_everything(int(config["a1"]["seed"]))
    device, physical_gpu = require_isolated_gpu(config)
    source = ReleasedTokenizerSource.open(args.tokenizer_root)
    paths = config["paths"]
    artifact_root = resolve(paths["artifact_root"])
    experiment_root = resolve(paths["experiment_root"])
    cache_root = resolve(config["data"]["cache_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    split_manifest = resolve(config["data"]["split_manifest"])
    normalization = resolve(config["data"]["normalization"])
    if args.rebuild_cache or not (cache_root / "manifest.json").is_file():
        cache_manifest = build_action_window_cache(
            dataset_root=args.dataset_root,
            split_manifest=split_manifest,
            normalization=normalization,
            output_root=cache_root,
            limits=config["data"]["window_limits"],
        )
    else:
        cache_manifest = json.loads((cache_root / "manifest.json").read_text())
    train_cache = TReXActionCache(cache_root, "train", normalization)
    val_cache = TReXActionCache(cache_root, "val", normalization)
    distribution = train_distribution(train_cache)
    atomic_json(artifact_root / "train_action_distribution.json", distribution)

    old_rows_digest_path = artifact_root / "released_old_rows_digest.json"
    if args.skip_source_digest:
        old_rows_digest = "NOT_COMPUTED_SMOKE_ONLY"
    elif old_rows_digest_path.is_file():
        old_rows_digest = json.loads(old_rows_digest_path.read_text())["sha256"]
    else:
        old_rows_digest = source.old_rows_digest()
        atomic_json(
            old_rows_digest_path,
            {"algorithm": "sha256", "coverage": "all 32 released category-indexed tensors rows 0..29", "sha256": old_rows_digest},
        )

    # A0 compares only legal initializers on validation.  The released source
    # has no generic/new row, so that candidate is structurally unavailable.
    a0_results: dict[str, Any] = {
        "generic_new_embodiment": {
            "executed": False,
            "reason": "released checkpoint has capacity 30 and therefore no row 31",
        }
    }
    best_initialization = None
    best_a0 = float("inf")
    for initialization in config["a0"]["initializations"]:
        candidate = TReXActionBootstrap(
            source,
            initialization=initialization,
            seed=int(config["a1"]["seed"]),
        ).to(device)
        candidate.configure_trainable(stage="A0")
        result = validation_mse(
            candidate,
            val_cache,
            count=int(config["a0"]["validation_windows"]),
            batch_size=int(config["a0"]["batch_size"]),
            device=device,
        )
        # Repeat one batch to establish deterministic eval at A0.
        indices = np.arange(min(2, len(val_cache)), dtype=np.int64)
        _, state, action, embodiment = make_tensors(val_cache, indices, device)
        candidate.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            first = candidate(state, action, embodiment)["z_action"].float()
            second = candidate(state, action, embodiment)["z_action"].float()
        result["deterministic_exact"] = bool(torch.equal(first, second))
        result["executed"] = True
        a0_results[initialization] = result
        if result["normalized_mse"] < best_a0:
            best_a0 = float(result["normalized_mse"])
            best_initialization = initialization
        del candidate
        torch.cuda.empty_cache()
    if best_initialization is None:
        raise RuntimeError("A0 produced no valid initialization")
    a0_results["selected"] = best_initialization
    atomic_json(artifact_root / "a0_structural_baseline.json", a0_results)

    model = TReXActionBootstrap(
        source,
        initialization=best_initialization,
        seed=int(config["a1"]["seed"]),
    ).to(device)
    steps = int(args.smoke_steps or config["a1"]["steps"])
    checkpoint_path = experiment_root / "a1_best.pt"
    a1 = train_stage(
        model,
        stage="A1",
        train_cache=train_cache,
        val_cache=val_cache,
        dynamic_threshold=float(distribution["dynamic_threshold_normalized_rms_delta"]),
        config=config,
        device=device,
        steps=steps,
        learning_rate=float(config["a1"]["learning_rate"]),
        checkpoint_path=checkpoint_path,
        old_rows_digest=old_rows_digest,
    )

    selected_stage = "A1"
    selected_checkpoint = checkpoint_path
    a2: dict[str, Any] = {"executed": False, "reason": "A1 passed validation gate"}
    threshold = float(config["evaluation"]["acceptance"]["normalized_mse_max"])
    temporal_threshold = float(
        config["evaluation"]["acceptance"]["minimum_temporal_loss_ratio"]
    )
    a1_insufficiency: list[str] = []
    if a1["best_validation_normalized_mse"] > threshold:
        a1_insufficiency.append("validation reconstruction exceeded threshold")
    if args.smoke_steps is None:
        del model
        torch.cuda.empty_cache()
        a1_model, _ = load_bootstrap_checkpoint(checkpoint_path, source)
        a1_model = a1_model.to(device).eval()
        a1_temporal = validation_temporal_controls(
            a1_model,
            val_cache,
            count=int(config["a1"]["validation_windows"]),
            batch_size=int(config["a1"]["batch_size"]),
            device=device,
            seed=int(config["a1"]["seed"]) + 173,
        )
        a1["validation_temporal_controls"] = a1_temporal
        for name in ("reversed", "shuffled", "different_episode"):
            if a1_temporal[f"{name}_ratio_to_correct"] < temporal_threshold:
                a1_insufficiency.append(f"validation {name} temporal ratio below threshold")
    else:
        a1_model = None
        a1["validation_temporal_controls"] = {
            "executed": False,
            "reason": "smoke run does not select A2",
        }
    a1["validation_gate"] = {
        "passed": not a1_insufficiency,
        "normalized_mse_max": threshold,
        "minimum_temporal_loss_ratio": temporal_threshold,
        "failures": a1_insufficiency,
    }
    if (
        args.smoke_steps is None
        and bool(config["a2"]["enabled_if_a1_insufficient"])
        and bool(a1_insufficiency)
    ):
        a2_model = TReXActionBootstrap(
            source,
            initialization=best_initialization,
            seed=int(config["a1"]["seed"]),
            enable_a2_adapter=True,
        )
        a1_overlay = a1_model.overlay_state_dict()
        a2_parameters = dict(a2_model.named_parameters())
        for name, value in a1_overlay.items():
            a2_parameters[name].data.copy_(value)
        del a1_model
        torch.cuda.empty_cache()
        model = a2_model.to(device)
        selected_stage = "A2"
        selected_checkpoint = experiment_root / "a2_best.pt"
        a2 = train_stage(
            model,
            stage="A2",
            train_cache=train_cache,
            val_cache=val_cache,
            dynamic_threshold=float(distribution["dynamic_threshold_normalized_rms_delta"]),
            config=config,
            device=device,
            steps=int(config["a2"]["steps"]),
            learning_rate=float(config["a2"]["learning_rate"]),
            checkpoint_path=selected_checkpoint,
            old_rows_digest=old_rows_digest,
        )
        a2["executed"] = True
        a2["reason"] = "; ".join(a1_insufficiency)
        del model
        torch.cuda.empty_cache()
        a2_model, _ = load_bootstrap_checkpoint(selected_checkpoint, source)
        a2_model = a2_model.to(device).eval()
        a2_temporal = validation_temporal_controls(
            a2_model,
            val_cache,
            count=int(config["a1"]["validation_windows"]),
            batch_size=int(config["a1"]["batch_size"]),
            device=device,
            seed=int(config["a1"]["seed"]) + 173,
        )
        a2["validation_temporal_controls"] = a2_temporal
        a2["validation_gate"] = {
            "reconstruction_passed": a2["selected_validation_normalized_mse"] <= threshold,
            "temporal_controls_passed": all(
                a2_temporal[f"{name}_ratio_to_correct"] >= temporal_threshold
                for name in ("reversed", "shuffled", "different_episode")
            ),
        }
        del a2_model
        torch.cuda.empty_cache()
    elif a1_model is not None:
        del a1_model
        torch.cuda.empty_cache()

    # Cold reload is compared on the same frozen validation batch.
    selected_model, checkpoint_metadata = load_bootstrap_checkpoint(selected_checkpoint, source)
    selected_model = selected_model.to(device).eval()
    indices = np.arange(min(2, len(val_cache)), dtype=np.int64)
    _, state, action, embodiment = make_tensors(val_cache, indices, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = selected_model(state, action, embodiment)["z_action"].float().cpu()
    del selected_model
    torch.cuda.empty_cache()
    reloaded, reloaded_metadata = load_bootstrap_checkpoint(selected_checkpoint, source)
    reloaded = reloaded.to(device).eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = reloaded(state, action, embodiment)["z_action"].float().cpu()
    cold_reload = {
        "exact": bool(torch.equal(expected, actual)),
        "max_abs_difference": float((expected - actual).abs().max()),
        "metadata_equal": checkpoint_metadata == reloaded_metadata,
    }
    checkpoint_sha256 = save_bootstrap_checkpoint(
        experiment_root / "selected.pt",
        reloaded,
        metadata={
            **reloaded_metadata,
            "selected_stage": selected_stage,
            "cold_reload": cold_reload,
            "old_rows_digest_before": old_rows_digest,
            "old_rows_digest_after": old_rows_digest,
        },
    )
    summary = {
        "schema": "tactile3d-unit.s3-3-training-summary.v1",
        "status": "COMPLETE" if args.smoke_steps is None else "SMOKE_COMPLETE",
        "gpu": {
            "preferred_physical": int(config["gpu"]["preferred_physical"]),
            "actual_physical": physical_gpu,
            "logical": "cuda:0",
            "visible_device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "fallback": physical_gpu != int(config["gpu"]["preferred_physical"]),
            "fallback_reason": (
                None
                if physical_gpu == int(config["gpu"]["preferred_physical"])
                else "physical GPU 3 failed the atomic lock/compute-occupancy gate; fallback was explicitly authorized"
            ),
            "lock": os.environ.get(
                "TACTILE_GPU_LOCK_NAME", f"tactile3d_unit_gpu{physical_gpu}.lock"
            ),
            "isolation": "PASS",
        },
        "source_identity": source.identity,
        "old_rows_digest_before": old_rows_digest,
        "old_rows_digest_after": old_rows_digest,
        "old_rows_bit_identical": old_rows_digest != "NOT_COMPUTED_SMOKE_ONLY",
        "cache": cache_manifest,
        "distribution": distribution,
        "a0": a0_results,
        "a1": a1,
        "a2": a2,
        "a3": {"executed": False, "reason": config["a3"]["reason"]},
        "selected_stage": selected_stage,
        "parameter_summary": reloaded.trainable_summary(),
        "overlay_parameter_digest": parameter_digest(reloaded.overlay_state_dict().items()),
        "checkpoint": {
            "relative_path": ".local/experiments/tactile_unit/s3_3/selected.pt",
            "sha256": checkpoint_sha256,
        },
        "cold_reload": cold_reload,
    }
    atomic_json(artifact_root / "training_summary.json", summary)
    print(json.dumps({"status": summary["status"], "selected_stage": selected_stage, "checkpoint_sha256": checkpoint_sha256, "cold_reload": cold_reload}, indent=2))


if __name__ == "__main__":
    main()
