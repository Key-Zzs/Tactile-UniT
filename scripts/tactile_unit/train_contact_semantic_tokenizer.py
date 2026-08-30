#!/usr/bin/env python3
"""Train the bounded Q1/Q2 Contact-native tokenizer candidates."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from contact_semantic_tokenizer_common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_SPEC,
    ContactSemanticTokenizer,
    WhiteningStatistics,
    load_runtime,
    same_episode_horizon_links,
    set_seed,
    sha256_file,
    verify_gpu,
    whitening_payload,
    write_json,
)
from gr00t.tactile_unit.contact_semantic_tokenizer import (
    deterministic_different_episode_permutation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--whitened-epochs", type=int)
    parser.add_argument("--predictive-epochs", type=int)
    parser.add_argument("--q2-semantic-epochs", type=int)
    parser.add_argument("--q2-private-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


@torch.inference_mode()
def validation_metrics(
    model: ContactSemanticTokenizer,
    codes: np.ndarray,
    arrays: dict[str, np.ndarray],
    frozen_decoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    full: bool,
) -> dict[str, float]:
    sums = {"frozen": 0.0, "frozen_dynamic": 0.0, "learned": 0.0, "vq": 0.0}
    counts = {"all": 0, "dynamic": 0, "values": 0}
    model.eval()
    for start in range(0, len(codes), batch_size):
        stop = min(start + batch_size, len(codes))
        z_c = torch.from_numpy(np.array(codes[start:stop], copy=True)).to(device)
        current = torch.from_numpy(np.array(arrays["current"][start:stop], copy=True)).to(device)
        future = torch.from_numpy(np.array(arrays["future"][start:stop], copy=True)).to(device)
        dynamic = torch.from_numpy(np.array(arrays["dynamic"][start:stop], copy=True)).to(device).bool()
        output = model(z_c)
        representation = output["full_native"] if full else output["semantic_native"]
        frozen_prediction = frozen_decoder(representation, current)
        learned_prediction = model.predict_horizon(representation, current, 16)
        per_sample = (frozen_prediction - future).square().mean(dim=1)
        sums["frozen"] += float(per_sample.sum().item())
        sums["learned"] += float((learned_prediction - future).square().sum().item())
        sums["vq"] += float(output["semantic_vq_loss"].item()) * len(z_c)
        if full:
            sums["vq"] += float(output["private_vq_loss"].item()) * len(z_c)
        counts["all"] += len(z_c)
        counts["values"] += future.numel()
        if dynamic.any():
            sums["frozen_dynamic"] += float(per_sample[dynamic].sum().item())
            counts["dynamic"] += int(dynamic.sum().item())
    return {
        "frozen_future_mse": sums["frozen"] / counts["all"],
        "frozen_dynamic_future_mse": sums["frozen_dynamic"] / counts["dynamic"],
        "learned_future_mse": sums["learned"] / counts["values"],
        "vq_loss": sums["vq"] / counts["all"],
    }


def fit_whitened_candidates(
    runtime: dict[str, Any],
    *,
    output_dir: Path,
    device: torch.device,
    physical_gpu: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> tuple[WhiteningStatistics, dict[str, torch.Tensor], dict[str, Any]]:
    spec = runtime["spec"]
    candidates = []
    train = runtime["codes"]["train"]
    validation = runtime["codes"]["validation"]
    validation_arrays = runtime["arrays"]["validation"]
    frozen_decoder = runtime["s2"].decoder
    generator = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
    for kind in spec["q1"]["whitening"]["kinds"]:
        for regularization in spec["q1"]["whitening"]["regularization_candidates"]:
            statistics = WhiteningStatistics.fit(
                train, kind=kind, regularization=float(regularization)
            )
            inverse_error = statistics.inverse_consistency_error(np.asarray(validation[:512]))
            if not np.isfinite(inverse_error) or inverse_error >= 1e-4:
                raise RuntimeError("whitening inverse consistency failed")
            model = ContactSemanticTokenizer(
                semantic_stages=2, whitening=statistics, private_stages=0
            ).to(device)
            model.semantic_encoder.requires_grad_(False)
            model.semantic_decoder.requires_grad_(False)
            model.horizon_decoders.requires_grad_(False)
            optimizer = torch.optim.AdamW(
                model.semantic_quantizer.parameters(), lr=learning_rate, weight_decay=0.0
            )
            best_score = float("inf")
            best_state = None
            best_epoch = 0
            history = []
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train), generator=generator).numpy()
                loss_sum = 0.0
                for start in range(0, len(order), batch_size):
                    selected = order[start : start + batch_size]
                    z_c = torch.from_numpy(np.array(train[selected], copy=True)).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model.semantic_forward(z_c)
                    loss = output["semantic_vq_loss"]
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.item()) * len(z_c)
                metrics = validation_metrics(
                    model,
                    validation,
                    validation_arrays,
                    frozen_decoder,
                    device,
                    batch_size,
                    full=False,
                )
                history.append({"epoch": epoch, "train_vq": loss_sum / len(train), "validation": metrics})
                if metrics["frozen_dynamic_future_mse"] < best_score:
                    best_score = metrics["frozen_dynamic_future_mse"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
            if best_state is None:
                raise RuntimeError("whitened candidate produced no checkpoint")
            name = f"whitened_{kind}_reg{float(regularization):g}"
            checkpoint = output_dir / "q1_whitened" / f"{name}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "tactile3d-unit.s3-2-q-tokenizer.v1",
                    "candidate": name,
                    "architecture": {"semantic_stages": 2, "private_stages": 0, "codes": 128},
                    "whitening": whitening_payload(statistics),
                    "state_dict": best_state,
                    "selection_partition": "validation",
                    "test_used_for_selection": False,
                    "frozen_identity": runtime["identity"],
                    "training": {"best_epoch": best_epoch, "best_score": best_score, "history": history},
                    "gpu": {"physical": physical_gpu, "logical": "cuda:0"},
                },
                checkpoint,
            )
            candidates.append(
                {
                    "name": name,
                    "kind": kind,
                    "regularization": float(regularization),
                    "inverse_consistency_max_abs": inverse_error,
                    "best_epoch": best_epoch,
                    "validation_dynamic_future_mse": best_score,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "statistics": statistics,
                    "state_dict": best_state,
                    "history": history,
                }
            )
    selected = min(candidates, key=lambda row: row["validation_dynamic_future_mse"])
    summary = {
        "schema": "tactile3d-unit.s3-2-q-whitened-selection.v1",
        "status": "COMPLETE",
        "selection_partition": "validation",
        "test_used_for_selection": False,
        "candidates": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in row.items()
                if key not in {"statistics", "state_dict"}
            }
            for row in candidates
        ],
        "selected": {
            "name": selected["name"],
            "checkpoint": str(selected["checkpoint"]),
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "validation_dynamic_future_mse": selected["validation_dynamic_future_mse"],
        },
    }
    write_json(output_dir / "q1_whitened/training_summary.json", summary)
    return selected["statistics"], selected["state_dict"], summary


def horizon_maps(arrays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    result = {}
    for horizon, offset in ((24, 8), (32, 16)):
        source, target = same_episode_horizon_links(
            arrays["episode_id"], arrays["anchor_frame"], offset
        )
        mapping = np.full(len(arrays["episode_id"]), -1, dtype=np.int64)
        mapping[source] = target
        result[horizon] = mapping
    return result


def predictive_batch_loss(
    model: ContactSemanticTokenizer,
    output: dict[str, torch.Tensor],
    *,
    representation: torch.Tensor,
    z_c: torch.Tensor,
    current: torch.Tensor,
    future: torch.Tensor,
    dynamic: torch.Tensor,
    mismatch_future: torch.Tensor,
    frozen_decoder: torch.nn.Module,
    dynamic_weight: float,
    lambda_future: float,
    lambda_delta: float,
    lambda_native: float,
    lambda_relational: float,
    lambda_temporal_mismatch: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = torch.where(dynamic, float(dynamic_weight), 1.0)
    frozen_prediction = frozen_decoder(representation, current)
    learned_prediction = model.predict_horizon(representation, current, 16)
    future_per_sample = (frozen_prediction - future).square().mean(dim=1)
    delta_per_sample = ((frozen_prediction - current) - (future - current)).square().mean(dim=1)
    learned_per_sample = (learned_prediction - future).square().mean(dim=1)
    mismatch_per_sample = (learned_prediction - mismatch_future).square().mean(dim=1)
    temporal_margin = F.relu(0.01 + learned_per_sample - mismatch_per_sample).mean()
    native_loss = F.mse_loss(representation, z_c)
    z_flat = z_c.flatten(1)
    representation_flat = representation.flatten(1)
    native_relation = F.cosine_similarity(z_flat[:-1], z_flat[1:], dim=1)
    token_relation = F.cosine_similarity(
        representation_flat[:-1], representation_flat[1:], dim=1
    )
    relational = F.mse_loss(token_relation, native_relation)
    loss = (
        lambda_future * (future_per_sample * weight).mean()
        + lambda_delta * (delta_per_sample * weight).mean()
        + (learned_per_sample * weight).mean()
        + lambda_native * native_loss
        + lambda_relational * relational
        + lambda_temporal_mismatch * temporal_margin
        + output["semantic_vq_loss"]
    )
    return loss, {
        "future": float(future_per_sample.mean().item()),
        "learned_future": float(learned_per_sample.mean().item()),
        "native": float(native_loss.item()),
        "relational": float(relational.item()),
        "temporal_mismatch_margin": float(temporal_margin.item()),
        "vq": float(output["semantic_vq_loss"].item()),
    }


def train_semantic_model(
    model: ContactSemanticTokenizer,
    runtime: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dynamic_weight: float,
    objective: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    train = runtime["codes"]["train"]
    arrays = runtime["arrays"]["train"]
    validation = runtime["codes"]["validation"]
    validation_arrays = runtime["arrays"]["validation"]
    frozen_decoder = runtime["s2"].decoder
    reverse_permutation = deterministic_different_episode_permutation(arrays["episode_id"], seed)
    maps = horizon_maps(arrays)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).numpy()
        sums: dict[str, float] = {}
        examples = 0
        horizon_examples = {24: 0, 32: 0}
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            z_c = torch.from_numpy(np.array(train[selected], copy=True)).to(device)
            current = torch.from_numpy(np.array(arrays["current"][selected], copy=True)).to(device)
            future = torch.from_numpy(np.array(arrays["future"][selected], copy=True)).to(device)
            dynamic = torch.from_numpy(np.array(arrays["dynamic"][selected], copy=True)).to(device).bool()
            mismatch_future = torch.from_numpy(
                np.array(arrays["future"][reverse_permutation[selected]], copy=True)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model.semantic_forward(z_c)
            representation = output["semantic_native"]
            loss, pieces = predictive_batch_loss(
                model,
                output,
                representation=representation,
                z_c=z_c,
                current=current,
                future=future,
                dynamic=dynamic,
                mismatch_future=mismatch_future,
                frozen_decoder=frozen_decoder,
                dynamic_weight=dynamic_weight,
                lambda_future=float(objective["lambda_future"]),
                lambda_delta=float(objective["lambda_delta"]),
                lambda_native=float(objective["lambda_native"]),
                lambda_relational=float(objective["lambda_relational"]),
                lambda_temporal_mismatch=float(objective["lambda_temporal_mismatch"]),
            )
            with torch.no_grad():
                reverse_z = runtime["s2"].encoder(future, current)
            reverse_output = model.semantic_forward(reverse_z)
            reverse_prediction = model.predict_horizon(
                reverse_output["semantic_native"], future, 16
            )
            reverse_loss = F.mse_loss(reverse_prediction, current)
            loss = (
                loss
                + float(objective["lambda_temporal_reverse"]) * reverse_loss
                + reverse_output["semantic_vq_loss"]
            )
            pieces["reverse"] = float(reverse_loss.item())
            for horizon in (24, 32):
                target_rows = maps[horizon][selected]
                mask = target_rows >= 0
                if np.any(mask):
                    mask_t = torch.from_numpy(mask).to(device)
                    target = torch.from_numpy(
                        np.array(arrays["future"][target_rows[mask]], copy=True)
                    ).to(device)
                    prediction = model.predict_horizon(
                        representation[mask_t], current[mask_t], horizon
                    )
                    horizon_loss = F.mse_loss(prediction, target)
                    loss = loss + float(objective["lambda_multi_horizon"]) * horizon_loss
                    pieces[f"horizon_{horizon}"] = float(horizon_loss.item())
                    horizon_examples[horizon] += int(mask.sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = len(z_c)
            examples += n
            sums["total"] = sums.get("total", 0.0) + float(loss.item()) * n
            for name, value in pieces.items():
                sums[name] = sums.get(name, 0.0) + value * n
        metrics = validation_metrics(
            model,
            validation,
            validation_arrays,
            frozen_decoder,
            device,
            batch_size,
            full=False,
        )
        selection_score = metrics["frozen_dynamic_future_mse"] + metrics["learned_future_mse"]
        history.append(
            {
                "epoch": epoch,
                "train": {name: value / examples for name, value in sums.items()},
                "multi_horizon_examples": horizon_examples,
                "validation": metrics,
                "selection_score": selection_score,
            }
        )
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("semantic training produced no checkpoint")
    return best_state, {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "history": history,
        "runtime_seconds": time.monotonic() - started,
        "selection_partition": "validation",
        "test_used_for_selection": False,
    }


def save_candidate(
    path: Path,
    *,
    candidate: str,
    model: ContactSemanticTokenizer,
    whitening: WhiteningStatistics,
    state_dict: dict[str, torch.Tensor],
    runtime: dict[str, Any],
    training: dict[str, Any],
    physical_gpu: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s3-2-q-tokenizer.v1",
            "candidate": candidate,
            "architecture": {
                "semantic_stages": model.semantic_stages,
                "private_stages": model.private_stages,
                "codes": model.codes,
            },
            "whitening": whitening_payload(whitening),
            "state_dict": state_dict,
            "selection_partition": "validation",
            "test_used_for_selection": False,
            "frozen_identity": runtime["identity"],
            "training": training,
            "gpu": {"physical": physical_gpu, "logical": "cuda:0"},
        },
        path,
    )
    return {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path)}


def train_private_residual(
    model: ContactSemanticTokenizer,
    runtime: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("private_quantizer."))
    optimizer = torch.optim.AdamW(
        model.private_quantizer.parameters(), lr=learning_rate, weight_decay=0.0
    )
    train = runtime["codes"]["train"]
    validation = runtime["codes"]["validation"]
    arrays = runtime["arrays"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).numpy()
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            z_c = torch.from_numpy(np.array(train[selected], copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(z_c)
            loss = output["private_vq_loss"]
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(z_c)
        metrics = validation_metrics(
            model,
            validation,
            arrays["validation"],
            runtime["s2"].decoder,
            device,
            batch_size,
            full=True,
        )
        history.append({"epoch": epoch, "train_private_vq": loss_sum / len(train), "validation": metrics})
        if metrics["frozen_dynamic_future_mse"] < best_score:
            best_score = metrics["frozen_dynamic_future_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("private residual training produced no checkpoint")
    return best_state, {
        "best_epoch": best_epoch,
        "best_validation_dynamic_future_mse": best_score,
        "history": history,
        "selection_partition": "validation",
        "test_used_for_selection": False,
    }


def main() -> int:
    args = parse_args()
    device, physical_gpu = verify_gpu()
    spec = json.loads(args.spec.read_text())
    seed = int(spec["seed"])
    set_seed(seed)
    runtime = load_runtime(spec_path=args.spec, source_root=args.source_root, device=device)
    batch_size = int(args.batch_size or spec["q1"]["predictive"]["batch_size"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    whitening, whitened_state, whitened_summary = fit_whitened_candidates(
        runtime,
        output_dir=args.output_dir,
        device=device,
        physical_gpu=physical_gpu,
        batch_size=batch_size,
        epochs=int(args.whitened_epochs or spec["q1"]["whitening"]["epochs"]),
        learning_rate=float(spec["q1"]["whitening"]["learning_rate"]),
    )
    predictive = ContactSemanticTokenizer(
        semantic_stages=2, whitening=whitening, private_stages=0
    ).to(device)
    predictive.load_state_dict(whitened_state, strict=True)
    predictive.semantic_encoder.requires_grad_(True)
    predictive.semantic_decoder.requires_grad_(True)
    predictive.semantic_quantizer.requires_grad_(True)
    predictive.horizon_decoders.requires_grad_(True)
    predictive_state, predictive_training = train_semantic_model(
        predictive,
        runtime,
        epochs=int(args.predictive_epochs or spec["q1"]["predictive"]["epochs"]),
        batch_size=batch_size,
        learning_rate=float(spec["q1"]["predictive"]["learning_rate"]),
        dynamic_weight=float(spec["q1"]["predictive"]["dynamic_weight"]),
        objective=spec["q1"]["predictive"],
        device=device,
        seed=seed + 100,
    )
    predictive.load_state_dict(predictive_state, strict=True)
    predictive_identity = save_candidate(
        args.output_dir / "q1_predictive/predictive.pt",
        candidate="predictive",
        model=predictive,
        whitening=whitening,
        state_dict=predictive_state,
        runtime=runtime,
        training=predictive_training,
        physical_gpu=physical_gpu,
    )
    q2 = ContactSemanticTokenizer(
        semantic_stages=1, whitening=whitening, private_stages=1
    ).to(device)
    predictive_named = predictive.state_dict()
    q2_state = q2.state_dict()
    for name in q2_state:
        if name in predictive_named and q2_state[name].shape == predictive_named[name].shape:
            q2_state[name] = predictive_named[name].detach().clone()
    q2.load_state_dict(q2_state, strict=True)
    q2.private_quantizer.requires_grad_(False)
    for parameter in q2.parameters():
        parameter.requires_grad_(True)
    q2.private_quantizer.requires_grad_(False)
    semantic_state, semantic_training = train_semantic_model(
        q2,
        runtime,
        epochs=int(args.q2_semantic_epochs or spec["q2"]["semantic_epochs"]),
        batch_size=batch_size,
        learning_rate=float(spec["q2"]["learning_rate"]),
        dynamic_weight=float(spec["q2"]["dynamic_weight"]),
        objective=spec["q1"]["predictive"],
        device=device,
        seed=seed + 200,
    )
    q2.load_state_dict(semantic_state, strict=True)
    private_state, private_training = train_private_residual(
        q2,
        runtime,
        epochs=int(args.q2_private_epochs or spec["q2"]["private_epochs"]),
        batch_size=batch_size,
        device=device,
        seed=seed + 300,
        learning_rate=float(spec["q2"]["private_learning_rate"]),
    )
    q2.load_state_dict(private_state, strict=True)
    q2_identity = save_candidate(
        args.output_dir / "q2_semantic_private/semantic_private.pt",
        candidate="semantic_private",
        model=q2,
        whitening=whitening,
        state_dict=private_state,
        runtime=runtime,
        training={"semantic": semantic_training, "private": private_training},
        physical_gpu=physical_gpu,
    )
    output = {
        "schema": "tactile3d-unit.s3-2-q-training-summary.v1",
        "status": "COMPLETE",
        "selection_partition": "validation",
        "test_used_for_selection": False,
        "physical_gpu": physical_gpu,
        "logical_device": str(device),
        "frozen_identity": runtime["identity"],
        "baseline_integrity": {
            "Q_BASE_2": runtime["baselines"]["Q_BASE_2"]["checkpoint_sha256"],
            "Q_BASE_3": runtime["baselines"]["Q_BASE_3"]["checkpoint_sha256"],
        },
        "whitened": whitened_summary,
        "predictive": {**predictive_identity, "training": predictive_training},
        "semantic_private": {
            **q2_identity,
            "training": {"semantic": semantic_training, "private": private_training},
        },
    }
    write_json(args.output_dir / "training_summary.json", output)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "predictive": predictive_identity,
                "semantic_private": q2_identity,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
