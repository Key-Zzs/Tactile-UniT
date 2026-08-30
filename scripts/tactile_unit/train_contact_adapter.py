#!/usr/bin/env python3
"""Train and validation-select lightweight S3.2 contact-codebook adaptors."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from audit_shared_rq_compatibility import probe_metric, sha256_file
from contact_adapter_common import (
    DEFAULT_CACHE,
    DEFAULT_CODES,
    DEFAULT_S1,
    DEFAULT_S2,
    DEFAULT_SPEC,
    DEFAULT_T4,
    DEFAULT_TRANSITIONS,
    component_digests,
    decode_codes,
    ensure_validation_codes,
    evaluate_transformed,
    load_arrays,
    load_runtime,
    reconstruction_bundle,
    transform_codes,
    verify_gpu,
)
from gr00t.contact_dynamics.evaluation import query_diversity
from gr00t.tactile_unit.compatibility import codebook_usage, parameter_digest
from gr00t.tactile_unit.contact_adapter import ContactCodebookAdaptor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS = ROOT / ".local/experiments/tactile_unit/s3_2"
DEFAULT_LOGS = ROOT / ".local/logs/tactile_unit/s3_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--transition-cache", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--code-cache", type=Path, default=DEFAULT_CODES)
    parser.add_argument("--runtime-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--s1-checkpoint", type=Path, default=DEFAULT_S1)
    parser.add_argument("--s2-checkpoint", type=Path, default=DEFAULT_S2)
    parser.add_argument("--t4-dir", type=Path, default=DEFAULT_T4)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pilot-epochs", type=int)
    parser.add_argument("--final-epochs", type=int)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient_smoke_test(
    rq: torch.nn.Module,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    codes: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    adaptor = ContactCodebookAdaptor("affine").to(device).train()
    rq.eval()
    encoder.eval()
    decoder.eval()
    rq_before = parameter_digest(rq)
    internal_before = [int(layer.internal_step) for layer in rq.layers]
    batch = 32
    contact = torch.from_numpy(np.array(codes[:batch], copy=True)).to(device)
    current = torch.from_numpy(np.array(arrays["current"][:batch], copy=True)).to(device)
    future = torch.from_numpy(np.array(arrays["future"][:batch], copy=True)).to(device)
    adapted = adaptor(contact)
    quantized, indices, _ = rq(adapted)
    prediction = decoder(quantized, current)
    loss = F.mse_loss(prediction, future) + F.mse_loss(adapted, quantized.detach())
    loss.backward()
    gradients = [parameter.grad for parameter in adaptor.parameters()]
    adaptor_norm = float(sum(gradient.norm().item() for gradient in gradients if gradient is not None))
    rq_gradients = [parameter.grad for parameter in rq.parameters()]
    decoder_gradients = [parameter.grad for parameter in decoder.parameters()]
    encoder_gradients = [parameter.grad for parameter in encoder.parameters()]
    internal_after = [int(layer.internal_step) for layer in rq.layers]
    result = {
        "loss": float(loss.item()),
        "adaptor_gradient_norm": adaptor_norm,
        "adaptor_gradient_finite": bool(
            gradients
            and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        ),
        "rq_gradient_none_or_zero": bool(
            all(gradient is None or not torch.count_nonzero(gradient) for gradient in rq_gradients)
        ),
        "decoder_gradient_none_or_zero": bool(
            all(gradient is None or not torch.count_nonzero(gradient) for gradient in decoder_gradients)
        ),
        "encoder_gradient_none_or_zero": bool(
            all(gradient is None or not torch.count_nonzero(gradient) for gradient in encoder_gradients)
        ),
        "rq_eval": bool(not rq.training and all(not layer.training for layer in rq.layers)),
        "rq_digest_unchanged": parameter_digest(rq) == rq_before,
        "rq_internal_steps_unchanged": internal_before == internal_after,
        "valid_indices": bool(indices.min() >= 0 and indices.max() < rq.layers[0].n_e),
    }
    result["status"] = "PASS" if (
        result["adaptor_gradient_finite"]
        and adaptor_norm > 0
        and all(
            result[key]
            for key in (
                "rq_gradient_none_or_zero",
                "decoder_gradient_none_or_zero",
                "encoder_gradient_none_or_zero",
                "rq_eval",
                "rq_digest_unchanged",
                "rq_internal_steps_unchanged",
                "valid_indices",
            )
        )
    ) else "FAIL"
    return result


def validation_score(metrics: dict[str, Any], identity: dict[str, Any]) -> float:
    dynamic = metrics["adapted_quantized_reconstruction"]["dynamic"]["future_mse"]
    identity_dynamic = identity["adapted_quantized_reconstruction"]["dynamic"]["future_mse"]
    distortion = metrics["quantization"]["relative_distortion"]
    identity_distortion = identity["quantization"]["relative_distortion"]
    return float(dynamic / identity_dynamic + distortion / identity_distortion)


def train_candidate(
    architecture: str,
    *,
    train_arrays: dict[str, np.ndarray],
    train_codes: np.ndarray,
    val_arrays: dict[str, np.ndarray],
    val_codes: np.ndarray,
    rq: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    lambda_delta: float,
    lambda_quant: float,
    dynamic_weight: float,
    identity_validation: dict[str, Any],
    seed: int,
) -> tuple[ContactCodebookAdaptor, dict[str, Any]]:
    set_seed(seed)
    adaptor = ContactCodebookAdaptor(architecture).to(device)
    optimizer = torch.optim.AdamW(
        adaptor.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    rq.eval()
    decoder.eval()
    for epoch in range(1, epochs + 1):
        adaptor.train()
        order = torch.randperm(len(train_codes), generator=generator).numpy()
        loss_sum = 0.0
        future_sum = 0.0
        quant_sum = 0.0
        examples = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            contact = torch.from_numpy(np.array(train_codes[selected], copy=True)).to(device)
            current = torch.from_numpy(np.array(train_arrays["current"][selected], copy=True)).to(
                device
            )
            future = torch.from_numpy(np.array(train_arrays["future"][selected], copy=True)).to(
                device
            )
            dynamic = torch.from_numpy(
                np.array(train_arrays["dynamic"][selected], dtype=np.float32, copy=True)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            adapted = adaptor(contact)
            quantized, _, _ = rq(adapted)
            prediction = decoder(quantized, current)
            weights = 1.0 + (dynamic_weight - 1.0) * dynamic
            future_per_sample = (prediction - future).square().mean(dim=1)
            delta_per_sample = ((prediction - current) - (future - current)).square().mean(dim=1)
            future_loss = (future_per_sample * weights).sum() / weights.sum()
            delta_loss = (delta_per_sample * weights).sum() / weights.sum()
            quant_loss = F.mse_loss(adapted, quantized.detach())
            loss = future_loss + lambda_delta * delta_loss + lambda_quant * quant_loss
            loss.backward()
            optimizer.step()
            count = len(selected)
            examples += count
            loss_sum += float(loss.item()) * count
            future_sum += float(future_loss.item()) * count
            quant_sum += float(quant_loss.item()) * count
        adapted_val, quantized_val, indices_val = transform_codes(
            adaptor, rq, val_codes, device, batch_size
        )
        validation = evaluate_transformed(
            decoder, adapted_val, quantized_val, val_arrays, device, batch_size
        )
        if indices_val.min() < 0 or indices_val.max() >= rq.layers[0].n_e:
            raise RuntimeError("candidate produced invalid frozen-RQ indices")
        score = validation_score(validation, identity_validation)
        row = {
            "epoch": epoch,
            "train_objective": loss_sum / examples,
            "train_weighted_future_mse": future_sum / examples,
            "train_quantization_mse": quant_sum / examples,
            "validation_score": score,
            "validation": validation,
        }
        history.append(row)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_validation = copy.deepcopy(validation)
            best_state = {
                name: value.detach().cpu().clone() for name, value in adaptor.state_dict().items()
            }
    if best_state is None or best_validation is None:
        raise AssertionError("candidate training did not produce a checkpoint")
    adaptor.load_state_dict(best_state, strict=True)
    adaptor.eval()
    return adaptor, {
        "architecture": architecture,
        "parameters": adaptor.parameter_count,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation": best_validation,
        "lambda_delta": lambda_delta,
        "lambda_quant": lambda_quant,
        "dynamic_weight": dynamic_weight,
        "history": history,
        "training_seconds": time.monotonic() - started,
    }


def usage_summary(indices: np.ndarray, codebook_size: int) -> list[dict[str, Any]]:
    return [
        {"stage": stage, **codebook_usage(indices[:, :, stage], codebook_size)}
        for stage in range(indices.shape[-1])
    ]


def validation_probes(
    train_feature: np.ndarray,
    val_feature: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    definitions = {
        "contact_transition": ("contact_transition", 4),
        "force_trend": ("force_trend_class", 3),
    }
    return {
        name: probe_metric(
            train_feature,
            val_feature,
            np.asarray(train_arrays[key]),
            np.asarray(val_arrays[key]),
            classes,
            device,
            batch_size,
            10.0,
        )
        for name, (key, classes) in definitions.items()
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    device = verify_gpu()
    runtime = load_runtime(
        args.spec,
        args.transition_cache,
        args.code_cache,
        args.s1_checkpoint,
        args.s2_checkpoint,
        args.t4_dir,
        device,
    )
    spec = runtime["spec"]
    optimization = spec["optimization"]
    selection = spec["selection"]
    seed = int(spec["seed"])
    batch_size = args.batch_size or int(optimization["batch_size"])
    pilot_epochs = args.pilot_epochs or int(selection["pilot_epochs"])
    final_epochs = args.final_epochs or int(selection["final_epochs"])
    if batch_size < 1 or pilot_epochs < 1 or final_epochs < 1:
        raise ValueError("batch size and epoch counts must be positive")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_cache.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    train_arrays = load_arrays(args.transition_cache, "train")
    val_arrays = load_arrays(args.transition_cache, "validation")
    train_codes = np.load(args.code_cache / "train.npy", mmap_mode="r")
    val_codes = ensure_validation_codes(
        runtime["s2"],
        val_arrays,
        args.runtime_cache / "validation_codes.npy",
        device,
        batch_size,
    )
    if tuple(train_codes.shape) != (279680, 8, 32) or tuple(val_codes.shape) != (17504, 8, 32):
        raise RuntimeError("canonical S3.2 train/validation code geometry mismatch")
    regenerated = runtime["s2"].encoder(
        torch.from_numpy(np.array(train_arrays["current"][:64], copy=True)).to(device),
        torch.from_numpy(np.array(train_arrays["future"][:64], copy=True)).to(device),
    ).detach().cpu().numpy()
    cache_match = bool(
        np.allclose(regenerated, np.asarray(train_codes[:64]), atol=1e-5, rtol=1e-5)
    )
    if not cache_match:
        raise RuntimeError("S2 train code cache does not match the frozen encoder")

    digests_before = component_digests(runtime)
    gradient = gradient_smoke_test(
        runtime["rq"],
        runtime["s2"].encoder,
        runtime["s2"].decoder,
        val_arrays,
        val_codes,
        device,
    )
    if gradient["status"] != "PASS":
        raise RuntimeError(f"frozen-RQ STE gradient smoke test failed: {gradient}")

    identity_adaptor = ContactCodebookAdaptor("identity").to(device).eval()
    identity_adapted, identity_quantized, identity_indices = transform_codes(
        identity_adaptor, runtime["rq"], val_codes, device, batch_size
    )
    identity_validation = evaluate_transformed(
        runtime["s2"].decoder,
        identity_adapted,
        identity_quantized,
        val_arrays,
        device,
        batch_size,
    )
    zero_prediction = decode_codes(
        runtime["s2"].decoder,
        np.zeros_like(val_codes),
        np.asarray(val_arrays["current"]),
        device,
        batch_size,
    )
    zero_validation = reconstruction_bundle(
        np.asarray(val_arrays["current"]),
        np.asarray(val_arrays["future"]),
        zero_prediction,
        np.asarray(val_arrays["dynamic"], dtype=bool),
    )

    pilot: list[dict[str, Any]] = []
    for dynamic_weight in spec["objective"]["dynamic_weight_pilot"]:
        for lambda_quant in spec["objective"]["lambda_quant_pilot"]:
            _, result = train_candidate(
                str(selection["pilot_architecture"]),
                train_arrays=train_arrays,
                train_codes=train_codes,
                val_arrays=val_arrays,
                val_codes=val_codes,
                rq=runtime["rq"],
                decoder=runtime["s2"].decoder,
                device=device,
                batch_size=batch_size,
                epochs=pilot_epochs,
                learning_rate=float(optimization["learning_rate"]),
                weight_decay=float(optimization["weight_decay"]),
                lambda_delta=float(spec["objective"]["lambda_delta"]),
                lambda_quant=float(lambda_quant),
                dynamic_weight=float(dynamic_weight),
                identity_validation=identity_validation,
                seed=seed,
            )
            pilot.append(result)
    selected_pilot = min(
        pilot,
        key=lambda item: (
            item["best_validation_score"],
            item["lambda_quant"],
            item["dynamic_weight"],
        ),
    )
    selected_lambda_quant = float(selected_pilot["lambda_quant"])
    selected_dynamic_weight = float(selected_pilot["dynamic_weight"])

    candidates: dict[str, Any] = {}
    adaptors: dict[str, ContactCodebookAdaptor] = {}
    candidate_val_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for architecture in ("affine", "mlp"):
        adaptor, result = train_candidate(
            architecture,
            train_arrays=train_arrays,
            train_codes=train_codes,
            val_arrays=val_arrays,
            val_codes=val_codes,
            rq=runtime["rq"],
            decoder=runtime["s2"].decoder,
            device=device,
            batch_size=batch_size,
            epochs=final_epochs,
            learning_rate=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            lambda_delta=float(spec["objective"]["lambda_delta"]),
            lambda_quant=selected_lambda_quant,
            dynamic_weight=selected_dynamic_weight,
            identity_validation=identity_validation,
            seed=seed,
        )
        adapted_val, quantized_val, indices_val = transform_codes(
            adaptor, runtime["rq"], val_codes, device, batch_size
        )
        result["best_validation"] = evaluate_transformed(
            runtime["s2"].decoder,
            adapted_val,
            quantized_val,
            val_arrays,
            device,
            batch_size,
        )
        result["best_validation_score"] = validation_score(
            result["best_validation"], identity_validation
        )
        adaptors[architecture] = adaptor
        candidates[architecture] = result
        candidate_val_arrays[architecture] = (adapted_val, quantized_val, indices_val)

    identity_train_quantized_path = ROOT / ".local/cache/tactile_unit/s3_0/contact_train_quantized.npy"
    if not identity_train_quantized_path.is_file():
        _, identity_train_quantized, _ = transform_codes(
            identity_adaptor,
            runtime["rq"],
            train_codes,
            device,
            batch_size,
            adapted_path=args.runtime_cache / "identity_train_adapted.npy",
            quantized_path=args.runtime_cache / "identity_train_quantized.npy",
            indices_path=args.runtime_cache / "identity_train_indices.npy",
        )
    else:
        identity_train_quantized = np.load(identity_train_quantized_path, mmap_mode="r")
    identity_probes = validation_probes(
        identity_train_quantized,
        identity_quantized,
        train_arrays,
        val_arrays,
        device,
        batch_size,
    )

    for architecture, adaptor in adaptors.items():
        train_adapted, train_quantized, train_indices = transform_codes(
            adaptor,
            runtime["rq"],
            train_codes,
            device,
            batch_size,
            adapted_path=args.runtime_cache / f"{architecture}_train_adapted.npy",
            quantized_path=args.runtime_cache / f"{architecture}_train_quantized.npy",
            indices_path=args.runtime_cache / f"{architecture}_train_indices.npy",
        )
        adapted_val, quantized_val, indices_val = candidate_val_arrays[architecture]
        probes = validation_probes(
            train_quantized,
            quantized_val,
            train_arrays,
            val_arrays,
            device,
            batch_size,
        )
        usage = usage_summary(indices_val, int(runtime["rq_identity"]["codes_per_stage"]))
        diversity = query_diversity(quantized_val)
        metrics = candidates[architecture]["best_validation"]
        comparative_gates = {
            "distortion_better_than_identity": metrics["quantization"]["relative_distortion"]
            < identity_validation["quantization"]["relative_distortion"],
            "dynamic_reconstruction_better_than_identity": metrics[
                "adapted_quantized_reconstruction"
            ]["dynamic"]["future_mse"]
            < identity_validation["adapted_quantized_reconstruction"]["dynamic"]["future_mse"],
            "dynamic_reconstruction_better_than_zero": metrics[
                "adapted_quantized_reconstruction"
            ]["dynamic"]["future_mse"]
            < zero_validation["dynamic"]["future_mse"],
            "semantic_probes_above_majority": all(
                probe["macro_f1"] > probe["majority"]["macro_f1"] for probe in probes.values()
            ),
            "no_query_collapse": diversity["collapsed_sample_fraction"] == 0.0,
            "no_codebook_collapse": all(
                row["active_codes"] > 1 and row["top1_frequency"] < 0.9 for row in usage
            ),
        }
        candidates[architecture].update(
            {
                "validation_probes": probes,
                "identity_validation_probes": identity_probes,
                "validation_codebook_usage": usage,
                "validation_query_diversity": diversity,
                "comparative_gates": comparative_gates,
                "comparative_gate_status": "PASS"
                if all(comparative_gates.values())
                else "FAIL",
            }
        )
        del train_adapted, train_quantized, train_indices

    passing = [
        architecture
        for architecture in ("affine", "mlp")
        if candidates[architecture]["comparative_gate_status"] == "PASS"
    ]
    if passing:
        selected_architecture = passing[0]
        selection_reason = (
            "smallest architecture satisfying every validation comparative, semantic, usage, "
            "and no-collapse gate"
        )
    else:
        selected_architecture = min(
            ("affine", "mlp"), key=lambda name: candidates[name]["best_validation_score"]
        )
        selection_reason = (
            "no candidate passed every validation gate; retained the lowest joint validation "
            "dynamic-reconstruction/distortion score for final diagnosis"
        )

    checkpoint_paths: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    for architecture, adaptor in adaptors.items():
        path = args.experiment_dir / f"{architecture}_best.pt"
        torch.save(
            {
                "schema": "tactile3d-unit.s3-2-contact-adaptor.v1",
                "architecture": architecture,
                "state_dict": {
                    name: value.detach().cpu() for name, value in adaptor.state_dict().items()
                },
                "parameters": adaptor.parameter_count,
                "seed": seed,
                "lambda_delta": float(spec["objective"]["lambda_delta"]),
                "lambda_quant": selected_lambda_quant,
                "dynamic_weight": selected_dynamic_weight,
                "best_epoch": candidates[architecture]["best_epoch"],
                "validation": candidates[architecture]["best_validation"],
                "frozen_identity": runtime["identity"],
            },
            path,
        )
        checkpoint_paths[architecture] = str(path)
        checkpoint_hashes[architecture] = sha256_file(path)
        candidates[architecture]["checkpoint"] = str(path)
        candidates[architecture]["checkpoint_sha256"] = checkpoint_hashes[architecture]

    digests_after = component_digests(runtime)
    integrity = {
        name: {
            "before": before,
            "after": digests_after[name],
            "unchanged": before == digests_after[name],
        }
        for name, before in digests_before.items()
    }
    if not all(item["unchanged"] for item in integrity.values()):
        raise RuntimeError("frozen parameter digest changed during S3.2 training")
    summary = {
        "schema": "tactile3d-unit.s3-2-contact-adaptor-training.v1",
        "status": "PASS",
        "seed": seed,
        "environment": {
            "physical_gpu": 3,
            "logical_device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "identity": runtime["identity"],
        "data": {
            "train_pairs": len(train_codes),
            "validation_pairs": len(val_codes),
            "test_pairs": int(spec["data"]["pairs"]["test"]),
            "canonical_horizon_frames": int(spec["data"]["canonical_horizon_frames"]),
            "train_cache_matches_frozen_encoder": cache_match,
            "validation_cache_sha256": sha256_file(args.runtime_cache / "validation_codes.npy"),
            "test_used_for_selection": False,
        },
        "gradient_integrity": gradient,
        "parameter_integrity": integrity,
        "identity_validation": identity_validation,
        "identity_validation_codebook_usage": usage_summary(
            identity_indices, int(runtime["rq_identity"]["codes_per_stage"])
        ),
        "zero_validation_reconstruction": zero_validation,
        "pilot": pilot,
        "selected_hyperparameters": {
            "lambda_delta": float(spec["objective"]["lambda_delta"]),
            "lambda_quant": selected_lambda_quant,
            "dynamic_weight": selected_dynamic_weight,
            "selection_partition": "validation",
            "pilot_architecture": str(selection["pilot_architecture"]),
        },
        "candidates": candidates,
        "selected_architecture": selected_architecture,
        "selection_reason": selection_reason,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_hashes": checkpoint_hashes,
        "runtime_seconds": time.monotonic() - started,
        "rq_training_mode_during_training": False,
        "test_used_for_selection": False,
        "s3_3_started": False,
    }
    write_json(args.experiment_dir / "training_summary.json", summary)
    write_json(
        args.log_dir / "training_result.json",
        {
            "status": summary["status"],
            "selected_architecture": selected_architecture,
            "selected_hyperparameters": summary["selected_hyperparameters"],
            "gradient_integrity": gradient["status"],
            "parameter_integrity": all(item["unchanged"] for item in integrity.values()),
            "runtime_seconds": summary["runtime_seconds"],
        },
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_architecture": selected_architecture,
                "selected_hyperparameters": summary["selected_hyperparameters"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
