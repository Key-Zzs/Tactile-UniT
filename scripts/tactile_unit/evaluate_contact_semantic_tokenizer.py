#!/usr/bin/env python3
"""Evaluate S3.2-Q candidates once on untouched test and apply frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from contact_semantic_tokenizer_common import (
    DEFAULT_ARTIFACTS,
    DEFAULT_CACHE,
    DEFAULT_EXPERIMENTS,
    DEFAULT_SPEC,
    ContactSemanticTokenizer,
    candidate_structure,
    decode_codes,
    load_runtime,
    materialize_candidate,
    probe_bundle,
    reconstruction_bundle,
    representation_metrics,
    set_seed,
    sha256_file,
    verify_gpu,
    whitening_from_payload,
    write_json,
)
from gr00t.tactile_unit.compatibility import parameter_digest
from gr00t.tactile_unit.contact_semantic_tokenizer import (
    classify_shared_private,
    classify_single_stream,
    deterministic_different_episode_permutation,
    private_stream_bypass,
    reconstruction_retention,
    same_episode_horizon_links,
    semantic_retention,
)
from gr00t.tactile_unit.s3_2_r import build_contact_rq
from s3_2_r_common import fit_ridge_recovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=DEFAULT_EXPERIMENTS / "training_summary.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "evaluation.json")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE / "evaluation")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def load_tokenizer(
    path: Path, expected_sha: str, device: torch.device
) -> tuple[ContactSemanticTokenizer, dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"tokenizer checkpoint SHA256 mismatch: {path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "tactile3d-unit.s3-2-q-tokenizer.v1":
        raise RuntimeError("invalid S3.2-Q tokenizer checkpoint schema")
    architecture = payload["architecture"]
    whitening = whitening_from_payload(payload["whitening"])
    model = ContactSemanticTokenizer(
        semantic_stages=int(architecture["semantic_stages"]),
        private_stages=int(architecture["private_stages"]),
        codes=int(architecture["codes"]),
        whitening=whitening,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device), payload


def load_baseline_rq(runtime: dict[str, Any], device: torch.device) -> torch.nn.Module:
    path = runtime["baselines"]["Q_BASE_2"]["checkpoint"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rq = build_contact_rq(stages=int(payload["stages"]), codes=int(payload["codes_per_stage"]))
    rq.load_state_dict(payload["state_dict"], strict=True)
    return rq.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def quantize_rq(
    rq: torch.nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    quantized = np.empty_like(values, dtype=np.float32)
    indices = np.empty((len(values), 8, len(rq.layers)), dtype=np.int64)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device)
        q_c, index, _ = rq(batch)
        quantized[start:stop] = q_c.cpu().numpy()
        indices[start:stop] = index.cpu().numpy()
    return quantized, indices


def reconstruction_condition(
    decoder: torch.nn.Module,
    representation: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    prediction = decode_codes(decoder, representation, arrays["current"], device, batch_size)
    return reconstruction_bundle(arrays, prediction), prediction


def controls(
    decoder: torch.nn.Module,
    representation: np.ndarray,
    arrays: dict[str, np.ndarray],
    permutation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    zero, _ = reconstruction_condition(decoder, np.zeros_like(representation), arrays, device, batch_size)
    shuffled, _ = reconstruction_condition(
        decoder, np.asarray(representation)[permutation], arrays, device, batch_size
    )
    return {"zero": zero, "shuffled": shuffled}


def native_recovery(
    train_rep: np.ndarray,
    validation_rep: np.ndarray,
    test_rep: np.ndarray,
    runtime: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return fit_ridge_recovery(
        train_rep,
        runtime["codes"]["train"],
        validation_rep,
        runtime["codes"]["validation"],
        test_rep,
        runtime["codes"]["test"],
        device=device,
    )


def temporal_controls(
    *,
    model: ContactSemanticTokenizer | None,
    rq: torch.nn.Module | None,
    representation: np.ndarray,
    arrays: dict[str, np.ndarray],
    runtime: dict[str, Any],
    permutation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    current = np.asarray(arrays["current"])
    future = np.asarray(arrays["future"])
    with torch.inference_mode():
        reverse_z = np.empty_like(runtime["codes"]["test"], dtype=np.float32)
        for start in range(0, len(current), batch_size):
            stop = min(start + batch_size, len(current))
            h_t = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
            h_future = torch.from_numpy(np.array(future[start:stop], copy=True)).to(device)
            reverse_z[start:stop] = runtime["s2"].encoder(h_future, h_t).cpu().numpy()
    if model is not None:
        reverse = materialize_candidate(
            model,
            reverse_z,
            device=device,
            batch_size=batch_size,
            output_dir=DEFAULT_CACHE / "reverse" / model.interface_type.lower(),
            split="test",
        )["semantic_native"]
    elif rq is not None:
        reverse, _ = quantize_rq(rq, reverse_z, device, batch_size)
    else:
        raise ValueError("temporal control requires a tokenizer or RQ")
    correct_prediction = decode_codes(runtime["s2"].decoder, representation, current, device, batch_size)
    reversed_prediction = decode_codes(runtime["s2"].decoder, reverse, current, device, batch_size)
    shuffled_prediction = decode_codes(
        runtime["s2"].decoder, np.asarray(representation)[permutation], current, device, batch_size
    )
    target = future.astype(np.float64)
    mismatch_target = target[permutation]
    metrics = {
        "correct_mse": float(np.square(correct_prediction - target).mean()),
        "reversed_mse": float(np.square(reversed_prediction - target).mean()),
        "shuffled_mse": float(np.square(shuffled_prediction - target).mean()),
        "mismatched_future_mse": float(np.square(correct_prediction - mismatch_target).mean()),
    }
    metrics["pass"] = bool(
        metrics["correct_mse"] < metrics["reversed_mse"]
        and metrics["correct_mse"] < metrics["shuffled_mse"]
        and metrics["correct_mse"] < metrics["mismatched_future_mse"]
    )
    return metrics


@torch.inference_mode()
def token_stability(
    *,
    model: ContactSemanticTokenizer | None,
    rq: torch.nn.Module | None,
    values: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    sample = torch.from_numpy(np.array(values[:2048], copy=True)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(sample.shape, generator=generator, device=device) * 1e-3
    if model is not None:
        clean = model(sample)
        perturbed = model(sample + noise)
        result = {
            "semantic_index_agreement": float(
                (clean["semantic_indices"] == perturbed["semantic_indices"]).float().mean().item()
            )
        }
        if model.private_quantizer is not None:
            result["private_index_agreement"] = float(
                (clean["private_indices"] == perturbed["private_indices"]).float().mean().item()
            )
        return result
    if rq is None:
        raise ValueError("stability requires model or RQ")
    _, clean, _ = rq(sample)
    _, perturbed, _ = rq(sample + noise)
    return {"semantic_index_agreement": float((clean == perturbed).float().mean().item())}


def multi_horizon_metrics(
    model: ContactSemanticTokenizer | None,
    representation: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    result = {}
    for horizon, offset in ((24, 8), (32, 16)):
        source, target = same_episode_horizon_links(
            arrays["episode_id"], arrays["anchor_frame"], offset
        )
        if len(source) == 0:
            result[str(horizon)] = {"status": "NOT_AVAILABLE", "windows": 0}
            continue
        if model is None:
            prediction = decode_codes(
                runtime_decoder := _MULTI_HORIZON_DECODER,
                np.asarray(representation)[source],
                np.asarray(arrays["current"])[source],
                device,
                batch_size,
            )
            decoder_name = "frozen_D_c_t16_extrapolation"
        else:
            predictions = []
            for start in range(0, len(source), batch_size):
                stop = min(start + batch_size, len(source))
                tokens = torch.from_numpy(np.array(representation[source[start:stop]], copy=True)).to(device)
                current = torch.from_numpy(
                    np.array(arrays["current"][source[start:stop]], copy=True)
                ).to(device)
                predictions.append(model.predict_horizon(tokens, current, horizon).cpu().numpy())
            prediction = np.concatenate(predictions)
            decoder_name = f"learned_horizon_{horizon}"
        target_value = np.asarray(arrays["future"])[target]
        result[str(horizon)] = {
            "status": "AVAILABLE",
            "windows": int(len(source)),
            "same_episode": True,
            "decoder": decoder_name,
            "future_mse": float(np.square(prediction - target_value).mean()),
        }
    return result


_MULTI_HORIZON_DECODER: torch.nn.Module


def evaluate_stream(
    *,
    name: str,
    direct: dict[str, np.ndarray],
    native_space: dict[str, np.ndarray],
    indices: np.ndarray,
    runtime: dict[str, Any],
    continuous: dict[str, Any],
    permutation: np.ndarray,
    device: torch.device,
    batch_size: int,
    bits: float,
    model: ContactSemanticTokenizer | None,
    rq: torch.nn.Module | None,
) -> dict[str, Any]:
    reconstruction, _ = reconstruction_condition(
        runtime["s2"].decoder,
        native_space["test"],
        runtime["arrays"]["test"],
        device,
        batch_size,
    )
    control = controls(
        runtime["s2"].decoder,
        native_space["test"],
        runtime["arrays"]["test"],
        permutation,
        device,
        batch_size,
    )
    probes = probe_bundle(
        direct["train"],
        direct["test"],
        runtime["arrays"]["train"],
        runtime["arrays"]["test"],
        device=device,
        batch_size=batch_size,
    )
    e_control = min(
        control["zero"]["dynamic"]["future_mse"],
        control["shuffled"]["dynamic"]["future_mse"],
    )
    r_recon = reconstruction_retention(
        reconstruction["dynamic"]["future_mse"],
        e_control,
        continuous["reconstruction"]["dynamic"]["future_mse"],
    )
    r_contact = semantic_retention(
        probes["contact_transition"]["macro_f1"],
        continuous["probes"]["contact_transition"]["macro_f1"],
        continuous["probes"]["contact_transition"]["majority"]["macro_f1"],
    )
    r_force = semantic_retention(
        probes["force_trend"]["macro_f1"],
        continuous["probes"]["force_trend"]["macro_f1"],
        continuous["probes"]["force_trend"]["majority"]["macro_f1"],
    )
    structure = candidate_structure(direct["test"], indices)
    return {
        "name": name,
        "bits": bits,
        "reconstruction": reconstruction,
        "controls": control,
        "probes": probes,
        "retention": {
            "r_recon_raw": r_recon,
            "r_recon_display_clipped": float(np.clip(r_recon, 0, 1)),
            "r_contact_raw": r_contact,
            "r_contact_display_clipped": float(np.clip(r_contact, 0, 1)),
            "r_force_raw": r_force,
            "r_force_display_clipped": float(np.clip(r_force, 0, 1)),
        },
        "native_recovery": native_recovery(
            direct["train"], direct["validation"], direct["test"], runtime, device
        ),
        "representation": representation_metrics(runtime["codes"]["test"], native_space["test"]),
        "collapse": structure,
        "temporal_controls": temporal_controls(
            model=model,
            rq=rq,
            representation=native_space["test"],
            arrays=runtime["arrays"]["test"],
            runtime=runtime,
            permutation=permutation,
            device=device,
            batch_size=batch_size,
        ),
        "multi_horizon": multi_horizon_metrics(
            model, native_space["test"], runtime["arrays"]["test"], device, batch_size
        ),
        "token_stability": token_stability(
            model=model,
            rq=rq,
            values=runtime["codes"]["test"],
            device=device,
            seed=int(runtime["spec"]["seed"]),
        ),
    }


def main() -> int:
    global _MULTI_HORIZON_DECODER
    args = parse_args()
    device, physical_gpu = verify_gpu()
    spec = json.loads(args.spec.read_text())
    set_seed(int(spec["seed"]))
    runtime = load_runtime(spec_path=args.spec, source_root=args.source_root, device=device)
    _MULTI_HORIZON_DECODER = runtime["s2"].decoder
    before = {
        "s2_encoder": parameter_digest(runtime["s2"].encoder),
        "s2_decoder": parameter_digest(runtime["s2"].decoder),
    }
    training = json.loads(args.training_summary.read_text())
    if training.get("status") != "COMPLETE" or training.get("test_used_for_selection"):
        raise RuntimeError("invalid S3.2-Q training summary")
    if training["frozen_identity"] != runtime["identity"]:
        raise RuntimeError("training/evaluation frozen identities differ")
    selected_whitened = training["whitened"]["selected"]
    models = {}
    payloads = {}
    for name, row in {
        "whitened": selected_whitened,
        "predictive": training["predictive"],
        "semantic_private": training["semantic_private"],
    }.items():
        model, payload = load_tokenizer(Path(row["checkpoint"]), row["checkpoint_sha256"], device)
        if payload["frozen_identity"] != runtime["identity"]:
            raise RuntimeError(f"{name} checkpoint frozen identity mismatch")
        models[name] = model
        payloads[name] = payload
    baseline_rq = load_baseline_rq(runtime, device)
    baseline_direct = {
        split: np.load(runtime["paths"]["r0_cache"] / f"private_{split}_quantized.npy", mmap_mode="r")
        for split in ("train", "validation", "test")
    }
    baseline_indices = np.load(runtime["paths"]["r0_cache"] / "private_test_indices.npy", mmap_mode="r")
    permutation = deterministic_different_episode_permutation(
        runtime["arrays"]["test"]["episode_id"], int(spec["seed"])
    )
    continuous_prediction = decode_codes(
        runtime["s2"].decoder,
        runtime["codes"]["test"],
        runtime["arrays"]["test"]["current"],
        device,
        args.batch_size,
    )
    continuous = {
        "reconstruction": reconstruction_bundle(runtime["arrays"]["test"], continuous_prediction),
        "probes": probe_bundle(
            runtime["codes"]["train"],
            runtime["codes"]["test"],
            runtime["arrays"]["train"],
            runtime["arrays"]["test"],
            device=device,
            batch_size=args.batch_size,
        ),
    }
    streams: dict[str, Any] = {}
    streams["ordinary"] = evaluate_stream(
        name="ordinary",
        direct=baseline_direct,
        native_space=baseline_direct,
        indices=baseline_indices,
        runtime=runtime,
        continuous=continuous,
        permutation=permutation,
        device=device,
        batch_size=args.batch_size,
        bits=112.0,
        model=None,
        rq=baseline_rq,
    )
    materialized = {}
    for name in ("whitened", "predictive", "semantic_private"):
        materialized[name] = {
            split: materialize_candidate(
                models[name],
                runtime["codes"][split],
                device=device,
                batch_size=args.batch_size,
                output_dir=args.cache_dir / name,
                split=split,
            )
            for split in ("train", "validation", "test")
        }
    for name in ("whitened", "predictive"):
        direct = {split: materialized[name][split]["semantic"] for split in materialized[name]}
        native_space = {
            split: materialized[name][split]["semantic_native"] for split in materialized[name]
        }
        streams[name] = evaluate_stream(
            name=name,
            direct=direct,
            native_space=native_space,
            indices=materialized[name]["test"]["semantic_indices"],
            runtime=runtime,
            continuous=continuous,
            permutation=permutation,
            device=device,
            batch_size=args.batch_size,
            bits=112.0,
            model=models[name],
            rq=None,
        )
    q2_values = materialized["semantic_private"]
    q2_model = models["semantic_private"]
    q2_streams = {}
    q2_streams["semantic_only"] = evaluate_stream(
        name="semantic_only",
        direct={split: q2_values[split]["semantic"] for split in q2_values},
        native_space={split: q2_values[split]["semantic_native"] for split in q2_values},
        indices=q2_values["test"]["semantic_indices"],
        runtime=runtime,
        continuous=continuous,
        permutation=permutation,
        device=device,
        batch_size=args.batch_size,
        bits=56.0,
        model=q2_model,
        rq=None,
    )
    q2_streams["private_only"] = evaluate_stream(
        name="private_only",
        direct={split: q2_values[split]["private"] for split in q2_values},
        native_space={split: q2_values[split]["private"] for split in q2_values},
        indices=q2_values["test"]["private_indices"],
        runtime=runtime,
        continuous=continuous,
        permutation=permutation,
        device=device,
        batch_size=args.batch_size,
        bits=56.0,
        model=None,
        rq=q2_model.private_quantizer,
    )
    full_indices = np.concatenate(
        [q2_values["test"]["semantic_indices"], q2_values["test"]["private_indices"]], axis=2
    )
    q2_streams["full"] = evaluate_stream(
        name="full",
        direct={split: q2_values[split]["full_native"] for split in q2_values},
        native_space={split: q2_values[split]["full_native"] for split in q2_values},
        indices=full_indices,
        runtime=runtime,
        continuous=continuous,
        permutation=permutation,
        device=device,
        batch_size=args.batch_size,
        bits=112.0,
        model=q2_model,
        rq=None,
    )
    test_arrays = runtime["arrays"]["test"]
    semantic = np.asarray(q2_values["test"]["semantic_native"])
    private = np.asarray(q2_values["test"]["private"])
    anti_representations = {
        "semantic_zero": private,
        "private_zero": semantic,
        "shuffled_semantic": semantic[permutation] + private,
        "shuffled_private": semantic + private[permutation],
    }
    anti_bypass = {}
    for name, representation in anti_representations.items():
        anti_bypass[name] = reconstruction_condition(
            runtime["s2"].decoder, representation, test_arrays, device, args.batch_size
        )[0]
    full_dynamic = q2_streams["full"]["reconstruction"]["dynamic"]["future_mse"]
    private_dynamic = q2_streams["private_only"]["reconstruction"]["dynamic"]["future_mse"]
    semantic_zero_impact = (
        anti_bypass["semantic_zero"]["dynamic"]["future_mse"] - full_dynamic
    ) / max(full_dynamic, 1e-12)
    bypass = private_stream_bypass(
        private_only_error=private_dynamic,
        full_error=full_dynamic,
        semantic_zero_error=anti_bypass["semantic_zero"]["dynamic"]["future_mse"],
        near_full_ratio=float(spec["q2"]["anti_bypass"]["private_near_full_ratio"]),
        minimum_relative_impact=float(
            spec["q2"]["anti_bypass"]["minimum_semantic_zero_relative_impact"]
        ),
    )
    anti_bypass["bypass"] = bypass
    anti_bypass["semantic_zero_relative_impact"] = semantic_zero_impact
    baseline_rare = streams["ordinary"]["probes"]["contact_transition"]["per_class"]
    minimum_rare_gain = float(spec["gates"]["rare_boundary"]["minimum_recall_gain_over_ordinary"])
    for candidate in list(streams.values()) + list(q2_streams.values()):
        rare = candidate["probes"]["contact_transition"]["per_class"]
        candidate["rare_boundary_pass"] = bool(
            rare["free_to_contact"]["recall"]
            >= baseline_rare["free_to_contact"]["recall"] + minimum_rare_gain
            and rare["contact_to_free"]["recall"]
            >= baseline_rare["contact_to_free"]["recall"] + minimum_rare_gain
        )
    best_single_name = max(
        ("ordinary", "whitened", "predictive"),
        key=lambda name: min(
            streams[name]["retention"]["r_recon_raw"],
            streams[name]["retention"]["r_contact_raw"],
            streams[name]["retention"]["r_force_raw"],
        ),
    )
    best_single = streams[best_single_name]
    single_ready = classify_single_stream(
        r_recon=best_single["retention"]["r_recon_raw"],
        r_contact=best_single["retention"]["r_contact_raw"],
        r_force=best_single["retention"]["r_force_raw"],
        rare_boundary_pass=best_single["rare_boundary_pass"],
        temporal_controls_pass=best_single["temporal_controls"]["pass"],
        collapse=bool(
            best_single["collapse"]["hard_code_collapse"]
            or best_single["collapse"]["query_collapse"]
        ),
    )
    semantic_q2 = q2_streams["semantic_only"]
    full_q2 = q2_streams["full"]
    shared_private_ready = classify_shared_private(
        semantic_r_contact=semantic_q2["retention"]["r_contact_raw"],
        semantic_r_force=semantic_q2["retention"]["r_force_raw"],
        full_r_recon=full_q2["retention"]["r_recon_raw"],
        rare_boundary_pass=semantic_q2["rare_boundary_pass"],
        temporal_controls_pass=semantic_q2["temporal_controls"]["pass"],
        bypass=bypass,
        collapse=bool(
            semantic_q2["collapse"]["hard_code_collapse"]
            or semantic_q2["collapse"]["query_collapse"]
            or full_q2["collapse"]["hard_code_collapse"]
            or full_q2["collapse"]["query_collapse"]
        ),
    )
    if single_ready:
        decision = "SEMANTIC_TOKENIZER_READY"
        contract = {
            "interface_type": "SINGLE_SEMANTIC",
            "semantic_component": best_single_name,
            "private_component": "NONE",
            "continuous_component": "NONE",
            "shapes": [8, 32],
            "index_shapes": [8, 2],
            "bitrate": 112,
            "checkpoint": payloads.get(best_single_name, {}).get("candidate", "Q_BASE_2"),
        }
    elif shared_private_ready:
        decision = "SHARED_PRIVATE_TOKENIZER_READY"
        contract = {
            "interface_type": "SEMANTIC_PLUS_PRIVATE",
            "semantic_component": "q_c_sem",
            "private_component": "q_c_priv",
            "continuous_component": "NONE",
            "shapes": {"semantic": [8, 32], "private": [8, 32]},
            "index_shapes": {"semantic": [8, 1], "private": [8, 1]},
            "bitrate": {"semantic": 56, "private": 56, "total": 112},
            "checkpoint": training["semantic_private"]["checkpoint"],
            "checkpoint_sha256": training["semantic_private"]["checkpoint_sha256"],
        }
    else:
        decision = "CONTINUOUS_CONTACT_RECOMMENDED"
        contract = {
            "interface_type": "CONTINUOUS",
            "semantic_component": "NONE",
            "private_component": "NONE",
            "continuous_component": "z_c",
            "shapes": [8, 32],
            "index_shapes": "NONE",
            "bitrate": "CONTINUOUS_FLOAT32",
            "checkpoint": "frozen S2 E_c/D_c",
            "checkpoint_sha256": runtime["identity"]["s2_checkpoint_sha256"],
        }
    contract["decoder_interface"] = "(representation [B,8,32], h_t [B,256]) -> h_future [B,256]"
    contract["cross_modal_alignment_allowed"] = (
        "continuous z_c through gated cross-attention"
        if decision == "CONTINUOUS_CONTACT_RECOMMENDED"
        else "semantic component only"
    )
    contract["contact_private"] = (
        "private residual only"
        if shared_private_ready
        else "native Contact detail remains modality-private"
    )
    after = {
        "s2_encoder": parameter_digest(runtime["s2"].encoder),
        "s2_decoder": parameter_digest(runtime["s2"].decoder),
    }
    if after != before:
        raise RuntimeError("frozen S2 identity changed during evaluation")
    output = {
        "schema": "tactile3d-unit.s3-2-q-evaluation.v1",
        "status": "COMPLETE",
        "test_used_once_after_validation_selection": True,
        "physical_gpu": physical_gpu,
        "logical_device": str(device),
        "frozen_identity": runtime["identity"],
        "frozen_integrity": {"before": before, "after": after, "unchanged": True},
        "continuous": continuous,
        "q1": streams,
        "q2": {**q2_streams, "anti_bypass": anti_bypass},
        "best_single": best_single_name,
        "gates": {"single_ready": single_ready, "shared_private_ready": shared_private_ready},
        "final_decision": decision,
        "track_c_contract": contract,
    }
    write_json(args.output, output)
    write_json(DEFAULT_ARTIFACTS / "final_decision.json", {
        "final_decision": decision,
        "best_single": best_single_name,
        "gates": output["gates"],
        "track_c_contract": contract,
    })
    print(json.dumps({"status": "COMPLETE", "decision": decision, "best_single": best_single_name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
