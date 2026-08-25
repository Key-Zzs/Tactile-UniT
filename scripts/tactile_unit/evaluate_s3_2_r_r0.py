#!/usr/bin/env python3
"""Evaluate the private Contact RQ ceiling and classify the R0 decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from contact_adapter_common import decode_codes, reconstruction_bundle, transform_codes
from evaluate_contact_adapter import load_adaptor
from gr00t.tactile_unit.s3_2_r import (
    build_contact_rq,
    classify_sufficiency,
    deterministic_different_episode_shuffle,
    reconstruction_retention,
    semantic_retention,
)
from s3_2_r_common import (
    DEFAULT_ARTIFACTS,
    DEFAULT_CACHE,
    DEFAULT_EXPERIMENTS,
    S3_0_CACHE,
    S3_2_CACHE,
    S3_2_EXPERIMENTS,
    candidate_metric_bundle,
    checkpoint_sha256,
    fit_ridge_recovery,
    frozen_guard,
    load_runtime,
    quantize_to_cache,
    semantic_probe_bundle,
    verify_frozen,
    verify_gpu,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_EXPERIMENTS / "r0/training_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS / "r0")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE / "r0")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def load_private_rq(training: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    selected = training["selected"]
    path = Path(selected["checkpoint"])
    if checkpoint_sha256(path) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected R0 checkpoint identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "tactile3d-unit.s3-2-r-private-contact-rq.v1":
        raise RuntimeError("invalid R0 private RQ checkpoint schema")
    rq = build_contact_rq(stages=int(payload["stages"]), codes=int(payload["codes_per_stage"]))
    rq.load_state_dict(payload["state_dict"], strict=True)
    return rq.eval().requires_grad_(False).to(device), payload


def reconstruct_condition(
    decoder: torch.nn.Module,
    representation: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    prediction = decode_codes(decoder, representation, arrays["current"], device, batch_size)
    return reconstruction_bundle(
        np.asarray(arrays["current"]),
        np.asarray(arrays["future"]),
        prediction,
        np.asarray(arrays["dynamic"], dtype=bool),
    )


def condition_controls(
    decoder: torch.nn.Module,
    representation: np.ndarray,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    permutation = deterministic_different_episode_shuffle(arrays["episode_id"], seed=seed)
    zero = reconstruct_condition(decoder, np.zeros_like(representation), arrays, device, batch_size)
    shuffled = reconstruct_condition(
        decoder, np.asarray(representation)[permutation], arrays, device, batch_size
    )
    return {"zero": zero, "shuffled": shuffled}


def main() -> int:
    args = parse_args()
    device = verify_gpu()
    runtime = load_runtime(device=device)
    guard = frozen_guard(runtime)
    training = json.loads(args.training_summary.read_text())
    if training.get("status") != "PASS" or training.get("test_used_for_selection"):
        raise RuntimeError("R0 training selection is invalid")
    if training["frozen_identity"] != runtime["identity"]:
        raise RuntimeError("R0 training/evaluation frozen identities disagree")
    private_rq, private_payload = load_private_rq(training, device)
    seed = int(runtime["spec"]["seed"])
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    private: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        q, indices, diagnostics = quantize_to_cache(
            private_rq,
            runtime["codes"][split],
            device=device,
            batch_size=args.batch_size,
            quantized_path=args.cache_dir / f"private_{split}_quantized.npy",
            indices_path=args.cache_dir / f"private_{split}_indices.npy",
        )
        private[split] = {"q": q, "indices": indices, "diagnostics": diagnostics}

    s3_2_training = json.loads((S3_2_EXPERIMENTS / "training_summary.json").read_text())
    affine = load_adaptor(
        Path(s3_2_training["checkpoint_paths"]["affine"]),
        s3_2_training["checkpoint_hashes"]["affine"],
        device,
    )
    p1_validation_u, p1_validation_q, p1_validation_i = transform_codes(
        affine, runtime["original_rq"], runtime["codes"]["validation"], device, args.batch_size
    )
    shared_conditions = {
        "frozen_shared_identity": {
            "train_q": np.load(S3_0_CACHE / "contact_train_quantized.npy", mmap_mode="r"),
            "validation_q": quantize_to_cache(
                runtime["original_rq"],
                runtime["codes"]["validation"],
                device=device,
                batch_size=args.batch_size,
                quantized_path=args.cache_dir / "identity_validation_quantized.npy",
                indices_path=args.cache_dir / "identity_validation_indices.npy",
            )[0],
            "test_q": np.load(S3_0_CACHE / "contact_test_quantized.npy", mmap_mode="r"),
            "test_indices": np.load(S3_0_CACHE / "contact_test_indices.npy", mmap_mode="r"),
            "train_u": runtime["codes"]["train"],
            "validation_u": runtime["codes"]["validation"],
            "test_u": runtime["codes"]["test"],
        },
        "p1_frozen_shared": {
            "train_q": np.load(S3_2_CACHE / "affine_train_quantized.npy", mmap_mode="r"),
            "validation_q": p1_validation_q,
            "test_q": np.load(S3_2_CACHE / "affine_test_quantized.npy", mmap_mode="r"),
            "test_indices": np.load(S3_2_CACHE / "affine_test_indices.npy", mmap_mode="r"),
            "train_u": np.load(S3_2_CACHE / "affine_train_adapted.npy", mmap_mode="r"),
            "validation_u": p1_validation_u,
            "test_u": np.load(S3_2_CACHE / "affine_test_adapted.npy", mmap_mode="r"),
        },
    }

    test_arrays = runtime["arrays"]["test"]
    decoder = runtime["s2"].decoder
    continuous_reconstruction = reconstruct_condition(
        decoder, runtime["codes"]["test"], test_arrays, device, args.batch_size
    )
    continuous_semantic = semantic_probe_bundle(
        runtime["codes"]["train"],
        runtime["codes"]["test"],
        runtime["arrays"]["train"],
        test_arrays,
        device,
        args.batch_size,
    )

    conditions: dict[str, Any] = {}
    all_recovery_inputs = {
        "separate_contact_rq": (
            private["train"]["q"], private["validation"]["q"], private["test"]["q"]
        ),
        **{
            name: (values["train_q"], values["validation_q"], values["test_q"])
            for name, values in shared_conditions.items()
        },
    }
    for name, (train_q, val_q, test_q) in all_recovery_inputs.items():
        if name == "separate_contact_rq":
            indices = private["test"]["indices"]
            diagnostics = private["test"]["diagnostics"]
            native_pre_rq = runtime["codes"]["test"]
            codebook_size = int(private_payload["codes_per_stage"])
        else:
            values = shared_conditions[name]
            indices = values["test_indices"]
            diagnostics = {"status": "recomputed caches; detailed stage distances reported in accepted S3.0/S3.2 artifacts"}
            native_pre_rq = values["test_u"]
            codebook_size = int(runtime["original_rq_identity"]["codes_per_stage"])
        bundle = candidate_metric_bundle(
            quantized=test_q,
            indices=indices,
            native=native_pre_rq,
            decoder=decoder,
            arrays=test_arrays,
            device=device,
            batch_size=args.batch_size,
            codebook_size=codebook_size,
            stage_diagnostics=diagnostics,
        )
        controls = condition_controls(decoder, test_q, test_arrays, device, args.batch_size, seed)
        semantics = semantic_probe_bundle(
            train_q,
            test_q,
            runtime["arrays"]["train"],
            test_arrays,
            device,
            args.batch_size,
        )
        r_recon = reconstruction_retention(
            bundle["reconstruction"]["dynamic"]["future_mse"],
            controls["zero"]["dynamic"]["future_mse"],
            controls["shuffled"]["dynamic"]["future_mse"],
            continuous_reconstruction["dynamic"]["future_mse"],
        )
        r_contact = semantic_retention(
            semantics["contact_transition"]["macro_f1"],
            continuous_semantic["contact_transition"]["macro_f1"],
            continuous_semantic["contact_transition"]["majority"]["macro_f1"],
        )
        r_force = semantic_retention(
            semantics["force_trend"]["macro_f1"],
            continuous_semantic["force_trend"]["macro_f1"],
            continuous_semantic["force_trend"]["majority"]["macro_f1"],
        )
        collapse = bundle["collapse"]
        category = classify_sufficiency(
            r_recon,
            r_contact,
            r_force,
            hard_code_collapse=bool(collapse["hard_code_collapse"]),
            query_collapse=bool(collapse["query_collapse"]),
        )
        recovery = fit_ridge_recovery(
            train_q,
            runtime["codes"]["train"],
            val_q,
            runtime["codes"]["validation"],
            test_q,
            runtime["codes"]["test"],
            device=device,
        )
        conditions[name] = {
            **bundle,
            "controls": controls,
            "semantic_probes": semantics,
            "retention": {
                "r_recon_raw": r_recon,
                "r_recon_display_clipped": float(np.clip(r_recon, 0, 1)),
                "r_contact_raw": r_contact,
                "r_contact_display_clipped": float(np.clip(r_contact, 0, 1)),
                "r_force_raw": r_force,
                "r_force_display_clipped": float(np.clip(r_force, 0, 1)),
            },
            "native_recoverability": recovery,
            "category": category,
        }

    separate_category = conditions["separate_contact_rq"]["category"]
    r0_final = {
        "STRONG_PASS": "SAME_CAPACITY_CONTACT_RQ_PASS",
        "PARTIAL": "SAME_CAPACITY_CONTACT_RQ_PARTIAL",
        "FAIL": "SAME_CAPACITY_CONTACT_RQ_FAIL",
    }[separate_category]
    integrity = verify_frozen(guard, runtime)
    output = {
        "schema": "tactile3d-unit.s3-2-r-r0-evaluation.v1",
        "status": "COMPLETE",
        "test_used_for_selection": False,
        "architecture": training["architecture"],
        "nominal_capacity_matches_original_unit": bool(
            training["architecture"] == {"queries": 8, "embedding_dim": 32, "stages": 2, "codes_per_stage": 128}
        ),
        "training": training["selected"],
        "continuous": {
            "reconstruction": continuous_reconstruction,
            "semantic_probes": continuous_semantic,
        },
        "conditions": conditions,
        "r0_final": r0_final,
        "capacity_sensitivity_required": separate_category == "FAIL",
        "interpretation": (
            "Contact is task-relevant compressible at the Original-UniT nominal discrete budget."
            if separate_category == "STRONG_PASS"
            else "Contact is only partially task-relevant compressible at the Original-UniT nominal discrete budget."
            if separate_category == "PARTIAL"
            else "Same-capacity private Contact RQ failed; run the pre-registered 3-stage capacity sensitivity before downstream shared-RQ stages."
        ),
        "frozen_identity": runtime["identity"],
        "frozen_integrity": integrity,
        "environment": {"physical_gpu": 3, "logical_device": str(device), "torch": torch.__version__},
    }
    write_json(args.output_dir / "r0_result.json", output)
    print(json.dumps({"status": "COMPLETE", "r0_final": r0_final, "category": separate_category, "retention": conditions["separate_contact_rq"]["retention"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
