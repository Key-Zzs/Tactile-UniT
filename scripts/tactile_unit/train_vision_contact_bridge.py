#!/usr/bin/env python3
"""Extract accepted paired features and train the small C0 bridge candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/tactile_unit"))

from continuous_contact_bridge_common import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_EXPERIMENTS,
    DEFAULT_SPEC,
    canonical_json_sha256,
    ensure_paired_caches,
    load_cache,
    load_frozen_vision,
    selected_indices,
    set_seed,
    state_dict_sha256,
    verify_file,
    verify_frozen_contact,
    verify_gpu,
)
from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.continuous_contact_bridge import (  # noqa: E402
    CausalContactGate,
    TokenSetCrossAttentionBridge,
    TwoTowerContinuousProjector,
    bridge_objective,
    parameter_count,
)
from gr00t.tactile_unit.paired_contract import sha256_file  # noqa: E402


def env_path(name: str, suffix: str = "") -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value) / suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--dataset-root", type=Path, default=env_path("TREX_DATASET_DIR"))
    parser.add_argument("--unit-checkpoint", type=Path, default=env_path("UNIT_FULLDATA_CKPT"))
    parser.add_argument("--s1-checkpoint", type=Path, default=env_path("TACTILE_TEACHER_CKPT"))
    parser.add_argument("--s2-checkpoint", type=Path, default=env_path("CONTACT_DYNAMICS_CKPT"))
    parser.add_argument("--transition-cache", type=Path, required=True)
    parser.add_argument("--code-cache", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--batch-size", type=int, default=16, help="Vision extraction batch size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args()


def require_paths(args: argparse.Namespace) -> None:
    for name in ("dataset_root", "unit_checkpoint", "s1_checkpoint", "s2_checkpoint"):
        if getattr(args, name) is None:
            raise ValueError(
                f"--{name.replace('_', '-')} or its machine-local environment variable is required"
            )


def batches(size: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(size, generator=generator)
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


@torch.inference_mode()
def validation_summary(
    model: torch.nn.Module, data: dict[str, np.ndarray], device: torch.device
) -> dict[str, float]:
    vision = torch.from_numpy(data["z_v"]).to(device)
    contact = torch.from_numpy(data["z_c"]).to(device)
    projected_vision, projected_contact = model(vision, contact)
    paired = F.cosine_similarity(projected_vision.flatten(1), projected_contact.flatten(1)).mean()
    shuffled = F.cosine_similarity(
        projected_vision.flatten(1), projected_contact.flip(0).flatten(1)
    ).mean()
    return {
        "paired_cosine": float(paired),
        "fixed_shuffle_cosine": float(shuffled),
        "margin": float(paired - shuffled),
        "contact_native_mse": float(F.mse_loss(projected_contact, contact)),
    }


def train_bridge(
    name: str,
    model: torch.nn.Module,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    spec: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["training"]["learning_rate"]), weight_decay=1e-4
    )
    batch_size = int(spec["training"]["batch_size"])
    generator = torch.Generator().manual_seed(int(spec["seed"]) + (1 if name == "B1" else 2))
    history: list[dict[str, float]] = []
    best_state = None
    best_margin = -float("inf")
    epochs = int(spec["training"]["epochs"])
    for epoch in range(epochs):
        totals = []
        for index in batches(len(train["z_v"]), batch_size, generator):
            rows = index.numpy()
            native_vision = torch.from_numpy(train["z_v"][rows]).to(device)
            native_contact = torch.from_numpy(train["z_c"][rows]).to(device)
            dynamic = torch.from_numpy(train["dynamic"][rows]).to(device)
            transition = torch.from_numpy(train["contact_transition"][rows]).to(device)
            weights = torch.ones(len(rows), device=device)
            weights = weights + dynamic.float() * (float(spec["training"]["dynamic_weight"]) - 1.0)
            boundary = (transition == 1) | (transition == 3)
            weights = torch.where(
                boundary,
                torch.full_like(weights, float(spec["training"]["rare_boundary_weight"])),
                weights,
            )
            projected_vision, projected_contact = model(native_vision, native_contact)
            losses = bridge_objective(
                projected_vision,
                projected_contact,
                native_vision,
                native_contact,
                weights=weights,
                temperature=0.07,
                prediction_weight=0.2,
                relational_weight=0.1,
            )
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(losses.total.detach()))
        model.eval()
        summary = validation_summary(model, validation, device)
        summary.update({"epoch": epoch + 1, "train_loss": float(np.mean(totals))})
        history.append(summary)
        if summary["margin"] > best_margin:
            best_margin = summary["margin"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        model.train()
    if best_state is None:
        raise RuntimeError(f"{name} produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "architecture": type(model).__name__,
        "parameter_count": parameter_count(model),
        "selected_epoch": int(np.argmax([row["margin"] for row in history])) + 1,
        "validation": validation_summary(model, validation, device),
        "history": history,
    }


def train_gate(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    spec: dict[str, Any],
    device: torch.device,
) -> tuple[CausalContactGate, dict[str, Any]]:
    model = CausalContactGate().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["training"]["learning_rate"]))
    generator = torch.Generator().manual_seed(int(spec["seed"]) + 3)
    batch_size = int(spec["training"]["batch_size"])
    best_state = None
    best_accuracy = -1.0
    history = []
    for epoch in range(int(spec["training"]["epochs"])):
        for index in batches(len(train["z_v_current"]), batch_size, generator):
            rows = index.numpy()
            vision = torch.from_numpy(train["z_v_current"][rows]).to(device)
            current = torch.from_numpy(train["h_current"][rows]).to(device)
            transition = torch.from_numpy(train["contact_transition"][rows]).to(device)
            target = ((transition == 2) | (transition == 3)).float()
            logits = model.network(torch.cat([vision.mean(dim=1), current], dim=-1)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            vision = torch.from_numpy(validation["z_v_current"]).to(device)
            current = torch.from_numpy(validation["h_current"]).to(device)
            transition = torch.from_numpy(validation["contact_transition"]).to(device)
            target = (transition == 2) | (transition == 3)
            score = model(vision, current).flatten()
            accuracy = float(((score >= 0.5) == target).float().mean())
            free_suppression = float(score[~target].mean()) if bool((~target).any()) else 0.0
            contact_activation = float(score[target].mean()) if bool(target.any()) else 0.0
        history.append(
            {
                "epoch": epoch + 1,
                "accuracy": accuracy,
                "free_gate_mean": free_suppression,
                "contact_gate_mean": contact_activation,
            }
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        model.train()
    if best_state is None:
        raise RuntimeError("B3 produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "architecture": type(model).__name__,
        "parameter_count": parameter_count(model),
        "selected_epoch": int(np.argmax([row["accuracy"] for row in history])) + 1,
        "validation": history[int(np.argmax([row["accuracy"] for row in history]))],
        "history": history,
        "causal_inputs": ["current visual context", "current h_t^c"],
    }


def main() -> int:
    args = parse_args()
    require_paths(args)
    if args.device == "gpu":
        device, physical_gpu = verify_gpu()
        resource_status = "GPU_ACQUIRED"
    else:
        device, physical_gpu = torch.device("cpu"), None
        resource_status = "GPU_RESOURCE_BUSY_CPU_FALLBACK"
    spec = json.loads(args.spec.read_text())
    set_seed(int(spec["seed"]))
    verify_file(
        args.transition_cache / "manifest.json",
        spec["frozen_identity"]["s2_transition_manifest_sha256"],
        "S2 transition manifest",
    )
    verify_file(
        args.code_cache / "train.npy",
        spec["frozen_identity"]["s2_train_codes_sha256"],
        "accepted train z_c",
    )
    verify_file(
        args.code_cache / "test.npy",
        spec["frozen_identity"]["s2_test_codes_sha256"],
        "accepted test z_c",
    )
    verify_file(
        args.paired_manifest,
        spec["paired_data"]["canonical_evaluation_manifest_sha256"],
        "canonical S3.1 960 manifest",
    )
    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s2, contact_identity = verify_frozen_contact(
        spec, args.s1_checkpoint, args.s2_checkpoint, device
    )
    contact_before = {
        "s2_encoder": parameter_digest(s2.encoder),
        "s2_decoder": parameter_digest(s2.decoder),
    }
    expected_cache_paths = [
        args.cache_root / f"paired_{split}.npz" for split in ("train", "validation", "test")
    ]
    if args.force_extract or not all(path.is_file() for path in expected_cache_paths):
        vision, vision_identity = load_frozen_vision(args.unit_checkpoint, spec, device)
        extraction = ensure_paired_caches(
            spec=spec,
            dataset_root=args.dataset_root,
            transition_cache=args.transition_cache,
            code_cache=args.code_cache,
            paired_manifest_path=args.paired_manifest,
            cache_root=args.cache_root,
            s2=s2,
            vision=vision,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        if vision_identity["trainable_parameters"] != 0:
            raise RuntimeError("Original UniT Vision was not frozen")
        del vision
        torch.cuda.empty_cache()
    else:
        extraction = {
            split: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for split, path in zip(("train", "validation", "test"), expected_cache_paths)
        }
        vision_identity = {
            "original_unit_tokenizer_files_sha256": {
                name: sha256_file(args.unit_checkpoint / "tokenizer" / name)
                for name in spec["frozen_identity"]["original_unit_tokenizer_files_sha256"]
            },
            "loading": "not reloaded; accepted cache reused",
            "trainable_parameters": 0,
        }
    train = load_cache(args.cache_root / "paired_train.npz")
    validation = load_cache(args.cache_root / "paired_validation.npz")
    test = load_cache(args.cache_root / "paired_test.npz")
    paired_manifest = json.loads(args.paired_manifest.read_text())
    expected_test_ids = [row["pair_id"] for row in paired_manifest["rows"]]
    expected_test_indices = np.asarray(
        [row["source"]["row_index"] for row in paired_manifest["rows"]], dtype=np.int64
    )
    if (
        test["pair_id"].tolist() != expected_test_ids
        or not np.array_equal(test["source_index"], expected_test_indices)
        or len(set(test["pair_id"].tolist())) != 960
    ):
        raise RuntimeError("training input does not preserve canonical 960 test identities")
    for order, (public, cache_name, values) in enumerate(
        (("train", "train", train), ("validation", "val", validation))
    ):
        arrays = {
            name: np.load(args.transition_cache / cache_name / f"{name}.npy", mmap_mode="r")
            for name in ("episode_id", "anchor_frame", "dynamic", "contact_transition")
        }
        expected = selected_indices(
            arrays,
            int(spec["preflight_sampling"][f"{public}_pairs"]),
            int(spec["seed"]) + order,
        )
        if not np.array_equal(values["source_index"], expected):
            raise RuntimeError(f"{public} paired cache selection/provenance mismatch")

    candidates: dict[str, Any] = {"B0": {"architecture": "identity/raw", "parameter_count": 0}}
    state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    for name, model in (
        ("B1", TwoTowerContinuousProjector("residual_mlp")),
        ("B2", TokenSetCrossAttentionBridge(heads=4)),
    ):
        trained, summary = train_bridge(name, model, train, validation, spec, device)
        state_dicts[name] = {
            key: value.detach().cpu() for key, value in trained.state_dict().items()
        }
        summary["state_dict_sha256"] = state_dict_sha256(state_dicts[name])
        candidates[name] = summary
    gate, gate_summary = train_gate(train, validation, spec, device)
    state_dicts["B3"] = {key: value.detach().cpu() for key, value in gate.state_dict().items()}
    gate_summary["state_dict_sha256"] = state_dict_sha256(state_dicts["B3"])
    candidates["B3"] = gate_summary
    contact_after = {
        "s2_encoder": parameter_digest(s2.encoder),
        "s2_decoder": parameter_digest(s2.decoder),
    }
    if contact_after != contact_before:
        raise RuntimeError("frozen S2 components changed during C0 training")
    checkpoint = args.output_dir / "continuous_bridge_candidates.pt"
    payload = {
        "schema": "tactile3d-unit.c0-continuous-bridge-checkpoint.v1",
        "spec_sha256": sha256_file(args.spec),
        "frozen_identity": {**contact_identity, **vision_identity},
        "frozen_integrity": {"before": contact_before, "after": contact_after, "unchanged": True},
        "cache_identity": {
            split: sha256_file(args.cache_root / f"paired_{split}.npz")
            for split in ("train", "validation", "test")
        },
        "state_dicts": state_dicts,
        "candidate_metadata": candidates,
        "test_used_for_selection": False,
        "physical_gpu": physical_gpu,
        "logical_device": str(device),
        "resource_status": resource_status,
    }
    torch.save(payload, checkpoint)
    summary = {
        "schema": "tactile3d-unit.c0-continuous-bridge-training.v1",
        "status": "COMPLETE",
        "physical_gpu": physical_gpu,
        "logical_device": str(device),
        "resource_status": resource_status,
        "spec_sha256": sha256_file(args.spec),
        "frozen_identity": {
            **contact_identity,
            "original_unit_tokenizer_files_sha256": vision_identity[
                "original_unit_tokenizer_files_sha256"
            ],
            "original_unit_loading": vision_identity["loading"],
            "original_unit_trainable_parameters": vision_identity["trainable_parameters"],
        },
        "frozen_integrity": payload["frozen_integrity"],
        "paired_extraction": extraction,
        "candidate_metadata": candidates,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_content_sha256": canonical_json_sha256(
            {
                name: row["state_dict_sha256"]
                for name, row in candidates.items()
                if "state_dict_sha256" in row
            }
        ),
        "test_used_for_selection": False,
    }
    output = args.output_dir / "training_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
