#!/usr/bin/env python3
"""Train exactly one preregistered matched teacher from a frozen snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_teacher import (  # noqa: E402
    MatchLossWeights,
    MatchedTeacher,
    StrictFieldStore,
    clip_active_grad_norm_,
    matched_teacher_loss,
)
from gr00t.simulation.pi2b_teacher.matched_teacher import (  # noqa: E402
    ALL_MODALITIES,
    COMMON_MODALITIES,
    common_state,
)
from scripts.simulation.pi2b_teacher.common import (  # noqa: E402
    CONFIG_ROOT,
    atomic_json,
    canonical_digest,
    load_json,
    sha256_file,
    tensor_digest,
)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def parameter_norm(parameters, *, gradients: bool) -> float:
    squares = []
    for parameter in parameters:
        value = parameter.grad if gradients else parameter
        if value is not None:
            squares.append(value.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu())


def batch_digest(pair_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in pair_ids:
        digest.update(str(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("T_VA_match", "T_VAC_match"), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runtime = load_json(args.runtime_manifest)
    if Path(runtime["snapshot_root"]).resolve() != ROOT.resolve():
        raise RuntimeError("training entrypoint is not running from the frozen snapshot")
    protocol = load_json(CONFIG_ROOT / "protocol.json")
    losses = load_json(CONFIG_ROOT / "losses.json")
    freeze = load_json(Path(runtime["write_root"]) / "artifacts/protocol_freeze.json")
    if freeze["freeze_identity"] != runtime["freeze_identity"]:
        raise RuntimeError("runtime/protocol freeze identity mismatch")
    if freeze["matching_decision"] != "PASS":
        raise RuntimeError("matching audit did not pass")
    if sha256_file(Path(runtime["common_va_normalization"])) != runtime["common_va_normalization_sha256"]:
        raise RuntimeError("common VA normalization identity changed")
    if int(protocol["scope"]["teacher_runs"]) != 2 or int(protocol["scope"]["teacher_retrains"]) != 0:
        raise RuntimeError("teacher run budget changed")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise RuntimeError("formal training requires exactly one supervisor-assigned GPU")
    seed = int(protocol["training_seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    modalities = COMMON_MODALITIES if args.mode == "T_VA_match" else ALL_MODALITIES
    required_fields = {"pair_id", "episode_id", "z_v", "z_a"}
    if args.mode == "T_VAC_match":
        required_fields.add("z_c")
    store = StrictFieldStore(Path(runtime["train_cache"]), required_fields)
    arrays = store.load()
    if args.mode == "T_VA_match" and "z_c" in store.accessed_fields:
        raise RuntimeError("VA loader accessed Contact")
    episode = np.unique(arrays["episode_id"], return_inverse=True)[1].astype(np.int64)
    run_root = Path(runtime["write_root"]) / "experiments" / args.mode / "seed42"
    final_path = run_root / "final.pt"
    resume_path = run_root / "resume.pt"
    status_path = Path(runtime["write_root"]) / "status" / f"{args.mode}.json"
    log_path = Path(runtime["write_root"]) / "logs" / f"{args.mode}.jsonl"
    if final_path.exists():
        raise RuntimeError(f"canonical run already exists and cannot be overwritten: {final_path}")
    if resume_path.exists() and not args.resume:
        raise RuntimeError("incomplete run exists; exact --resume is required")
    run_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model = MatchedTeacher(modalities, initialization_seed=int(protocol["initialization_seed"])).to(device)
    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(int(training["batch_rng_seed"]))
    start_step = 0
    batch_hashes: list[str] = []
    history: list[dict[str, Any]] = []
    if args.resume:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["mode"] != args.mode or payload["freeze_identity"] != runtime["freeze_identity"]:
            raise RuntimeError("resume identity mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        start_step = int(payload["step"])
        batch_hashes = list(payload["batch_hashes"])
        history = list(payload["history"])
    weights = MatchLossWeights(**losses["weights"])
    common_names = set(common_state(model))
    common_parameters = [parameter for name, parameter in model.named_parameters() if name in common_names]
    contact_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(("projectors.contact.", "recovery.contact."))
    ]
    start_time = time.monotonic()
    model.train()
    log_mode = "a" if args.resume else "x"
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log:
        for step in range(start_step + 1, int(training["steps"]) + 1):
            rows = rng.choice(len(arrays["pair_id"]), size=int(training["batch_size"]), replace=False)
            digest = batch_digest(arrays["pair_id"][rows])
            batch_hashes.append(digest)
            native = {
                "vision": torch.from_numpy(arrays["z_v"][rows]).to(device),
                "action": torch.from_numpy(arrays["z_a"][rows]).to(device),
            }
            if args.mode == "T_VAC_match":
                native["contact"] = torch.from_numpy(arrays["z_c"][rows]).to(device)
            episode_tensor = torch.from_numpy(episode[rows]).to(device)
            loss, parts = matched_teacher_loss(
                model,
                native,
                episode_tensor,
                temperature=float(losses["temperature"]),
                weights=weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            common_gradient_norm = parameter_norm(common_parameters, gradients=True)
            contact_gradient_norm = parameter_norm(contact_parameters, gradients=True)
            total_preclip = float(
                clip_active_grad_norm_(
                    model.parameters(), float(training["gradient_clip_global"])
                ).detach().cpu()
            )
            optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            if step == 1 or step % 10 == 0 or step == int(training["steps"]):
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start_time
                completed_here = step - start_step
                record = {
                    "mode": args.mode,
                    "step": step,
                    "steps": int(training["steps"]),
                    "batch_sha256": digest,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **{name: float(value.detach().cpu()) for name, value in parts.items()},
                    "common_gradient_norm": common_gradient_norm,
                    "contact_gradient_norm": contact_gradient_norm,
                    "global_gradient_norm_preclip": total_preclip,
                    "global_clip_activated": total_preclip > float(training["gradient_clip_global"]),
                    "elapsed_seconds_this_process": elapsed,
                    "seconds_per_step_this_process": elapsed / completed_here,
                }
                history.append(record)
                log.write(json.dumps(record, sort_keys=True) + "\n")
                atomic_json(
                    status_path,
                    {
                        "state": "RUNNING",
                        "pid": os.getpid(),
                        "mode": args.mode,
                        "step": step,
                        "steps": int(training["steps"]),
                        "seconds_per_step": elapsed / completed_here,
                        "log": str(log_path),
                        "run_root": str(run_root),
                        "freeze_identity": runtime["freeze_identity"],
                    },
                )
            if step % 100 == 0 and step < int(training["steps"]):
                atomic_torch(
                    resume_path,
                    {
                        "schema": "tactile3d-unit.s4-3-pi2b-teacher-resume.v1",
                        "mode": args.mode,
                        "step": step,
                        "freeze_identity": runtime["freeze_identity"],
                        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(),
                        "numpy_rng_state": rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all(),
                        "batch_hashes": batch_hashes,
                        "history": history,
                    },
                )
    exposure_digest = canonical_digest(batch_hashes)
    checkpoint = {
        "schema": "tactile3d-unit.s4-3-pi2b-matched-teacher.v1",
        "mode": args.mode,
        "modalities": list(modalities),
        "training_seed": seed,
        "initialization_seed": int(protocol["initialization_seed"]),
        "final_step": int(training["steps"]),
        "freeze_identity": runtime["freeze_identity"],
        "common_initialization_digest": load_json(Path(runtime["write_root"]) / "artifacts/common_initialization_digest.json")["sha256"],
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_updates": int(training["steps"]),
        "sample_exposure": int(training["steps"]) * int(training["batch_size"]),
        "batch_schedule_sha256": exposure_digest,
        "history": history,
        "source_snapshot": str(ROOT),
        "source_hashes": runtime["snapshot_hashes"],
        "common_va_normalization_sha256": runtime["common_va_normalization_sha256"],
        "va_contact_field_access": store.accessed_fields,
    }
    atomic_torch(final_path, checkpoint)
    final_sha = sha256_file(final_path)
    cold = torch.load(final_path, map_location="cpu", weights_only=False)
    reloaded = MatchedTeacher(tuple(cold["modalities"]), initialization_seed=cold["initialization_seed"])
    reloaded.load_state_dict(cold["state_dict"], strict=True)
    with torch.inference_mode():
        probe_native = {
            "vision": torch.from_numpy(arrays["z_v"][:8]).float(),
            "action": torch.from_numpy(arrays["z_a"][:8]).float(),
        }
        if args.mode == "T_VAC_match":
            probe_native["contact"] = torch.from_numpy(arrays["z_c"][:8]).float()
        finite_reload = all(torch.isfinite(value).all() for value in reloaded(probe_native).values())
    if not finite_reload or int(cold["final_step"]) != int(training["steps"]):
        raise RuntimeError("cold reload verification failed")
    if resume_path.exists():
        resume_path.unlink()
    completion = {
        "state": "COMPLETE",
        "mode": args.mode,
        "pid": os.getpid(),
        "final_step": int(training["steps"]),
        "checkpoint": str(final_path),
        "checkpoint_sha256": final_sha,
        "cold_reload_finite": finite_reload,
        "batch_schedule_sha256": exposure_digest,
        "sample_exposure": checkpoint["sample_exposure"],
        "common_final_digest": tensor_digest(common_state(reloaded)),
        "log": str(log_path),
        "run_root": str(run_root),
    }
    atomic_json(status_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
