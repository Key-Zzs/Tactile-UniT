#!/usr/bin/env python3
"""Train only the DexJoCo embodiment slot under the Original UniT objective."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts/simulation"
OFFICIAL_ROOT = ROOT / ".local/external/s4_3_pi2u/unit_official"
CHECKPOINT_ROOT = (
    ROOT
    / ".local/external/s4_3_pi2u/unit_fulldata/"
    "VLA-UniT-3B-fulldata/tokenizer"
)
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
NORMALIZATION = ROOT / ".local/cache/simulation/s4_3_pi2v/adapter_normalization.npz"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2v"
DEFAULT_OUTPUT = ROOT / ".local/experiments/simulation/s4_3_pi2v/unit_adapter"
DEFAULT_LOG = ROOT / ".local/logs/simulation/s4_3_pi2v/unit_adapter/train.jsonl"

sys.path.insert(0, str(SCRIPT_ROOT))
from s4_3_pi2v_unit_adapter import (  # noqa: E402
    DEXJOCO_CATEGORY_ID,
    EXPECTED_ADAPTER_PARAMETERS,
    adapter_named_parameters,
    adapter_state_dict,
    dexjoco_action_view,
    dexjoco_state_view,
    finite_nonzero_adapter_gradients,
    frozen_parameter_digest,
    install_dexjoco_adapter,
    load_adapter_state_dict,
    normalize_and_pad,
    sha256_file,
    trainable_audit,
)

# Make the audited upstream implementation authoritative for all gr00t imports.
sys.path.insert(0, str(OFFICIAL_ROOT))
from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer  # noqa: E402


OFFICIAL_SOURCE_SHA = "0d762e32180bddd765694ef3846a3a5053f9d37f"
OFFICIAL_CHECKPOINT_SHA = {
    "config.json": "7a651f488c93521e0d507880fc250a475e6a08aa9307aa1349f9d3509844971e",
    "model-00001-of-00002.safetensors": "32d5c326f6c83d12185b6954d2a52511f66ad18b6fdf814aecc5726dd39c243c",
    "model-00002-of-00002.safetensors": "2f8093a900330e5111b63e44dc1687b3212bec343e5b3832bf2e40f2bf18a768",
    "model.safetensors.index.json": "3b6d73d2442ce694287c5cd8b93db1bb232909becf35f08ecabadb614b9a1b86",
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class OriginalUniTPairDataset(Dataset):
    """Leakage-clean S4.2 transitions in the official tokenizer tensor schema."""

    def __init__(self, split: str, augment: bool, seed: int) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("adapter dataset is TRAIN/DEV only")
        with np.load(PAIR_ROOT / f"{split}.npz", allow_pickle=False) as source:
            self.episode_id = source["episode_id"]
            self.anchor_step = source["anchor_step"].astype(np.int64)
            self.future_step = source["future_step"].astype(np.int64)
            self.state = dexjoco_state_view(source["current_state"])
            self.action = dexjoco_action_view(source["action_chunk"])
        with np.load(NORMALIZATION, allow_pickle=False) as stats:
            self.state, self.state_mask = normalize_and_pad(
                self.state, stats["state_mean"], stats["state_std"], 128
            )
            self.action, self.action_mask = normalize_and_pad(
                self.action, stats["action_mean"], stats["action_std"], 128
            )
        self.augment = augment
        self.seed = seed
        self.random_crop = transforms.RandomCrop((608, 608))
        self.center_crop = transforms.CenterCrop((608, 608))
        self.resize = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
        )
        self.color = transforms.ColorJitter(
            brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self) -> int:
        return len(self.episode_id)

    def _images(self, index: int, epoch: int) -> torch.Tensor:
        episode = self.episode_id[index]
        episode_root = ROOT / ".local/datasets/simulation/s4_2/episodes" / episode / "frames"
        paths = (
            episode_root / f"{self.anchor_step[index]:06d}.jpg",
            episode_root / f"{self.future_step[index]:06d}.jpg",
        )
        frames = []
        for path in paths:
            with Image.open(path) as image:
                value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            frames.append(torch.from_numpy(value).permute(2, 0, 1))
        video = torch.stack(frames)
        # Make augmentation a pure function of the sample and sampler epoch so
        # an adapter checkpoint can resume without depending on worker timing.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + 1_000_003 * epoch + index)
            video = self.random_crop(video) if self.augment else self.center_crop(video)
            video = self.resize(video)
            if self.augment:
                video = self.color(video)
        video = video.float().div_(255.0)
        video = (video - self.mean) / self.std
        if video.shape != (2, 3, 224, 224) or not bool(torch.isfinite(video).all()):
            raise RuntimeError("invalid Original UniT visual transition")
        return video

    def __getitem__(self, key: tuple[int, int]) -> dict[str, torch.Tensor]:
        index, epoch = key
        video = self._images(index, epoch)
        return {
            "imagenet_obs_images": video[0:1],
            "imagenet_goal_images": video[1:2],
            "state": torch.from_numpy(self.state[index]).unsqueeze(0),
            "state_mask": torch.from_numpy(self.state_mask[index]).unsqueeze(0),
            "action": torch.from_numpy(self.action[index]),
            "action_mask": torch.from_numpy(self.action_mask[index]),
            "embodiment_id": torch.tensor(DEXJOCO_CATEGORY_ID, dtype=torch.long),
        }


class EpochTaggedSampler(Sampler[tuple[int, int]]):
    """Attach the deterministic sampler epoch to every dataset index."""

    def __init__(self, sampler: DistributedSampler) -> None:
        self.sampler = sampler
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        return iter((index, self.epoch) for index in self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--save-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--startup-audit-step", type=int, default=55)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return rank, local_rank, world_size


def seed_all(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_identities() -> None:
    import subprocess

    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=OFFICIAL_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ("git", "status", "--short"), cwd=OFFICIAL_ROOT, text=True
    ).strip()
    if head != OFFICIAL_SOURCE_SHA or status:
        raise RuntimeError("S4_3_PI2V_UNIT_SOURCE_DRIFT")
    actual = {name: sha256_file(CHECKPOINT_ROOT / name) for name in OFFICIAL_CHECKPOINT_SHA}
    if actual != OFFICIAL_CHECKPOINT_SHA:
        raise RuntimeError("S4_3_PI2V_UNIT_SOURCE_DRIFT")


def load_model(device: torch.device, seed: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = GR00T_Tokenizer.from_pretrained(
        CHECKPOINT_ROOT,
        tune_vision_model=False,
        tune_vision_m_former=False,
        tune_bridge_projector=False,
        tune_action_encoder=False,
        tune_fusion=False,
        tune_vq=False,
        tune_vision_decoder=False,
        tune_action_decoder_projector=False,
        tune_action_decoder_diffusion=False,
    )
    contract = install_dexjoco_adapter(model, seed=seed)
    model.config.use_multi_scenario_training = True
    model.config.compute_dtype = "bfloat16"
    model.compute_dtype = "bfloat16"
    model.config.vision_decoder_cfg["is_io_hidden_states"] = True
    model.config.use_lpips_loss = False
    model.is_dino_mode = True
    model.use_lpips_loss = False
    model.train().to(device=device, dtype=torch.bfloat16)
    # The released standard RVQ performs in-place dead-code restarts whenever
    # it is in train mode, independently of requires_grad.  PI2V freezes the
    # released codebook byte-for-byte, so quantization/loss stay active while
    # its mutable training-time restart path is disabled.
    model.vq.eval()
    return model, contract


def scheduler_multiplier(step: int, max_steps: int, warmup_ratio: float) -> float:
    warmup = max(1, round(max_steps * warmup_ratio))
    if step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def reduce_metrics(metrics: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    names = sorted(metrics)
    values = torch.tensor([metrics[name] for name in names], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
    return {name: float(value) for name, value in zip(names, values.cpu().tolist())}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> None:
    local_rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    rng_by_rank = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank = [local_rng]
    if rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "tactile3d-unit.s4-3-pi2v-dexjoco-adapter.v1",
            "step": step,
            "official_source_sha": OFFICIAL_SOURCE_SHA,
            "official_checkpoint_sha": OFFICIAL_CHECKPOINT_SHA,
            "dexjoco_category_id": DEXJOCO_CATEGORY_ID,
            "adapter_state_dict": adapter_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "world_size": world_size,
            "rng_by_rank": rng_by_rank,
            "protocol": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    if args.microbatch * args.gradient_accumulation * world_size != args.effective_batch:
        raise RuntimeError("declared effective batch does not match execution")
    if args.mode == "formal" and args.max_steps != 80_000:
        raise RuntimeError("formal adapter budget must be the source-defined 80k optimizer steps")
    seed_all(args.seed, rank)
    verify_identities()
    model, adapter_contract = load_model(device, args.seed)
    adapter_parameters = [value for _, value in adapter_named_parameters(model)]
    optimizer = torch.optim.AdamW(
        adapter_parameters,
        lr=args.learning_rate,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_multiplier(step, args.max_steps, args.warmup_ratio),
    )
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior = checkpoint["protocol"]
        for field in (
            "max_steps",
            "microbatch",
            "gradient_accumulation",
            "effective_batch",
            "seed",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
        ):
            if prior[field] != getattr(args, field):
                raise RuntimeError(f"resume protocol drift: {field}")
        if checkpoint["world_size"] != world_size:
            raise RuntimeError("resume requires the frozen formal world size")
        load_adapter_state_dict(model, checkpoint["adapter_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        rng = checkpoint["rng_by_rank"][rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        start_step = int(checkpoint["step"])
    distributed_model: torch.nn.Module = model
    if world_size > 1:
        distributed_model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    dataset = OriginalUniTPairDataset("train", augment=True, seed=args.seed)
    distributed_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True
    )
    sampler = EpochTaggedSampler(distributed_sampler)
    loader = DataLoader(
        dataset,
        batch_size=args.microbatch,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 10_000 + rank),
    )
    consumed_microbatches = start_step * args.gradient_accumulation
    epoch, batch_offset = divmod(consumed_microbatches, len(loader))
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(batch_offset):
        next(iterator)
    frozen_before = frozen_parameter_digest(model) if rank == 0 else None
    if world_size > 1:
        dist.barrier()
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        parameter_audit = trainable_audit(model)
        if (
            parameter_audit["trainable_parameter_count"] != EXPECTED_ADAPTER_PARAMETERS
            or parameter_audit["shared_official_parameters_trainable"] != 0
        ):
            raise RuntimeError("adapter-only trainable parameter gate failed")
        atomic_json(
            ARTIFACTS / "adapter_trainable_parameters.json",
            {
                "schema": "tactile3d-unit.s4-3-pi2v-adapter-parameters.v1",
                "status": "PASS",
                **parameter_audit,
            },
        )
        atomic_json(
            ARTIFACTS / "dexjoco_unit_adapter_contract.json",
            {
                "schema": "tactile3d-unit.s4-3-pi2v-dexjoco-adapter-contract.v1",
                "status": "PASS",
                "description": "Original UniT official implementation and released tokenizer initialization with a DexJoCo-specific embodiment adapter",
                "new_embodiment_slot": 30,
                "existing_slots": list(range(30)),
                "official_bank_storage": "frozen original tensors",
                "optimizer_scope": "appended DexJoCo tensors only",
                "shared_components_trainable": 0,
                "adapter": adapter_contract,
            },
        )
        launch = {
            "schema": "tactile3d-unit.s4-3-pi2v-adapter-launch.v1",
            "status": "RUNNING",
            "mode": args.mode,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "official_source_sha": OFFICIAL_SOURCE_SHA,
            "official_checkpoint_sha": OFFICIAL_CHECKPOINT_SHA,
            "dexjoco_adapter": adapter_contract,
            "world_size": world_size,
            "visible_physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "microbatch_per_device": args.microbatch,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch": args.effective_batch,
            "precision": "BF16",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": [0.95, 0.999],
            "warmup_ratio": args.warmup_ratio,
            "scheduler": "cosine",
            "target_steps": args.max_steps,
            "frozen_parameter_digest_at_launch": frozen_before,
            "log": f"$PI2V_ROOT/logs/{args.output.name}/train.jsonl",
            "output": f"$PI2V_ROOT/experiments/{args.output.name}",
        }
        atomic_json(
            ARTIFACTS / ("adapter_launch.json" if args.mode == "formal" else "adapter_smoke_launch.json"),
            launch,
        )

    optimizer.zero_grad(set_to_none=True)
    timings: list[float] = []
    latest_metrics: dict[str, float] = {}
    last_gradient_gate: dict[str, Any] = {}
    for step in range(start_step, args.max_steps):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        accum_metrics: dict[str, float] = {}
        for accumulation_index in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = {
                key: value.to(device=device, non_blocking=True)
                for key, value in batch.items()
            }
            sync = accumulation_index == args.gradient_accumulation - 1
            context = (
                nullcontext()
                if sync or not isinstance(distributed_model, DistributedDataParallel)
                else distributed_model.no_sync()
            )
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = distributed_model(batch)
                    loss = outputs["loss"] / args.gradient_accumulation
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("REPRESENTATION_TRAINING_FAILURE: non-finite loss")
                loss.backward()
            for name, value in outputs.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    accum_metrics[name] = accum_metrics.get(name, 0.0) + float(
                        value.detach().float()
                    ) / args.gradient_accumulation
                elif isinstance(value, (int, float)):
                    accum_metrics[name] = accum_metrics.get(name, 0.0) + float(
                        value
                    ) / args.gradient_accumulation
        last_gradient_gate = finite_nonzero_adapter_gradients(adapter_parameters)
        if not last_gradient_gate["all_present"] or not last_gradient_gate["all_finite"]:
            raise RuntimeError("REPRESENTATION_TRAINING_FAILURE: invalid adapter gradient")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(adapter_parameters, 1.0))
        if not math.isfinite(gradient_norm):
            raise RuntimeError("REPRESENTATION_TRAINING_FAILURE: non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        latest_metrics = reduce_metrics(accum_metrics, device, world_size)
        current_step = step + 1
        row = {
            "step": current_step,
            "total_steps": args.max_steps,
            "sec_per_step": elapsed,
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": gradient_norm,
            "adapter_gradient": last_gradient_gate,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            **latest_metrics,
        }
        if rank == 0:
            append_log(args.metrics_log, row)
            print(json.dumps(row, sort_keys=True), flush=True)

        startup_due = args.mode == "formal" and current_step == args.startup_audit_step
        if startup_due:
            if world_size > 1:
                dist.barrier()
            local_device = {
                "rank": rank,
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[rank],
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                "adapter_gradient": last_gradient_gate,
            }
            devices = [None] * world_size if rank == 0 else None
            if world_size > 1:
                dist.gather_object(local_device, devices, dst=0)
            else:
                devices = [local_device]
            if rank == 0:
                frozen_now = frozen_parameter_digest(model)
                steady = timings[5:] if len(timings) > 5 else timings
                finite_losses = all(math.isfinite(value) for value in latest_metrics.values())
                losses_present = all(
                    key in latest_metrics
                    for key in (
                        "vision_recon_loss",
                        "action_recon_loss",
                        "both_vision_recon_loss",
                        "both_action_recon_loss",
                        "vision_only_vision_recon_loss",
                        "vision_only_action_recon_loss",
                        "action_only_vision_recon_loss",
                        "action_only_action_recon_loss",
                        "vq_loss",
                    )
                )
                rvq_active = any("vq_active_codes_layer_0" in key for key in latest_metrics)
                gradients_valid = all(
                    row["adapter_gradient"]["all_present"]
                    and row["adapter_gradient"]["all_finite"]
                    and row["adapter_gradient"]["nonzero_tensor_count"]
                    == row["adapter_gradient"]["tensor_count"]
                    for row in devices
                )
                startup_pass = all(
                    (frozen_now == frozen_before, finite_losses, losses_present, rvq_active,
                     gradients_valid, len(steady) >= 50, os.access(args.output, os.W_OK))
                )
                startup = {
                    "schema": "tactile3d-unit.s4-3-pi2v-adapter-startup-validation.v1",
                    "status": "PASS" if startup_pass else "FAIL",
                    "step": current_step,
                    "target_step": args.max_steps,
                    "steady_timing_samples": len(steady),
                    "median_sec_per_step": statistics.median(steady),
                    "finite_losses": finite_losses,
                    "original_unit_losses_present": losses_present,
                    "rvq_active": rvq_active,
                    "all_rank_gradients_valid_and_nonzero": gradients_valid,
                    "devices": devices,
                    "frozen_parameter_digest_before": frozen_before,
                    "frozen_parameter_digest_after": frozen_now,
                    "frozen_parameters_unchanged": frozen_now == frozen_before,
                    "latest_metrics": latest_metrics,
                    "checkpoint_root_writable": os.access(args.output, os.W_OK),
                }
                atomic_json(ARTIFACTS / "adapter_startup_validation.json", startup)
                if startup["status"] != "PASS":
                    raise RuntimeError("frozen official tokenizer parameters changed")
            if world_size > 1:
                dist.barrier()

        if current_step % args.save_steps == 0 or current_step == args.max_steps:
            if world_size > 1:
                dist.barrier()
            checkpoint_path = args.output / f"checkpoint-step{current_step}.pt"
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, current_step, args, rank, world_size
            )
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        frozen_after = frozen_parameter_digest(model)
        steady = timings[5:] if len(timings) > 5 else timings
        final_checkpoint = args.output / f"checkpoint-step{args.max_steps}.pt"
        result = {
            "schema": "tactile3d-unit.s4-3-pi2v-adapter-training-completion.v1",
            "status": "PASS" if frozen_after == frozen_before else "FAIL",
            "mode": args.mode,
            "steps": args.max_steps,
            "optimizer_steps": args.max_steps - start_step,
            "effective_batch": args.effective_batch,
            "samples_seen": args.max_steps * args.effective_batch,
            "median_steady_sec_per_step": statistics.median(steady),
            "timing_samples": len(steady),
            "final_metrics": latest_metrics,
            "adapter_gradient": last_gradient_gate,
            "frozen_parameter_digest_before": frozen_before,
            "frozen_parameter_digest_after": frozen_after,
            "frozen_parameters_unchanged": frozen_after == frozen_before,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "final_checkpoint": f"$PI2V_ROOT/experiments/{args.output.name}/"
            + final_checkpoint.name,
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
        }
        artifact_name = (
            "adapter_training_completion.json"
            if args.mode == "formal"
            else "adapter_memory_smoke.json"
        )
        atomic_json(ARTIFACTS / artifact_name, result)
        if result["status"] != "PASS":
            raise RuntimeError("frozen official tokenizer parameters changed")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
