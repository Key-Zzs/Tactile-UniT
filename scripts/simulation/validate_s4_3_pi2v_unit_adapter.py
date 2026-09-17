#!/usr/bin/env python3
"""Cold-load and validate the frozen PI2V Original-UniT DexJoCo adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers.image_processing_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/simulation"))

from s4_3_pi2v_unit_adapter import (  # noqa: E402
    DEXJOCO_CATEGORY_ID,
    adapter_state_dict,
    frozen_parameter_digest,
    load_adapter_state_dict,
    sha256_file,
)
from train_s4_3_pi2v_unit_adapter import (  # noqa: E402
    ARTIFACTS,
    CHECKPOINT_ROOT,
    OFFICIAL_CHECKPOINT_SHA,
    OFFICIAL_SOURCE_SHA,
    OriginalUniTPairDataset,
    load_model,
)


CHECKPOINT = (
    ROOT / ".local/experiments/simulation/s4_3_pi2v/unit_adapter/checkpoint-step80000.pt"
)
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2v_unit_adapter_protocol.json"
DEV_PAIRS = ROOT / ".local/cache/simulation/s4_2/pairs/validation.npz"
COMPLETION = ARTIFACTS / "adapter_training_completion.json"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tensor_dict_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(str(tuple(array.shape)).encode() + b"\0")
        digest.update(array.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class IndexedDataset(Dataset):
    def __init__(self, dataset: OriginalUniTPairDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[(index, 0)]


def group_derangement(groups: np.ndarray) -> np.ndarray:
    """Return a deterministic bijection with a different source group per row."""

    order = np.argsort(groups, kind="stable")
    _, counts = np.unique(groups[order], return_counts=True)
    shift = int(counts.max())
    permutation = np.empty(len(groups), dtype=np.int64)
    permutation[order] = np.roll(order, shift)
    if len(np.unique(permutation)) != len(groups) or np.any(groups == groups[permutation]):
        raise RuntimeError("could not construct a source-group-disjoint DEV control")
    return permutation


def route(model, vision, action, pv: int, pa: int):
    batch = vision.shape[0]
    device = vision.device
    unit = model.fusion(
        visual_tokens=vision,
        action_tokens=action,
        pv=torch.full((batch,), pv, dtype=torch.long, device=device),
        pa=torch.full((batch,), pa, dtype=torch.long, device=device),
    )
    # The official forward executes this path under BF16 autocast. Presence
    # mask arithmetic promotes the fusion output to FP32 when called directly,
    # so restore the audited compute dtype before the frozen down-resampler.
    unit = unit.to(dtype=model.dtype)
    down = model.vq_down_resampler(unit)
    quantized, indices, _ = model.vq(down)
    up = model.bridge_projector(quantized)
    return unit, down, indices, up


def action_prediction(model, up, state_features, action_inputs):
    features = BatchFeature(
        data={"backbone_features": up + model.pos_embed.unsqueeze(0), "state_features": state_features}
    )
    return model.action_decoder.get_action(features, action_inputs)["action_pred"]


def action_smooth_l1_per_sample(prediction, action_inputs):
    target = action_inputs["action"]
    mask = action_inputs["action_mask"]
    values = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.5) * mask
    return values.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)


def vision_cosine_loss_per_sample(model, up, obs_embeddings, goal_embeddings):
    prediction = model.vision_decoder(
        cond_input=obs_embeddings, latent_motion_tokens=up
    )
    return 1.0 - F.cosine_similarity(prediction, goal_embeddings, dim=-1).mean(dim=-1)


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    block = 100
    for start in range(0, BOOTSTRAP_RESAMPLES, block):
        stop = min(start + block, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def code_usage(indices: np.ndarray) -> dict[str, dict[str, float | int]]:
    result = {}
    for level in range(indices.shape[-1]):
        counts = np.bincount(indices[..., level].reshape(-1), minlength=128)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        result[f"level_{level + 1}"] = {
            "active_codes": int(np.count_nonzero(counts)),
            "entropy_nats": entropy,
            "normalized_entropy": entropy / math.log(128),
            "perplexity": math.exp(entropy),
            "minimum_count": int(counts[counts > 0].min()),
            "maximum_count": int(counts.max()),
        }
    return result


def retrieval_metrics(vision: np.ndarray, action: np.ndarray, device: torch.device) -> dict[str, float]:
    vision_tensor = F.normalize(torch.from_numpy(vision).float().flatten(1).to(device), dim=-1)
    action_tensor = F.normalize(torch.from_numpy(action).float().flatten(1).to(device), dim=-1)
    ranks_va = []
    ranks_av = []
    chunk = 256
    for start in range(0, len(vision), chunk):
        stop = min(start + chunk, len(vision))
        similarity = vision_tensor[start:stop] @ action_tensor.T
        target = torch.arange(start, stop, device=device)
        paired = similarity[torch.arange(stop - start, device=device), target]
        ranks_va.append((similarity > paired[:, None]).sum(dim=1).cpu())
        similarity_t = action_tensor[start:stop] @ vision_tensor.T
        paired_t = similarity_t[torch.arange(stop - start, device=device), target]
        ranks_av.append((similarity_t > paired_t[:, None]).sum(dim=1).cpu())
    rank_va = torch.cat(ranks_va).numpy()
    rank_av = torch.cat(ranks_av).numpy()
    return {
        "vision_to_action_recall_at_1": float(np.mean(rank_va < 1)),
        "vision_to_action_recall_at_5": float(np.mean(rank_va < 5)),
        "vision_to_action_recall_at_10": float(np.mean(rank_va < 10)),
        "action_to_vision_recall_at_1": float(np.mean(rank_av < 1)),
        "action_to_vision_recall_at_5": float(np.mean(rank_av < 5)),
        "action_to_vision_recall_at_10": float(np.mean(rank_av < 10)),
        "vision_to_action_median_rank": float(np.median(rank_va + 1)),
        "action_to_vision_median_rank": float(np.median(rank_av + 1)),
    }


def concatenate(rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    outputs = (
        ARTIFACTS / "adapter_checkpoint_manifest.json",
        ARTIFACTS / "adapter_representation_metrics.json",
        ARTIFACTS / "adapter_codebook_usage.json",
        ARTIFACTS / "adapter_cross_reconstruction.json",
    )
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite frozen PI2V adapter validation artifacts")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible or torch.cuda.device_count() != 1:
        raise SystemExit("adapter validation requires exactly one explicitly visible GPU")

    protocol = json.loads(PROTOCOL.read_text())
    completion = json.loads(COMPLETION.read_text())
    checkpoint_sha = sha256_file(CHECKPOINT)
    if completion.get("status") != "PASS" or checkpoint_sha != completion["final_checkpoint_sha256"]:
        raise RuntimeError("formal adapter completion/checkpoint identity failed")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["step"] != 80_000 or checkpoint["official_source_sha"] != OFFICIAL_SOURCE_SHA:
        raise RuntimeError("formal adapter checkpoint contract failed")

    device = torch.device("cuda:0")
    model, adapter_contract = load_model(device, protocol["seed"])
    load_adapter_state_dict(model, checkpoint["adapter_state_dict"])
    model.eval()
    frozen_before = frozen_parameter_digest(model)
    if frozen_before != completion["frozen_parameter_digest_after"]:
        raise RuntimeError("frozen official parameter identity failed on cold load")

    dataset = OriginalUniTPairDataset("validation", augment=False, seed=protocol["seed"])
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    with np.load(DEV_PAIRS, allow_pickle=False) as source:
        groups = source["source_trajectory_id"].copy()
        pair_ids = source["pair_id"].copy()
    permutation = group_derangement(groups)

    stored: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "index",
            "vision_query",
            "action_query",
            "vision_down",
            "action_down",
            "fused_indices",
            "vision_indices",
            "action_indices",
            "fused_action",
            "target_action",
            "fused_action_loss",
            "vision_action_loss",
            "action_action_loss",
            "mean_action_loss",
            "fused_vision_loss",
            "vision_vision_loss",
            "action_vision_loss",
            "no_motion_vision_loss",
        )
    }
    branch_norms = {"vision": [], "action": [], "fusion": []}

    for batch_number, (indices, batch) in enumerate(loader, start=1):
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        obs, goal, action_inputs, _ = model.prepare_input(batch)
        vision, obs_embeddings, goal_embeddings = model.vision_branch(obs, goal, batch_size=len(indices))
        action, state_features = model.action_branch(
            actions=action_inputs["action"],
            state=action_inputs["state"],
            cat_ids=action_inputs["embodiment_id"],
        )
        fused_unit, _, fused_codes, fused_up = route(model, vision, action, 1, 1)
        _, vision_down, vision_codes, vision_up = route(model, vision, action, 1, 0)
        _, action_down, action_codes, action_up = route(model, vision, action, 0, 1)
        fused_action = action_prediction(model, fused_up, state_features, action_inputs)
        vision_action = action_prediction(model, vision_up, state_features, action_inputs)
        action_action = action_prediction(model, action_up, state_features, action_inputs)

        tensors = (vision, action, fused_unit, vision_down, action_down, fused_up)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("non-finite branch output on representation DEV")
        branch_norms["vision"].append(vision.float().norm(dim=-1).mean().item())
        branch_norms["action"].append(action.float().norm(dim=-1).mean().item())
        branch_norms["fusion"].append(fused_unit.float().norm(dim=-1).mean().item())

        stored["index"].append(indices.numpy())
        stored["vision_query"].append(vision.cpu().float().numpy())
        stored["action_query"].append(action.cpu().float().numpy())
        stored["vision_down"].append(vision_down.cpu().float().numpy())
        stored["action_down"].append(action_down.cpu().float().numpy())
        stored["fused_indices"].append(fused_codes.cpu().numpy().astype(np.int16))
        stored["vision_indices"].append(vision_codes.cpu().numpy().astype(np.int16))
        stored["action_indices"].append(action_codes.cpu().numpy().astype(np.int16))
        stored["fused_action"].append(fused_action[..., :22].cpu().float().numpy())
        stored["target_action"].append(action_inputs["action"][..., :22].cpu().float().numpy())
        for name, prediction in (
            ("fused_action_loss", fused_action),
            ("vision_action_loss", vision_action),
            ("action_action_loss", action_action),
        ):
            stored[name].append(
                action_smooth_l1_per_sample(prediction, action_inputs).float().cpu().numpy()
            )
        stored["mean_action_loss"].append(
            action_smooth_l1_per_sample(torch.zeros_like(fused_action), action_inputs)
            .float()
            .cpu()
            .numpy()
        )
        for name, up in (
            ("fused_vision_loss", fused_up),
            ("vision_vision_loss", vision_up),
            ("action_vision_loss", action_up),
        ):
            stored[name].append(
                vision_cosine_loss_per_sample(model, up, obs_embeddings, goal_embeddings)
                .float()
                .cpu()
                .numpy()
            )
        stored["no_motion_vision_loss"].append(
            (1.0 - F.cosine_similarity(obs_embeddings, goal_embeddings, dim=-1).mean(dim=-1))
            .float()
            .cpu()
            .numpy()
        )
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(f"representation pass {batch_number}/{len(loader)}", flush=True)

    values = concatenate(stored)
    if not np.array_equal(values["index"], np.arange(len(dataset))):
        raise RuntimeError("DEV iteration order changed")

    controls: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "shuffled_vision_to_action",
            "shuffled_action_to_vision",
            "shuffled_fusion_to_action",
            "shuffled_fusion_to_vision",
        )
    }
    vision_queries = torch.from_numpy(values["vision_query"])
    action_queries = torch.from_numpy(values["action_query"])
    for batch_number, (indices, batch) in enumerate(loader, start=1):
        original = indices.numpy()
        shuffled = permutation[original]
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        obs, goal, action_inputs, _ = model.prepare_input(batch)
        _, obs_embeddings, goal_embeddings = model.vision_branch(obs, goal, batch_size=len(indices))
        _, state_features = model.action_branch(
            actions=action_inputs["action"],
            state=action_inputs["state"],
            cat_ids=action_inputs["embodiment_id"],
        )
        shuffled_vision = vision_queries[shuffled].to(device=device, dtype=model.dtype)
        shuffled_action = action_queries[shuffled].to(device=device, dtype=model.dtype)
        current_vision = vision_queries[original].to(device=device, dtype=model.dtype)
        _, _, _, vision_up = route(model, shuffled_vision, shuffled_action, 1, 0)
        _, _, _, action_up = route(model, shuffled_vision, shuffled_action, 0, 1)
        _, _, _, fusion_up = route(model, current_vision, shuffled_action, 1, 1)
        controls["shuffled_vision_to_action"].append(
            action_smooth_l1_per_sample(
                action_prediction(model, vision_up, state_features, action_inputs), action_inputs
            ).float().cpu().numpy()
        )
        controls["shuffled_action_to_vision"].append(
            vision_cosine_loss_per_sample(model, action_up, obs_embeddings, goal_embeddings)
            .float()
            .cpu()
            .numpy()
        )
        controls["shuffled_fusion_to_action"].append(
            action_smooth_l1_per_sample(
                action_prediction(model, fusion_up, state_features, action_inputs), action_inputs
            ).float().cpu().numpy()
        )
        controls["shuffled_fusion_to_vision"].append(
            vision_cosine_loss_per_sample(model, fusion_up, obs_embeddings, goal_embeddings)
            .float()
            .cpu()
            .numpy()
        )
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(f"control pass {batch_number}/{len(loader)}", flush=True)
    control_values = concatenate(controls)

    vision_flat = values["vision_down"].reshape(len(dataset), -1)
    action_flat = values["action_down"].reshape(len(dataset), -1)
    paired_relation = np.sum(vision_flat * action_flat, axis=1) / np.maximum(
        np.linalg.norm(vision_flat, axis=1) * np.linalg.norm(action_flat, axis=1), 1e-12
    )
    shuffled_action_flat = action_flat[permutation]
    shuffled_relation = np.sum(vision_flat * shuffled_action_flat, axis=1) / np.maximum(
        np.linalg.norm(vision_flat, axis=1) * np.linalg.norm(shuffled_action_flat, axis=1), 1e-12
    )
    relation_delta = paired_relation - shuffled_relation
    relation_ci = bootstrap_mean_ci(relation_delta)
    retrieval = retrieval_metrics(values["vision_down"], values["action_down"], device)

    prediction = values["fused_action"].reshape(-1, 22)
    target = values["target_action"].reshape(-1, 22)
    target_centered = target - target.mean(axis=0, keepdims=True)
    target_variance_sum = np.sum(target_centered**2, axis=0)
    constant_dimensions = np.flatnonzero(target_variance_sum < 1e-8)
    evaluable_dimensions = np.flatnonzero(target_variance_sum >= 1e-8)
    r2 = np.full(22, np.nan, dtype=np.float64)
    r2[evaluable_dimensions] = 1.0 - np.sum(
        (prediction[:, evaluable_dimensions] - target[:, evaluable_dimensions]) ** 2, axis=0
    ) / target_variance_sum[evaluable_dimensions]
    correlations: list[float | None] = []
    for dimension in range(22):
        correlations.append(
            None
            if dimension in constant_dimensions
            else float(np.corrcoef(prediction[:, dimension], target[:, dimension])[0, 1])
        )

    usage = {
        "schema": "tactile3d-unit.s4-3-pi2v-adapter-codebook-usage.v1",
        "status": "PASS",
        "rows": len(dataset),
        "slots_per_row": 8,
        "classes_per_level": 128,
        "routes": {
            "fused": code_usage(values["fused_indices"]),
            "vision_only": code_usage(values["vision_indices"]),
            "action_only": code_usage(values["action_indices"]),
        },
    }
    thresholds = protocol["preregistered_dev_gate"]
    code_gate = all(
        level["active_codes"] >= thresholds["minimum_active_codes_each_level_each_route"]
        and level["normalized_entropy"] >= thresholds["minimum_normalized_code_entropy_each_level_each_route"]
        for route_values in usage["routes"].values()
        for level in route_values.values()
    )
    cross = {
        "schema": "tactile3d-unit.s4-3-pi2v-adapter-cross-reconstruction.v1",
        "status": "PASS",
        "source_group_disjoint_shuffle": True,
        "shuffle_pair_id_sha256": hashlib.sha256(
            np.stack((np.arange(len(dataset)), permutation), axis=1).tobytes()
        ).hexdigest(),
        "vision_to_action": {
            "paired_smooth_l1": float(values["vision_action_loss"].mean()),
            "shuffled_smooth_l1": float(control_values["shuffled_vision_to_action"].mean()),
        },
        "action_to_vision": {
            "paired_cosine_loss": float(values["action_vision_loss"].mean()),
            "shuffled_cosine_loss": float(control_values["shuffled_action_to_vision"].mean()),
        },
        "fusion_to_action": {
            "paired_smooth_l1": float(values["fused_action_loss"].mean()),
            "shuffled_smooth_l1": float(control_values["shuffled_fusion_to_action"].mean()),
        },
        "fusion_to_vision": {
            "paired_cosine_loss": float(values["fused_vision_loss"].mean()),
            "shuffled_cosine_loss": float(control_values["shuffled_fusion_to_vision"].mean()),
        },
    }
    cross_gate = all(
        values["paired_smooth_l1"] < values["shuffled_smooth_l1"]
        if "paired_smooth_l1" in values
        else values["paired_cosine_loss"] < values["shuffled_cosine_loss"]
        for values in (
            cross["vision_to_action"],
            cross["action_to_vision"],
            cross["fusion_to_action"],
            cross["fusion_to_vision"],
        )
    )
    frozen_after = frozen_parameter_digest(model)
    gates = {
        "full_frozen_dev_4860": len(dataset) == 4_860,
        "all_outputs_finite": all(
            np.isfinite(value).all()
            for value in (*values.values(), *control_values.values())
        ),
        "vision_action_fusion_branches_finite_nonzero": all(
            np.isfinite(rows).all() and min(rows) > 0 for rows in branch_norms.values()
        ),
        "no_catastrophic_codebook_collapse": code_gate,
        "paired_relation_bootstrap_ci95_lower_above_zero": relation_ci[0]
        > thresholds["paired_fused_vs_shuffled_alignment_bootstrap_ci95_lower"],
        "cross_reconstruction_better_than_group_shuffled": cross_gate,
        "fused_action_better_than_train_mean_action": float(values["fused_action_loss"].mean())
        < float(values["mean_action_loss"].mean()),
        "fused_vision_better_than_no_motion": float(values["fused_vision_loss"].mean())
        < float(values["no_motion_vision_loss"].mean()),
        "frozen_official_parameters_unchanged": frozen_after == frozen_before,
        "official_parent_matches": checkpoint["official_checkpoint_sha"] == OFFICIAL_CHECKPOINT_SHA,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    metrics = {
        "schema": "tactile3d-unit.s4-3-pi2v-adapter-representation-metrics.v1",
        "status": status,
        "decision": (
            "S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_PASS"
            if status == "PASS"
            else "S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL"
        ),
        "split": "frozen source-group-disjoint DEV",
        "rows": len(dataset),
        "source_groups": int(len(np.unique(groups))),
        "pair_id_sha256": hashlib.sha256(pair_ids.tobytes()).hexdigest(),
        "branch_output_mean_l2_norm": {
            name: float(np.mean(rows)) for name, rows in branch_norms.items()
        },
        "reconstruction": {
            "fused_action_smooth_l1": float(values["fused_action_loss"].mean()),
            "vision_to_action_smooth_l1": float(values["vision_action_loss"].mean()),
            "action_to_action_smooth_l1": float(values["action_action_loss"].mean()),
            "train_mean_action_smooth_l1": float(values["mean_action_loss"].mean()),
            "fused_vision_cosine_loss": float(values["fused_vision_loss"].mean()),
            "vision_to_vision_cosine_loss": float(values["vision_vision_loss"].mean()),
            "action_to_vision_cosine_loss": float(values["action_vision_loss"].mean()),
            "no_motion_vision_cosine_loss": float(values["no_motion_vision_loss"].mean()),
        },
        "paired_vs_shuffled_relation": {
            "paired_mean_cosine": float(paired_relation.mean()),
            "group_shuffled_mean_cosine": float(shuffled_relation.mean()),
            "paired_minus_shuffled_mean": float(relation_delta.mean()),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_ci95": list(relation_ci),
        },
        "retrieval": retrieval,
        "action_semantic_retention": {
            "normalized_action_dimensions": 22,
            "constant_target_dimensions": constant_dimensions.tolist(),
            "constant_target_dimension_prediction_rmse": {
                str(index): float(
                    np.sqrt(np.mean((prediction[:, index] - target[:, index]) ** 2))
                )
                for index in constant_dimensions
            },
            "evaluable_target_dimensions": evaluable_dimensions.tolist(),
            "median_dimension_r2": float(np.nanmedian(r2)),
            "minimum_dimension_r2": float(np.nanmin(r2)),
            "median_dimension_pearson": float(
                np.median([value for value in correlations if value is not None])
            ),
            "minimum_dimension_pearson": float(
                np.min([value for value in correlations if value is not None])
            ),
            "dimension_r2": [None if not np.isfinite(value) else float(value) for value in r2],
            "dimension_pearson": correlations,
        },
        "frozen_parameter_digest_before": frozen_before,
        "frozen_parameter_digest_after": frozen_after,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2v-adapter-checkpoint-manifest.v1",
        "status": status,
        "checkpoint": "$PI2V_ROOT/experiments/unit_adapter/checkpoint-step80000.pt",
        "checkpoint_sha256": checkpoint_sha,
        "adapter_state_sha256": tensor_dict_sha256(adapter_state_dict(model)),
        "adapter_trainable_parameters": adapter_contract["trainable_parameters"],
        "official_source_commit": OFFICIAL_SOURCE_SHA,
        "official_checkpoint_parent_sha256": OFFICIAL_CHECKPOINT_SHA,
        "frozen_official_parameter_digest": frozen_after,
        "protocol_sha256": sha256_file(PROTOCOL),
        "training_entrypoint_sha256": sha256_file(
            ROOT / "scripts/simulation/train_s4_3_pi2v_unit_adapter.py"
        ),
        "validation_entrypoint_sha256": sha256_file(Path(__file__)),
        "train_pair_sha256": protocol["data"]["train"]["sha256"],
        "dev_pair_sha256": protocol["data"]["dev"]["sha256"],
        "selected_checkpoint_rule": protocol["formal_selection"],
        "codebook_identity": frozen_after,
        "representation_decision": metrics["decision"],
    }
    usage["status"] = "PASS" if code_gate else "FAIL"
    cross["status"] = "PASS" if cross_gate else "FAIL"
    atomic_json(ARTIFACTS / "adapter_codebook_usage.json", usage)
    atomic_json(ARTIFACTS / "adapter_cross_reconstruction.json", cross)
    atomic_json(ARTIFACTS / "adapter_representation_metrics.json", metrics)
    atomic_json(ARTIFACTS / "adapter_checkpoint_manifest.json", manifest)
    print(json.dumps({"status": status, "decision": metrics["decision"], "gates": gates}))
    if status != "PASS":
        raise SystemExit("S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL")


if __name__ == "__main__":
    main()
