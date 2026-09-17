#!/usr/bin/env python3
"""Bounded read-only native-GR1 sanity check for the PI2W custom gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / ".local/external/s4_3_pi2u/unit_official"
CHECKPOINT = ROOT / ".local/external/s4_3_pi2u/unit_fulldata/VLA-UniT-3B-fulldata/tokenizer"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi2w/native_unit_gate_sanity.json"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2w/native_loader"
DATA_CONFIG = "fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only"
TASKS = (
    "gr1_unified.PnPBottleToCabinetClose",
    "gr1_unified.PnPCanToDrawerClose",
    "gr1_unified.PnPCupToDrawerClose",
    "gr1_unified.PnPMilkToMicrowaveClose",
    "gr1_unified.PnPPotatoToMicrowaveClose",
    "gr1_unified.PnPWineToCabinetClose",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA",
)

sys.path.insert(0, str(OFFICIAL))
sys.path.insert(1, str(ROOT))
from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage  # noqa: E402
from gr00t.experiment.data_config_unit import load_data_config  # noqa: E402
from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer  # noqa: E402
from gr00t.model.transforms import collate  # noqa: E402
from scripts.reproduce.check_gr1_data_contract import redirect_loader_stats_writes  # noqa: E402


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def deterministic_frames(length: int, count: int = 4, goal_horizon: int = 16) -> list[int]:
    maximum = length - 1 - goal_horizon
    if maximum < 0:
        return []
    return [int(value) for value in np.linspace(0, maximum, count, dtype=np.int64)]


def native_samples(dataset_root: Path, data_config: Any, maximum: int) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    labels: list[str] = []
    CACHE.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        if len(samples) >= maximum:
            break
        task_root = dataset_root / task
        episodes = [json.loads(line) for line in (task_root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        held_out = episodes[-10:]
        episode_ids = [int(row["episode_index"]) for row in held_out]
        with redirect_loader_stats_writes(task_root, CACHE / task):
            dataset = LeRobotSingleDatasetWithGoalImage(
                dataset_path=task_root,
                modality_configs=data_config.modality_config(),
                transforms=data_config.transform(),
                embodiment_tag="gr1",
                video_backend="decord",
                episode_ids=episode_ids,
            )
            dataset.transforms.eval()
            for row in held_out:
                episode = int(row["episode_index"])
                for frame in deterministic_frames(int(row["length"])):
                    raw = dataset.get_step_data(episode, frame)
                    samples.append(dataset.transforms(raw))
                    labels.append(task)
                    if len(samples) >= maximum:
                        return samples, labels
    return samples, labels


def route(model, vision, action, obs_features, pv: int, pa: int) -> torch.Tensor:
    batch = vision.shape[0]
    device = vision.device
    unit = model.fusion(
        visual_tokens=vision,
        action_tokens=action,
        pv=torch.full((batch,), pv, dtype=torch.long, device=device),
        pa=torch.full((batch,), pa, dtype=torch.long, device=device),
    ).to(model.dtype)
    down = model.vq_down_resampler(unit)
    quantized, _, _ = model.vq(down)
    projected = model.bridge_projector(quantized)
    return model.vision_decoder(cond_input=obs_features, latent_motion_tokens=projected)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if ARTIFACT.exists():
        raise SystemExit("refusing to overwrite native UniT sanity artifact")
    if args.max_samples > 512:
        raise SystemExit("native sanity is bounded to at most 512 samples")
    if not args.dataset_root.is_dir():
        atomic_json(ARTIFACT, {
            "schema": "tactile3d-unit.s4-3-pi2w-native-unit-gate-sanity.v1",
            "status": "NATIVE_UNIT_GATE_SANITY_NOT_AVAILABLE",
            "reason": "Configured local GR1 dataset root is absent; no download attempted.",
        })
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible or torch.cuda.device_count() != 1:
        raise SystemExit("native sanity requires exactly one explicitly visible idle GPU")

    full_config = json.loads(
        (ROOT / ".local/external/s4_3_pi2u/unit_checkpoints_metadata/VLA-UniT-3B-fulldata/config.json").read_text()
    )
    data_config = load_data_config(
        DATA_CONFIG,
        eagle_path=full_config["backbone_cfg"]["eagle_path"],
        use_bridge=False,
        ignore_lang_prefix=False,
        enable_imagenet_preprocessing=True,
        tokenizer_only=True,
    )
    samples, labels = native_samples(args.dataset_root, data_config, args.max_samples)
    if not samples:
        raise RuntimeError("local GR1 root produced no bounded native samples")

    device = torch.device("cuda:0")
    model = GR00T_Tokenizer.from_pretrained(
        CHECKPOINT,
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
    model.config.compute_dtype = "bfloat16"
    model.compute_dtype = "bfloat16"
    model.eval().to(device=device, dtype=torch.bfloat16)
    model.vq.eval()
    paths = ("D0_no_motion", "D1_vision_only", "D2_action_only", "D3_fusion")
    values: dict[str, list[np.ndarray]] = {name: [] for name in paths}
    for start in range(0, len(samples), args.batch_size):
        batch = collate(samples[start:start + args.batch_size], None)
        obs, goal, action_inputs, _ = model.prepare_input(batch)
        batch_size = len(samples[start:start + args.batch_size])
        vision, obs_features, future_features = model.vision_branch(obs, goal, batch_size=batch_size)
        action, _ = model.action_branch(
            actions=action_inputs["action"],
            state=action_inputs["state"],
            cat_ids=action_inputs["embodiment_id"],
        )
        predictions = {
            "D0_no_motion": obs_features,
            "D1_vision_only": route(model, vision, action, obs_features, 1, 0),
            "D2_action_only": route(model, vision, action, obs_features, 0, 1),
            "D3_fusion": route(model, vision, action, obs_features, 1, 1),
        }
        for name, prediction in predictions.items():
            loss = 1.0 - F.cosine_similarity(prediction.float(), future_features.float(), dim=-1).mean(dim=-1)
            values[name].append(loss.cpu().numpy())
        print(f"native sanity {min(start + args.batch_size, len(samples))}/{len(samples)}", flush=True)
    arrays = {name: np.concatenate(rows) for name, rows in values.items()}
    label_array = np.asarray(labels)
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2w-native-unit-gate-sanity.v1",
        "status": "PASS",
        "execution": "READ_ONLY_INFERENCE",
        "released_checkpoint": "VLA-UniT-3B-fulldata/tokenizer",
        "official_source_commit": "0d762e32180bddd765694ef3846a3a5053f9d37f",
        "dataset": "$GR1_DATASET_ROOT",
        "sampling": "first 512 rows of the frozen representation-benchmark rule: last 10 episodes/task, four deterministic valid frames/episode, task order frozen in source",
        "samples": len(samples),
        "tasks": int(len(np.unique(label_array))),
        "metric": "same FP32 patch cosine and reductions as PI2W D0-D3 DEV diagnostic",
        "paths": {},
    }
    for name in paths:
        row: dict[str, Any] = describe(arrays[name])
        row["per_task_mean"] = {
            task: float(arrays[name][label_array == task].mean()) for task in np.unique(label_array)
        }
        if name != "D0_no_motion":
            delta = arrays[name] - arrays["D0_no_motion"]
            row["minus_D0"] = describe(delta)
            row["fraction_samples_beating_D0"] = float(np.mean(delta < 0))
        payload["paths"][name] = row
    payload["interpretive_test"] = {
        "all_learned_paths_beat_D0_by_mean": all(
            arrays[name].mean() < arrays["D0_no_motion"].mean() for name in paths[1:]
        ),
        "vision_only_beats_D0_by_mean": bool(
            arrays["D1_vision_only"].mean() < arrays["D0_no_motion"].mean()
        ),
        "fusion_beats_D0_by_mean": bool(
            arrays["D3_fusion"].mean() < arrays["D0_no_motion"].mean()
        ),
    }
    atomic_json(ARTIFACT, payload)
    print(json.dumps({"status": "PASS", "samples": len(samples), "paths": payload["paths"]}, sort_keys=True))


if __name__ == "__main__":
    main()
