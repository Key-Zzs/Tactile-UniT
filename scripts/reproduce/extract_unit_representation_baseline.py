#!/usr/bin/env python3
"""Extract the canonical original-UniT representation benchmark features.

The extractor intentionally mirrors the encoding/VQ portion of the existing
``GR00T_Tokenizer.forward`` without changing that model's output contract.
It is evaluation-only and requires physical GPU3 to be exposed as logical
``cuda:0`` by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "configs/reproduction/baselines/unit_representation_gr1.json"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/reproduction/t4"
MODALITIES = ("vision", "action", "multimodal")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    tokenizer = checkpoint / "tokenizer"
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"Official nested tokenizer directory not found: {tokenizer}")
    names = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    hashes: dict[str, str] = {}
    for name in names:
        path = tokenizer / name
        if not path.exists():
            raise FileNotFoundError(f"Missing nested tokenizer file: {path}")
        hashes[f"tokenizer/{name}"] = sha256_file(path)
    return {
        "variant": "VLA-UniT-3B-fulldata",
        "source": "official released checkpoint nested tokenizer",
        "relative_files": hashes,
    }


def load_spec(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_episode_metadata(task_root: Path) -> list[dict[str, Any]]:
    path = task_root / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Episode metadata not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def deterministic_frames(length: int, goal_horizon: int, count: int) -> list[int]:
    max_start = int(length) - 1 - int(goal_horizon)
    if max_start < 0:
        return []
    values = np.linspace(0, max_start, count, dtype=np.int64).tolist()
    if len(set(values)) == count:
        return [int(value) for value in values]
    # Deterministic fallback for unusually short episodes. The canonical GR1
    # data has enough valid frames; this branch makes the rule explicit.
    candidates = list(range(max_start + 1))
    if not candidates:
        return []
    return [int(candidates[min(i, len(candidates) - 1)]) for i in range(count)]


def build_manifest(dataset_root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
    dataset_spec = spec["dataset"]
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    tasks = dataset_spec["task_directories"]
    for task in tasks:
        metadata = read_episode_metadata(dataset_root / task)
        held_out = metadata[-int(dataset_spec["episodes_per_task"]):]
        if len(held_out) != int(dataset_spec["episodes_per_task"]):
            raise RuntimeError(f"Task {task} has only {len(metadata)} episodes; cannot select held-out set")
        for episode_row in held_out:
            episode = int(episode_row["episode_index"])
            length = int(episode_row["length"])
            frames = deterministic_frames(
                length=length,
                goal_horizon=int(dataset_spec["goal_horizon"]),
                count=int(dataset_spec["intervals_per_episode"]),
            )
            if len(frames) != int(dataset_spec["intervals_per_episode"]):
                skipped.append({"task": task, "episode": episode, "length": length, "reason": "no valid interval"})
                continue
            for frame in frames:
                samples.append({
                    "pair_id": f"{task}::episode-{episode}::frame-{frame}",
                    "task": task,
                    "episode": episode,
                    "frame": frame,
                    "goal_frame": frame + int(dataset_spec["goal_horizon"]),
                    "action_start": frame,
                    "action_end_exclusive": frame + int(dataset_spec["action_horizon"]),
                    "episode_length": length,
                })
    manifest = {
        "benchmark_name": spec["benchmark_name"],
        "benchmark_version": spec["benchmark_version"],
        "sampling": dataset_spec,
        "samples": samples,
        "skipped": skipped,
    }
    digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest, digest


def install_loading_capture() -> list[dict[str, Any]]:
    from transformers import PreTrainedModel

    original = PreTrainedModel.from_pretrained
    original_function = original.__func__
    records: list[dict[str, Any]] = []

    def capture(cls, *args, **kwargs):
        if cls.__name__ != "GR00T_Tokenizer":
            return original_function(cls, *args, **kwargs)
        kwargs = dict(kwargs)
        kwargs["output_loading_info"] = True
        loaded, info = original_function(cls, *args, **kwargs)
        records.append({
            "class": cls.__name__,
            "missing_keys": list(info.get("missing_keys", [])),
            "unexpected_keys": list(info.get("unexpected_keys", [])),
            "mismatched_keys": [list(item) for item in info.get("mismatched_keys", [])],
            "error_msgs": list(info.get("error_msgs", [])),
        })
        return loaded

    PreTrainedModel.from_pretrained = classmethod(capture)
    return records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_codebook_size(model: torch.nn.Module) -> int:
    layers = getattr(model.vq, "layers", None)
    if layers is not None and len(layers):
        return int(getattr(layers[0], "n_e", getattr(layers[0].config, "n_e", 0)))
    return int(model.config.vq_cfg["n_e"])


def encode_batch(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Mirror the existing forward encoding/VQ route for three explicit masks."""

    # prepare_input is the model-owned input normalization/device/dtype route.
    obs_input, goal_input, action_inputs, tokenizer_inputs = model.prepare_input(batch)
    batch_size = int(batch["state"].shape[0])
    vision_query_features, _, _ = model.vision_branch(
        obs_input=obs_input, goal_input=goal_input, batch_size=batch_size
    )
    action_query_features, _ = model.action_branch(
        actions=action_inputs["action"],
        state=action_inputs["state"],
        cat_ids=action_inputs["embodiment_id"],
    )
    outputs: dict[str, list[torch.Tensor]] = {
        "l1": [], "l2": [], "l3": [], "l4": [], "l5": []
    }
    masks = ((1, 0), (0, 1), (1, 1))
    for pv, pa in masks:
        pv_tensor = torch.full((batch_size,), pv, dtype=torch.long, device=vision_query_features.device)
        pa_tensor = torch.full((batch_size,), pa, dtype=torch.long, device=vision_query_features.device)
        unit_tokens = model.fusion(
            visual_tokens=vision_query_features,
            action_tokens=action_query_features,
            pv=pv_tensor,
            pa=pa_tensor,
        )
        vq_input = model.vq_down_resampler(unit_tokens)
        quantized, indices, _ = model.vq(vq_input)
        projected = model.bridge_projector(quantized)
        if indices.ndim == 2:
            indices = indices.unsqueeze(-1)
        outputs["l1"].append(unit_tokens)
        outputs["l2"].append(vq_input)
        outputs["l3"].append(quantized)
        outputs["l4"].append(indices)
        outputs["l5"].append(projected)
    return {
        "vision_query_features": vision_query_features,
        "action_query_features": action_query_features,
        **{key: torch.stack(value, dim=1) for key, value in outputs.items()},
    }


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    detached = tensor.detach().cpu()
    return detached.numpy() if not torch.is_floating_point(detached) else detached.float().numpy()


def make_dataset(task_root: Path, data_config: Any, episode_ids: list[int]) -> Any:
    from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage

    transform = data_config.transform()
    dataset = LeRobotSingleDatasetWithGoalImage(
        dataset_path=task_root,
        modality_configs=data_config.modality_config(),
        embodiment_tag="gr1",
        transforms=transform,
        video_backend="decord",
        episode_ids=episode_ids,
    )
    # Freeze all augmentation and crop randomness. The model is also eval-only.
    dataset.transforms.eval()
    return dataset


def load_samples(
    samples: list[dict[str, Any]], dataset_root: Path, data_config: Any
) -> list[dict[str, Any]]:
    datasets: dict[str, Any] = {}
    episode_ids_by_task: dict[str, list[int]] = {}
    for sample in samples:
        episode_ids_by_task.setdefault(sample["task"], []).append(int(sample["episode"]))
    for task, episode_ids in episode_ids_by_task.items():
        datasets[task] = make_dataset(dataset_root / task, data_config, sorted(set(episode_ids)))

    transformed: list[dict[str, Any]] = []
    for sample in samples:
        dataset = datasets[sample["task"]]
        raw = dataset.get_step_data(int(sample["episode"]), int(sample["frame"]))
        transformed.append(dataset.transforms(raw))
    return transformed


def batches(values: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    from gr00t.model.transforms import collate

    result = []
    for start in range(0, len(values), batch_size):
        result.append(collate(values[start:start + batch_size], None))
    return result


def compare_determinism(first: dict[str, np.ndarray], second: dict[str, np.ndarray], atol: float, rtol: float) -> dict[str, Any]:
    result: dict[str, Any] = {"continuous": {}, "discrete": {}}
    for key in ("l1", "l2", "l3"):
        delta = np.max(np.abs(first[key].astype(np.float64) - second[key].astype(np.float64)))
        ok = bool(np.allclose(first[key], second[key], atol=atol, rtol=rtol))
        result["continuous"][key] = {"max_abs_diff": float(delta), "allclose": ok}
    delta = np.max(np.abs(first["l4"].astype(np.int64) - second["l4"].astype(np.int64)))
    result["discrete"] = {"max_abs_diff": int(delta), "exact_equal": bool(np.array_equal(first["l4"], second["l4"]))}
    result["pass"] = all(item["allclose"] for item in result["continuous"].values()) and result["discrete"]["exact_equal"]
    result["atol"] = float(atol)
    result["rtol"] = float(rtol)
    return result


def extract(
    model: torch.nn.Module,
    transformed: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, np.ndarray]:
    all_values: dict[str, list[np.ndarray]] = {
        "vision_query_features": [], "action_query_features": [],
        "l1": [], "l2": [], "l3": [], "l4": [], "l5": []
    }
    for batch in batches(transformed, batch_size):
        with torch.inference_mode():
            values = encode_batch(model, batch)
        for key, value in values.items():
            all_values[key].append(to_numpy(value))
    return {key: np.concatenate(value, axis=0) for key, value in all_values.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional local smoke limit; omit for canonical extraction")
    parser.add_argument("--determinism-samples", type=int, default=16)
    parser.add_argument("--determinism-atol", type=float, default=1e-4)
    parser.add_argument("--determinism-rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("T4 requires CUDA_VISIBLE_DEVICES=3 so physical GPU3 is logical cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"T4 requires exactly one visible CUDA device, got {torch.cuda.device_count()}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    checkpoint = (args.checkpoint or Path(os.environ.get("UNIT_FULLDATA_CKPT", ""))).resolve()
    dataset_root = (args.dataset_root or Path(os.environ.get("GR1_DATASET_DIR", ""))).resolve()
    if not checkpoint.is_dir() or not dataset_root.is_dir():
        raise RuntimeError("Provide --checkpoint/--dataset-root or source .local/config/reproduction.env")

    spec = load_spec(args.spec)
    manifest, manifest_hash = build_manifest(dataset_root, spec)
    if args.max_samples is not None:
        manifest["samples"] = manifest["samples"][:args.max_samples]
        manifest["smoke_limit"] = int(args.max_samples)
        manifest_hash = hashlib.sha256(canonical_json(manifest)).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "features").mkdir(exist_ok=True)
    (args.output_dir / "metrics").mkdir(exist_ok=True)
    (args.output_dir / "visualization").mkdir(exist_ok=True)
    manifest_path = args.output_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_file_hash = sha256_file(manifest_path)

    seed = int(spec["dataset"]["seed"])
    seed_everything(seed)
    tokenizer_identity = checkpoint_identity(checkpoint)
    tokenizer_path = checkpoint / "tokenizer"

    # Importing the class registers the custom Transformers config/model types.
    from gr00t.experiment.data_config_unit import load_data_config
    from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer

    config_path = ROOT / "gr00t/model/configs/shared_tokenizer/gr00t_tokenizer_mix_unified_eef_dino.json"
    base_config = json.loads(config_path.read_text())
    data_config = load_data_config(
        spec["dataset"]["data_config"],
        eagle_path=base_config["backbone_cfg"]["eagle_path"],
        use_bridge=False,
        ignore_lang_prefix=False,
        enable_imagenet_preprocessing=True,
        tokenizer_only=True,
    )
    transformed = load_samples(manifest["samples"], dataset_root, data_config)
    loading_records = install_loading_capture()
    model = GR00T_Tokenizer.from_pretrained(
        pretrained_model_name_or_path=str(tokenizer_path),
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
    model.use_lpips_loss = False
    model.eval().to("cuda")

    values = extract(model, transformed, args.batch_size)
    np.savez_compressed(args.output_dir / "features" / "unit_representation_features.npz", **values,
                        pair_id=np.asarray([s["pair_id"] for s in manifest["samples"]]),
                        task=np.asarray([s["task"] for s in manifest["samples"]]),
                        episode=np.asarray([s["episode"] for s in manifest["samples"]], dtype=np.int64),
                        frame=np.asarray([s["frame"] for s in manifest["samples"]], dtype=np.int64),
                        modality=np.asarray(MODALITIES),
                        modality_masks=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int64))

    determinant_count = min(args.determinism_samples, len(transformed))
    deterministic_first = extract(model, transformed[:determinant_count], args.batch_size)
    deterministic_second = extract(model, transformed[:determinant_count], args.batch_size)
    deterministic = compare_determinism(
        deterministic_first, deterministic_second, args.determinism_atol, args.determinism_rtol
    )
    finite = all(bool(np.isfinite(value).all()) for value in values.values() if value.dtype.kind == "f")
    indices = values["l4"]
    codebook_size = get_codebook_size(model)
    valid_indices = bool(indices.min() >= 0 and indices.max() < codebook_size)
    config = model.config.to_dict()
    extraction_summary = {
        "status": "PASS" if finite and valid_indices and deterministic["pass"] else "FAIL",
        "physical_gpu": 3,
        "logical_device": str(next(model.parameters()).device),
        "checkpoint": str(checkpoint),
        "tokenizer_identity": tokenizer_identity,
        "manifest_sha256": manifest_file_hash,
        "manifest_canonical_sha256": manifest_hash,
        "sample_count": len(manifest["samples"]),
        "loading_records": loading_records,
        "model_dimensions": {
            "query_num": int(config["query_num"]),
            "hidden_size": int(config["hidden_size"]),
            "vq_embedding_dim": int(config["vq_cfg"]["e_dim"]),
            "vq_stages": int(config["vq_cfg"].get("num_stages", len(getattr(model.vq, "layers", [])))),
            "codes_per_stage": int(config["vq_cfg"]["n_e"]),
            "action_horizon": int(config["action_horizon"]),
            "goal_horizon": int(spec["dataset"]["goal_horizon"]),
        },
        "shapes": {key: list(value.shape) for key, value in values.items()},
        "l4_index_range": [int(indices.min()), int(indices.max())],
        "codebook_size": codebook_size,
        "finite": finite,
        "valid_vq_indices": valid_indices,
        "determinism": deterministic,
        "sample_manifest": str(args.output_dir / "sample_manifest.json"),
    }
    (args.output_dir / "extraction_summary.json").write_text(json.dumps(extraction_summary, indent=2) + "\n")
    print(json.dumps(extraction_summary, indent=2))
    return 0 if extraction_summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
