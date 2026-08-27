"""Shared C0 runtime, extraction, and evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.compatibility import (
    deterministic_contact_subset,
    parameter_digest,
)  # noqa: E402
from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    TReXPairedDataset,
    decode_rgb_frame,
    preprocess_trex_rgb,
    sha256_file,
)

DEFAULT_SPEC = ROOT / "configs/tactile_unit/c0_continuous_contact_bridge.json"
DEFAULT_CACHE = ROOT / ".local/cache/tactile_unit/c0"
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/c0"
DEFAULT_EXPERIMENTS = ROOT / ".local/experiments/tactile_unit/c0"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch")
    return actual


def verify_gpu() -> tuple[torch.device, int]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("C0 GPU jobs require CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"2", "3"}:
        raise RuntimeError("C0 permits only physical GPU2 or GPU3")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C0 requires exactly one visible CUDA device")
    return torch.device("cuda:0"), int(visible)


def load_s2_model(path: Path, device: torch.device) -> torch.nn.Module:
    from gr00t.contact_dynamics.models import (
        ContactDynamicsEncoder,
        ContactDynamicsModel,
        LatentTransitionDecoder,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "proposed":
        raise RuntimeError("S2 checkpoint is not the accepted proposed model")
    model = ContactDynamicsModel(ContactDynamicsEncoder(), LatentTransitionDecoder())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def load_s1_teacher(path: Path, device: torch.device) -> torch.nn.Module:
    from gr00t.tactile_teacher.models import PredictiveContactTeacher

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "tactile3d-unit.s1-contact-teacher-checkpoint.v1":
        raise RuntimeError("S1 checkpoint schema mismatch")
    model = PredictiveContactTeacher(
        latent_dim=int(checkpoint["latent_dim"]), channels=int(checkpoint["channels"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def verify_frozen_contact(
    spec: dict[str, Any], s1_checkpoint: Path, s2_checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, str]]:
    identity = spec["frozen_identity"]
    hashes = {
        "s1_checkpoint_sha256": verify_file(
            s1_checkpoint, identity["s1_teacher_checkpoint_sha256"], "S1 checkpoint"
        ),
        "s2_checkpoint_sha256": verify_file(
            s2_checkpoint, identity["s2_checkpoint_sha256"], "S2 checkpoint"
        ),
    }
    s2 = load_s2_model(s2_checkpoint, device)
    encoder_digest = parameter_digest(s2.encoder)
    decoder_digest = parameter_digest(s2.decoder)
    if encoder_digest != identity["s2_encoder_parameter_digest"]:
        raise RuntimeError("S2 checkpoint hash passed but E_c parameter digest failed")
    if decoder_digest != identity["s2_decoder_parameter_digest"]:
        raise RuntimeError("S2 checkpoint hash passed but D_c parameter digest failed")
    return s2, {
        **hashes,
        "s2_encoder_parameter_digest": encoder_digest,
        "s2_decoder_parameter_digest": decoder_digest,
    }


def install_loading_capture() -> list[dict[str, Any]]:
    from transformers import PreTrainedModel

    original = PreTrainedModel.from_pretrained
    original_function = original.__func__
    records: list[dict[str, Any]] = []

    def capture(cls, *args, **kwargs):
        if cls.__name__ != "GR00T_Tokenizer":
            return original_function(cls, *args, **kwargs)
        options = dict(kwargs)
        options["output_loading_info"] = True
        loaded, info = original_function(cls, *args, **options)
        records.append(
            {
                "class": cls.__name__,
                "missing_keys": list(info.get("missing_keys", [])),
                "unexpected_keys": list(info.get("unexpected_keys", [])),
                "mismatched_keys": [list(value) for value in info.get("mismatched_keys", [])],
                "error_msgs": list(info.get("error_msgs", [])),
            }
        )
        return loaded

    PreTrainedModel.from_pretrained = classmethod(capture)
    return records


def load_frozen_vision(
    checkpoint_root: Path, spec: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    expected = spec["frozen_identity"]["original_unit_tokenizer_files_sha256"]
    tokenizer_path = checkpoint_root / "tokenizer"
    actual = {
        name: verify_file(tokenizer_path / name, digest, f"Original UniT {name}")
        for name, digest in expected.items()
    }
    from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer

    loading = install_loading_capture()
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
    errors = [] if len(loading) == 1 else ["load count"]
    if loading:
        errors.extend(
            key
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading[0][key]
        )
    if errors:
        raise RuntimeError(f"Original UniT did not load exactly: {errors}")
    model.use_lpips_loss = False
    model.eval().requires_grad_(False).to(device)
    return model, {
        "original_unit_tokenizer_files_sha256": actual,
        "loading": loading,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def _decode_pair(dataset_root: Path, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_root / record["vision"]["relative_path"]
    current = decode_rgb_frame(path, record["vision"]["current"]["packed_timestamp"])
    future = decode_rgb_frame(path, record["vision"]["future"]["packed_timestamp"])
    return preprocess_trex_rgb(current), preprocess_trex_rgb(future)


def selected_indices(arrays: dict[str, np.ndarray], count: int, seed: int) -> np.ndarray:
    return deterministic_contact_subset(
        arrays["episode_id"],
        arrays["anchor_frame"],
        arrays["dynamic"],
        arrays["contact_transition"],
        count=count,
        seed=seed,
    )


def _manifest_test_indices(manifest: dict[str, Any]) -> np.ndarray:
    rows = manifest["rows"]
    if len(rows) != 960 or len({row["pair_id"] for row in rows}) != 960:
        raise RuntimeError("C0 requires exactly the canonical 960 unique pair IDs")
    if any(row["source"]["split"] != "test" for row in rows):
        raise RuntimeError("canonical evaluation manifest contains non-test rows")
    return np.asarray([row["source"]["row_index"] for row in rows], dtype=np.int64)


@torch.inference_mode()
def extract_split_cache(
    *,
    split: str,
    dataset: TReXPairedDataset,
    indices: np.ndarray,
    codes: np.ndarray | None,
    s2: torch.nn.Module,
    vision: torch.nn.Module,
    dataset_root: Path,
    output: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    n = len(indices)
    z_v = np.empty((n, 8, 32), dtype=np.float32)
    z_v_current = np.empty_like(z_v)
    z_c = np.empty_like(z_v)
    z_c_reversed = np.empty_like(z_v)
    h_current = np.empty((n, 256), dtype=np.float32)
    episode_id = np.empty(n, dtype=np.int64)
    anchor_frame = np.empty(n, dtype=np.int64)
    dynamic = np.empty(n, dtype=np.bool_)
    contact_transition = np.empty(n, dtype=np.int64)
    force_trend_class = np.empty(n, dtype=np.int64)
    pair_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            source_indices = [int(value) for value in indices[start:stop]]
            records = [dataset.record(index) for index in source_indices]
            decoded = list(executor.map(lambda row: _decode_pair(dataset_root, row), records))
            obs = torch.from_numpy(np.stack([row[0] for row in decoded]))[:, None].to(
                vision.device, dtype=vision.dtype
            )
            goal = torch.from_numpy(np.stack([row[1] for row in decoded]))[:, None].to(
                vision.device, dtype=vision.dtype
            )
            transition_values, _, _ = vision.vision_branch(obs, goal, batch_size=len(records))
            current_values, _, _ = vision.vision_branch(obs, obs, batch_size=len(records))
            if start == 0:
                repeated, _, _ = vision.vision_branch(obs, goal, batch_size=len(records))
                repeated_l2 = vision.vq_down_resampler(repeated)
                original_l2 = vision.vq_down_resampler(transition_values)
                if not torch.equal(original_l2, repeated_l2):
                    raise RuntimeError(
                        f"frozen Original UniT Vision is non-deterministic for {split}"
                    )
            z_v[start:stop] = vision.vq_down_resampler(transition_values).float().cpu().numpy()
            z_v_current[start:stop] = vision.vq_down_resampler(current_values).float().cpu().numpy()
            current = np.stack(
                [np.asarray(dataset.arrays["current"][index]) for index in source_indices]
            ).astype(np.float32)
            future = np.stack(
                [np.asarray(dataset.arrays["future"][index]) for index in source_indices]
            ).astype(np.float32)
            h_current[start:stop] = current
            if codes is None:
                z_c[start:stop] = (
                    s2.encoder(
                        torch.from_numpy(current).to(vision.device),
                        torch.from_numpy(future).to(vision.device),
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
            else:
                z_c[start:stop] = np.asarray(codes[source_indices], dtype=np.float32)
            z_c_reversed[start:stop] = (
                s2.encoder(
                    torch.from_numpy(future).to(vision.device),
                    torch.from_numpy(current).to(vision.device),
                )
                .float()
                .cpu()
                .numpy()
            )
            for offset, (index, record) in enumerate(zip(source_indices, records), start=start):
                episode_id[offset] = int(dataset.arrays["episode_id"][index])
                anchor_frame[offset] = int(dataset.arrays["anchor_frame"][index])
                dynamic[offset] = bool(dataset.arrays["dynamic"][index])
                contact_transition[offset] = int(dataset.arrays["contact_transition"][index])
                force_trend_class[offset] = int(dataset.arrays["force_trend_class"][index])
                pair_ids.append(record["pair_id"])
    if not all(
        np.isfinite(value).all() for value in (z_v, z_v_current, z_c, z_c_reversed, h_current)
    ):
        raise RuntimeError(f"non-finite {split} extraction")
    np.savez_compressed(
        output,
        z_v=z_v,
        z_v_current=z_v_current,
        z_c=z_c,
        z_c_reversed=z_c_reversed,
        h_current=h_current,
        episode_id=episode_id,
        anchor_frame=anchor_frame,
        dynamic=dynamic,
        contact_transition=contact_transition,
        force_trend_class=force_trend_class,
        source_index=indices,
        pair_id=np.asarray(pair_ids),
    )
    return {
        "split": split,
        "count": n,
        "path": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "z_v_shape": list(z_v.shape),
        "z_c_shape": list(z_c.shape),
        "unique_pair_ids": len(set(pair_ids)),
        "dynamic": int(dynamic.sum()),
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    required = {
        "z_v",
        "z_v_current",
        "z_c",
        "z_c_reversed",
        "h_current",
        "episode_id",
        "anchor_frame",
        "dynamic",
        "contact_transition",
        "force_trend_class",
        "source_index",
        "pair_id",
    }
    if set(payload.files) != required:
        raise RuntimeError(f"invalid C0 paired cache schema: {path}")
    values = {name: payload[name] for name in payload.files}
    n = len(values["z_v"])
    if values["z_v"].shape != (n, 8, 32) or values["z_c"].shape != (n, 8, 32):
        raise RuntimeError("invalid C0 transition cache geometry")
    if values["z_c_reversed"].shape != (n, 8, 32):
        raise RuntimeError("invalid C0 reversed transition cache geometry")
    if values["z_v_current"].shape != (n, 8, 32) or values["h_current"].shape != (n, 256):
        raise RuntimeError("invalid C0 causal cache geometry")
    return values


def ensure_paired_caches(
    *,
    spec: dict[str, Any],
    dataset_root: Path,
    transition_cache: Path,
    code_cache: Path,
    paired_manifest_path: Path,
    cache_root: Path,
    s2: torch.nn.Module,
    vision: torch.nn.Module,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    manifest = json.loads(paired_manifest_path.read_text())
    counts = spec["preflight_sampling"]
    summaries: dict[str, Any] = {}
    for order, (public, cache_name) in enumerate(
        (("train", "train"), ("validation", "val"), ("test", "test"))
    ):
        dataset = TReXPairedDataset(dataset_root, transition_cache, split=cache_name)
        if public == "test":
            indices = _manifest_test_indices(manifest)
        else:
            indices = selected_indices(
                dataset.arrays, int(counts[f"{public}_pairs"]), int(spec["seed"]) + order
            )
        output = cache_root / f"paired_{public}.npz"
        codes_path = code_cache / f"{cache_name}.npy"
        codes = np.load(codes_path, mmap_mode="r") if codes_path.is_file() else None
        summaries[public] = extract_split_cache(
            split=public,
            dataset=dataset,
            indices=indices,
            codes=codes,
            s2=s2,
            vision=vision,
            dataset_root=dataset_root,
            output=output,
            batch_size=batch_size,
            workers=workers,
        )
    if summaries["test"]["count"] != 960:
        raise RuntimeError("test extraction changed the canonical 960 set")
    return summaries


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def different_episode_permutation(episode_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    episodes = np.asarray(episode_ids)
    for _ in range(10_000):
        permutation = rng.permutation(len(episodes))
        if np.all(episodes[permutation] != episodes):
            return permutation
    order = np.argsort(episodes, kind="stable")
    _, counts = np.unique(episodes[order], return_counts=True)
    if int(counts.max()) * 2 > len(episodes):
        raise RuntimeError("different-episode negatives are impossible")
    rotated = np.roll(order, int(counts.max()))
    permutation = np.empty(len(episodes), dtype=np.int64)
    permutation[order] = rotated
    if not np.all(episodes[permutation] != episodes):
        raise RuntimeError("failed to construct different-episode negatives")
    return permutation


def same_episode_wrong_time_permutation(
    episode_ids: np.ndarray, anchors: np.ndarray, minimum_offset: int = 32
) -> np.ndarray:
    episodes = np.asarray(episode_ids)
    anchors = np.asarray(anchors)
    result = np.full(len(episodes), -1, dtype=np.int64)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        for row in rows:
            valid = rows[np.abs(anchors[rows] - anchors[row]) >= minimum_offset]
            if len(valid):
                result[row] = int(valid[np.argmin(np.abs(anchors[valid] - anchors[row]))])
    return result


def flatten_normalized(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(flatten_normalized(left) * flatten_normalized(right), axis=1)


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        stop = min(start + 250, samples)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def retrieval_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    similarities = flatten_normalized(left) @ flatten_normalized(right).T
    order = np.argsort(-similarities, axis=1)
    ranks = np.argmax(order == np.arange(len(order))[:, None], axis=1) + 1
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "chance_recall_at_1": float(1 / len(ranks)),
        "chance_recall_at_5": float(min(5 / len(ranks), 1.0)),
        "chance_recall_at_10": float(min(10 / len(ranks), 1.0)),
    }


def effective_rank(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    centered = flat - flat.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    probabilities = np.square(singular) / np.maximum(np.sum(np.square(singular)), 1e-12)
    return float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))))


def query_diversity(values: np.ndarray) -> float:
    normalized = np.asarray(values, dtype=np.float64)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=-1, keepdims=True), 1e-12)
    similarity = normalized @ np.swapaxes(normalized, 1, 2)
    mask = ~np.eye(8, dtype=bool)
    return float(1.0 - np.mean(similarity[:, mask]))


def distribution_metrics(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(values, axis=-1)
    return {
        "token_norm_mean": float(norms.mean()),
        "token_norm_std": float(norms.std()),
        "effective_rank": effective_rank(values),
        "query_diversity": query_diversity(values),
        "global_std": float(np.std(values)),
    }
