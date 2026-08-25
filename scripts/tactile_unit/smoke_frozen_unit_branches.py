#!/usr/bin/env python3
"""Frozen Original-UniT vision smoke and non-semantic T-Rex action shape smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    TREX_EMBODIMENT_ID,
    TReXPairedDataset,
    decode_rgb_frame,
    normalize_and_pad_trex_state_action,
    preprocess_trex_rgb,
    sha256_file,
    sha256_json,
)


DEFAULT_MANIFEST = ROOT / ".local/artifacts/tactile_unit/s3_1/paired_eval_manifest.json"
DEFAULT_TRANSITIONS = ROOT / ".local/cache/contact_dynamics/s2_transition_pairs"
DEFAULT_CODES = ROOT / ".local/cache/contact_dynamics/s2_codes/test.npy"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_1/frozen_branch_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--transition-cache", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--contact-codes", type=Path, default=DEFAULT_CODES)
    parser.add_argument(
        "--state-action-stats",
        type=Path,
        default=ROOT / ".local/artifacts/tactile_unit/s3_1/state_action_normalization.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--action-smoke-count", type=int, default=4)
    return parser.parse_args()


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


def decode_pair(dataset_root: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_root / row["vision"]["relative_path"]
    current = decode_rgb_frame(path, row["vision"]["current"]["packed_timestamp"])
    future = decode_rgb_frame(path, row["vision"]["future"]["packed_timestamp"])
    return preprocess_trex_rgb(current), preprocess_trex_rgb(future)


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("S3.1 frozen branch smoke requires physical GPU3 as logical cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S3.1 frozen branch smoke requires exactly one visible CUDA device")
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    if len(rows) != 960:
        raise ValueError("frozen vision smoke must use the canonical 960-pair manifest")

    expected_files = {
        "config.json": "7a651f488c93521e0d507880fc250a475e6a08aa9307aa1349f9d3509844971e",
        "model.safetensors.index.json": "3b6d73d2442ce694287c5cd8b93db1bb232909becf35f08ecabadb614b9a1b86",
        "model-00001-of-00002.safetensors": "32d5c326f6c83d12185b6954d2a52511f66ad18b6fdf814aecc5726dd39c243c",
        "model-00002-of-00002.safetensors": "2f8093a900330e5111b63e44dc1687b3212bec343e5b3832bf2e40f2bf18a768",
    }
    tokenizer_path = args.checkpoint / "tokenizer"
    actual_files = {name: sha256_file(tokenizer_path / name) for name in expected_files}
    if actual_files != expected_files:
        raise ValueError("Original UniT tokenizer identity mismatch")

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
    if len(loading) != 1 or any(
        loading[0][key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Original UniT tokenizer did not load exactly: {loading}")
    model.use_lpips_loss = False
    model.eval().requires_grad_(False).to("cuda")
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    vision_count = 0
    vision_finite = True
    vision_norm_sum = 0.0
    vision_l2_norm_sum = 0.0
    vision_shape = None
    vision_l2_shape = None
    with ThreadPoolExecutor(max_workers=8) as executor, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            pairs = list(executor.map(lambda row: decode_pair(args.dataset_root, row), batch_rows))
            obs = torch.from_numpy(np.stack([pair[0] for pair in pairs]))[:, None].to("cuda")
            goal = torch.from_numpy(np.stack([pair[1] for pair in pairs]))[:, None].to("cuda")
            obs = obs.to(dtype=model.dtype)
            goal = goal.to(dtype=model.dtype)
            values, _, _ = model.vision_branch(obs, goal, batch_size=len(batch_rows))
            l2 = model.vq_down_resampler(values)
            vision_shape = list(values.shape[1:])
            vision_l2_shape = list(l2.shape[1:])
            vision_finite = vision_finite and bool(torch.isfinite(values).all() and torch.isfinite(l2).all())
            vision_norm_sum += float(torch.linalg.vector_norm(values.float(), dim=-1).mean()) * len(batch_rows)
            vision_l2_norm_sum += float(torch.linalg.vector_norm(l2.float(), dim=-1).mean()) * len(batch_rows)
            vision_count += len(batch_rows)

    # The framework maps generic NEW_EMBODIMENT to 31, but the released
    # tokenizer has only 30 category slots. Do not alias GR1 or index an
    # out-of-range category. If a future checkpoint includes slot 31, a
    # shape-only smoke is allowed, but it remains non-semantic until trained.
    max_num_embodiments = int(model.config.action_encoder_cfg["max_num_embodiments"])
    action_forward_attempted = TREX_EMBODIMENT_ID < max_num_embodiments
    action_finite = None
    action_query_shape = None
    action_l2_shape = None
    action_smoke_count = 0
    if action_forward_attempted:
        state_action_stats = json.loads(args.state_action_stats.read_text())
        paired = TReXPairedDataset(
            args.dataset_root,
            args.transition_cache,
            split="test",
            contact_codes=args.contact_codes,
        )
        action_rows = rows[: args.action_smoke_count]
        states, actions = [], []
        for row in action_rows:
            state, action = paired.load_state_action(int(row["source"]["row_index"]))
            transformed = normalize_and_pad_trex_state_action(state, action, state_action_stats)
            states.append(transformed["state"])
            actions.append(transformed["action"])
        with torch.inference_mode():
            state_tensor = torch.from_numpy(np.stack(states))[:, None].to("cuda", dtype=model.dtype)
            action_tensor = torch.from_numpy(np.stack(actions)).to("cuda", dtype=model.dtype)
            embodiment = torch.full(
                (len(action_rows),), TREX_EMBODIMENT_ID, dtype=torch.long, device="cuda"
            )
            action_values, state_values = model.action_branch(action_tensor, state_tensor, embodiment)
            action_l2 = model.vq_down_resampler(action_values)
        action_finite = bool(
            torch.isfinite(action_values).all()
            and torch.isfinite(state_values).all()
            and torch.isfinite(action_l2).all()
        )
        action_query_shape = list(action_values.shape[1:])
        action_l2_shape = list(action_l2.shape[1:])
        action_smoke_count = len(action_rows)
    summary: dict[str, Any] = {
        "schema": "tactile3d-unit.s3-1-frozen-branch-smoke.v1",
        "original_unit_tokenizer_files_sha256": actual_files,
        "loading": loading,
        "model_mode": "eval",
        "trainable_parameter_count": trainable_parameters,
        "optimizer_created": False,
        "vision": {
            "canonical_pair_count": vision_count,
            "query_shape": vision_shape,
            "l2_shape": vision_l2_shape,
            "finite": vision_finite,
            "query_norm_mean": vision_norm_sum / vision_count,
            "l2_norm_mean": vision_l2_norm_sum / vision_count,
            "status": "PASS" if vision_count == 960 and vision_finite else "FAIL",
        },
        "action": {
            "real_pair_smoke_count": action_smoke_count,
            "embodiment_id": TREX_EMBODIMENT_ID,
            "released_tokenizer_max_num_embodiments": max_num_embodiments,
            "released_tokenizer_valid_ids": [0, max_num_embodiments - 1],
            "category_expansion_required": not action_forward_attempted,
            "forward_attempted": action_forward_attempted,
            "query_shape": action_query_shape,
            "l2_shape": action_l2_shape,
            "finite": action_finite,
            "semantic_status": "ACTION_BRANCH_TRAINING_REQUIRED_IN_LATER_STAGE",
            "canonical_semantic_baseline": False,
            "status": (
                "SHAPE_PASS_WITH_WARNING"
                if action_finite
                else "NOT_RUN_REQUIRES_NEW_CATEGORY_PARAMETERS"
            ),
        },
    }
    passed = (
        trainable_parameters == 0
        and summary["vision"]["status"] == "PASS"
        and vision_shape == [8, 1024]
        and vision_l2_shape == [8, 32]
        and (
            (not action_forward_attempted and TREX_EMBODIMENT_ID == 31 and max_num_embodiments == 30)
            or (action_finite and action_query_shape == [8, 1024] and action_l2_shape == [8, 32])
        )
    )
    summary["status"] = "PASS_WITH_ACTION_BRANCH_WARNING" if passed else "FAIL"
    summary["summary_sha256"] = sha256_json(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
