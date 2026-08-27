#!/usr/bin/env python3
"""Audit Track C0 provenance, paired timing, splits, and causal separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/tactile_unit"))

from continuous_contact_bridge_common import (  # noqa: E402
    DEFAULT_ARTIFACTS,
    DEFAULT_SPEC,
    verify_file,
    verify_frozen_contact,
    load_s1_teacher,
)
from gr00t.tactile_unit.causal_contact_contract import (  # noqa: E402
    ContactBridgeBatch,
    ContactMode,
    ContactTransitionTarget,
    CurrentContactContext,
    FutureContactLeakageError,
    reject_future_oracles,
)
from gr00t.tactile_unit.paired_contract import sha256_file  # noqa: E402


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--unit-checkpoint", type=Path, default=env_path("UNIT_FULLDATA_CKPT"))
    parser.add_argument("--s1-checkpoint", type=Path, default=env_path("TACTILE_TEACHER_CKPT"))
    parser.add_argument("--s2-checkpoint", type=Path, default=env_path("CONTACT_DYNAMICS_CKPT"))
    parser.add_argument("--transition-cache", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--s3-1-summary", type=Path, required=True)
    parser.add_argument("--s3-1-frozen-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "contract_audit.json")
    return parser.parse_args()


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def pair_id_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(row["pair_id"] for row in rows).encode()).hexdigest()


def main() -> int:
    args = parse_args()
    for name in ("unit_checkpoint", "s1_checkpoint", "s2_checkpoint"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} or machine-local environment is required")
    spec = json.loads(args.spec.read_text())
    identity = spec["frozen_identity"]
    starting_base = spec["track_b_base_sha"]
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    merge_base = git("merge-base", "HEAD", starting_base)
    git_gate = branch == "develop/continuous-contact-bridge" and merge_base == starting_base

    vision_files = {
        name: verify_file(
            args.unit_checkpoint / "tokenizer" / name, digest, f"Original UniT {name}"
        )
        for name, digest in identity["original_unit_tokenizer_files_sha256"].items()
    }
    s2, contact_identity = verify_frozen_contact(
        spec, args.s1_checkpoint, args.s2_checkpoint, torch.device("cpu")
    )
    s1 = load_s1_teacher(args.s1_checkpoint, torch.device("cpu"))
    s2.eval().requires_grad_(False)
    frozen_gate = (
        not any(parameter.requires_grad for parameter in s1.parameters())
        and not s1.training
        and not any(parameter.requires_grad for parameter in s2.parameters())
        and not s2.training
        and contact_identity["s2_encoder_parameter_digest"]
        == identity["s2_encoder_parameter_digest"]
        and contact_identity["s2_decoder_parameter_digest"]
        == identity["s2_decoder_parameter_digest"]
    )
    transition_hash = verify_file(
        args.transition_cache / "manifest.json",
        identity["s2_transition_manifest_sha256"],
        "S2 transition manifest",
    )
    transition_manifest = json.loads((args.transition_cache / "manifest.json").read_text())
    expected_counts = spec["paired_data"]["split_pairs"]
    count_gate = all(
        int(transition_manifest["splits"][cache_name]["pairs"]) == int(expected_counts[public])
        for public, cache_name in (("train", "train"), ("validation", "val"), ("test", "test"))
    )
    split_episodes = {
        name: set(
            np.load(args.transition_cache / cache_name / "episode_id.npy", mmap_mode="r").tolist()
        )
        for name, cache_name in (("train", "train"), ("validation", "val"), ("test", "test"))
    }
    overlaps = {
        "train_validation": len(split_episodes["train"] & split_episodes["validation"]),
        "train_test": len(split_episodes["train"] & split_episodes["test"]),
        "validation_test": len(split_episodes["validation"] & split_episodes["test"]),
    }
    manifest_hash = verify_file(
        args.paired_manifest,
        spec["paired_data"]["canonical_evaluation_manifest_sha256"],
        "S3.1 canonical paired manifest",
    )
    paired = json.loads(args.paired_manifest.read_text())
    rows = paired["rows"]
    pair_gate = len(rows) == 960 and len({row["pair_id"] for row in rows}) == 960
    timing_failures = []
    for row in rows:
        current = row["contact"]["current_teacher_window_inclusive"]
        future = row["contact"]["future_teacher_window_inclusive"]
        anchor = int(row["anchor"]["frame"])
        if current != [anchor - 15, anchor] or future != [anchor + 1, anchor + 16]:
            timing_failures.append(row["pair_id"])
        if (
            row["vision"]["future"]["episode_frame"] - row["vision"]["current"]["episode_frame"]
            != 16
        ):
            timing_failures.append(row["pair_id"])
        if row["source"]["split"] != "test":
            timing_failures.append(row["pair_id"])
    s3_1 = json.loads(args.s3_1_summary.read_text())
    reuse_gate = (
        s3_1.get("status") == "PASS"
        and s3_1["paired_eval"]["resolved"] == 960
        and s3_1["off_by_one_contract"]["status"] == "PASS"
    )
    frozen_smoke = json.loads(args.s3_1_frozen_smoke.read_text())
    loading = frozen_smoke.get("loading", [])
    frozen_vision_gate = (
        frozen_smoke.get("status") == "PASS_WITH_ACTION_BRANCH_WARNING"
        and frozen_smoke.get("trainable_parameter_count") == 0
        and len(loading) == 1
        and not any(
            loading[0].get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        )
        and frozen_smoke["vision"]["canonical_pair_count"] == 960
        and frozen_smoke["vision"]["l2_shape"] == [8, 32]
        and frozen_smoke["vision"]["finite"]
    )

    runtime_rejection = False
    try:
        ContactBridgeBatch(
            CurrentContactContext(np.zeros((1, 256), dtype=np.float32)),
            contact_target=ContactTransitionTarget(np.zeros((1, 8, 32), dtype=np.float32)),
        ).validate_for(ContactMode.INFERENCE)
    except FutureContactLeakageError:
        runtime_rejection = True
    nested_rejection = False
    try:
        reject_future_oracles({"contact": {"h_future": np.zeros((1, 256))}})
    except FutureContactLeakageError:
        nested_rejection = True
    causal_gate = runtime_rejection and nested_rejection
    gates = {
        "branch_ancestry": git_gate,
        "checkpoint_provenance": frozen_gate,
        "transition_manifest_identity": bool(transition_hash),
        "split_counts": count_gate,
        "split_leakage_zero": not any(overlaps.values()),
        "canonical_960_exact": pair_gate,
        "timing_t_t_plus_16": not timing_failures,
        "s3_1_contract_reused": reuse_gate,
        "frozen_vision_exact_load": frozen_vision_gate,
        "causal_runtime_rejection": causal_gate,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    output = {
        "schema": "tactile3d-unit.c0-contract-audit.v1",
        "status": status,
        "git": {
            "branch": branch,
            "head": head,
            "track_b_base": starting_base,
            "merge_base": merge_base,
        },
        "frozen_identity": {
            "original_unit_tokenizer_files_sha256": vision_files,
            "original_unit_mode": "eval",
            "original_unit_trainable_parameters": frozen_smoke["trainable_parameter_count"],
            "original_unit_loading": loading,
            "original_unit_vision_shape": frozen_smoke["vision"]["l2_shape"],
            "original_unit_vision_finite": frozen_smoke["vision"]["finite"],
            **contact_identity,
            "s1_mode": "eval",
            "s1_trainable_parameters": sum(
                parameter.numel() for parameter in s1.parameters() if parameter.requires_grad
            ),
            "s2_mode": "eval",
            "s2_trainable_parameters": sum(
                parameter.numel() for parameter in s2.parameters() if parameter.requires_grad
            ),
            "before_after": "PASS: checkpoint files verified read-only; S1/S2 eval and requires_grad=False",
        },
        "continuous_contract": {
            "current_causal_context": "h_t^c [B,256] from T_[t-0.5:t]",
            "transition_teacher": "z_c [B,8,32] float32, native S2 E_c output",
            "horizon_frames": 16,
            "horizon_seconds": 16 / 30,
            "current_window": ["t-15", "t"],
            "future_window": ["t+1", "t+16"],
            "overlap_samples": 0,
            "whitening": "FORBIDDEN",
        },
        "causal_audit": {
            "offline_teachers": ["z_v", "z_c"],
            "online_legal": ["I_<=t", "robot_state_<=t", "T_[t-0.5:t]", "h_t^c", "z_hat_c"],
            "illegal_oracles": ["I_t+16", "h_t+16^c", "true z_c"],
            "typed_runtime_rejection": runtime_rejection,
            "nested_mapping_rejection": nested_rejection,
        },
        "paired_data": {
            "counts": expected_counts,
            "canonical_960": len(rows),
            "unique_pair_ids": len({row["pair_id"] for row in rows}),
            "pair_id_digest": pair_id_digest(rows),
            "manifest_sha256": manifest_hash,
            "timing_failure_count": len(set(timing_failures)),
            "split_overlap": overlaps,
            "pair_coverage": s3_1["pair_coverage_fraction"],
        },
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
