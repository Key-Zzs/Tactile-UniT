#!/usr/bin/env python3
"""Audit C1 identity/provenance and freeze native VAC baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.continuous_vac_shared_space import pairwise_alignment_metrics  # noqa: E402
from gr00t.tactile_unit.paired_contract import sha256_file  # noqa: E402
from gr00t.tactile_unit.vac_latent_dataset import load_split, validate_cache  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c1_vac_latent_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--paired-manifest", type=Path, default=ROOT / ".local/artifacts/tactile_unit/s3_1/paired_eval_manifest.json")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--integration-cache", type=Path, default=ROOT / ".local/cache/tactile_unit/integration/canonical_vac_teachers.npz")
    parser.add_argument("--integration-baseline", type=Path, default=ROOT / ".local/artifacts/tactile_unit/integration/native_pairwise_baseline.json")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--retrieval-chunk", type=int, default=512)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def subset_metrics(
    left: np.ndarray,
    right: np.ndarray,
    episode: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
    retrieval_chunk: int,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    if len(indices) < 2 or len(np.unique(episode[indices])) < 2:
        return {"count": len(indices), "status": "INSUFFICIENT"}
    return {
        "count": len(indices),
        **pairwise_alignment_metrics(
            np.asarray(left[indices]), np.asarray(right[indices]), np.asarray(episode[indices]),
            bootstrap_samples=bootstrap_samples, seed=seed, retrieval_chunk=retrieval_chunk,
        ),
    }


def pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
    episode: np.ndarray,
    dynamic: np.ndarray,
    contact_transition: np.ndarray,
    canonical_mask: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
    retrieval_chunk: int,
) -> dict[str, Any]:
    masks = {
        "all": np.ones(len(episode), dtype=bool),
        "dynamic": np.asarray(dynamic, dtype=bool),
        "rare_boundary": np.isin(contact_transition, [1, 2]),
        "canonical_960": canonical_mask,
        "free_to_contact": contact_transition == 1,
        "contact_to_free": contact_transition == 2,
    }
    return {
        name: subset_metrics(
            left, right, episode, mask, seed=seed + index,
            bootstrap_samples=bootstrap_samples, retrieval_chunk=retrieval_chunk,
        )
        for index, (name, mask) in enumerate(masks.items())
    }


def human_acceptance(audit: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    counts = audit["counts"]
    lines = [
        "# C1 VAC Latent Dataset Human Acceptance",
        "",
        f"Decision: **{audit['status']}**",
        "",
        "1. Pair manifest: frozen before training; hash-locked array-sharded NPY.",
        f"2. Split counts: train {counts['train']}; validation {counts['validation']}; test {counts['test']}.",
        "3. Checkpoint provenance: Vision / A-R / S1 / S2 / E_c / D_c identities recorded and validated.",
        "4. V/A/C cache ordering: PASS; one shared pair_id/source_index order.",
        f"5. Exact canonical 960: {'PASS' if audit['canonical_960'] else 'FAIL'}.",
        "6. Native pairwise baseline: READ-ONLY; no model parameters trained.",
        f"7. Dynamic test rows: {baseline['distribution']['dynamic']}; rare boundaries: {baseline['distribution']['rare_boundary']}.",
        f"8. Final C1 decision: {audit['status']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    spec = json.loads(args.config.read_text())
    cache_root = args.cache_root or ROOT / spec["runtime"]["cache_root"]
    artifact_root = args.artifact_root or ROOT / spec["runtime"]["artifact_root"]
    accepted = json.loads(args.paired_manifest.read_text())
    canonical_ids = [str(row["pair_id"]) for row in accepted["rows"]]
    audit = validate_cache(
        cache_root,
        expected_counts={name: int(value) for name, value in spec["cached_pairs"].items()},
        canonical_pair_ids_960=canonical_ids,
        verify_hashes=True,
    )
    manifest = json.loads((cache_root / "manifest.json").read_text())
    identity = spec["frozen_identity"]
    required_provenance = {
        "paired_manifest_sha256": identity["paired_manifest_sha256"],
        "s2_transition_manifest_sha256": identity["s2_transition_manifest_sha256"],
        "action_checkpoint_sha256": identity["action_checkpoint_sha256"],
        "action_feature_stats_canonical_sha256": identity["action_feature_stats_canonical_sha256"],
        "s1_teacher_checkpoint_sha256": identity["s1_teacher_checkpoint_sha256"],
        "s2_checkpoint_sha256": identity["s2_checkpoint_sha256"],
        "s2_encoder_parameter_digest": identity["s2_encoder_parameter_digest"],
        "s2_decoder_parameter_digest": identity["s2_decoder_parameter_digest"],
        "old_action_rows_digest": identity["old_action_rows_digest"],
        "vision_preprocessing_identity": identity["vision_preprocessing_identity"],
    }
    if any(manifest["provenance"].get(name) != value for name, value in required_provenance.items()):
        raise RuntimeError("C1 root manifest frozen provenance mismatch")
    audit.update({
        "schema": "tactile3d-unit.vac-c1-audit.v1",
        "manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "manifest_canonical_sha256": manifest["canonical_sha256"],
        "provenance": manifest["provenance"],
        "status": "C1_READY",
    })

    test = load_split(cache_root, "test", verify_hashes=False)
    pair_ids = np.asarray(test.arrays["pair_id"])
    canonical_mask = np.isin(pair_ids, canonical_ids)
    episode = np.asarray(test.arrays["episode_id"])
    dynamic = np.asarray(test.arrays["dynamic"])
    transition = np.asarray(test.arrays["contact_transition"])
    representations = {
        "V": test.arrays["z_v"], "A": test.arrays["z_a"], "C": test.arrays["z_c"]
    }
    row_by_id = {str(value): index for index, value in enumerate(pair_ids)}
    canonical_rows = np.asarray([row_by_id[value] for value in canonical_ids], dtype=np.int64)
    with np.load(args.integration_cache, allow_pickle=False) as integration:
        if not np.array_equal(integration["pair_id"], np.asarray(canonical_ids)):
            raise RuntimeError("accepted integration cache no longer matches canonical manifest order")
        integration_identity = {
            "pair_id": True,
            "z_v_exact": bool(np.array_equal(np.asarray(test.arrays["z_v"][canonical_rows]), integration["z_v"])),
            "z_a_exact": bool(np.array_equal(np.asarray(test.arrays["z_a"][canonical_rows]), integration["z_a"])),
            "z_c_exact": bool(np.array_equal(np.asarray(test.arrays["z_c"][canonical_rows]), integration["z_c"])),
        }
    integration_identity["status"] = "PASS" if all(
        integration_identity[name] for name in ("pair_id", "z_v_exact", "z_a_exact", "z_c_exact")
    ) else "FAIL"
    if integration_identity["status"] != "PASS":
        raise RuntimeError("C1 canonical 960 latent identity differs from accepted integration")
    baseline: dict[str, Any] = {
        "schema": "tactile3d-unit.vac-c1-native-baseline.v1",
        "role": "READ_ONLY_NATIVE_BASELINE",
        "distribution": {
            "all": len(test),
            "dynamic": int(dynamic.sum()),
            "rare_boundary": int(np.isin(transition, [1, 2]).sum()),
            "free_to_contact": int((transition == 1).sum()),
            "contact_to_free": int((transition == 2).sum()),
            "canonical_960": int(canonical_mask.sum()),
        },
        "pairs": {},
        "canonical_960_integration_identity": integration_identity,
        "accepted_canonical_960_baseline": json.loads(args.integration_baseline.read_text()),
    }
    for index, (name, left, right) in enumerate((
        ("V-A", "V", "A"), ("V-C", "V", "C"), ("A-C", "A", "C")
    )):
        baseline["pairs"][name] = pair_metrics(
            representations[left], representations[right], episode, dynamic, transition,
            canonical_mask, seed=int(spec["seed"]) + index * 100,
            bootstrap_samples=args.bootstrap_samples, retrieval_chunk=args.retrieval_chunk,
        )
    atomic_json(artifact_root / "cache_audit.json", audit)
    atomic_json(artifact_root / "native_pairwise_baseline.json", baseline)
    (artifact_root / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance(audit, baseline))
    print(json.dumps({"audit": audit, "baseline": baseline}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
