#!/usr/bin/env python3
"""Audit the accepted C2 Contact probe before any C2-R training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c2r_contact_preservation import (  # noqa: E402
    bootstrap_probe_metrics,
    canonical_contact_probe,
    retention,
    sha256_file,
    verify_accepted_c2_checkpoint,
)
from gr00t.tactile_unit.continuous_vac_shared_space import load_checkpoint  # noqa: E402
from gr00t.tactile_unit.vac_latent_dataset import load_split  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c2r_contact_preservation_remediation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def encode_contact(model, split, device: torch.device, batch_size: int) -> np.ndarray:
    result = np.empty((len(split), 8, 32), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            native = torch.from_numpy(
                np.array(split.arrays["z_c"][start:stop], copy=True)
            ).to(device)
            result[start:stop] = model.encode("contact", native).float().cpu().numpy()
    return result


def clean_probe(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    runtime = config["runtime"]
    cache_root = ROOT / runtime["cache_root"]
    checkpoint = ROOT / runtime["accepted_c2_checkpoint"]
    accepted_evaluation_path = ROOT / runtime["accepted_c2_evaluation"]
    artifact = ROOT / runtime["artifact_root"] / "metric_audit.json"
    checkpoint_sha = verify_accepted_c2_checkpoint(checkpoint)
    manifest_sha_before = sha256_file(cache_root / "manifest.json")
    accepted = json.loads(accepted_evaluation_path.read_text())
    device, lock_handle, gpu = resolve_device(
        args.device, allowed_physical=("2", "3")
    )
    try:
        set_seed(int(config["seed"]))
        train = load_split(cache_root, "train", verify_hashes=True)
        test = load_split(cache_root, "test", verify_hashes=True)
        model, _ = load_checkpoint(checkpoint, device)
        model.eval().requires_grad_(False).to(device)
        shared_train = encode_contact(model, train, device, args.batch_size)
        shared_test = encode_contact(model, test, device, args.batch_size)
        probe_results: dict[str, Any] = {}
        implementation_bug = False
        for name, key, classes in (
            ("contact_transition", "contact_transition", 4),
            ("force_trend", "force_trend_class", 3),
        ):
            native = canonical_contact_probe(
                train.arrays["z_c"], test.arrays["z_c"],
                train.arrays[key], test.arrays[key], classes,
                return_prediction=True,
            )
            shared = canonical_contact_probe(
                shared_train, shared_test,
                train.arrays[key], test.arrays[key], classes,
                return_prediction=True,
            )
            value = retention(shared, native)
            expected = accepted["contact"]["probes"][name]
            equality = {
                "native_f1": bool(np.isclose(native["macro_f1"], expected["native"]["macro_f1"], atol=1e-12, rtol=0.0)),
                "shared_f1": bool(np.isclose(shared["macro_f1"], expected["shared"]["macro_f1"], atol=1e-12, rtol=0.0)),
                "retention": bool(np.isclose(value, expected["retention"], atol=1e-12, rtol=0.0)),
                "protocol": native["protocol"] == shared["protocol"],
            }
            implementation_bug = implementation_bug or not all(equality.values())
            probe_results[name] = {
                "native": clean_probe(native),
                "shared": clean_probe(shared),
                "retention": value,
                "accepted_equality": equality,
            }
            if name == "contact_transition":
                bootstrap_samples = args.bootstrap_samples or int(
                    config["metric_audit"]["bootstrap_samples"]
                )
                probe_results[name]["bootstrap"] = bootstrap_probe_metrics(
                    np.asarray(test.arrays[key]),
                    native["_prediction"],
                    shared["_prediction"],
                    native["_majority_prediction"],
                    samples=bootstrap_samples,
                    seed=int(config["seed"]) + 500,
                )

        sensitivity = []
        for seed in config["metric_audit"]["probe_seeds"]:
            order = np.random.default_rng(int(seed)).permutation(len(train))
            native = canonical_contact_probe(
                train.arrays["z_c"], test.arrays["z_c"],
                train.arrays["contact_transition"], test.arrays["contact_transition"], 4,
                train_order=order,
            )
            shared = canonical_contact_probe(
                shared_train, shared_test,
                train.arrays["contact_transition"], test.arrays["contact_transition"], 4,
                train_order=order,
            )
            sensitivity.append({
                "seed": int(seed),
                "native_f1": native["macro_f1"],
                "shared_f1": shared["macro_f1"],
                "retention": retention(shared, native),
            })
        diagnostic = {}
        for name in ("native_f1", "shared_f1", "retention"):
            values = np.asarray([row[name] for row in sensitivity], dtype=np.float64)
            diagnostic[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

        manifest_sha_after = sha256_file(cache_root / "manifest.json")
        if manifest_sha_before != manifest_sha_after:
            implementation_bug = True
        result = {
            "schema": "tactile3d-unit.vac-c2r-metric-audit.v1",
            "decision": (
                "C2R_METRIC_IMPLEMENTATION_INVALID" if implementation_bug
                else "C2R0_METRIC_AUDIT_PASS"
            ),
            "accepted_c2_checkpoint_sha256": checkpoint_sha,
            "accepted_c2_result_preserved": True,
            "canonical_original_r_contact": accepted["contact"]["probes"]["contact_transition"]["retention"],
            "gpu": gpu,
            "rows": {"train": len(train), "test": len(test)},
            "feature_protocol_equality": True,
            "probe_protocol_equality": all(
                row["accepted_equality"]["protocol"] for row in probe_results.values()
            ),
            "probe_results": probe_results,
            "fixed_seed_sensitivity": {"runs": sensitivity, "summary": diagnostic, "role": "DIAGNOSTIC ONLY"},
            "cache_manifest_sha256_before": manifest_sha_before,
            "cache_manifest_sha256_after": manifest_sha_after,
            "implementation_bug": implementation_bug,
            "test_role": "accepted C2 diagnostic audit only; never model selection",
        }
        atomic_json(artifact, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if implementation_bug:
            raise SystemExit(2)
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
