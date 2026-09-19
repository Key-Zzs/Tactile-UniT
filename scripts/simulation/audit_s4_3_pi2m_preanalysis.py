#!/usr/bin/env python3
"""Prove PI2M rollout completeness and frozen provenance before statistics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2m"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2m/evaluation/seed7"
TMP = ROOT / ".local/tmp/s43m7"
PRE_FREEZE = ARTIFACTS / "pre_eval_freeze.json"
RECOVERY_LAUNCH = ARTIFACTS / "b2_recovery_launch.json"
RECOVERY_COMPLETION = ARTIFACTS / "b2_recovery_completion.json"
RECOVERY_ARCHIVE = (
    ARTIFACTS
    / "evaluation_recovery/event_001/interruption_archive_manifest.json"
)
OUTPUT = ARTIFACTS / "preanalysis_completeness_audit.json"
FAILED_EVENT = ARTIFACTS / "preanalysis_completeness_audit_event_001.json"
MODELS = ("B1", "B_HVA", "B2")
EXPECTED_CONTROL_RAW = {
    "B1": "8e6ac39c60d6a84e0df953f6c4076ebd55e0680b312b486a0814699e346680cf",
    "B_HVA": "078d8abb6fd933a9b4a42053212ec808098c6111de2e309e06599bc6bf69a18e",
}
EXPECTED_SIDECARS = {
    "B2_contact": (
        ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz",
        "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e",
    ),
    "B_HVA_corrected_VA": (
        ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz",
        "7d51a23672273ec3ea46f947080c0c3ea66f947080c0c3ea66db55332522b199caab00ad4fc2126",
    ),
    "historical_BVA_0p32s": (
        ROOT / ".local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz",
        "c596b2f56880a969148f7cf06268ecfa9ad23bac01014be6c73ad20afd0d0612",
    ),
}
EPISODE_PATTERN = re.compile(r"^episode_(\d+)_(success|failure)$")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def checkpoint_tree_hash(path: Path) -> str:
    rows = []
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": file.relative_to(path).as_posix(),
                "bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def checkpoint_path(symbolic: str) -> Path:
    return Path(symbolic.replace("$REPO_ROOT", str(ROOT)))


def cache_results(model: str) -> tuple[dict[int, bool], list[str]]:
    output = CACHE / model.lower()
    results: dict[int, bool] = {}
    unexpected: list[str] = []
    for item in sorted(output.iterdir()):
        if not item.is_dir():
            continue
        match = EPISODE_PATTERN.fullmatch(item.name)
        if match is None:
            unexpected.append(item.name)
            continue
        index = int(match.group(1))
        if index in results:
            raise RuntimeError(f"duplicate cache episode index for {model}: {index}")
        results[index] = match.group(2) == "success"
    return results, unexpected


def diagnostics_rows(model: str) -> list[dict[str, Any]]:
    path = TMP / f"{model.lower()}_inference.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    if OUTPUT.exists():
        previous = read_json(OUTPUT)
        if previous.get("status") != "FAIL" or FAILED_EVENT.exists():
            raise SystemExit(f"refusing to overwrite {OUTPUT}")
        OUTPUT.rename(FAILED_EVENT)
    for forbidden in (
        ARTIFACTS / "rollout_completeness.json",
        ARTIFACTS / "paired_statistics.json",
        ARTIFACTS / "claim_freeze.json",
    ):
        if forbidden.exists():
            raise SystemExit(f"statistics/completeness already exists before preanalysis audit: {forbidden}")

    pre = read_json(PRE_FREEZE)
    recovery_launch = read_json(RECOVERY_LAUNCH)
    recovery_completion = read_json(RECOVERY_COMPLETION)
    recovery_archive = read_json(RECOVERY_ARCHIVE)
    job = read_json(ARTIFACTS / "evaluation_job_status.json")
    raw_paths = {model: ARTIFACTS / f"{model.lower()}_raw_rollouts.json" for model in MODELS}
    raw = {model: read_json(path) for model, path in raw_paths.items()}
    raw_hashes = {model: sha256_file(path) for model, path in raw_paths.items()}
    identities = {
        model: [row["reset_identity"] for row in payload["episode_results"]]
        for model, payload in raw.items()
    }
    source_hashes = {
        symbolic: sha256_file(checkpoint_path(symbolic))
        for symbolic in pre["sources_sha256"]
    }
    checkpoint_hashes = {
        model: checkpoint_tree_hash(checkpoint_path(pre["checkpoint_paths"][model]))
        for model in MODELS
    }
    sidecar_hashes = {
        name: sha256_file(path) for name, (path, _) in EXPECTED_SIDECARS.items()
    }

    cache_by_model: dict[str, dict[int, bool]] = {}
    cache_unexpected: dict[str, list[str]] = {}
    chunks_by_model: dict[str, list[dict[str, Any]]] = {}
    errors_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        cache_by_model[model], cache_unexpected[model] = cache_results(model)
        diagnostics = diagnostics_rows(model)
        chunks_by_model[model] = [
            row for row in diagnostics if row.get("type") == "action_chunk"
        ]
        errors_by_model[model] = [
            row for row in diagnostics if row.get("type") == "server_client_error"
        ]

    archived_cache = RECOVERY_ARCHIVE.parent / "interrupted/cache/b2"
    old_prefix, old_unexpected = {}, []
    for item in sorted(archived_cache.iterdir()):
        if not item.is_dir() or item.name.endswith("_temp"):
            continue
        match = EPISODE_PATTERN.fullmatch(item.name)
        if match is None:
            old_unexpected.append(item.name)
            continue
        old_prefix[int(match.group(1))] = match.group(2) == "success"
    prefix_indices = sorted(old_prefix)
    prefix_disagreements = [
        index
        for index in prefix_indices
        if old_prefix[index] != cache_by_model["B2"].get(index)
    ]
    prefix_agreements = len(prefix_indices) - len(prefix_disagreements)

    per_model_gates: dict[str, dict[str, bool]] = {}
    for model in MODELS:
        payload = raw[model]
        episodes = payload["episode_results"]
        chunks = chunks_by_model[model]
        per_model_gates[model] = {
            "raw_PASS_200_seed7": payload.get("status") == "PASS"
            and payload.get("episodes") == 200
            and payload.get("evaluator_seed") == 7
            and len(episodes) == 200,
            "indices_exact_0_199": [row["episode_index"] for row in episodes]
            == list(range(200)),
            "cache_exact_0_199": sorted(cache_by_model[model]) == list(range(200))
            and not cache_unexpected[model],
            "cache_outcomes_equal_raw": all(
                cache_by_model[model][index] is bool(episodes[index]["success"])
                for index in range(200)
            ),
            "all_raw_gates_PASS": all(value == "PASS" for value in payload["gates"].values()),
            "runtime_contact_state": payload.get("mode") == "CONTACT_STATE_TOKENS"
            and payload.get("contact_state_sent_to_policy") is True,
            "no_raw_server_client_errors": not payload.get("server_client_errors")
            and all(not row.get("server_client_errors") for row in episodes),
            "telemetry_present_finite": all(
                row.get("contact_state_diagnostics", {}).get("finite") is True
                and row.get("executed_action_stats", {}).get("finite") is True
                and row.get("tactile_diagnostics", {}).get("samples") == row.get("steps") + 1
                for row in episodes
            ),
            "diagnostic_chunks_present": bool(chunks),
            "diagnostic_chunks_finite_30x22": all(
                row.get("finite") is True and row.get("shape") == [30, 22]
                for row in chunks
            ),
            "diagnostic_identity_matches_raw": all(
                row.get("reset_identity")
                == identities[model][int(row["episode_index"])]
                for row in chunks
            ),
            "contact_sent_no_training_targets": all(
                row.get("contact_state_sent") is True
                and row.get("training_only_fields_sent") == []
                for row in chunks
            ),
            "no_diagnostic_server_client_errors": not errors_by_model[model],
        }

    repo_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    repo_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    repo_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip()
    dexjoco_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/dexjoco", text=True
    ).strip()

    global_gates = {
        "three_models_exact": set(raw) == set(MODELS),
        "exactly_600_complete": all(
            len(payload["episode_results"]) == 200 for payload in raw.values()
        ),
        "triple_aligned_200_unique_resets": identities["B1"]
        == identities["B_HVA"]
        == identities["B2"]
        and len(identities["B1"]) == len(set(identities["B1"])) == 200,
        "B1_B_HVA_raw_unchanged": all(
            raw_hashes[model] == expected
            for model, expected in EXPECTED_CONTROL_RAW.items()
        ),
        "B2_raw_matches_recovery_completion": raw_hashes["B2"]
        == recovery_completion.get("B2_raw_sha256"),
        "checkpoint_hashes_exact": checkpoint_hashes == pre["checkpoint_tree_sha256"],
        "frozen_sources_exact": all(
            source_hashes[symbolic] == expected
            for symbolic, expected in pre["sources_sha256"].items()
        ),
        "sidecar_hashes_exact": all(
            sidecar_hashes[name] == expected
            for name, (_, expected) in EXPECTED_SIDECARS.items()
        ),
        "dexjoco_revision_exact": dexjoco_head
        == "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
        "branch_exact": repo_branch == "develop/sim-benchmark",
        "worktree_clean": not repo_status,
        "job_DONE_exit0_600": job.get("state") == "DONE"
        and job.get("exit_code") == 0
        and job.get("total_rollouts_complete") == 600,
        "recovery_archive_PASS_preserved": recovery_archive.get("status") == "PASS"
        and recovery_archive.get("B1_B_HVA_modified") is False,
        "recovery_clean_B2_only": recovery_completion.get("status") == "PASS"
        and recovery_completion.get("models_relaunched") == ["B2"]
        and recovery_completion.get("models_not_relaunched") == ["B1", "B_HVA"]
        and recovery_completion.get("episodes") == 200,
        "recovery_decision_frozen_before_completion": recovery_launch.get(
            "prefix_replay_comparison_role"
        )
        == "integrity diagnostic only; not a selection or acceptance gate"
        and datetime.fromisoformat(recovery_launch["launched_at"])
        < datetime.fromisoformat(recovery_completion["completed_at"]),
        "interrupted_prefix_exactly_31_preserved": prefix_indices == list(range(31))
        and not old_unexpected,
        "interrupted_prefix_nondeterminism_measured_not_hidden": prefix_agreements == 18
        and prefix_disagreements
        == [2, 3, 4, 8, 9, 10, 19, 22, 24, 25, 26, 27, 29],
        "no_splicing_final_B2_is_one_clean_200_run": recovery_launch.get(
            "supervision"
        )
        == "B2_ONLY_CLEAN_REPLAY"
        and recovery_completion.get("recovery_method") == "B2_ONLY_CLEAN_REPLAY",
        "no_statistics_before_completeness": True,
        "PI2B_not_started": recovery_completion.get("PI2B_started") is False,
    }
    all_gates = list(global_gates.values()) + [
        value for gates in per_model_gates.values() for value in gates.values()
    ]
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2m-preanalysis-completeness-audit.v1",
        "status": "PASS" if all(all_gates) else "FAIL",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "branch": repo_branch,
            "head": repo_head,
            "tracked_worktree_clean": not repo_status,
            "dexjoco_head": dexjoco_head,
        },
        "counts": {model: len(raw[model]["episode_results"]) for model in MODELS},
        "total": sum(len(raw[model]["episode_results"]) for model in MODELS),
        "raw_sha256": raw_hashes,
        "checkpoint_tree_sha256": checkpoint_hashes,
        "source_sha256": source_hashes,
        "sidecar_sha256": sidecar_hashes,
        "reset_identity_sequence_canonical_sha256": canonical_sha256(identities["B1"]),
        "action_chunk_counts": {model: len(chunks_by_model[model]) for model in MODELS},
        "termination_counts": {
            model: dict(Counter(row["termination"] for row in raw[model]["episode_results"]))
            for model in MODELS
        },
        "recovery_provenance": {
            "method": "B2_ONLY_CLEAN_REPLAY",
            "interrupted_complete_prefix": len(prefix_indices),
            "prefix_outcome_agreements": prefix_agreements,
            "prefix_outcome_disagreements": len(prefix_disagreements),
            "prefix_outcome_disagreement_indices": prefix_disagreements,
            "reset_identity_parity": recovery_completion.get(
                "reset_identity_parity_B1_B_HVA_B2"
            ),
            "interpretation": "same reset identities but official asynchronous evaluator is not bit/outcome deterministic across retries; the canonical B2 artifact is one complete clean replay selected by failure policy before completion, never a splice or best-of-run choice",
        },
        "global_gates": {
            name: "PASS" if value else "FAIL" for name, value in global_gates.items()
        },
        "per_model_gates": {
            model: {name: "PASS" if value else "FAIL" for name, value in gates.items()}
            for model, gates in per_model_gates.items()
        },
        "statistics_performed": False,
        "PI2B_started": False,
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "counts": payload["counts"], "total": payload["total"], "recovery_provenance": payload["recovery_provenance"]}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("PI2M_PREANALYSIS_COMPLETENESS_FAIL")


if __name__ == "__main__":
    main()
