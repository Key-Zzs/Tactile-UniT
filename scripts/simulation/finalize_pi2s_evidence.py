#!/usr/bin/env python3
"""Derive the bounded S4.3-PI2S diagnosis from frozen evidence.

This script is intentionally CPU-only.  It does not import model runtimes, load
checkpoints, launch a simulator, or authorize additional rollouts.  ``--write``
publishes deterministic derived artifacts; ``--check`` verifies them byte for
byte against the current frozen inputs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PI2S_ROOT = REPOSITORY_ROOT / ".local" / "experiments" / "simulation" / "s4_3_pi2s"


class EvidenceError(RuntimeError):
    """Raised when frozen evidence is missing or internally inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def episode_key(episode: dict[str, Any]) -> tuple[str, str, int]:
    item = episode["tuple"]
    return item["checkpoint_id"], item["reset_identity"], int(item["sampling_seed"])


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        item = episode["tuple"]
        grouped[(item["checkpoint_id"], item["h_condition"])].append(episode)
    summary: dict[str, Any] = {}
    for (checkpoint_id, condition), rows in sorted(grouped.items()):
        pinch_counts = Counter(int(row["max_pinch_count"]) for row in rows)
        summary.setdefault(checkpoint_id, {})[condition] = {
            "episodes": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "failures": sum(not bool(row["success"]) for row in rows),
            "ever_object_contact": sum(bool(row["ever_object_contact"]) for row in rows),
            "ever_lifted": sum(bool(row["ever_lifted"]) for row in rows),
            "ever_native_trigger": sum(bool(row["ever_native_trigger"]) for row in rows),
            "max_pinch_count_distribution": {
                str(key): value for key, value in sorted(pinch_counts.items())
            },
            "native_max_steps": sum(row["termination"] == "native_max_steps" for row in rows),
        }
    return summary


def paired_optional_effects(
    base_episodes: list[dict[str, Any]], optional_episodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    base_by_key = {episode_key(row): row for row in base_episodes}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in optional_episodes:
        item = row["tuple"]
        grouped[(item["checkpoint_id"], item["h_condition"])].append(row)
    results = []
    for (checkpoint_id, condition), rows in sorted(grouped.items()):
        paired = []
        for row in rows:
            key = episode_key(row)
            if key not in base_by_key:
                raise EvidenceError(f"optional tuple has no correct-H base pair: {key}")
            paired.append((base_by_key[key], row))
        wins = sum((not base["success"]) and optional["success"] for base, optional in paired)
        losses = sum(base["success"] and (not optional["success"]) for base, optional in paired)
        results.append(
            {
                "checkpoint_id": checkpoint_id,
                "h_condition": condition,
                "paired_episodes": len(paired),
                "correct_h_successes": sum(bool(base["success"]) for base, _ in paired),
                "intervention_successes": sum(bool(optional["success"]) for _, optional in paired),
                "wins_vs_correct": wins,
                "losses_vs_correct": losses,
                "agreements": len(paired) - wins - losses,
                "success_difference": sum(
                    int(optional["success"]) - int(base["success"]) for base, optional in paired
                ),
            }
        )
    return results


def input_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_rollout_contract(result: dict[str, Any], planned: int, label: str) -> None:
    expected = {
        "status": "PASS",
        "canonical_tuples_planned": planned,
        "canonical_tuples_completed": planned,
        "canonical_tuples_invalid": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "real_robot_used": False,
        "performance_selection_used": False,
        "formal_track_a_scores_replaced": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise EvidenceError(f"{label} contract mismatch: {key}={result.get(key)!r}")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != planned:
        raise EvidenceError(f"{label} episode count is not {planned}")
    keys = [episode_key(row) + (row["tuple"]["h_condition"],) for row in episodes]
    if len(set(keys)) != planned:
        raise EvidenceError(f"{label} contains duplicate canonical tuples")


def gradient_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for checkpoint_id, metrics in artifact["per_checkpoint_summary"].items():
        cosine = metrics["all_common_trainable/cosine_g_main_g_aux"]
        output[checkpoint_id] = {
            "common_trainable_main_aux_cosine_count": cosine["count"],
            "common_trainable_main_aux_cosine_min": cosine["min"],
            "common_trainable_main_aux_cosine_median": cosine["median"],
            "common_trainable_main_aux_cosine_max": cosine["max"],
        }
    return output


def fixed_action_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for checkpoint_id, conditions in artifact["per_checkpoint_condition_summary"].items():
        output[checkpoint_id] = {}
        for condition, metrics in conditions.items():
            output[checkpoint_id][condition] = {
                "mean_chunk_l2_vs_correct": metrics["chunk_l2/all"]["mean"],
                "mean_delta_jerk_proxy": metrics["delta_jerk_proxy"]["mean"],
            }
    return output


def build_outputs(pi2s_root: Path) -> dict[Path, bytes]:
    artifacts = pi2s_root / "artifacts"
    paths = {
        "base": artifacts / "diagnostic_rollout_results.json",
        "optional": artifacts / "diagnostic_rollout_optional_results_v21.json",
        "parity": artifacts / "offline_online_h_parity.json",
        "time": artifacts / "h_physical_time_audit.json",
        "distribution": artifacts / "h_distribution_diagnostics.json",
        "prefix": artifacts / "prefix_contract_audit.json",
        "fixed": artifacts / "fixed_observation_interventions.json",
        "gradient": artifacts / "read_only_gradient_diagnostics.json",
        "historical": artifacts / "historical_failure_stage_analysis.json",
        "recipe": artifacts / "checkpoint_recipe_matrix.json",
        "rng": artifacts / "initialization_rng_audit_v2.json",
    }
    loaded = {key: load_json(path) for key, path in paths.items()}
    require_rollout_contract(loaded["base"], 72, "base rollout")
    require_rollout_contract(loaded["optional"], 48, "optional rollout")
    if loaded["optional"].get("further_rollout_expansion_authorized") is not False:
        raise EvidenceError("optional result does not freeze further expansion as false")
    if loaded["time"].get("status") != "PASS":
        raise EvidenceError("physical time/history audit is not PASS")
    for key in ("prefix", "fixed", "gradient"):
        if loaded[key].get("status") != "PASS":
            raise EvidenceError(f"{key} diagnostic is not PASS")

    parity = loaded["parity"]
    expected_parity_status = "COMPLETE_MISMATCH_EXCEEDS_FROZEN_TOLERANCE"
    if parity.get("status") != expected_parity_status:
        raise EvidenceError(f"unexpected parity status: {parity.get('status')}")
    parity_pairs = parity["metrics"]["pairs"]
    if not parity_pairs["cached_vs_direct"]["bitwise_equal"]:
        raise EvidenceError("cached and direct E_T are no longer bitwise equal")
    if parity_pairs["direct_vs_live"]["allclose_at_protocol_tolerance"]:
        raise EvidenceError("direct/live mismatch unexpectedly disappeared")

    base_episodes = loaded["base"]["episodes"]
    optional_episodes = loaded["optional"]["episodes"]
    base_summary = summarize_episodes(base_episodes)
    optional_summary = summarize_episodes(optional_episodes)
    paired = paired_optional_effects(base_episodes, optional_episodes)
    hva43 = base_summary["B_HVA_43"]["correct"]
    if (
        hva43["episodes"] != 12
        or hva43["successes"] != 0
        or hva43["ever_object_contact"] != 12
        or hva43["ever_lifted"] != 12
        or hva43["ever_native_trigger"] != 0
    ):
        raise EvidenceError("unexpected B_HVA_43 milestone evidence")

    inputs = [input_record(path, key) for key, path in sorted(paths.items())]
    generated_from = {
        "input_manifest": inputs,
        "input_manifest_sha256": hashlib.sha256(canonical_json_bytes(inputs)).hexdigest(),
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "execution": "CPU_ONLY_DERIVATION_NO_MODEL_LOAD_NO_SIMULATOR_NO_GPU",
        },
        "latest_frozen_evidence_time_utc": loaded["optional"]["created_at_utc"],
    }

    diagnosis = {
        "schema": "tactile3d-unit.s4-3-pi2s-diagnosis-and-evidence-strength.v1",
        "status": "COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE",
        "scientific_role": "POST_HOC_DEVELOPMENT_DIAGNOSIS_DOES_NOT_REPLACE_TRACK_A_FORMAL_SCORES",
        "budget": {
            "base_development_rollouts": 72,
            "optional_h_rollouts": 48,
            "total_development_rollouts": 120,
            "maximum_authorized": 120,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "real_robot_runs": 0,
            "further_expansion_authorized": False,
        },
        "rollout_summary": {"base": base_summary, "optional": optional_summary},
        "paired_optional_effects": paired,
        "h_parity": {
            "status": parity["status"],
            "cached_vs_direct_bitwise_equal": True,
            "direct_vs_live_allclose_at_frozen_tolerance": False,
            "direct_vs_live_max_absolute_error": parity_pairs["direct_vs_live"][
                "max_absolute_error"
            ],
            "direct_vs_live_mean_absolute_error": parity_pairs["direct_vs_live"][
                "mean_absolute_error"
            ],
            "frozen_tolerance": parity["protocol_tolerance"],
            "causal_scope": "MEASURED_INTERFACE_NONPARITY_NOT_SHOWN_TO_CAUSE_ROLLOUT_FAILURE",
        },
        "fixed_observation_action_sensitivity": fixed_action_summary(loaded["fixed"]),
        "read_only_gradient_summary": gradient_summary(loaded["gradient"]),
        "failure_localization": {
            "B_HVA_43_correct_h": hva43,
            "coarse_conclusion": (
                "All 12 episodes reached object contact and native lift, but none reached the "
                "native trigger or success. Failure is localized after contact/lift and before "
                "completion of the native pinch-cycle trigger."
            ),
            "exact_first_failure_stage": "N/A_DETAILED_APPROACH_AND_STABLE_GRASP_NOT_DEFINED",
            "why_exact_is_unavailable": (
                "APPROACH and STABLE_GRASP had no preregistered native predicate; historical "
                "3,000-outcome telemetry is episode-aggregate only."
            ),
        },
        "evidence_strength": [
            {
                "diagnosis": "H_HISTORY_OR_PHYSICAL_TIME_ASSEMBLY_ERROR",
                "strength": "NOT_SUPPORTED_BY_FROZEN_AUDIT",
                "evidence": "raw 50 Hz ticks, LEFT_REPEAT_FIRST, reset isolation and t+27 masks PASS",
            },
            {
                "diagnosis": "H_CACHED_OR_DIRECT_ENCODER_MISMATCH",
                "strength": "CONTRADICTED_ON_64_FROZEN_ROWS",
                "evidence": "cached sidecar and direct E_T are float32 bitwise equal",
            },
            {
                "diagnosis": "H_LIVE_SERVICE_NUMERIC_NONPARITY",
                "strength": "DIRECTLY_OBSERVED_ENGINEERING_MISMATCH_CAUSAL_ROLE_UNPROVEN",
                "evidence": "direct/live max absolute error 2.7418136596679688e-6 exceeds frozen tolerance",
            },
            {
                "diagnosis": "H_CONDITIONING_REACHES_POLICY_ACTIONS",
                "strength": "DIRECTLY_SUPPORTED_AT_FIXED_OBSERVATIONS",
                "evidence": "lag5/other/mean/zero H produce nonzero action deltas for H-enabled models",
            },
            {
                "diagnosis": "H_CONDITIONING_IS_THE_UNIQUE_HVA43_ROOT_CAUSE",
                "strength": "NOT_SUPPORTED",
                "evidence": "48 paired development interventions are small-sample and direction-mixed",
            },
            {
                "diagnosis": "HVA43_SIMPLE_LATENT_NORM_COLLAPSE",
                "strength": "NOT_SUPPORTED",
                "evidence": "historical H norms remain finite and similar across HVA seeds42/43/44",
            },
            {
                "diagnosis": "PERSISTENT_MAIN_AUX_GRADIENT_CONFLICT",
                "strength": "MIXED_NOT_UNIQUE",
                "evidence": "fixed-time common-parameter cosines contain both signs and vary by batch/time",
            },
            {
                "diagnosis": "SEED42_AND_43_44_DIFFER_ONLY_BY_NUMERIC_SEED",
                "strength": "CONTRADICTED_BY_PROVENANCE",
                "evidence": "seed42 uses multiple historical entrypoints/commits; 43/44 share Track A snapshot",
            },
            {
                "diagnosis": "EXACT_FIRST_FAILURE_STAGE",
                "strength": "UNOBSERVABLE_FROM_AVAILABLE_DEFINITIONS_AND_HISTORICAL_TELEMETRY",
                "evidence": "only native contact/lift/pinch trigger milestones are defensible",
            },
        ],
        "required_questions": {
            "why_H_before_teacher": (
                "Online H is the common inference interface used by B1/B_HVA/B2 and can fail "
                "independently of a teacher; Track B is a separate matched-representation study "
                "and cannot isolate or replace Track A runtime H."
            ),
            "HVA43_first_failure": (
                "Coarsely after object contact and lift but before native pinch-cycle trigger; "
                "exact APPROACH/STABLE_GRASP first failure is not observable."
            ),
            "seed_only_difference": "No; source entrypoint/commit and unverifiable RNG histories also differ.",
            "same_history_reproducible": (
                "Cached and direct E_T are bitwise reproducible; the live Unix-service path is "
                "finite but exceeds the frozen numeric tolerance."
            ),
            "what_is_proven": (
                "Contracts, provenance differences, action sensitivity, a small live-path numeric "
                "mismatch, and coarse closed-loop milestones are measured. No unique causal root "
                "cause or universal method ranking is proven."
            ),
            "minimum_next_change": (
                "Version and qualify one deterministic H transport/runtime contract first; any "
                "later retraining must use matched source/RNG telemetry and unchanged teachers."
            ),
            "expand_policy_search": "No. Stop at the frozen 120-rollout diagnostic budget.",
        },
        "final_interpretation": (
            "Evidence supports a real H-conditioned action pathway and detects a small live-service "
            "numeric mismatch plus source-cohort nonidentity, but neither explains HVA43 uniquely. "
            "The correct closeout is diagnosis inconclusive, with all negative results retained."
        ),
        **generated_from,
    }

    next_action = {
        "schema": "tactile3d-unit.s4-3-pi2s-next-action-recommendation.v1",
        "status": "RECOMMEND_STOP_AUTOMATIC_POLICY_SEARCH",
        "automatic_training_authorized": False,
        "automatic_rollout_expansion_authorized": False,
        "teacher_replacement_authorized": False,
        "recommended_sequence": [
            {
                "order": 1,
                "action": "VERSION_AND_QUALIFY_H_RUNTIME_TRANSPORT",
                "scope": (
                    "Explain or remove the measured direct-vs-live numeric mismatch under a frozen "
                    "device/kernel/thread contract; append an erratum artifact and never overwrite v1."
                ),
                "matched_controls": [
                    "same raw float32 history bytes",
                    "same accepted E_T checkpoint and architecture",
                    "same device, dtype, thread and kernel settings",
                    "cached, direct and live routes",
                ],
            },
            {
                "order": 2,
                "action": "ADD_PROSPECTIVE_FAILURE_TELEMETRY_BEFORE_ANY_NEW_SCIENCE",
                "scope": (
                    "Predefine approach/stable-grasp predicates and record stepwise TCP, tongs, "
                    "contact, pinch counter, success counter, H and executed action telemetry."
                ),
            },
            {
                "order": 3,
                "action": "ONLY_IF_SEPARATELY_AUTHORIZED_RUN_ONE_MATCHED_SOURCE_COHORT",
                "scope": (
                    "Use one immutable source/config/runtime, persist initialization and PRNG traces, "
                    "and compare B1/B_HVA/B2 without changing the frozen teacher or formal history."
                ),
            },
        ],
        "not_recommended": [
            "replace Track A teacher with Track B teacher to seek a positive result",
            "open additional policy candidates or seeds automatically",
            "reinterpret development rollouts as Track A formal scores",
            "infer stable grasp or exact first failure from videos after seeing outcomes",
        ],
        **generated_from,
    }

    real_robot = {
        "schema": "tactile3d-unit.s4-3-pi2s-real-robot-readiness.v1",
        "status": "NO_REAL_ROBOT_EXECUTION_PERFORMED",
        "S5.0_sensor_interface_data_audit": {
            "decision": "BLOCKED",
            "missing": [
                "RH56DFTP available channels, units, range, frequency and latency contract",
                "causal history/resampling and synchronization validation",
                "robot action adapter and bounded action semantics",
                "independent emergency stop and safety envelope",
            ],
        },
        "S5.1_supervised_pilot": {
            "decision": "BLOCKED",
            "condition": "requires completed S5.0 plus separate human hardware/safety authorization",
        },
        "S5.2_formal_method_ablation": {
            "decision": "NOT_READY",
            "condition": (
                "requires a completed supervised pilot, independent TRAIN/DEV/TEST split and "
                "preregistered success, failure, retry and human-intervention rules"
            ),
        },
        "domain_reporting": (
            "Simulation and real-hardware conclusions must be reported separately; no real-domain "
            "advantage or transfer claim is supported here."
        ),
        **generated_from,
    }

    plots = {
        "schema": "tactile3d-unit.s4-3-pi2s-plots-manifest.v1",
        "status": "NO_NEW_RENDERED_PLOTS_REQUIRED_FOR_DIAGNOSTIC_CLOSEOUT",
        "reason": (
            "The bounded diagnosis is fully represented by machine-readable per-checkpoint tables, "
            "paired tuple counts and frozen upstream metrics. No visual was used to select samples "
            "or change scientific meaning."
        ),
        "table_sources": [
            "diagnosis_and_evidence_strength.json#/rollout_summary",
            "diagnosis_and_evidence_strength.json#/paired_optional_effects",
            "fixed_observation_interventions.json#/per_checkpoint_condition_summary",
            "read_only_gradient_diagnostics.json#/per_checkpoint_summary",
            "historical_failure_stage_analysis.json",
        ],
        "future_plot_policy": (
            "Any paper plot must be regenerated from these frozen artifacts and preserve values, "
            "uncertainty, sample identity and the formal-vs-post-hoc distinction."
        ),
        **generated_from,
    }

    acceptance = """# S4.3-PI2S human acceptance boundary

Status: `COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE`

This local closeout does not authorize a push, pull request, merge to `main`, tag,
release, model retraining, teacher replacement, extra rollout, worktree deletion,
or real-robot execution.

The 120-rollout post-hoc development budget is exhausted (72 base + 48 optional).
All original positive and negative results, including failed v1/v2 recovery attempts,
remain preserved.  A separate human decision is required for any next experiment,
hardware activity, or worktree removal.
"""

    outputs = {
        artifacts / "diagnosis_and_evidence_strength.json": canonical_json_bytes(diagnosis),
        artifacts / "next_action_recommendation.json": canonical_json_bytes(next_action),
        artifacts / "real_robot_readiness.json": canonical_json_bytes(real_robot),
        artifacts / "plots_manifest.json": canonical_json_bytes(plots),
        artifacts / "HUMAN_ACCEPTANCE.md": acceptance.encode("utf-8"),
    }
    paper = REPOSITORY_ROOT / ".local/paper/PAPER_CORE.md"
    paper_before_pi2s = pi2s_root / "integration/paper_core_after_i3.md"
    if not paper.is_file() or not paper_before_pi2s.is_file():
        raise EvidenceError("PAPER_CORE or post-I3 paper snapshot is missing")
    before_text = paper_before_pi2s.read_text(encoding="utf-8").splitlines(keepends=True)
    after_bytes = paper.read_bytes()
    after_text = after_bytes.decode("utf-8").splitlines(keepends=True)
    delta = "".join(
        difflib.unified_diff(
            before_text,
            after_text,
            fromfile="integration/paper_core_after_i3.md",
            tofile=".local/paper/PAPER_CORE.md",
        )
    ).encode("utf-8")
    outputs[pi2s_root / "integration/paper_core_after_pi2s.md"] = after_bytes
    outputs[pi2s_root / "integration/paper_core_delta_pi2s.md"] = delta
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--pi2s-root", type=Path, default=DEFAULT_PI2S_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_outputs(args.pi2s_root.resolve())
    if args.write:
        for path, data in outputs.items():
            atomic_write(path, data)
        status = "WRITE_PASS"
    else:
        mismatches = [
            str(path)
            for path, data in outputs.items()
            if not path.is_file() or path.read_bytes() != data
        ]
        if mismatches:
            raise EvidenceError("derived outputs are missing or stale: " + ", ".join(mismatches))
        status = "CHECK_PASS"
    print(
        json.dumps(
            {
                "status": status,
                "outputs": {
                    str(path): hashlib.sha256(data).hexdigest() for path, data in outputs.items()
                },
                "gpu_used": False,
                "model_loaded": False,
                "simulator_started": False,
                "additional_rollouts": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
