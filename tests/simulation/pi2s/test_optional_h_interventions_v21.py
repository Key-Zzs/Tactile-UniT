from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.simulation import run_pi2s_optional_h_interventions_v21 as recovery


def test_audit_only_correction_is_exact_and_does_not_expand_budget() -> None:
    _protocol, _protocol_sha, _v2_decision, decision = recovery._load_contract()

    assert decision["root_cause"]["classification"] == "AUDIT_HASH_DOMAIN_MISMATCH_CONFIRMED"
    assert decision["root_cause"]["scientific_runtime_changed"] is False
    assert decision["root_cause"]["post_result_tolerance_relaxation"] is False
    assert decision["recovery_budget"] == {
        "valid_v1_protocol_tuples_reused": 6,
        "completed_v2_episodes_reaudited_without_rerun": 3,
        "v21_unexecuted_protocol_tuples_to_run": 39,
        "final_unique_optional_protocol_tuples": 48,
        "final_total_s5_protocol_tuples_including_base": 120,
        "original_v1_invalid_attempts_preserved": 3,
        "original_v2_false_positive_audit_failures_preserved": 3,
        "further_expansion_authorized": False,
    }
    assert decision["guardrails"]["completed_v2_episodes_may_be_rerun"] is False


def test_v21_partition_is_six_plus_three_plus_thirty_nine() -> None:
    protocol, _protocol_sha, _v2_decision, decision = recovery._load_contract()
    pending = recovery.build_recovery_tuples(protocol)
    completed = recovery._completed_v2_keys(decision)
    full_v2 = recovery.base.read_json(recovery.V2_MANIFEST)["tuples"]

    assert len(pending) == 39
    assert len(completed) == 3
    assert {row["tuple_key"] for row in pending}.isdisjoint(completed)
    assert {row["tuple_key"] for row in pending} | completed == {
        row["tuple_key"] for row in full_v2
    }
    assert {
        identifier: sum(row["checkpoint_id"] == identifier for row in pending)
        for identifier in recovery.FOCUS_IDS
    } == {"B_HVA_42": 13, "B_HVA_43": 13, "B1_43": 13}


def test_v2_positive_and_false_positive_failure_bytes_are_immutable() -> None:
    _protocol, _protocol_sha, v2_decision, decision = recovery._load_contract()

    recovery._verify_v2_evidence(v2_decision, decision)

    assert not recovery.v2.RESULTS.exists()
    assert len(decision["completed_v2_episodes_for_reaudit"]) == 3


def test_real_completed_v2_episodes_pass_domain_aware_reaudit() -> None:
    _protocol, _protocol_sha, _v2_decision, decision = recovery._load_contract()
    v2_index = {
        row["tuple_key"]: row for row in recovery.base.read_json(recovery.V2_MANIFEST)["tuples"]
    }
    observed_checks = 0
    cross_domain_mismatches = 0
    for row in decision["completed_v2_episodes_for_reaudit"]:
        artifact = recovery._resolve_pi2s_symbolic(row["tuple_artifact"])
        payload = recovery.base.read_json(artifact)
        events = {int(value["control_step"]): value for value in payload["observation_events"]}
        post = {
            int(value["post_step_correct_h"]["control_tick"]): value["post_step_correct_h"]
            for value in payload["step_telemetry"]
        }
        for tick in set(events) & set(post):
            observed_checks += 1
            cross_domain_mismatches += int(events[tick]["correct_h_sha256"] != post[tick]["sha256"])
            assert events[tick]["correct_h_l2"] == post[tick]["l2"]
        audit = recovery.build_corrected_audit(artifact, v2_index[row["v2_tuple_key"]])
        assert audit["status"] == "PASS"
        assert all(value == "PASS" for value in audit["gates"].values())
        assert audit["original_v2_false_positive_failure_preserved"] is True

    assert observed_checks == 388
    assert cross_domain_mismatches == 388


def test_corrected_audit_rejects_a_changed_raw_lag_source(tmp_path: Path) -> None:
    _protocol, _protocol_sha, _v2_decision, decision = recovery._load_contract()
    row = decision["completed_v2_episodes_for_reaudit"][0]
    source = recovery._resolve_pi2s_symbolic(row["tuple_artifact"])
    payload = recovery.base.read_json(source)
    expected = payload["tuple"]
    payload["observation_events"][1]["lag_source_correct_h_sha256"] = "0" * 64
    artifact = tmp_path / "tuple_result.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    audit = recovery.build_corrected_audit(artifact, expected)

    assert audit["status"] == "FAIL"
    assert audit["gates"]["all_observation_events_conditioned_exactly"] == "FAIL"


def test_progress_counts_thirteen_tuple_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(recovery.base, "MANIFEST", manifest_path)
    assignments = [
        {"tuple_key": f"b_hva_42__r11_i00__p{index}__hsame_episode_lag5__v2"} for index in range(13)
    ] + [{"tuple_key": "b_hva_43__r11_i00__p4318__hsame_episode_lag5__v2"}]

    payload = recovery.progress_payload(
        {"source_commit": "a" * 40, "protocol_sha256": "b" * 64},
        assignments,
        started=time.time() - 1.0,
        status="RUNNING",
    )

    assert payload["completed_blocks"] == 1
    assert payload["completed_recovery_tuples"] == 14
    assert payload["completed_optional_protocol_tuples"] == 23
    assert payload["planned_optional_protocol_tuples"] == 48


def test_v21_scientific_audit_failure_never_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "tuple_result.json"
    artifact.write_text("{}", encoding="utf-8")
    recovery._audit_failure_path(artifact).write_text("{}", encoding="utf-8")

    def fail_attempt(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RuntimeError("evaluator exited")

    monkeypatch.setattr(recovery, "BASE_RUN_TUPLE_ATTEMPT", fail_attempt)
    monkeypatch.setattr(
        recovery.base,
        "attempt_paths",
        lambda _row, _attempt: SimpleNamespace(artifact=artifact),
    )

    with pytest.raises(recovery.base.ContractError, match="without retry"):
        recovery.run_tuple_attempt(
            {"tuple_key": "key__v2"},
            1,
            9000,
            tmp_path / "contact.sock",
            1,
        )


def test_v21_source_has_no_training_expansion_or_destructive_entrypoint() -> None:
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    forbidden = (
        "optimizer.step(",
        "train_state.save(",
        "git push",
        "shutil.rmtree(",
        "MAX_INFRASTRUCTURE_ATTEMPTS =",
    )

    assert all(token not in source for token in forbidden)
