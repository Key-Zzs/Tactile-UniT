from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.simulation import run_pi2s_optional_h_interventions_v2 as recovery


def test_recovery_partition_is_exact_and_never_reruns_valid_v1() -> None:
    protocol, _protocol_sha, decision = recovery._load_contract()
    full = recovery._full_protocol_tuples(protocol)
    accepted = recovery._accepted_protocol_keys(decision)
    pending = recovery.build_recovery_tuples(protocol)

    assert len(full) == 48
    assert len(accepted) == 6
    assert len(pending) == 42
    assert not (accepted & {row["protocol_tuple_key"] for row in pending})
    assert accepted | {row["protocol_tuple_key"] for row in pending} == {
        row["tuple_key"] for row in full
    }
    assert all(row["tuple_key"].endswith("__v2") for row in pending)
    assert all(row["diagnostic_runtime_version"] == recovery.RUNTIME_VERSION for row in pending)
    assert decision["recovery_budget"]["further_expansion_authorized"] is False


def test_preserved_v1_positive_and_negative_evidence_is_immutable() -> None:
    _protocol, _protocol_sha, decision = recovery._load_contract()

    recovery._verify_v1_evidence(decision)

    assert not recovery.V1_RESULT.exists()
    assert len(decision["preserved_valid_v1_tuples"]) == 6
    assert len(decision["preserved_invalid_v1_attempts"]) == 3


class _FakeContactRuntime:
    def __init__(self, environment: "_FakeParent") -> None:
        self.environment = environment
        self.contact_state_calls = 0

    def contact_state(self) -> np.ndarray:
        self.contact_state_calls += 1
        return np.full(256, self.environment._control_step, dtype=np.float32)


class _FakeParent:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._control_step = 0
        self._observation_events: list[dict[str, Any]] = []
        self._steps: list[dict[str, Any]] = []
        self._last_h: dict[str, Any] | None = None
        self._contact_runtime = _FakeContactRuntime(self)

    def step(self, _action: np.ndarray) -> str:
        self._steps.append({"control_step": self._control_step})
        self._control_step += 1
        return "STEP"

    def get_obs(self) -> dict[str, np.ndarray]:
        contact = self._contact_runtime.contact_state()
        row = {
            "control_step": self._control_step,
            "h_sha256": recovery._sha256_array(contact),
            "h_l2": float(np.linalg.norm(contact)),
        }
        self._observation_events.append(row)
        self._last_h = row
        return {"contact_state": contact}


def _fake_environment(
    monkeypatch: pytest.MonkeyPatch, condition: str
) -> recovery.RawTickBufferedHEnvironment:  # type: ignore[name-defined]
    monkeypatch.setattr(recovery, "BASE_BUILD_ENVIRONMENT", lambda _base: _FakeParent)
    monkeypatch.setattr(
        recovery.base,
        "TUPLE_SPEC",
        {"model": "B_HVA", "h_condition": condition},
    )
    return recovery.build_v2_environment(object)()


def test_lag5_uses_raw_physical_tick_two_at_policy_tick_seven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _fake_environment(monkeypatch, "same_episode_lag5")
    first = environment.get_obs()
    for _ in range(7):
        assert environment.step(np.zeros(22, dtype=np.float32)) == "STEP"
    observation = environment.get_obs()
    event = environment._observation_events[-1]

    assert np.array_equal(first["contact_state"], np.zeros(256, dtype=np.float32))
    assert np.array_equal(observation["contact_state"], np.full(256, 2, dtype=np.float32))
    assert set(environment._pi2s_correct_h_by_tick) == set(range(8))
    assert environment._contact_runtime.contact_state_calls == 9
    assert event["requested_control_tick"] == 2
    assert event["actual_control_tick"] == 2
    assert event["lag_source_correct_h_sha256"] == recovery._sha256_array(
        np.full(256, 2, dtype=np.float32)
    )
    assert all("post_step_correct_h" in row for row in environment._steps)


def test_train_mean_does_not_add_per_physical_step_encoder_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _fake_environment(monkeypatch, "train_mean")
    environment.get_obs()
    for _ in range(7):
        environment.step(np.zeros(22, dtype=np.float32))
    observation = environment.get_obs()

    assert np.array_equal(observation["contact_state"], recovery.v1.train_mean_h())
    assert environment._contact_runtime.contact_state_calls == 2
    assert all("post_step_correct_h" not in row for row in environment._steps)


def test_v2_audit_proves_continuous_lag_buffer_and_policy_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = {
        "tuple_key": "synthetic__v2",
        "protocol_tuple_key": "synthetic",
        "tuple_identity_sha256": "a" * 64,
        "diagnostic_runtime_version": recovery.RUNTIME_VERSION,
        "h_condition": "same_episode_lag5",
    }
    environment = _fake_environment(monkeypatch, "same_episode_lag5")
    environment.get_obs()
    for _ in range(7):
        environment.step(np.zeros(22, dtype=np.float32))
    environment.get_obs()
    chunks = [
        {
            "observation_timestamp": row["control_step"],
            "training_only_fields_sent": [],
            "sampling_control": {
                "h_condition": "same_episode_lag5",
                "delivered_h_sha256": row["delivered_h_sha256"],
                "delivered_h_l2": row["delivered_h_l2"],
            },
        }
        for row in environment._observation_events
    ]
    artifact = tmp_path / "tuple_result.json"
    artifact.write_text(
        json.dumps(
            {
                "tuple": expected,
                "observation_events": environment._observation_events,
                "action_chunks": chunks,
                "step_telemetry": environment._steps,
            }
        ),
        encoding="utf-8",
    )

    audit = recovery.build_v2_audit(artifact, expected)

    assert audit["status"] == "PASS"
    assert all(value == "PASS" for value in audit["gates"].values())
    assert audit["post_step_h_query_count"] == 7
    assert audit["buffered_correct_tick_count_through_last_observation"] == 8


def test_v2_audit_rejects_a_gap_disguised_by_duplicate_raw_ticks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = {
        "tuple_key": "synthetic__v2",
        "protocol_tuple_key": "synthetic",
        "tuple_identity_sha256": "a" * 64,
        "diagnostic_runtime_version": recovery.RUNTIME_VERSION,
        "h_condition": "same_episode_lag5",
    }
    environment = _fake_environment(monkeypatch, "same_episode_lag5")
    environment.get_obs()
    for _ in range(7):
        environment.step(np.zeros(22, dtype=np.float32))
    environment.get_obs()
    environment._steps[-1]["post_step_correct_h"]["control_tick"] = 6
    chunks = [
        {
            "observation_timestamp": row["control_step"],
            "training_only_fields_sent": [],
            "sampling_control": {
                "h_condition": "same_episode_lag5",
                "delivered_h_sha256": row["delivered_h_sha256"],
                "delivered_h_l2": row["delivered_h_l2"],
            },
        }
        for row in environment._observation_events
    ]
    artifact = tmp_path / "tuple_result.json"
    artifact.write_text(
        json.dumps(
            {
                "tuple": expected,
                "observation_events": environment._observation_events,
                "action_chunks": chunks,
                "step_telemetry": environment._steps,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(recovery.base.ContractError, match="contradictory correct H"):
        recovery.build_v2_audit(artifact, expected)


def test_recovery_progress_counts_fourteen_tuple_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(recovery.base, "MANIFEST", manifest_path)
    assignments = [
        {"tuple_key": f"b_hva_42__r11_i00__p{index}__htrain_mean__v2"} for index in range(14)
    ] + [{"tuple_key": "b_hva_43__r11_i00__p4317__hsame_episode_lag5__v2"}]

    payload = recovery.progress_payload(
        {"source_commit": "a" * 40, "protocol_sha256": "b" * 64},
        assignments,
        started=time.time() - 1.0,
        status="RUNNING",
    )

    assert payload["completed_blocks"] == 1
    assert payload["planned_blocks"] == 3
    assert payload["completed_recovery_tuples"] == 15
    assert payload["completed_optional_protocol_tuples"] == 21
    assert payload["planned_optional_protocol_tuples"] == 48


def test_episode_summary_preserves_actual_v2_attempt(tmp_path: Path) -> None:
    artifact = tmp_path / "tuple.json"
    audit = tmp_path / "audit.json"
    artifact.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    payload = {
        "success": False,
        "termination": "native_max_steps",
        "steps": 1000,
        "ever_object_contact": True,
        "ever_lifted": True,
        "max_pinch_count": 1,
        "ever_native_trigger": False,
        "max_success_counter": 0,
        "max_consecutive_object_contact_run": 8,
        "exact_first_failure_stage": "N/A",
    }

    row = recovery._episode_summary(
        {"tuple_key": "key", "protocol_tuple_key": "protocol-key"},
        payload,
        artifact,
        audit,
        recovery.RUNTIME_VERSION,
        "V2_RECOVERY",
        2,
    )

    assert row["attempt"] == 2
    assert row["protocol_tuple_key"] == "protocol-key"
    assert row["provenance"] == "V2_RECOVERY"


def test_failed_h_audit_is_published_before_evaluator_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = {
        "tuple_key": "key__v2",
        "protocol_tuple_key": "key",
        "tuple_identity_sha256": "a" * 64,
    }
    spec_path = tmp_path / "spec.json"
    artifact = tmp_path / "tuple_result.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    def fake_base_evaluate(*_args: object, **_kwargs: object) -> None:
        artifact.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(recovery.base, "evaluate_tuple", fake_base_evaluate)
    monkeypatch.setattr(
        recovery,
        "build_v2_audit",
        lambda _artifact, _spec: {
            "schema": recovery.SCHEMA_AUDIT,
            "status": "FAIL",
            "gates": {"raw_tick_buffer_complete": "FAIL"},
        },
    )

    with pytest.raises(recovery.base.ContractError, match="H delivery audit failed"):
        recovery.evaluate_tuple(
            spec_path,
            tmp_path / "contact.sock",
            tmp_path / "output",
            tmp_path / "diagnostics.jsonl",
            artifact,
            9000,
        )

    published = recovery.base.read_json(recovery._audit_path(artifact))
    assert published["status"] == "FAIL"


def test_scientific_audit_failure_never_consumes_attempt_two(
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


def test_recovery_source_contains_no_expansion_or_training_entrypoint() -> None:
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    forbidden = (
        "optimizer.step(",
        "train_state.save(",
        "git push",
        "shutil.rmtree(",
        "MAX_INFRASTRUCTURE_ATTEMPTS =",
    )

    assert all(token not in source for token in forbidden)
