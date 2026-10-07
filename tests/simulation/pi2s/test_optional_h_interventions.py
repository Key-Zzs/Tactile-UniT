from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from scripts.simulation import run_pi2s_optional_h_interventions as optional


def test_post_base_decision_is_exact_and_not_result_seeking() -> None:
    protocol, protocol_sha, decision = optional._load_contract()

    assert protocol_sha == decision["protocol"]["sha256"]
    assert decision["decision"] == "RUN_EXACTLY_THE_FROZEN_OPTIONAL_48_AND_DO_NOT_EXPAND_FURTHER"
    assert decision["guardrails"]["purpose"] == "MECHANISM_DIAGNOSIS_NOT_POSITIVE_RESULT_SEARCH"
    assert decision["guardrails"]["formal_track_a_scores_may_be_replaced"] is False
    assert decision["guardrails"]["best_intervention_may_be_selected_for_deployment"] is False
    assert decision["frozen_budget"]["further_rollout_expansion_authorized"] is False
    assert protocol["development_rollouts"]["maximum_canonical_tuples"] == 120


def test_optional_tuple_contract_is_exact_unique_and_outcome_blind() -> None:
    protocol, _protocol_sha, _decision = optional._load_contract()
    tuples = optional.build_optional_tuples(protocol)

    assert len(tuples) == 48
    assert len({row["tuple_key"] for row in tuples}) == 48
    assert {row["checkpoint_id"] for row in tuples} == set(optional.FOCUS_IDS)
    assert {row["h_condition"] for row in tuples} == set(optional.CONDITIONS)
    assert {row["sampling_seed"] for row in tuples} == {4317, 4318}
    assert all(row["runtime_mode"] == "CONTACT_STATE_TOKENS" for row in tuples)
    assert all("__h" in row["tuple_key"] for row in tuples)
    forbidden = {"success", "failure", "reward", "termination", "score"}
    assert all(not (forbidden & set(row)) for row in tuples)


def test_train_mean_is_recomputed_from_persistent_snapshot_and_hash_bound() -> None:
    protocol, _protocol_sha, _decision = optional._load_contract()
    mean = optional.train_mean_h()

    assert mean.shape == (256,)
    assert mean.dtype == np.float32
    assert np.isfinite(mean).all()
    assert optional._sha256_array(mean) == protocol["h_conditions"]["train_mean_h_float32_sha256"]
    assert np.linalg.norm(mean.astype(np.float64)) == pytest.approx(
        protocol["h_conditions"]["train_mean_h_l2"], abs=1e-12, rel=0.0
    )


class _FakeParent:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._control_step = 0
        self._observation_events: list[dict] = []
        self._last_h = None

    def get_obs(self) -> dict[str, np.ndarray]:
        contact = np.full(256, self._control_step, dtype=np.float32)
        row = {
            "control_step": self._control_step,
            "h_sha256": optional._sha256_array(contact),
            "h_l2": float(np.linalg.norm(contact)),
        }
        self._observation_events.append(row)
        self._last_h = row
        return {"contact_state": contact}


def test_optional_environment_delivers_train_mean_and_retains_correct_h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(optional, "BASE_BUILD_ENVIRONMENT", lambda _base: _FakeParent)
    monkeypatch.setattr(
        optional.base,
        "TUPLE_SPEC",
        {"model": "B_HVA", "h_condition": "train_mean"},
    )
    environment = optional.build_optional_environment(object)()
    environment._control_step = 7

    observation = environment.get_obs()
    event = environment._observation_events[-1]

    assert np.array_equal(observation["contact_state"], optional.train_mean_h())
    assert observation["_pi2s_h_condition"] == "train_mean"
    assert event["correct_h_sha256"] == optional._sha256_array(np.full(256, 7, dtype=np.float32))
    assert event["delivered_h_sha256"] == optional._sha256_array(optional.train_mean_h())
    assert event["requested_control_step"] is None
    assert event["bootstrap_affected"] is False


def test_optional_environment_lag5_is_causal_and_marks_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(optional, "BASE_BUILD_ENVIRONMENT", lambda _base: _FakeParent)
    monkeypatch.setattr(
        optional.base,
        "TUPLE_SPEC",
        {"model": "B1", "h_condition": "same_episode_lag5"},
    )
    environment = optional.build_optional_environment(object)()
    observations = []
    for step in range(7):
        environment._control_step = step
        observations.append(environment.get_obs())

    assert np.array_equal(observations[0]["contact_state"], np.zeros(256, dtype=np.float32))
    assert np.array_equal(observations[4]["contact_state"], np.zeros(256, dtype=np.float32))
    assert np.array_equal(observations[5]["contact_state"], np.zeros(256, dtype=np.float32))
    assert np.array_equal(observations[6]["contact_state"], np.ones(256, dtype=np.float32))
    assert environment._observation_events[4]["requested_control_step"] == -1
    assert environment._observation_events[4]["actual_control_step"] == 0
    assert environment._observation_events[4]["bootstrap_affected"] is True
    assert environment._observation_events[5]["bootstrap_affected"] is False


class _FakeRandom:
    def key(self, value: int) -> tuple[str, int]:
        return ("key", value)


class _FakeJax:
    random = _FakeRandom()


class _FakePolicy:
    def __init__(self) -> None:
        self._rng = None
        self.metadata = {}
        self.calls: list[dict] = []

    def infer(self, observation: dict) -> dict:
        self.calls.append(dict(observation))
        return {"actions": np.zeros((30, 22), dtype=np.float32)}


def test_policy_wrapper_strips_condition_and_reports_delivered_h() -> None:
    policy = _FakePolicy()
    wrapper = optional.OptionalSeedResetPolicy(policy, _FakeJax())
    contact = np.arange(256, dtype=np.float32)
    result = wrapper.infer(
        {
            "contact_state": contact,
            "_pi2s_h_condition": "same_episode_lag5",
            "_pi2s_sampling_seed": 4317,
            "_pi2s_episode_token": "tuple",
            "_pi2s_reset_rng": True,
        }
    )

    assert "_pi2s_h_condition" not in policy.calls[0]
    assert np.array_equal(policy.calls[0]["contact_state"], contact)
    control = result["pi2s_sampling_control"]
    assert control["h_condition"] == "same_episode_lag5"
    assert control["delivered_h_sha256"] == optional._sha256_array(contact)


def _synthetic_tuple_artifact(path: Path, expected: dict, condition: str) -> None:
    correct0 = np.zeros(256, dtype=np.float32)
    correct5 = np.full(256, 5, dtype=np.float32)
    mean = optional.train_mean_h()
    if condition == "train_mean":
        delivered0 = delivered5 = mean
        requested0 = actual0 = requested5 = actual5 = None
        bootstrap0 = bootstrap5 = False
    else:
        delivered0 = delivered5 = correct0
        requested0, actual0, bootstrap0 = -5, 0, True
        requested5, actual5, bootstrap5 = 0, 0, False
    events = [
        {
            "control_step": 0,
            "h_condition": condition,
            "correct_h_sha256": optional._sha256_array(correct0),
            "correct_h_l2": 0.0,
            "delivered_h_sha256": optional._sha256_array(delivered0),
            "delivered_h_l2": float(np.linalg.norm(delivered0)),
            "h_sha256": optional._sha256_array(delivered0),
            "h_l2": float(np.linalg.norm(delivered0)),
            "requested_control_step": requested0,
            "actual_control_step": actual0,
            "bootstrap_affected": bootstrap0,
        },
        {
            "control_step": 5,
            "h_condition": condition,
            "correct_h_sha256": optional._sha256_array(correct5),
            "correct_h_l2": float(np.linalg.norm(correct5)),
            "delivered_h_sha256": optional._sha256_array(delivered5),
            "delivered_h_l2": float(np.linalg.norm(delivered5)),
            "h_sha256": optional._sha256_array(delivered5),
            "h_l2": float(np.linalg.norm(delivered5)),
            "requested_control_step": requested5,
            "actual_control_step": actual5,
            "bootstrap_affected": bootstrap5,
        },
    ]
    chunks = [
        {
            "observation_timestamp": row["control_step"],
            "training_only_fields_sent": [],
            "sampling_control": {
                "h_condition": condition,
                "delivered_h_sha256": row["delivered_h_sha256"],
                "delivered_h_l2": row["delivered_h_l2"],
            },
        }
        for row in events
    ]
    path.write_text(
        json.dumps({"tuple": expected, "observation_events": events, "action_chunks": chunks})
    )


@pytest.mark.parametrize("condition", optional.CONDITIONS)
def test_intervention_audit_checks_event_and_policy_delivery(
    tmp_path: Path, condition: str
) -> None:
    expected = {
        "tuple_key": "tuple-" + condition,
        "tuple_identity_sha256": "a" * 64,
        "h_condition": condition,
    }
    artifact = tmp_path / "tuple.json"
    _synthetic_tuple_artifact(artifact, expected, condition)

    audit = optional.build_intervention_audit(artifact, expected)

    assert audit["status"] == "PASS"
    assert all(value == "PASS" for value in audit["gates"].values())
    assert audit["policy_query_count"] == 2


def test_optional_progress_counts_sixteen_tuple_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(optional.base, "MANIFEST", manifest)
    assignments = [
        {"tuple_key": f"b_hva_42__r11_i00__p{index}__htrain_mean"} for index in range(16)
    ] + [{"tuple_key": "b_hva_43__r11_i00__p4317__htrain_mean"}]
    payload = optional.progress_payload(
        {
            "source_commit": "a" * 40,
            "protocol_sha256": "b" * 64,
            "base_results_sha256": "c" * 64,
        },
        assignments,
        started=time.time() - 1.0,
        status="RUNNING",
    )

    assert payload["completed_canonical_tuples"] == 17
    assert payload["completed_blocks"] == 1
    assert payload["planned_blocks"] == 3
    assert payload["optional_48_started"] is True


def test_optional_source_has_no_training_push_deletion_or_robot_operation() -> None:
    source = Path(optional.__file__).read_text()
    assert "optimizer.step(" not in source
    assert "train_state.save" not in source
    assert "git push" not in source
    assert "worktree remove" not in source
    assert '"real_robot_used": False' in source
    assert optional.EXPECTED_TUPLES == 48
