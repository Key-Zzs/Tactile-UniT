from __future__ import annotations

from collections import deque
import importlib.util
import json
from pathlib import Path
import queue
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/simulation/run_pi2s_diagnostic_rollouts.py"
SPEC = importlib.util.spec_from_file_location("run_pi2s_diagnostic_rollouts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
rollouts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rollouts
SPEC.loader.exec_module(rollouts)


def test_frozen_base_tuple_contract_is_exact_and_outcome_blind() -> None:
    protocol = json.loads(rollouts.PROTOCOL.read_text())
    tuples = rollouts.build_base_tuples(protocol)

    assert len(tuples) == 72
    assert len({row["tuple_key"] for row in tuples}) == 72
    assert {row["checkpoint_id"] for row in tuples} == {
        "B0_43",
        "B_VA27_43",
        "B1_43",
        "B_HVA_42",
        "B_HVA_43",
        "B2_43",
    }
    assert {row["sampling_seed"] for row in tuples} == {4317, 4318}
    assert {row["h_condition"] for row in tuples} == {"correct"}
    assert all(len(row["reset_identity"]) == 64 for row in tuples)
    assert all(len(row["checkpoint_tree_sha256"]) == 64 for row in tuples)
    forbidden = {"success", "failure", "reward", "termination", "score"}
    assert all(not (forbidden & set(row)) for row in tuples)
    assert protocol["development_rollouts"]["optional_h_intervention"]["canonical_tuples"] == 48
    assert protocol["development_rollouts"]["maximum_canonical_tuples"] == 120


def test_stage_contract_does_not_invent_approach_or_stable_grasp() -> None:
    assert rollouts.STAGE_DEFINITION["APPROACH"].startswith("N/A:")
    assert rollouts.STAGE_DEFINITION["STABLE_GRASP"].startswith("N/A:")
    assert "contact_count > 0" in rollouts.STAGE_DEFINITION["OBJECT_CONTACT"]
    assert "tongs_pos[2]" in rollouts.STAGE_DEFINITION["LIFT"]
    assert "success_counter >= 30" in rollouts.STAGE_DEFINITION["NATIVE_SUCCESS"]


class _FakeRandom:
    def key(self, value: int) -> tuple[str, int]:
        return ("key", value)


class _FakeJax:
    random = _FakeRandom()


class _FakePolicy:
    def __init__(self) -> None:
        self._rng = None
        self.metadata = {"model": "fake"}
        self.calls: list[dict] = []

    def infer(self, observation: dict) -> dict:
        self.calls.append(dict(observation))
        return {"actions": np.zeros((30, 22), dtype=np.float32)}


def _seeded_observation(*, token: str, seed: int, reset: bool) -> dict:
    return {
        "state": np.zeros(23, dtype=np.float32),
        "_pi2s_sampling_seed": seed,
        "_pi2s_episode_token": token,
        "_pi2s_reset_rng": reset,
    }


def test_sampling_wrapper_resets_once_and_strips_diagnostic_metadata() -> None:
    policy = _FakePolicy()
    wrapped = rollouts.SeedResetPolicy(policy, _FakeJax())

    first = wrapped.infer(_seeded_observation(token="episode-a", seed=4317, reset=True))
    second = wrapped.infer(_seeded_observation(token="episode-a", seed=4317, reset=False))

    assert policy._rng == ("key", 4317)
    assert policy.calls == [
        {"state": pytest.approx(np.zeros(23))},
        {"state": pytest.approx(np.zeros(23))},
    ]
    assert first["pi2s_sampling_control"] == {
        "sampling_seed": 4317,
        "episode_token": "episode-a",
        "query_index": 0,
        "rng_reset_before_query": True,
    }
    assert second["pi2s_sampling_control"]["query_index"] == 1
    assert second["pi2s_sampling_control"]["rng_reset_before_query"] is False


def test_sampling_wrapper_rejects_silent_episode_change() -> None:
    policy = _FakePolicy()
    wrapped = rollouts.SeedResetPolicy(policy, _FakeJax())
    wrapped.infer(_seeded_observation(token="episode-a", seed=4317, reset=True))
    with pytest.raises(rollouts.ContractError, match="changed without"):
        wrapped.infer(_seeded_observation(token="episode-b", seed=4318, reset=False))


def test_retry_can_explicitly_reset_same_episode_token() -> None:
    policy = _FakePolicy()
    wrapped = rollouts.SeedResetPolicy(policy, _FakeJax())
    wrapped.infer(_seeded_observation(token="same-tuple", seed=4317, reset=True))
    wrapped.infer(_seeded_observation(token="same-tuple", seed=4317, reset=False))
    retried = wrapped.infer(_seeded_observation(token="same-tuple", seed=4317, reset=True))
    assert retried["pi2s_sampling_control"]["query_index"] == 0
    assert retried["pi2s_sampling_control"]["rng_reset_before_query"] is True


def test_action_merge_preserves_values_and_adds_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_package = ModuleType("dexjoco_openpi_client")
    interpolate = lambda old, new, t: (1.0 - t) * old + t * new  # noqa: E731
    fake_package.eval_dexjoco_openpi = SimpleNamespace(
        _interp_single_arm_action=interpolate,
        _interp_dual_arm_action=interpolate,
    )
    monkeypatch.setitem(sys.modules, "dexjoco_openpi_client", fake_package)
    first = np.zeros((4, 22), dtype=np.float32)
    first[:, 0] = 1.0
    second = np.zeros((4, 22), dtype=np.float32)
    second[:, 0] = 3.0
    action_queue: queue.Queue[dict] = queue.Queue()
    action_queue.put({"action": first, "timestamp": 0, "chunk_id": "a"})
    buffer: deque = deque()

    rollouts.receive_actions_with_provenance(action_queue, buffer, 0, False)
    assert [row.timestamp for row in buffer] == [0, 1, 2, 3]
    assert all(row.chunk_ids == ("a",) for row in buffer)
    assert np.array_equal(np.stack([row.action for row in buffer]), first)

    buffer.popleft()
    action_queue.put({"action": second, "timestamp": 1, "chunk_id": "b"})
    rollouts.receive_actions_with_provenance(action_queue, buffer, 1, False)
    assert [row.timestamp for row in buffer] == [1, 2, 3, 4]
    assert all(row.chunk_ids == ("a", "b") for row in list(buffer)[:3])
    assert buffer[-1].chunk_ids == ("b",)
    # Three overlapping steps use the accepted interpolation weights 1/4, 2/4, 3/4.
    assert [row.action[0] for row in list(buffer)[:3]] == pytest.approx([1.5, 2.0, 2.5])
    assert buffer[-1].action[0] == pytest.approx(3.0)
    assert rollouts.ACTION_PROVENANCE_BY_TIMESTAMP == {
        1: ("a", "b"),
        2: ("a", "b"),
        3: ("a", "b"),
        4: ("b",),
    }


def test_scientific_publication_is_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    rollouts.publish_json_no_clobber(destination, {"value": 1})
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        rollouts.publish_json_no_clobber(destination, {"value": 2})
    assert destination.read_bytes() == before


def test_exact_byte_backup_is_no_clobber(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"spacing" : true}\n')
    destination = tmp_path / "nested/backup.json"
    rollouts.copy_file_no_clobber(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        rollouts.copy_file_no_clobber(source, destination)


def test_progress_counts_only_complete_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    monkeypatch.setattr(rollouts, "MANIFEST", manifest_path)
    assignments = [{"tuple_key": f"b0_43__r9_i00__p{index}"} for index in range(12)] + [
        {"tuple_key": "b1_43__r9_i00__p4317"}
    ]
    payload = rollouts.progress_payload(
        {"source_commit": "a" * 40, "protocol_sha256": "b" * 64},
        assignments,
        started=rollouts.time.time() - 1.0,
        status="RUNNING",
    )
    assert payload["completed_canonical_tuples"] == 13
    assert payload["completed_blocks"] == 1
    assert payload["optional_48_started"] is False


def test_tuple_artifact_validation_preserves_negative_outcome(tmp_path: Path) -> None:
    expected = {"tuple_key": "x", "tuple_identity_sha256": "a" * 64}
    path = tmp_path / "tuple.json"
    path.write_text(
        json.dumps(
            {
                "schema": rollouts.SCHEMA_TUPLE,
                "status": "PASS",
                "tuple": expected,
                "success": False,
                "steps": 1000,
                "step_telemetry": [{}] * 1000,
                "gates": {"negative_result_retained": "PASS"},
            }
        )
    )
    payload = rollouts.validate_tuple_artifact(path, expected)
    assert payload["success"] is False
    assert payload["steps"] == 1000


def test_checkpoint_presence_check_never_hashes_checkpoint_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    for relative in (
        "_CHECKPOINT_METADATA",
        "params/manifest.ocdbt",
        "train_state/manifest.ocdbt",
    ):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("present")
    monkeypatch.setattr(rollouts, "symbolic_checkpoint", lambda _value: checkpoint)
    monkeypatch.setattr(
        rollouts,
        "sha256_file",
        lambda _path: pytest.fail("checkpoint presence validation must not hash trees"),
    )
    rollouts.validate_checkpoint_presence(
        {"checkpoint_path": "$EXPERIMENT_ROOT/example", "checkpoint_id": "B0_43"}
    )


def test_fixed_contract_has_no_training_or_real_robot_operation() -> None:
    source = SCRIPT.read_text()
    assert "optimizer.step(" not in source
    assert "train_state.save" not in source
    assert "worktree remove" not in source
    assert "git push" not in source
    assert 'real_robot_used": False' in source
    assert rollouts.MAX_INFRASTRUCTURE_ATTEMPTS == 2
