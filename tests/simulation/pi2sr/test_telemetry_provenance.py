from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from gr00t.simulation.pi2sr_provenance import (
    build_provenance_manifest,
    compare_provenance,
)
from gr00t.simulation.pi2sr_telemetry import (
    SCHEMA,
    AsyncTelemetryWriter,
    StableGraspTracker,
    build_step_record,
    storage_probe,
    tactile_summary,
)
from scripts.simulation.run_pi2sr_telemetry_smoke import policy_stub, synthetic_tactile

ROOT = Path(__file__).resolve().parents[3]


def tactile(occupied_regions: int = 0, normal_force: float = 0.0) -> np.ndarray:
    value = np.zeros((5, 6), dtype=np.float32)
    value[:occupied_regions, 0] = 1.0
    value[:occupied_regions, 1] = normal_force
    value[:occupied_regions, 2] = normal_force / 2.0
    value[:occupied_regions, 3:6] = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    return value.reshape(-1)


def record(step: int = 0) -> dict:
    return build_step_record(
        run_id="run-1",
        episode_id="episode-1",
        task="pinch_tongs",
        reset_identity="development-reset-1",
        control_step_index=step,
        simulation_time_sec=step * 0.02,
        observation_timestamp_ns=1000 + step,
        policy_request_id=f"request-{step // 5}",
        policy_request_index=step // 5,
        action_chunk_generated_ns=1100 + step,
        action_applied_ns=1200 + step,
        policy_facing_state=np.zeros(22, dtype=np.float32),
        environment_facing_state=np.zeros(23, dtype=np.float32),
        commanded_action=np.zeros(22, dtype=np.float32),
        environment_applied_action=np.zeros(23, dtype=np.float32),
        raw_canonical_tactile=tactile(2, 0.3),
        canonical_h=np.zeros(256, dtype=np.float32),
        canonical_h_reference=None,
        h_runtime_identity="DIRECT_IN_PROCESS:cpu:float32:batch1",
        history_valid_samples=26,
        history_bootstrap_status="WARM",
        replan_required=step % 5 == 0,
        replan_stride=5,
        action_queue_index=step % 5,
        action_queue_remaining=4 - step % 5,
        terminated=False,
        truncated=False,
        stage_predicates={
            "CONTACT": True,
            "LIFT": None,
            "NATIVE_TRIGGER": None,
            "SUCCESS": False,
            "STABLE_GRASP": step >= 4,
        },
        evaluation_only_fields={"object_height": {"available": False, "value": None}},
        wall_monotonic_ns=900 + step,
    )


def test_tactile_summary_marks_cop_only_when_defined() -> None:
    summary = tactile_summary(tactile(2, 0.3))
    assert summary["occupancy_count_by_region"] == [1, 1, 0, 0, 0]
    assert summary["normal_force_by_region"][:2] == pytest_approx_list([0.3, 0.3])
    assert summary["cop_by_region"][0] == pytest_approx_list([0.1, 0.2, 0.3])
    assert summary["cop_by_region"][2] is None


def pytest_approx_list(values: list[float]) -> list[float]:
    return [float(np.float32(value)) for value in values]


def test_stable_grasp_is_prospective_five_step_diagnostic() -> None:
    tracker = StableGraspTracker()
    assert [tracker.update(tactile(2, 0.3)) for _ in range(5)] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert tracker.update(tactile()) is False


def test_record_has_frozen_schema_and_evaluation_only_boundary() -> None:
    value = record()
    assert value["schema"] == SCHEMA
    assert value["policy_input"] is False
    assert value["evaluation_only"]["evaluation_only"] is True
    assert value["evaluation_only"]["policy_input"] is False
    assert set(value["stage_predicates"]) == {
        "CONTACT",
        "LIFT",
        "NATIVE_TRIGGER",
        "SUCCESS",
        "STABLE_GRASP",
    }
    assert value["canonical_h_reference"] is None
    assert len(value["canonical_h"]) == 256


def test_async_writer_is_complete_and_gzip_chunked(tmp_path: Path) -> None:
    destination = tmp_path / "telemetry" / "episode.jsonl.gz"
    writer = AsyncTelemetryWriter(destination, queue_capacity=16)
    accepted = [writer.submit(record(step)) for step in range(8)]
    summary = writer.close()
    assert all(accepted)
    assert summary["status"] == "PASS"
    assert summary["submitted_records"] == summary["written_records"] == 8
    assert summary["dropped_records"] == 0
    with gzip.open(destination, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    assert [row["control_step_index"] for row in rows] == list(range(8))


def test_writer_failure_is_reported_without_touching_action(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("block", encoding="utf-8")
    action = np.arange(22, dtype=np.float32)
    before = action.tobytes()
    writer = AsyncTelemetryWriter(blocked_parent / "episode.jsonl.gz", queue_capacity=2)
    time.sleep(0.05)
    writer.submit(record())
    summary = writer.close()
    assert summary["status"] == "FAIL"
    assert summary["writer_error"] is not None
    assert action.tobytes() == before


def test_storage_probe_exercises_atomic_same_directory_path(tmp_path: Path) -> None:
    assert storage_probe(tmp_path)["status"] == "PASS"
    assert list(tmp_path.iterdir()) == []


def test_provenance_manifest_and_seed_comparison() -> None:
    manifest = build_provenance_manifest(
        repo_root=ROOT,
        config_paths=[ROOT / "configs/simulation/pi2sr/provenance_contract_v1.json"],
        base_model_checkpoint_tree_sha256=None,
        teacher_or_contact_encoder_sha256="3d4519",
        data_identity={
            "dataset_manifest_sha256": "dataset",
            "split_identity": "development",
            "source_group_identities": ["group-a"],
            "normalization_sha256": "normalization",
            "target_mask_sha256": None,
            "episode_reset_manifest_sha256": "reset",
        },
        randomness={"evaluator_reset_seed": 1, "policy_request_counter": 0},
        device="cpu",
        dtype="float32",
    )
    assert manifest["historical_retroactive_prng_reconstruction_claimed"] is False
    assert manifest["randomness"]["evaluator_reset_seed"] == 1
    assert "root_training_seed" in manifest["missing_randomness"]
    identical = json.loads(json.dumps(manifest))
    assert compare_provenance(manifest, identical) == "same_source_same_config_same_seed"
    different_seed = json.loads(json.dumps(manifest))
    different_seed["randomness"]["evaluator_reset_seed"] = 2
    assert compare_provenance(manifest, different_seed) == "same_source_same_config_different_seed"
    different_source = json.loads(json.dumps(manifest))
    different_source["source_identity"]["git_commit"] = "other"
    assert compare_provenance(manifest, different_source) == "different_source_or_config"


def test_smoke_fixture_exercises_contact_without_stochastic_policy() -> None:
    assert np.count_nonzero(synthetic_tactile(0, 0)) == 0
    contact = synthetic_tactile(1, 5).reshape(5, 6)
    assert int(np.sum(contact[:, 0] > 0.5)) == 2
    state = np.linspace(-0.2, 0.2, 22, dtype=np.float32)
    h_value = np.ones(256, dtype=np.float32)
    first = policy_stub(state, h_value, 1)
    second = policy_stub(state, h_value, 1)
    assert first.shape == (27, 22)
    assert first.tobytes() == second.tobytes()
