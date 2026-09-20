from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/simulation/status_s4_3_pi2n.py"


def load_module():
    spec = importlib.util.spec_from_file_location("status_s4_3_pi2n", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def running_inputs():
    return {
        "job": {"state": "RUNNING", "exit_code": None},
        "heartbeat": {"state": "RUNNING", "exit_code": None},
        "heartbeat_age": 30.0,
        "supervisor": {"pid": 123, "live": True},
        "matching_trainers": 1,
        "runtime_identity_matches": True,
        "training_identity_frozen": True,
        "final_checkpoint_present": False,
        "completion_audit_passed": False,
    }


def test_status_requires_every_live_identity_gate_for_running() -> None:
    module = load_module()
    values = running_inputs()
    state, failed = module.classify_run(**values)
    assert state == "RUNNING"
    assert failed == []
    for key in (
        "job",
        "heartbeat",
        "heartbeat_age",
        "supervisor",
        "matching_trainers",
        "runtime_identity_matches",
        "training_identity_frozen",
    ):
        changed = dict(values)
        if key == "job":
            changed[key] = {"state": "DONE", "exit_code": 0}
        elif key == "heartbeat":
            changed[key] = {"state": "FAILED", "exit_code": 1}
        elif key == "heartbeat_age":
            changed[key] = module.HEARTBEAT_MAX_AGE_SECONDS + 1
        elif key == "supervisor":
            changed[key] = {"pid": 123, "live": False}
        elif key == "matching_trainers":
            changed[key] = 0
        else:
            changed[key] = False
        assert module.classify_run(**changed)[0] == "EXITED_UNVERIFIED"


def test_done_is_unverified_until_checkpoint_audit_passes() -> None:
    module = load_module()
    values = running_inputs()
    values.update(
        {
            "job": {"state": "DONE", "exit_code": 0},
            "heartbeat": {"state": "DONE", "exit_code": 0},
            "heartbeat_age": 30.0,
            "supervisor": {"pid": 123, "live": False},
            "matching_trainers": 0,
            "runtime_identity_matches": True,
            "training_identity_frozen": True,
            "final_checkpoint_present": True,
        }
    )
    assert module.classify_run(**values)[0] == "EXITED_UNVERIFIED"
    values["completion_audit_passed"] = True
    assert module.classify_run(**values)[0] == "COMPLETE_VERIFIED"


def test_completion_manifest_binds_model_step_hash_and_checkpoint(tmp_path: Path) -> None:
    module = load_module()
    spec = module.MODEL_SPECS[0]
    checkpoint = tmp_path / spec.experiment / "29999"
    payload = {
        "status": "PASS",
        "model_id": spec.model_id,
        "optimizer_steps": 30_000,
        "checkpoint": str(checkpoint),
        "checkpoint_tree_sha256": "a" * 64,
    }
    assert module.completion_manifest_valid(payload, spec, checkpoint)
    for key, value in (
        ("status", "FAIL"),
        ("model_id", "B_VAC_V"),
        ("optimizer_steps", 29_999),
        ("checkpoint_tree_sha256", "short"),
    ):
        changed = dict(payload)
        changed[key] = value
        assert not module.completion_manifest_valid(changed, spec, checkpoint)


def test_completion_status_requires_exact_referenced_pass_chain(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    spec = module.MODEL_SPECS[0]
    checkpoint = tmp_path / spec.experiment / "29999"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    completion_path = artifacts / "b_va27_training_completion.json"
    completion = {
        "status": "PASS",
        "model_id": "B_VA27",
        "optimizer_steps": 30_000,
        "gates": {"cold_load": "PASS", "finite": "PASS"},
    }
    completion_path.write_text(json.dumps(completion))
    manifest = {
        "status": "PASS",
        "frozen": True,
        "model_id": "B_VA27",
        "optimizer_steps": 30_000,
        "checkpoint": str(checkpoint),
        "checkpoint_tree_sha256": "a" * 64,
        "training_completion": "$REPO_ROOT/artifacts/b_va27_training_completion.json",
        "training_completion_sha256": hashlib.sha256(
            completion_path.read_bytes()
        ).hexdigest(),
        "gates": {"manifest": "PASS"},
    }
    assert module.completion_chain_status_valid(
        manifest, completion, completion_path, spec, checkpoint
    )
    manifest["training_completion_sha256"] = "0" * 64
    assert not module.completion_chain_status_valid(
        manifest, completion, completion_path, spec, checkpoint
    )
    manifest["training_completion_sha256"] = hashlib.sha256(
        completion_path.read_bytes()
    ).hexdigest()
    completion["gates"]["finite"] = "FAIL"
    assert not module.completion_chain_status_valid(
        manifest, completion, completion_path, spec, checkpoint
    )


def test_training_freeze_binds_the_final_only_candidate_contract() -> None:
    module = load_module()
    spec = module.MODEL_SPECS[0]
    payload = {
        "status": "FROZEN_BEFORE_TRAINING",
        "model_id": spec.model_id,
        "seed": 42,
        "steps": 30_000,
        "global_batch_size": 32,
        "final_checkpoint_step": 29_999,
        "checkpoint_selection": "FINAL_ONLY",
        "base_manifest_status": "PASS",
        "target_sha256": "a" * 64,
        "source_snapshot_sha256": "b" * 64,
    }
    assert module.training_freeze_valid(payload, spec)
    for key, value in (
        ("status", "RUNNING"),
        ("seed", 43),
        ("steps", 20_000),
        ("global_batch_size", 16),
        ("checkpoint_selection", "BEST"),
        ("target_sha256", "short"),
    ):
        changed = dict(payload)
        changed[key] = value
        assert not module.training_freeze_valid(changed, spec)


def test_status_entry_is_read_only_and_model_free() -> None:
    source = SCRIPT.read_text()
    assert "write_text" not in source
    assert "openpi" not in source.lower()
    assert "orbax" not in source.lower()
    assert "torch" not in source.lower()
    assert "jax" not in source.lower()
    assert '"read_only": True' in source
    assert '"runtime_files_written": False' in source


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_evaluation_status_requires_live_identity_and_fresh_heartbeat(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        module,
        "proc_identity",
        lambda pid: {
            "pid": pid,
            "live": True,
            "cmdline": "python launch_s4_3_pi2n_development.py supervise --gpus 1",
        },
    )
    monkeypatch.setattr(module, "tmux_session_exists", lambda name: True)
    _write(tmp_path / "pre_dev_freeze.json", {"status": "PASS"})
    _write(
        tmp_path / "development_launch.json",
        {
            "status": "LAUNCHING",
            "session": "s43_pi2n_dev",
            "physical_gpu_ids": [1],
            "physical_gpu_uuids": ["GPU-test"],
        },
    )
    common = {
        "state": "RUNNING",
        "exit_code": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "session": "s43_pi2n_dev",
        "supervisor_pid": 123,
        "physical_gpu_ids": [1],
        "physical_gpu_uuids": ["GPU-test"],
    }
    _write(tmp_path / "development_job_status.json", common)
    _write(tmp_path / "development_heartbeat.json", common)
    row = module.summarize_evaluation(
        module.EVALUATION_SPECS[0], datetime.now(timezone.utc)
    )
    assert row is not None
    assert row["state"] == "RUNNING"
    assert row["performance_values_read"] is False

    stale = dict(common)
    stale["updated_at"] = "2000-01-01T00:00:00+00:00"
    _write(tmp_path / "development_heartbeat.json", stale)
    row = module.summarize_evaluation(
        module.EVALUATION_SPECS[0], datetime.now(timezone.utc)
    )
    assert row is not None
    assert row["state"] == "EXITED_UNVERIFIED"


def test_evaluation_done_requires_all_atomic_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(
        module,
        "proc_identity",
        lambda pid: {"pid": pid, "live": False, "error": "process not present"},
    )
    monkeypatch.setattr(module, "tmux_session_exists", lambda name: False)
    _write(tmp_path / "pre_dev_freeze.json", {"status": "PASS"})
    _write(
        tmp_path / "development_launch.json",
        {
            "status": "LAUNCHING",
            "session": "s43_pi2n_dev",
            "physical_gpu_ids": [1, 2],
            "physical_gpu_uuids": ["GPU-a", "GPU-b"],
        },
    )
    done = {
        "state": "DONE",
        "exit_code": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "session": "s43_pi2n_dev",
        "supervisor_pid": 123,
        "physical_gpu_ids": [1, 2],
        "physical_gpu_uuids": ["GPU-a", "GPU-b"],
    }
    _write(tmp_path / "development_job_status.json", done)
    _write(tmp_path / "development_heartbeat.json", done)
    _write(
        tmp_path / "development_results.json",
        {"status": "PASS", "total_canonical_outcomes": 90},
    )
    _write(
        tmp_path / "vac_star_selection.json",
        {"status": "PASS", "selected_vac_star": "B_VAC_V"},
    )
    _write(
        tmp_path / "development_gpu_execution.json",
        {"status": "PASS", "maximum_heavy_workers": 2},
    )
    row = module.summarize_evaluation(
        module.EVALUATION_SPECS[0], datetime.now(timezone.utc)
    )
    assert row is not None
    assert row["state"] == "COMPLETE_VERIFIED"

    (tmp_path / "vac_star_selection.json").unlink()
    row = module.summarize_evaluation(
        module.EVALUATION_SPECS[0], datetime.now(timezone.utc)
    )
    assert row is not None
    assert row["state"] == "EXITED_UNVERIFIED"


def test_candidate_audit_status_is_read_only_and_identity_checked(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    artifacts = tmp_path / "artifacts"
    jobs = artifacts / "candidate_completion_audit_jobs"
    attempt = jobs / "b_va27" / "attempt_001"
    attempt.mkdir(parents=True)
    monkeypatch.setattr(module, "ARTIFACTS", artifacts)
    monkeypatch.setattr(module, "AUDIT_JOB_ROOT", jobs)
    monkeypatch.setattr(
        module,
        "proc_identity",
        lambda pid: {
            "pid": pid,
            "live": True,
            "cmdline": (
                "python launch_s4_3_pi2n_candidate_audit.py supervise "
                "--model-id B_VA27 --attempt-id attempt_001"
            ),
        },
    )
    monkeypatch.setattr(module, "tmux_session_exists", lambda name: True)
    session = "s43_pi2n_audit_b_va27_001"
    _write(
        attempt / "launch.json",
        {
            "status": "LAUNCHING",
            "model_id": "B_VA27",
            "attempt_id": "attempt_001",
            "session": session,
        },
    )
    running = {
        "state": "RUNNING",
        "exit_code": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": "B_VA27",
        "attempt_id": "attempt_001",
        "session": session,
        "supervisor_pid": 123,
    }
    _write(attempt / "job_status.json", running)
    _write(attempt / "heartbeat.json", running)
    rows = module.summarize_candidate_audits(datetime.now(timezone.utc))
    assert len(rows) == 1
    assert rows[0]["state"] == "RUNNING"
    assert rows[0]["runtime_identity_matches"] is True
    assert rows[0]["performance_values_read"] is False

    stale = dict(running)
    stale["updated_at"] = "2000-01-01T00:00:00+00:00"
    _write(attempt / "heartbeat.json", stale)
    rows = module.summarize_candidate_audits(datetime.now(timezone.utc))
    assert rows[0]["state"] == "EXITED_UNVERIFIED"
