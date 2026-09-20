from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
