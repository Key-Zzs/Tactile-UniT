from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.simulation import launch_s4_3_pi2n_candidate_audit as launch


def test_candidate_audit_launcher_scope_is_exact_and_cpu_only() -> None:
    assert set(launch.CANDIDATES) == {"B_VA27", "B_VAC_V"}
    assert launch.SESSION_PREFIX == "s43_pi2n_audit_"
    assert launch.AUDIT_PYTHON == Path(
        "/home/wbcd/miniconda3/envs/openpi/bin/python"
    )
    source = Path(launch.__file__).read_text()
    assert '"CUDA_VISIBLE_DEVICES": ""' in source
    assert '"JAX_PLATFORMS": "cpu"' in source
    assert '"gpu_required": False' in source
    assert '"policy_inference": False' in source
    assert '"training_or_evaluation_launched": False' in source
    assert '"automatic_pre_dev_or_followup": False' in source
    assert "audit_runtime_identity()" in source
    assert '[str(AUDIT_PYTHON), str(AUDITOR)' in source
    assert "freeze_s4_3_pi2n_development.py" not in source
    assert "launch_s4_3_pi2n_development.py" not in source


def test_candidate_audit_job_attempts_are_append_only(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(launch, "JOB_ROOT", tmp_path)
    spec = launch.CANDIDATES["B_VAC_V"]
    first = launch.reserve_attempt(spec)
    second = launch.reserve_attempt(spec)
    assert first.name == "attempt_001"
    assert second.name == "attempt_002"
    assert first.is_dir() and second.is_dir()


def test_attempt_id_validation_cannot_escape_job_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(launch, "JOB_ROOT", tmp_path)
    spec = launch.CANDIDATES["B_VA27"]
    valid = launch.reserve_attempt(spec)
    assert launch.attempt_dir(spec, valid.name) == valid
    for invalid in ("../attempt_001", "attempt_1", "/tmp/attempt_001"):
        try:
            launch.attempt_dir(spec, invalid)
        except SystemExit as error:
            assert "invalid" in str(error)
        else:
            raise AssertionError(f"unsafe attempt id accepted: {invalid}")


def test_audit_runtime_dependency_failure_is_preflight_error(
    tmp_path: Path, monkeypatch
) -> None:
    python = tmp_path / "python"
    python.write_text("")
    monkeypatch.setattr(launch, "AUDIT_PYTHON", python)
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'jax'",
        ),
    )
    try:
        launch.audit_runtime_identity()
    except SystemExit as error:
        assert "dependency probe failed" in str(error)
        assert "No module named 'jax'" in str(error)
    else:
        raise AssertionError("missing model-runtime dependency was accepted")
