from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/simulation/audit_s4_3_pi2n_candidate_completion.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_s4_3_pi2n_candidate_completion", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_contracts_are_exact_and_bounded() -> None:
    module = load_module()
    assert set(module.CANDIDATES) == {"B_VA27", "B_VAC_V"}
    va = module.CANDIDATES["B_VA27"]
    vac = module.CANDIDATES["B_VAC_V"]
    assert va.mode_name == "VA_PHYSICAL_AUX"
    assert not va.contact_adapter_expected
    assert vac.mode_name == "CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX"
    assert vac.contact_adapter_expected
    assert (va.physical_gpu_id, vac.physical_gpu_id) == (1, 3)


def test_checkpoint_tree_hash_binds_path_content_and_size(tmp_path: Path) -> None:
    module = load_module()
    checkpoint = tmp_path / "29999"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "train_state").mkdir()
    (checkpoint / "params/a").write_bytes(b"same")
    (checkpoint / "train_state/b").write_bytes(b"state")
    first = module.checkpoint_hashes(checkpoint)
    second = module.checkpoint_hashes(checkpoint)
    assert first == second
    assert first[0] != first[1]
    (checkpoint / "params/a").write_bytes(b"changed")
    changed = module.checkpoint_hashes(checkpoint)
    assert changed[0] != first[0]
    assert changed[1] != first[1]


def test_log_parser_requires_actual_progress_and_periodic_metrics() -> None:
    module = load_module()
    text = """
Step 0: grad_norm=1.0, loss=0.5, lambda_phys=0.03
Progress on: 1.00kit/30.0kit rate:4.5s/it remaining:36:00:00 elapsed:01:00:00
Step 100: grad_norm=0.9, loss=0.4, lambda_phys=0.03
Progress on: 30.0kit/30.0kit rate:4.0s/it remaining:00:00:00 elapsed:33:20:00
"""
    parsed = module.parse_training_log(text)
    assert [row["step"] for row in parsed["metric_rows"]] == [0, 100]
    assert parsed["progress_steps"] == [1000, 30000]
    assert parsed["rates_seconds_per_step"] == [4.5, 4.0]
    assert parsed["elapsed"] == "33:20:00"
    assert module.elapsed_seconds(parsed["elapsed"]) == 120_000


def test_completion_audit_is_non_overwriting_and_performance_blind() -> None:
    source = SCRIPT.read_text()
    assert "refusing to overwrite completed" in source
    assert "TRAINING_NOT_DONE_EXIT_ZERO" in source
    assert "TRAINING_PROCESS_STILL_LIVE" in source
    assert '"policy_evaluation_performed": False' in source
    assert '"dev_and_final_performance_still_unseen"' in source
    assert "restored_train_state_step_30000" in source
    assert "checkpoint_steps_exact" in source
    assert "protected_policy_checkpoints_unchanged" in source
    assert "candidate_completion_attempts" in source
    assert 'if status == "PASS"' in source
    assert "COMPLETION_AUDIT_FAIL_PRESERVED" in source


def test_failed_audit_attempts_are_append_only_and_leave_canonical_free(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)
    spec = module.CANDIDATES["B_VA27"]
    first = module.reserve_failed_audit_attempt(spec)
    second = module.reserve_failed_audit_attempt(spec)
    assert first.name == "attempt_001"
    assert second.name == "attempt_002"
    assert first.is_dir() and second.is_dir()
    assert not (tmp_path / "b_va27_training_completion.json").exists()
    assert not (tmp_path / "b_va27_checkpoint_manifest.json").exists()
