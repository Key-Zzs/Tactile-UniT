from __future__ import annotations

import json
from pathlib import Path

from gr00t.simulation.s4_3_pi1 import TactileUnitMode


ROOT = Path(__file__).resolve().parents[2]


def test_pi2n_budget_and_dag_are_frozen() -> None:
    dag = json.loads((ROOT / "configs/simulation/s4_3_pi2n_dag.json").read_text())
    assert dag["global_limits"]["new_policy_runs_max"] == 3
    assert dag["global_limits"]["physical_gpus_max"] == 4
    assert dag["global_limits"]["pi2b_training"] is False
    assert dag["nodes"]["N3_C"]["depends_on"] == ["N1_D"]
    assert dag["nodes"]["N4_V"]["depends_on"] == ["N1_D", "N2_T"]


def test_vac_v_target_identity_is_exact_plus_27_and_nas_resident() -> None:
    protocol = json.loads((ROOT / "configs/simulation/s4_3_pi2n_vac_v_target_protocol.json").read_text())
    target = protocol["target"]
    assert protocol["status"] == "FROZEN_BEFORE_TARGET_BUILD"
    assert target["name"] == "u_v^VAC"
    assert target["future_row_offset"] == 27
    assert target["physical_horizon_seconds"] == 0.54
    assert target["output"].startswith("$PI2N_RUN_ROOT/caches/")
    assert target["allowed_fields"] == ["index", "vac_vision_target", "vac_aux_valid"]
    assert protocol["teacher"]["bridge_role"] == "B3"
    assert protocol["teacher"]["bridge_checkpoint_candidate"] == "B3_slot_t0.07_w1"


def test_new_mode_and_training_fields_are_explicit() -> None:
    assert TactileUnitMode.CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX.value == "CONTACT_STATE_TOKENS_VAC_V_PHYSICAL_AUX"
    source = (ROOT / "gr00t/simulation/pi05_tactile_unit.py").read_text()
    assert "vac_vision_target" in source
    assert "vac_aux_valid" in source
    assert 'raise ValueError("training-only vac_vision_target is forbidden during inference")' in source
    branch = source.split("def _physical_loss", 1)[1].split("def compute_physical_auxiliary_loss", 1)[0]
    assert "observation.vac_vision_target, observation.vac_aux_valid" in branch


def test_b_va27_and_b_vac_v_are_independent_base_runs() -> None:
    protocol = json.loads((ROOT / "configs/simulation/s4_3_pi2n_candidate_protocol.json").read_text())
    trainer = (ROOT / "scripts/simulation/train_s4_3_pi2n.py").read_text()
    assert protocol["common_training"]["initialization"] == "exact pi05_base independently for every run"
    assert protocol["common_training"]["training_seed"] == 42
    assert protocol["common_training"]["optimizer_steps"] == 30_000
    assert protocol["common_training"]["global_batch_size"] == 32
    assert protocol["B_VA27"]["online_contact_history"] is False
    assert protocol["B_VAC_V"]["comparison_level"] == "PACKAGE_LEVEL_TEACHER_COMPARISON_ONLY"
    assert "seed=42" in trainer
    assert "batch_size=32" in trainer
    assert "num_train_steps=30_000" in trainer
    assert "resume=False" in trainer
    assert "overwrite=False" in trainer


def test_refinement_route_is_frozen_before_results() -> None:
    diagnostics = json.loads((ROOT / "configs/simulation/s4_3_pi2n_diagnostics.json").read_text())
    route = diagnostics["refinement_route"]
    assert diagnostics["status"] == "FROZEN_BEFORE_NUMERICAL_DIAGNOSTICS"
    assert route["priority"] == ["BLOCKED_TEACHER", "C_REBALANCE", "C_ORDER", "C_JOINT", "NOT_RUN_NOT_JUSTIFIED"]
    assert route["single_route_only"] is True
    assert route["closed_loop_candidate_results_forbidden"] is True


def test_supervisor_requires_gpu_identity_and_never_resumes() -> None:
    source = (ROOT / "scripts/simulation/supervise_s4_3_pi2n_training.sh").read_text()
    assert "expected_uuid" in source
    assert "flock -n" in source
    assert "memory_used > 64 || utilization > 5" in source
    assert "--fsdp-devices 1" in source
    trainer = (ROOT / "scripts/simulation/train_s4_3_pi2n.py").read_text()
    assert "resume=False" in trainer
    assert "overwrite=False" in trainer
