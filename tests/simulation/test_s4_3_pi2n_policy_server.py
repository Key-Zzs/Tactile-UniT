from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/simulation/serve_s4_3_pi2n_policy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("serve_s4_3_pi2n_policy", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_cohort_and_runtime_modes_are_explicit() -> None:
    module = load_module()
    assert tuple(module.CHECKPOINTS) == (
        "B0",
        "B_VA27",
        "B1",
        "B_HVA",
        "B2",
        "B_VAC_V",
    )
    assert module.RUNTIME_MODES == {
        "B0": "NONE",
        "B_VA27": "NONE",
        "B1": "CONTACT_STATE_TOKENS",
        "B_HVA": "CONTACT_STATE_TOKENS",
        "B2": "CONTACT_STATE_TOKENS",
        "B_VAC_V": "CONTACT_STATE_TOKENS",
    }


def test_new_candidate_paths_are_final_only_on_audited_run_root() -> None:
    module = load_module()
    assert module.CHECKPOINTS["B_VA27"].relative_to(module.PI2N_RUN_ROOT).as_posix() == (
        "runs/B_VA27/pinch_tongs/s43_pi2n_b_va27_seed42/29999"
    )
    assert module.CHECKPOINTS["B_VAC_V"].relative_to(module.PI2N_RUN_ROOT).as_posix() == (
        "runs/B_VAC_V/pinch_tongs/s43_pi2n_b_vac_v_seed42/29999"
    )


def test_production_server_removes_training_only_targets() -> None:
    source = SCRIPT.read_text()
    assert "data=official.data" in source
    assert "mode=TactileUnitMode.CONTACT_STATE_TOKENS" in source
    assert "target_sidecar_path=None" in source
    assert "serve_forever" in source
    assert "resume=" not in source
    assert "PI2B" not in source
