from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_runtime_audit_protocol.json"
SCRIPT = ROOT / "scripts/simulation/run_s4_3_pi2n_runtime_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_s4_3_pi2n_runtime_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_audit_budget_and_exposure_are_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    cohort = protocol["cohort"]
    assert protocol["status"] == "FROZEN_BEFORE_RUNTIME_AUDIT"
    assert cohort["models"] == ["B1", "B_HVA", "B2"]
    assert cohort["evaluator_seed"] == 8
    assert cohort["episodes_per_replicate"] == 10
    assert cohort["replicates"] == 3
    assert cohort["maximum_rollouts"] == 90
    assert cohort["scientific_performance_role"] is False
    assert sorted(cohort["execution_order"]) == ["replicate_1", "replicate_2", "replicate_3"]
    assert all(sorted(models) == sorted(cohort["models"]) for models in cohort["execution_order"].values())


def test_runtime_sampling_and_retry_limits_are_explicit() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["randomness"]["per_episode_policy_rng_reset"] is False
    assert "jax.random.key(0)" in protocol["randomness"]["policy_sampling"]
    assert protocol["retry"]["native_failure_or_timeout_in_denominator"] is True
    assert protocol["retry"]["best_of_or_per_reset_splicing"] is False
    assert protocol["runtime"]["fresh_policy_server_each_model_replicate"] is True
    assert protocol["fixed_observation_fixture"]["interpretation"].startswith("tests fixed-input")


def test_payload_and_action_hashes_are_stable_and_sensitive() -> None:
    module = load_module()
    payload = {
        "state": np.arange(23, dtype=np.float32),
        "base": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "prompt": "pinch",
    }
    reordered = {"prompt": payload["prompt"], "base": payload["base"], "state": payload["state"]}
    assert module.canonical_payload_sha256(payload) == module.canonical_payload_sha256(reordered)
    changed = dict(payload)
    changed["state"] = payload["state"].copy()
    changed["state"][0] += 1
    assert module.canonical_payload_sha256(payload) != module.canonical_payload_sha256(changed)
    action = np.zeros((30, 22), dtype=np.float32)
    assert module.action_sha256(action) == module.action_sha256(action.copy())
    action[0, 0] = 1
    assert module.action_sha256(action) != module.action_sha256(np.zeros((30, 22), dtype=np.float32))


def test_runtime_runner_is_non_overwriting_and_not_formal_selection() -> None:
    source = SCRIPT.read_text()
    assert "refusing to overwrite or resume a PI2N runtime audit output" in source
    assert "runtime.gpu_is_idle" in source
    assert "runtime.acquire_gpu_lock" in source
    assert '"candidate_policy_results_read": False' in source
    assert '"formal_results_read": False' in source
    assert "PI2N_RUNTIME_AUDIT_BLOCKS_SCIENTIFIC_ROLLOUTS" in source
    assert "serve_s4_3_pi2m_policy.py" in source
    assert "recover_s4_3_pi2m_b2_evaluation.py" not in source
