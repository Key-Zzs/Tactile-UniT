from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_runtime_determinism_repair.json"
SCRIPT = ROOT / "scripts/simulation/run_s4_3_pi2n_determinism_repair.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_s4_3_pi2n_determinism_repair", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_is_fixed_input_only_and_does_not_expand_rollout_budget() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["scope"]["engineering_fix_index"] == 1
    assert protocol["scope"]["maximum_engineering_fixes"] == 2
    assert protocol["scope"]["task_rollouts"] == 0
    assert protocol["scope"]["models"] == ["B1", "B_HVA", "B2"]
    assert protocol["scope"]["fresh_server_repeats_per_model"] == 3
    assert protocol["runtime_only_change"]["checkpoint_change"] is False
    assert protocol["runtime_only_change"]["policy_rng_change"] is False
    assert protocol["runtime_only_change"]["sampling_algorithm_change"] is False
    assert protocol["runtime_only_change"]["evaluator_change"] is False


def test_fixture_payload_and_hashes_are_stable_and_sensitive() -> None:
    module = load_module()
    first = module.fixture_payload()
    second = module.fixture_payload()
    assert module.canonical_payload_sha256(first) == module.canonical_payload_sha256(second)
    assert first["base"].shape == (640, 640, 3)
    assert first["wrist"].shape == (640, 640, 3)
    assert first["state"].shape == (23,)
    assert first["contact_state"].shape == (256,)
    changed = dict(second)
    changed["state"] = second["state"].copy()
    changed["state"][0] += 1
    assert module.canonical_payload_sha256(first) != module.canonical_payload_sha256(changed)
    action = np.zeros((30, 22), dtype=np.float32)
    assert module.action_sha256(action) == module.action_sha256(action.copy())
    action[0, 0] = 1
    assert module.action_sha256(action) != module.action_sha256(np.zeros((30, 22), dtype=np.float32))


def test_repair_requires_exact_action_hash_and_non_overwrite() -> None:
    source = SCRIPT.read_text()
    assert "same_float32_action_sha256_within_each_model" in source
    assert "refusing to overwrite or resume the determinism repair fixture" in source
    assert "--xla_gpu_deterministic_ops=true" in source
    assert "--xla_gpu_exclude_nondeterministic_ops=true" in source
    assert "--xla_gpu_autotune_level=0" in source
    assert '"task_rollouts": 0' in source
