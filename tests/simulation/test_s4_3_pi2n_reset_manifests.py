from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_evaluation_protocol.json"
SCRIPT = ROOT / "scripts/simulation/freeze_s4_3_pi2n_reset_manifests.py"


def load_module():
    spec = importlib.util.spec_from_file_location("freeze_s4_3_pi2n_reset_manifests", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reset_blocks_match_the_frozen_evaluation_protocol() -> None:
    module = load_module()
    protocol = json.loads(PROTOCOL.read_text())
    assert module.DEV_BLOCKS == ((9, 10), (10, 10), (11, 10))
    assert module.FINAL_BLOCKS == ((12, 50), (13, 50), (14, 50), (15, 50))
    assert protocol["cohorts"]["PI2N_DEV"]["resets"] == sum(
        count for _, count in module.DEV_BLOCKS
    )
    assert protocol["cohorts"]["PI2N_FINAL"]["resets"] == sum(
        count for _, count in module.FINAL_BLOCKS
    )


def test_reset_identity_hash_is_stable_sensitive_and_ordered() -> None:
    module = load_module()
    state = np.arange(19, dtype=np.float64)
    processed = np.arange(23, dtype=np.float64)
    assert module.sha256_array(state, processed) == module.sha256_array(
        state.copy(), processed.copy()
    )
    changed = processed.copy()
    changed[0] += 1
    assert module.sha256_array(state, processed) != module.sha256_array(state, changed)
    assert module.sha256_array(state, processed) != module.sha256_array(processed, state)
    identities = [module.sha256_array(state), module.sha256_array(processed)]
    assert module.sequence_sha256(identities) != module.sequence_sha256(identities[::-1])


def test_freezer_is_non_overwriting_and_policy_free() -> None:
    source = SCRIPT.read_text()
    assert "refusing to overwrite PI2N reset manifests or exposure ledger" in source
    assert "runtime_seed8_generator_reproduces_frozen_identity_sequence" in source
    assert "dev_disjoint_all_historical_raw_rollouts" in source
    assert "final_disjoint_all_historical_raw_rollouts" in source
    assert "scientific_policy_inference_performed" in source
    assert "policy_config" not in source
    assert "sample_actions" not in source
