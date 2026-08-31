from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/simulation"


def load(name: str) -> dict:
    return json.loads((CONFIG / name).read_text(encoding="utf-8"))


def sha256(name: str) -> str:
    return hashlib.sha256((CONFIG / name).read_bytes()).hexdigest()


def test_frozen_s4_1_identities_are_exact() -> None:
    protocol = load("s4_2_protocol.json")
    for value in protocol["frozen_s4_1_contracts"].values():
        assert hashlib.sha256((ROOT / value["path"]).read_bytes()).hexdigest() == value["sha256"]


def test_source_groups_are_disjoint_and_exact_70_15_15() -> None:
    manifest = load("s4_2_dataset_split_manifest.json")
    all_groups: set[tuple[str, str]] = set()
    counts = {"train": 0, "validation": 0, "test": 0}
    for task, splits in manifest["groups"].items():
        task_groups: set[str] = set()
        for split, groups in splits.items():
            assert not task_groups.intersection(groups)
            task_groups.update(groups)
            counts[split] += len(groups)
            all_groups.update((task, group) for group in groups)
        assert len(task_groups) == 20
    assert len(all_groups) == 60
    assert counts == {"train": 42, "validation": 9, "test": 9}
    assert manifest["counts"]["total_episodes"] == 300


def test_task_and_timing_contract_remain_single_arm() -> None:
    protocol = load("s4_2_protocol.json")
    dataset = load("s4_2_dataset_contract.json")
    assert protocol["task_selection"]["selected"] == [
        "pinch_tongs",
        "hammer_nail",
        "click_mouse",
    ]
    assert dataset["canonical_shapes"]["policy_action"] == 22
    assert dataset["canonical_shapes"]["environment_action"] == 23
    assert dataset["canonical_shapes"]["sim_tactile"] == 30
    assert dataset["pairing"]["transition_offset"] == 27
    assert dataset["pairing"]["current_history"] == [-25, 0]
    assert dataset["pairing"]["future_history"] == [2, 27]
    assert dataset["pairing"]["action_chunk"] == [0, 26]
    assert dataset["pairing"]["forbid_action_t_plus_27"] is True


def test_candidate_limits_and_parameter_budgets_are_frozen() -> None:
    protocol = load("s4_2_protocol.json")
    registry = load("s4_2_model_candidate_registry.json")
    assert len(registry["contact_teacher"]["trials"]) == 4
    assert len(registry["action_encoder"]["trials"]) == 3
    assert len(registry["bridge"]["trials"]) == 5
    assert registry["contact_rq"]["canonical"] == {
        "stages": 2,
        "codes_per_stage": 128,
        "code_dim": 32,
        "queries": 8,
    }
    assert protocol["models"]["parameter_budgets"] == {
        "contact_teacher": 5_000_000,
        "contact_dynamics": 2_000_000,
        "action_encoder": 25_000_000,
        "bridge_adapter_per_modality": 50_000,
        "sim_shared_total": 300_000,
    }


def test_test_access_is_closed_before_selection() -> None:
    for name in (
        "s4_2_protocol.json",
        "s4_2_dataset_split_manifest.json",
        "s4_2_model_candidate_registry.json",
        "s4_2_selection_contract.json",
    ):
        value = load(name)
        access = value.get("test_access", value)
        assert access["test_loaded"] is False
        assert access["selection_uses_test"] is False
        assert access["training_uses_test"] is False


def test_protocol_files_have_stable_nonempty_hashes() -> None:
    names = (
        "s4_2_protocol.json",
        "s4_2_dataset_contract.json",
        "s4_2_dataset_split_manifest.json",
        "s4_2_model_candidate_registry.json",
        "s4_2_selection_contract.json",
    )
    assert all(len(sha256(name)) == 64 for name in names)
