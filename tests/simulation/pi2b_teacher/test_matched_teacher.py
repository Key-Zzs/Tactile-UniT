from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.simulation.pi2b_teacher import (
    MatchLossWeights,
    StrictFieldStore,
    build_matched_pair,
    clip_active_grad_norm_,
    matched_teacher_loss,
)
from gr00t.simulation.pi2b_teacher.matched_teacher import common_state
from scripts.simulation.pi2b_teacher.launch_confirmation import resolve_dexjoco_python

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs/simulation/pi2b_teacher"


def inputs(count: int = 8):
    generator = torch.Generator().manual_seed(7)
    native = {
        "vision": torch.randn(count, 8, 32, generator=generator),
        "action": torch.randn(count, 8, 32, generator=generator),
        "contact": torch.randn(count, 8, 32, generator=generator),
    }
    episode = torch.arange(count)
    return native, episode


def test_common_structure_and_initialization_are_byte_equal() -> None:
    va, vac = build_matched_pair(42)
    left, right = common_state(va), common_state(vac)
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def test_contact_disabled_forward_loss_grad_and_update_are_exact() -> None:
    native, episode = inputs()
    va, vac = build_matched_pair(42)
    va_native = {name: native[name] for name in ("vision", "action")}
    va_loss, va_parts = matched_teacher_loss(
        va, va_native, episode, temperature=0.07, weights=MatchLossWeights()
    )
    vac_loss, vac_parts = matched_teacher_loss(
        vac,
        native,
        episode,
        temperature=0.07,
        weights=replace(MatchLossWeights(), contact_pair=0.0, contact_native=0.0, contact_relational=0.0, contact_variance=0.0),
    )
    assert torch.equal(va_loss, vac_loss)
    assert torch.equal(va_parts["va_total"], vac_parts["va_total"])
    left_grad = torch.autograd.grad(va_loss, tuple(va.parameters()))
    common_vac = [parameter for name, parameter in vac.named_parameters() if name in common_state(vac)]
    right_grad = torch.autograd.grad(vac_loss, common_vac)
    assert all(torch.equal(left, right) for left, right in zip(left_grad, right_grad))
    left_optimizer = torch.optim.AdamW(va.parameters(), lr=3e-4, weight_decay=1e-4)
    right_optimizer = torch.optim.AdamW(vac.parameters(), lr=3e-4, weight_decay=1e-4)
    left_optimizer.zero_grad(set_to_none=True)
    right_optimizer.zero_grad(set_to_none=True)
    va_loss, _ = matched_teacher_loss(
        va, va_native, episode, temperature=0.07, weights=MatchLossWeights()
    )
    vac_loss, _ = matched_teacher_loss(
        vac,
        native,
        episode,
        temperature=0.07,
        weights=replace(MatchLossWeights(), contact_pair=0.0, contact_native=0.0, contact_relational=0.0, contact_variance=0.0),
    )
    va_loss.backward()
    vac_loss.backward()
    assert torch.equal(clip_active_grad_norm_(va.parameters(), 1.0), clip_active_grad_norm_(vac.parameters(), 1.0))
    left_optimizer.step()
    right_optimizer.step()
    assert all(
        torch.equal(common_state(va)[name], common_state(vac)[name])
        for name in common_state(va)
    )


def test_contact_is_additive_and_reaches_common_path_only_when_enabled() -> None:
    native, episode = inputs()
    _, vac = build_matched_pair(42)
    total, parts = matched_teacher_loss(
        vac, native, episode, temperature=0.07, weights=MatchLossWeights()
    )
    assert torch.equal(total, parts["va_total"] + parts["contact_total"])
    assert parts["contact_total"].item() > 0.0
    shared_slot_gradient = torch.autograd.grad(parts["contact_total"], vac.shared_slots)[0]
    assert torch.count_nonzero(shared_slot_gradient)
    contact_parameters = [
        parameter
        for name, parameter in vac.named_parameters()
        if name.startswith(("projectors.contact.", "recovery.contact."))
    ]
    gradients = torch.autograd.grad(parts["va_total"], contact_parameters, allow_unused=True)
    assert all(value is None for value in gradients)


def test_va_store_physically_denies_contact_fields(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    np.savez(path, pair_id=np.asarray(["x"]), z_v=np.zeros((1, 8, 32)), z_a=np.zeros((1, 8, 32)), z_c=np.ones((1, 8, 32)))
    store = StrictFieldStore(path, {"pair_id", "z_v", "z_a"})
    values = store.load()
    assert set(values) == {"pair_id", "z_v", "z_a"}
    with pytest.raises(PermissionError):
        store.read("z_c")
    assert "z_c" not in store.accessed_fields


def test_protocol_is_exactly_one_preregistered_pair_and_no_policy() -> None:
    protocol = json.loads((CONFIG_ROOT / "protocol.json").read_text())
    assert protocol["models"] == {
        "T_VA_match": ["vision", "action"],
        "T_VAC_match": ["vision", "action", "contact"],
    }
    assert protocol["training"] == {
        "steps": 800,
        "batch_size": 512,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip_global": 1.0,
        "schedule": "constant",
        "checkpoint": "final_step_only_canonical",
        "batch_sampling": "numpy Generator.choice without replacement",
        "batch_rng_seed": 42,
    }
    assert protocol["scope"]["teacher_runs"] == 2
    assert protocol["scope"]["teacher_retrains"] == 0
    assert protocol["scope"]["architecture_sweeps"] == 0
    assert protocol["scope"]["policy_training"] is False
    assert protocol["scope"]["pi05_training"] is False
    assert protocol["confirmation"]["source_group_indices"] == [23, 24, 25]


def test_confirmation_interpreter_is_runtime_derived(tmp_path: Path) -> None:
    envs = tmp_path / "conda/envs"
    unit_python = envs / "unit/bin/python"
    dex_python = envs / "tactile-unit-dexjoco/bin/python"
    unit_python.parent.mkdir(parents=True)
    dex_python.parent.mkdir(parents=True)
    unit_python.touch()
    dex_python.touch()
    assert resolve_dexjoco_python({"python": str(unit_python)}) == dex_python
    assert resolve_dexjoco_python(
        {"python": "not-used", "dexjoco_python": str(dex_python)}
    ) == dex_python
