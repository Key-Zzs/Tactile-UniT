from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gr00t.tactile_unit.vac_transition_contract import (
    ActionTransitionTarget,
    FutureOracleLeakageError,
    ModalityAvailability,
    OfflineVACTransitionTeachers,
    OnlineCausalContext,
    PredictedOrPlannedActionTransition,
    TransitionAnchor,
    VACContractError,
    reject_online_oracles,
    validate_integrated_manifest_row,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/tactile_unit/integration_continuous_vac_contract.json"


def values(batch: int = 2) -> dict[str, np.ndarray]:
    return {
        "z_v": np.zeros((batch, 8, 32), dtype=np.float32),
        "z_a": np.ones((batch, 8, 32), dtype=np.float32),
        "z_c": np.full((batch, 8, 32), 2, dtype=np.float32),
        "h_t_c": np.zeros((batch, 256), dtype=np.float32),
        "state": np.zeros((batch, 128), dtype=np.float32),
        "action": np.zeros((batch, 16, 128), dtype=np.float32),
    }


def offline_batch(batch: int = 2, **overrides: object) -> OfflineVACTransitionTeachers:
    arguments: dict[str, object] = {
        "pair_id": [f"pair-{index}" for index in range(batch)],
        "episode_id": np.arange(batch),
        "t": np.full(batch, 15),
        "t_future": np.full(batch, 31),
        **values(batch),
        "modality_masks": {
            "vision": np.ones(batch, dtype=bool),
            "action": np.ones(batch, dtype=bool),
            "contact": np.ones(batch, dtype=bool),
        },
        "provenance": {"checkpoint": "immutable digest"},
    }
    arguments.update(overrides)
    return OfflineVACTransitionTeachers(**arguments)  # type: ignore[arg-type]


def test_integration_branch_contains_both_accepted_lineages() -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    assert branch in {"develop/tactile-unit-integration", "develop/tactile-unit-vac"}
    if branch == "develop/tactile-unit-vac":
        assert subprocess.run(
            ["git", "merge-base", "--is-ancestor", "develop/tactile-unit-integration", "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode == 0
    for branch in (
        "origin/develop/contact-semantic-tokenizer",
        "origin/develop/continuous-contact-bridge",
        "origin/develop/tactile-action-bootstrap",
        "origin/develop/action-transition-remediation",
    ):
        assert subprocess.run(
            ["git", "merge-base", "--is-ancestor", branch, "HEAD"], cwd=ROOT
        ).returncode == 0


def test_transition_anchor_is_exactly_k16() -> None:
    assert TransitionAnchor("pair", 1, 15, 31).t_future == 31
    with pytest.raises(VACContractError, match=r"t -> t\+16"):
        TransitionAnchor("pair", 1, 15, 32)


def test_offline_teachers_validate_same_batch_shapes_dtype_and_order() -> None:
    batch = offline_batch()
    assert batch.batch_size == 2
    assert batch.z_v.shape == batch.z_a.shape == batch.z_c.shape == (2, 8, 32)
    with pytest.raises(VACContractError, match="same batch length"):
        offline_batch(z_a=np.zeros((1, 8, 32), dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        offline_batch(z_c=np.zeros((2, 8, 32), dtype=np.float64))


def test_offline_teachers_reject_nonfinite_and_duplicate_pairs() -> None:
    invalid = values()["z_v"]
    invalid[0, 0, 0] = np.nan
    with pytest.raises(VACContractError, match="non-finite"):
        offline_batch(z_v=invalid)
    with pytest.raises(VACContractError, match="unique"):
        offline_batch(pair_id=["same", "same"])


def test_missing_modality_masks_are_explicit_not_zero_inferred() -> None:
    masks = {
        "vision": np.asarray([True, True]),
        "action": np.asarray([True, False]),
        "contact": np.asarray([False, True]),
    }
    batch = offline_batch(modality_masks=masks)
    assert not batch.modality_masks["contact"][0]
    with pytest.raises(VACContractError, match="vision, action, and contact"):
        offline_batch(modality_masks={"vision": [True, True]})
    with pytest.raises(TypeError, match="boolean"):
        offline_batch(
            modality_masks={"vision": [1, 1], "action": [1, 1], "contact": [1, 1]}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"z_c": "teacher"},
        {"observation": {"future_image": "oracle"}},
        {"nested": [{"teacher": {"z_v_target": "oracle"}}]},
        {"nested": ({"z_a": "demonstration teacher"},)},
        {"contact": {"h_t+16^c": "future"}},
    ],
)
def test_online_guard_rejects_top_level_and_nested_oracles(payload: object) -> None:
    with pytest.raises(FutureOracleLeakageError):
        reject_online_oracles(payload)
    reject_online_oracles(payload, oracle_eval=True)


def test_online_guard_accepts_current_and_policy_generated_fields() -> None:
    reject_online_oracles(
        {
            "current": {"image": "I_t", "h_t_c": "current tactile context"},
            "policy": {"z_hat_a": "planned", "z_hat_c": "predicted"},
        }
    )


def test_action_target_and_policy_plan_are_distinct_types() -> None:
    z = np.zeros((1, 8, 32), dtype=np.float32)
    assert ActionTransitionTarget(z).teacher_only
    assert PredictedOrPlannedActionTransition(z).policy_generated
    assert type(ActionTransitionTarget(z)) is not type(PredictedOrPlannedActionTransition(z))


def test_online_context_has_no_offline_teacher_slot() -> None:
    z = np.zeros((1, 8, 32), dtype=np.float32)
    context = OnlineCausalContext(
        current_visual_observation="I_t",
        robot_state=np.zeros((1, 128), dtype=np.float32),
        predicted_or_planned_action=PredictedOrPlannedActionTransition(z),
        modality_available=ModalityAvailability(True, True, False),
    )
    assert context.predicted_or_planned_action is not None
    assert "z_a" not in OnlineCausalContext.__dataclass_fields__
    with pytest.raises(FutureOracleLeakageError, match="not a teacher"):
        OnlineCausalContext(
            current_visual_observation="I_t",
            robot_state=np.zeros((1, 128), dtype=np.float32),
            predicted_or_planned_action=ActionTransitionTarget(z),  # type: ignore[arg-type]
        )
    with pytest.raises(FutureOracleLeakageError):
        OnlineCausalContext(
            current_visual_observation={"nested": {"future_image": "I_t+16"}},
            robot_state=np.zeros((1, 128), dtype=np.float32),
        )


def test_all_960_manifest_rows_have_exact_timing_and_pair_identity() -> None:
    config = json.loads(CONFIG.read_text())
    manifest_path = ROOT / config["canonical_data"]["paired_manifest"]
    if not manifest_path.is_file():
        pytest.skip("canonical local acceptance manifest is unavailable")
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == config["canonical_data"][
        "paired_manifest_sha256"
    ]
    manifest = json.loads(manifest_path.read_text())
    anchors = [validate_integrated_manifest_row(row) for row in manifest["rows"]]
    assert len(anchors) == len({anchor.pair_id for anchor in anchors}) == 960
    assert all(anchor.t_future == anchor.t + 16 for anchor in anchors)


def test_manifest_validator_rejects_action_off_by_one_and_contact_overlap() -> None:
    config = json.loads(CONFIG.read_text())
    manifest_path = ROOT / config["canonical_data"]["paired_manifest"]
    if not manifest_path.is_file():
        pytest.skip("canonical local acceptance manifest is unavailable")
    manifest = json.loads(manifest_path.read_text())
    row = json.loads(json.dumps(manifest["rows"][0]))
    row["action"]["episode_frames_inclusive"][1] += 1
    with pytest.raises(VACContractError, match="timing mismatch"):
        validate_integrated_manifest_row(row)
    row = json.loads(json.dumps(manifest["rows"][0]))
    row["contact"]["future_teacher_window_inclusive"][0] -= 1
    with pytest.raises(VACContractError, match="timing mismatch"):
        validate_integrated_manifest_row(row)


def test_config_freezes_canonical_continuous_paths_and_identities() -> None:
    config = json.loads(CONFIG.read_text())
    scope = config["scope"]
    assert not any(
        scope[key]
        for key in (
            "full_track_c_started",
            "m3_established",
            "training_allowed",
            "optimizer_allowed",
            "backward_allowed",
            "shared_rq_adaptation",
            "contact_rq",
            "contact_whitening",
        )
    )
    assert config["representations"]["action"]["interface"] == "continuous pre-RQ"
    assert config["representations"]["contact"]["interface"] == "native continuous S2 E_c output"
    assert config["representations"]["contact"]["rq"] is False
    assert config["representations"]["contact"]["whitening"] is False
    assert config["frozen_identity"]["old_action_rows_digest"] == (
        "e92ced68df2247c19dd99f5be8b165922fbcc06ffa0c27597999ebb4b54d803c"
    )


def test_m3_remains_unexecuted_and_supports_continuous_hybrid_gates() -> None:
    value = json.loads(
        (ROOT / "configs/tactile_unit/m3_continuous_vac_evaluation.json").read_text()
    )
    gates = " ".join(row["gate"].lower() for row in value["preregistered_gates"])
    assert value["status"] == "SPEC_ONLY_NOT_EXECUTED"
    assert value["same_codebook_required"] is False
    assert value["contact_interface"]["type"] == "continuous"
    for phrase in (
        "paired v-a",
        "paired v-c",
        "paired a-c",
        "retrieval",
        "cross-modal",
        "missing modalities",
        "temporal semantics",
        "offline-only",
    ):
        assert phrase in gates


def test_integration_auditor_contains_no_training_or_optimizer_path() -> None:
    source = (ROOT / "scripts/tactile_unit/audit_tactile_unit_integration.py").read_text()
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer_instantiated\": False" in source
    assert "training_performed\": False" in source


def test_integration_tracked_files_contain_no_private_paths_or_credentials() -> None:
    paths = (
        CONFIG,
        ROOT / "configs/tactile_unit/m3_continuous_vac_evaluation.json",
        ROOT / "gr00t/tactile_unit/vac_transition_contract.py",
        ROOT / "scripts/tactile_unit/audit_tactile_unit_integration.py",
        ROOT / "docs/research/tactile_unit_integration.md",
        Path(__file__),
    )
    forbidden = ("/" + "home/", "/" + "mnt/", "Bear" + "er ", "Author" + "ization:")
    for path in paths:
        text = path.read_text()
        assert not any(value in text for value in forbidden), path
