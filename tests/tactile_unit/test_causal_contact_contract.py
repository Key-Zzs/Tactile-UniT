from __future__ import annotations

import numpy as np
import pytest

from gr00t.tactile_unit.causal_contact_contract import (
    ContactBridgeBatch,
    ContactMode,
    ContactTransitionTarget,
    CurrentContactContext,
    FutureContactLeakageError,
    PredictedContactTransition,
    VisionTransitionTarget,
    reject_future_oracles,
    runtime_contact_batch,
)


def values(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_explicit_contact_roles_and_shapes() -> None:
    current = CurrentContactContext(values((2, 256)))
    contact = ContactTransitionTarget(values((2, 8, 32)))
    vision = VisionTransitionTarget(values((2, 8, 32)))
    predicted = PredictedContactTransition(values((2, 8, 32)))
    assert current.inference_available
    assert contact.teacher_only and vision.teacher_only
    assert predicted.inference_generated


def test_contract_rejects_wrong_shape_dtype_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="shape"):
        CurrentContactContext(values((2, 255)))
    with pytest.raises(TypeError, match="float32"):
        ContactTransitionTarget(np.zeros((2, 8, 32), dtype=np.float64))
    invalid = values((2, 8, 32))
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        VisionTransitionTarget(invalid)


def test_inference_rejects_true_future_transition_targets() -> None:
    batch = ContactBridgeBatch(
        CurrentContactContext(values((2, 256))),
        contact_target=ContactTransitionTarget(values((2, 8, 32))),
    )
    with pytest.raises(FutureContactLeakageError, match="future-derived"):
        batch.validate_for(ContactMode.INFERENCE)
    batch.validate_for(ContactMode.OFFLINE_TRAINING)
    batch.validate_for(ContactMode.ORACLE_EVALUATION)


@pytest.mark.parametrize(
    "payload",
    [
        {"z_c": values((1, 8, 32))},
        {"contact": {"h_future": values((1, 256))}},
        {"vision": {"goal_image": values((1, 3, 224, 224))}},
        {"contact_transition_target": values((1, 8, 32))},
    ],
)
def test_mapping_runtime_rejects_nested_oracle_fields(payload: dict) -> None:
    with pytest.raises(FutureContactLeakageError):
        reject_future_oracles(payload)
    reject_future_oracles(payload, mode=ContactMode.OFFLINE_TRAINING)
    reject_future_oracles(payload, mode=ContactMode.OFFLINE_EVALUATION)
    reject_future_oracles(payload, mode=ContactMode.ORACLE_EVALUATION)


def test_runtime_builder_accepts_only_current_and_predicted_contact() -> None:
    batch = runtime_contact_batch(values((3, 256)), values((3, 8, 32)))
    assert batch.current_contact is not None
    assert batch.predicted_contact is not None
    assert batch.contact_target is None
    batch.validate_for(ContactMode.INFERENCE)


def test_runtime_builder_supports_missing_contact() -> None:
    batch = runtime_contact_batch(None)
    assert batch.current_contact is None
    assert batch.predicted_contact is None
    batch.validate_for(ContactMode.INFERENCE)
