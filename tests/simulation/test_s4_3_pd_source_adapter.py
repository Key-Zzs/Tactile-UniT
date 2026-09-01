import numpy as np

from gr00t.simulation.dexjoco_adapter import (
    SimObservation,
    policy_action_to_env_action,
)
from gr00t.simulation.s4_3_pd import (
    S43PDAttemptLogger,
    official_env_action_to_policy_action,
    policy_proprio_22,
    quaternion_wxyz_to_signed_rotvec,
)


def test_signed_rotvec_preserves_negative_quaternion_sign():
    quaternion = np.asarray([-0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    action = np.concatenate([np.asarray([0.1, -0.2, 1.1]), quaternion, np.arange(16) / 10])
    policy = official_env_action_to_policy_action(action)
    reconstructed = policy_action_to_env_action(policy).values
    assert np.dot(reconstructed[3:7], quaternion) > 1.0 - 2e-6
    np.testing.assert_allclose(reconstructed[:3], action[:3], atol=2e-6)
    np.testing.assert_allclose(reconstructed[7:], action[7:], atol=2e-6)
    assert np.linalg.norm(quaternion_wxyz_to_signed_rotvec(quaternion)) > np.pi


def test_policy_proprio_filters_task_state_and_is_22d():
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    robot = np.concatenate([np.asarray([0.1, 0.2, 1.3]), quaternion, np.arange(16) / 20])
    full = np.concatenate([robot, np.asarray([9.0, 8.0, 7.0])])
    result = policy_proprio_22(full)
    assert result.shape == (22,)
    np.testing.assert_allclose(result[:3], robot[:3])
    np.testing.assert_allclose(result[6:], robot[7:23])


def test_zero_quaternion_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "norm is zero"):
        quaternion_wxyz_to_signed_rotvec(np.zeros(4))


def test_attempt_logger_records_only_student_visible_shapes(tmp_path):
    robot = np.concatenate(
        [
            np.asarray([0.1, 0.2, 1.3, 1.0, 0.0, 0.0, 0.0]),
            np.arange(16) / 20,
            np.asarray([9.0, 8.0, 7.0]),
        ]
    )
    current = SimObservation(
        timestamp_sec=0.0,
        control_step=0,
        episode_id="attempt-0",
        task_name="click_mouse",
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        proprio=robot,
        sim_tactile=np.zeros(30, dtype=np.float32),
        terminated=False,
        truncated=False,
        success=False,
    )
    next_observation = SimObservation(
        **{**current.__dict__, "timestamp_sec": 0.02, "control_step": 1, "success": True}
    )
    source = np.concatenate(
        [np.asarray([0.1, 0.2, 1.3, 1.0, 0.0, 0.0, 0.0]), np.arange(16) / 20]
    )
    action = official_env_action_to_policy_action(source)
    env_action = policy_action_to_env_action(action)
    logger = S43PDAttemptLogger(
        tmp_path / "attempt-0", {"attempt_id": "attempt-0", "task": "click_mouse"}
    )
    logger.append(
        current,
        action,
        env_action,
        next_observation,
        1.0,
        {"succeed": True},
        "REPLAY",
        0,
        {"contact_count": 0},
        np.asarray([1.0, 1.0, 10.0, 0.01]),
        1.0,
    )
    manifest = logger.finish(
        termination_reason="NATIVE_SUCCESS",
        native_state_names=("a", "b", "c", "d"),
    )
    assert manifest["success"] is True
    assert manifest["fields"]["proprio"] == [1, 22]
    assert manifest["fields"]["sim_tactile"] == [1, 30]
    assert manifest["fields"]["policy_action"] == [1, 22]
