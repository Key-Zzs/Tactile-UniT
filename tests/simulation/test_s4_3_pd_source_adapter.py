import numpy as np

from gr00t.simulation.dexjoco_adapter import policy_action_to_env_action
from gr00t.simulation.s4_3_pd import (
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
