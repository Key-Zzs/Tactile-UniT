from scripts.simulation.freeze_s4_3_policy_dataset import valid_bc_windows


def test_valid_bc_window_count_requires_full_history_and_target():
    assert valid_bc_windows(51) == 0
    assert valid_bc_windows(52) == 1
    assert valid_bc_windows(100) == 49


def test_valid_bc_window_count_never_crosses_short_episode_boundary():
    assert valid_bc_windows(0) == 0
    assert valid_bc_windows(25) == 0
    assert valid_bc_windows(27) == 0
