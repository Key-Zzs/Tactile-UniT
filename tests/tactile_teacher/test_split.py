from gr00t.tactile_teacher.split import build_episode_split


def test_episode_split_is_deterministic_stratified_and_leak_free():
    labels = {i: "press" if i < 20 else "wipe" for i in range(40)}
    first = build_episode_split(labels, seed=42)
    second = build_episode_split(labels, seed=42)

    assert first == second
    first.validate(labels)
    assert len(first.train) == 32
    assert len(first.val) == 4
    assert len(first.test) == 4
    for part in (first.train, first.val, first.test):
        assert {labels[x] for x in part} == {"press", "wipe"}


def test_episode_split_changes_with_seed():
    labels = {i: "primitive" for i in range(30)}
    assert build_episode_split(labels, seed=1) != build_episode_split(labels, seed=2)
