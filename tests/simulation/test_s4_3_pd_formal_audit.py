from scripts.simulation.audit_s4_3_policy_expert_data import (
    classify_numeric_duplicates,
)


def _row(attempt_index, *, task="pinch_tongs", group="group-0", rgb=None, steps="s"):
    return {
        "attempt_id": f"attempt-{attempt_index}",
        "task": task,
        "source_group_id": group,
        "attempt_index": attempt_index,
        "steps_npz_sha256": steps,
        "rgb_tree_sha256": rgb or f"rgb-{attempt_index}",
    }


def test_numeric_duplicates_are_expected_only_within_one_five_attempt_group():
    expected, unexpected = classify_numeric_duplicates([_row(index) for index in range(5)])
    assert len(expected) == 1
    assert unexpected == []
    assert expected[0]["classification"] == "EXPECTED_WITHIN_SOURCE_VISUAL_PERTURBATIONS"


def test_cross_group_numeric_duplicate_is_unexpected():
    rows = [_row(index) for index in range(4)]
    rows.append(_row(4, group="group-1"))
    expected, unexpected = classify_numeric_duplicates(rows)
    assert expected == []
    assert len(unexpected) == 1
    assert unexpected[0]["classification"] == "UNEXPECTED_NUMERIC_DUPLICATE"


def test_duplicate_rgb_tree_prevents_expected_classification():
    rows = [_row(index, rgb="same-rgb") for index in range(5)]
    expected, unexpected = classify_numeric_duplicates(rows)
    assert expected == []
    assert len(unexpected) == 1
