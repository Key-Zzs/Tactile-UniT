from __future__ import annotations

from scripts.simulation.finalize_pi2s_evidence import paired_optional_effects, summarize_episodes


def _episode(checkpoint: str, condition: str, reset: str, seed: int, success: bool) -> dict:
    return {
        "tuple": {
            "checkpoint_id": checkpoint,
            "h_condition": condition,
            "reset_identity": reset,
            "sampling_seed": seed,
        },
        "success": success,
        "ever_object_contact": True,
        "ever_lifted": True,
        "ever_native_trigger": success,
        "max_pinch_count": 3 if success else 1,
        "termination": "native_success" if success else "native_max_steps",
    }


def test_summarize_episodes_preserves_negative_milestones() -> None:
    rows = [
        _episode("B_HVA_43", "correct", "a", 1, False),
        _episode("B_HVA_43", "correct", "b", 1, True),
    ]
    summary = summarize_episodes(rows)["B_HVA_43"]["correct"]
    assert summary == {
        "episodes": 2,
        "successes": 1,
        "failures": 1,
        "ever_object_contact": 2,
        "ever_lifted": 2,
        "ever_native_trigger": 1,
        "max_pinch_count_distribution": {"1": 1, "3": 1},
        "native_max_steps": 1,
    }


def test_paired_optional_effects_counts_wins_losses_and_agreements() -> None:
    base = [
        _episode("B1_43", "correct", "a", 7, False),
        _episode("B1_43", "correct", "b", 7, True),
        _episode("B1_43", "correct", "c", 7, False),
    ]
    optional = [
        _episode("B1_43", "train_mean", "a", 7, True),
        _episode("B1_43", "train_mean", "b", 7, False),
        _episode("B1_43", "train_mean", "c", 7, False),
    ]
    result = paired_optional_effects(base, optional)
    assert result == [
        {
            "checkpoint_id": "B1_43",
            "h_condition": "train_mean",
            "paired_episodes": 3,
            "correct_h_successes": 1,
            "intervention_successes": 1,
            "wins_vs_correct": 1,
            "losses_vs_correct": 1,
            "agreements": 1,
            "success_difference": 0,
        }
    ]
