#!/usr/bin/env python3
"""Compare released and locally reproduced checkpoints under the frozen protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
OFFICIAL_EVAL = ARTIFACT_ROOT / "official_checkpoint_eval.json"
REPRODUCED_EVAL = ARTIFACT_ROOT / "reproduced_checkpoint_eval.json"
TRAINING = ARTIFACT_ROOT / "training_completion.json"
CHECKPOINT_SMOKE = ARTIFACT_ROOT / "reproduced_checkpoint_smoke.json"
SERVER_SMOKE = ARTIFACT_ROOT / "reproduced_checkpoint_server_smoke.json"
TRAINING_CONFIG = ARTIFACT_ROOT / "official_training_config.json"
OUTPUT = ARTIFACT_ROOT / "official_vs_reproduced_comparison.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def intervals_overlap(left: list[float], right: list[float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def exact_mcnemar_pvalue(official_only: int, reproduced_only: int) -> float:
    discordant = official_only + reproduced_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, value) for value in range(min(official_only, reproduced_only) + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * lower_tail)


def main() -> None:
    official = load(OFFICIAL_EVAL)
    reproduced = load(REPRODUCED_EVAL)
    training = load(TRAINING)
    checkpoint_smoke = load(CHECKPOINT_SMOKE)
    server_smoke = load(SERVER_SMOKE)
    config = load(TRAINING_CONFIG)

    official_results = official["episode_results"]
    reproduced_results = reproduced["episode_results"]
    episode_ids = sorted(set(official_results) | set(reproduced_results))
    paired = [
        {
            "episode": episode,
            "official": official_results.get(episode),
            "reproduced": reproduced_results.get(episode),
        }
        for episode in episode_ids
    ]
    both_success = sum(row["official"] == "success" and row["reproduced"] == "success" for row in paired)
    official_only = sum(row["official"] == "success" and row["reproduced"] == "failure" for row in paired)
    reproduced_only = sum(row["official"] == "failure" and row["reproduced"] == "success" for row in paired)
    both_failure = sum(row["official"] == "failure" and row["reproduced"] == "failure" for row in paired)

    official_ci = official["wilson_95ci"]
    reproduced_ci = reproduced["wilson_95ci"]
    paper_mean = official["published_reference"]["success_rate_mean"]
    paper_std = official["published_reference"]["success_rate_std"]
    paper_band = [max(0.0, paper_mean - paper_std), min(1.0, paper_mean + paper_std)]
    official_ci_compatible = intervals_overlap(official_ci, reproduced_ci)
    paper_compatible = intervals_overlap(reproduced_ci, paper_band)

    same_conditions = (
        official["task"] == reproduced["task"] == "pinch_tongs"
        and official["regime"] == reproduced["regime"] == "rand_obj"
        and official["seed"] == reproduced["seed"] == 0
        and official["episodes"] == reproduced["episodes"] == 20
        and official["evaluation_conditions"] == reproduced["evaluation_conditions"]
        and set(official_results) == set(reproduced_results) == {f"{index:02d}" for index in range(20)}
    )
    official_pipeline_pass = official["status"] == "PASS"
    local_training_pass = training["status"] == "PASS"
    local_checkpoint_pass = checkpoint_smoke["status"] == "PASS"
    local_server_eval_pass = server_smoke["status"] == "PASS" and reproduced["status"] == "PASS"
    no_algorithm_changes = not config["official_algorithmic_defaults_overridden"]
    nontrivial_local_success = reproduced["successes"] > 0
    statistically_operationally_compatible = official_ci_compatible and paper_compatible

    if not official_pipeline_pass:
        decision = "S4_3_PI0_OFFICIAL_BASELINE_REPRODUCTION_FAIL"
    elif not same_conditions or not local_server_eval_pass:
        decision = "S4_3_PI0_OFFICIAL_PIPELINE_INCOMPATIBLE"
    elif not local_training_pass or not local_checkpoint_pass:
        decision = "S4_3_PI0_OFFICIAL_TRAINING_REPRODUCTION_GAP"
    elif not nontrivial_local_success or not statistically_operationally_compatible:
        decision = "S4_3_PI0_OFFICIAL_TRAINING_REPRODUCTION_GAP"
    elif no_algorithm_changes:
        decision = "S4_3_PI0_FULL_OFFICIAL_BASELINE_REPRODUCTION"
    else:
        decision = "STRUCTURAL_FAIL"

    full_reproduction = decision == "S4_3_PI0_FULL_OFFICIAL_BASELINE_REPRODUCTION"
    readiness = "READY_WITH_WARNINGS" if full_reproduction else "NOT_READY"
    gates = {
        "official_checkpoint_pipeline": official_pipeline_pass,
        "local_training_30000": local_training_pass,
        "local_checkpoint_loadable": local_checkpoint_pass,
        "local_official_server_and_evaluator": local_server_eval_pass,
        "same_paired_conditions": same_conditions,
        "local_success_nontrivial": nontrivial_local_success,
        "wilson_intervals_overlap": official_ci_compatible,
        "local_wilson_overlaps_published_mean_plus_minus_std": paper_compatible,
        "official_algorithm_unchanged": no_algorithm_changes,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-official-vs-reproduced-comparison.v1",
        "stage": "PI0-11/PI0-12",
        "scope": {
            "task": "pinch_tongs",
            "regime": "rand_obj",
            "episodes": 20,
            "seed": 0,
            "full_dexjoco_benchmark": False,
            "claim": (
                "Official DexJoCo pi0.5 rand-obj pipeline reproduced on pinch_tongs "
                "under a reduced-cost one-task, 20-episode protocol."
            ),
        },
        "official_checkpoint": {
            "successes": official["successes"],
            "episodes": official["episodes"],
            "success_rate": official["success_rate"],
            "wilson_95ci": official_ci,
        },
        "reproduced_checkpoint": {
            "successes": reproduced["successes"],
            "episodes": reproduced["episodes"],
            "success_rate": reproduced["success_rate"],
            "wilson_95ci": reproduced_ci,
        },
        "published_reference": {
            "success_rate_mean": paper_mean,
            "success_rate_std": paper_std,
            "mean_plus_minus_std_band": paper_band,
            "episodes_per_task": official["published_reference"]["paper_episodes_per_task"],
            "source": official["published_reference"]["source"],
        },
        "difference_reproduced_minus_official": (reproduced["success_rate"] - official["success_rate"]),
        "difference_reproduced_minus_published_mean": (reproduced["success_rate"] - paper_mean),
        "paired_counts": {
            "both_success": both_success,
            "official_only_success": official_only,
            "reproduced_only_success": reproduced_only,
            "both_failure": both_failure,
            "discordant": official_only + reproduced_only,
            "exact_mcnemar_two_sided_pvalue": exact_mcnemar_pvalue(official_only, reproduced_only),
        },
        "paired_episode_outcomes": paired,
        "compatibility": {
            "official_wilson_overlap": official_ci_compatible,
            "published_reference_overlap": paper_compatible,
            "statistically_operationally_compatible": statistically_operationally_compatible,
        },
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "decision": decision,
        "pi1_readiness": readiness,
        "readiness_warnings": [
            "Only one DexJoCo task was evaluated.",
            "Each checkpoint has only 20 fixed-seed episodes, so Wilson intervals are wide.",
            "rand-full and dynamics randomization were not evaluated.",
            "Renderer-level nondeterminism remains possible.",
        ],
        "reproduction_status": "PASS" if full_reproduction else "FAIL",
        "status": "PASS",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(
        json.dumps(
            {
                "decision": decision,
                "official": f"{official['successes']}/{official['episodes']}",
                "reproduced": f"{reproduced['successes']}/{reproduced['episodes']}",
                "paired_counts": payload["paired_counts"],
                "pi1_readiness": readiness,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
