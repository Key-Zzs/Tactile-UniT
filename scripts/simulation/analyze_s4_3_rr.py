#!/usr/bin/env python3
"""Route S4.3-RR artifacts through the byte-frozen statistical analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import read_json, sha256_file  # noqa: E402
from scripts.simulation import analyze_s4_3_closed_loop as frozen  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr"
FREEZE = ARTIFACT_ROOT / "pre_rollout_freeze_v2.json"
COMPLETENESS = ARTIFACT_ROOT / "rollout_completeness.json"


def main() -> None:
    freeze = read_json(FREEZE)
    completeness = read_json(COMPLETENESS)
    frozen_source = ROOT / freeze["statistical_code"]
    if sha256_file(frozen_source) != freeze["statistical_code_sha256"]:
        raise SystemExit("S4_3_RR_ROLLOUT_HARNESS_FAIL: statistics hash mismatch")
    if completeness.get("status") != "PASS":
        raise SystemExit("S4_3_RR_ROLLOUT_COMPLETENESS_FAIL")
    frozen.ARTIFACT_ROOT = ARTIFACT_ROOT
    frozen.ROLLOUTS = ARTIFACT_ROOT / "closed_loop_rollouts.json"
    frozen.main()


if __name__ == "__main__":
    main()
