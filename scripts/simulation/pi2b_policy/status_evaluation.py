#!/usr/bin/env python3
"""Report Track-A FINAL progress without opening performance-bearing rows."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2b_policy"
PROGRESS = ARTIFACTS / "final_progress.json"


def main() -> None:
    if PROGRESS.is_file():
        print(PROGRESS.read_text(), end="")
        return
    raw = ARTIFACTS / "final_raw"
    count = len(list(raw.glob("reset_seed_*/*/train_seed_*.json"))) if raw.is_dir() else 0
    print(json.dumps({"status": "STARTING", "completed_blocks": count, "total_blocks": 60, "completed_rollouts": count * 50, "total_rollouts": 3000, "performance_values_read": False}, sort_keys=True))


if __name__ == "__main__":
    main()
