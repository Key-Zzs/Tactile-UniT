#!/usr/bin/env python3
"""Route the established S4.3 regression suite to S4.3-RR artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.simulation import run_s4_3_regressions as established  # noqa: E402


def main() -> None:
    established.ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_rr"
    established.LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_rr/regressions"
    established.main()


if __name__ == "__main__":
    main()
