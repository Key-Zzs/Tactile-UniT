#!/usr/bin/env python3
"""Render compact, dependency-free C6 decision plots from frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".local/artifacts/tactile_unit/vac_c6"


def svg(path: Path, body: str) -> None:
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520"><rect width="900" height="520" fill="white"/>' + body + '</svg>')


def main() -> None:
    final = json.loads((OUT / "final_decision.json").read_text())
    rank = json.loads((OUT / "rank_geometry.json").read_text())
    plots = OUT / "plots"; plots.mkdir(parents=True, exist_ok=True)
    rows = ''.join(f'<text x="60" y="{75 + i * 29}" font-family="sans-serif" font-size="18">{key}: {"PASS" if value else "FAIL"}</text>' for i, (key, value) in enumerate(final["gates"].items()))
    svg(plots / "01_m3_hard_gate_matrix.svg", f'<text x="60" y="38" font-family="sans-serif" font-size="24" font-weight="bold">{final["decision"]}</text>{rows}')
    values = [("oracle", rank["oracle_u_c_effective_rank"]), ("FULL_AH", rank["full_ah_effective_rank"]), ("FALLBACK_A", rank["fallback_a_effective_rank"]), ("offline F_VA", rank["offline_va_effective_rank"]), ("causal visual", rank["causal_visual_effective_rank"])]
    bars = ''.join(f'<text x="60" y="{78 + i * 72}" font-family="sans-serif" font-size="18">{name}</text><rect x="250" y="{55 + i * 72}" width="{value * 18:.1f}" height="30" fill="#376996"/><text x="{260 + value * 18:.1f}" y="{78 + i * 72}" font-family="sans-serif" font-size="15">{value:.3f}</text>' for i, (name, value) in enumerate(values))
    svg(plots / "02_effective_rank_warning.svg", '<text x="60" y="35" font-family="sans-serif" font-size="24" font-weight="bold">Shared Contact effective rank</text>' + bars)
    (OUT / "visualization_summary.json").write_text(json.dumps({"schema":"tactile3d-unit.vac-c6-visualization.v1", "plots":["01_m3_hard_gate_matrix.svg", "02_effective_rank_warning.svg"], "decision":final["decision"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
