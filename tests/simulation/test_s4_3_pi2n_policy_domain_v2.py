from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sparse_force_remediation_is_versioned_and_narrow() -> None:
    protocol = json.loads((ROOT / "configs/simulation/s4_3_pi2n_policy_domain_probe_v2.json").read_text())
    assert protocol["status"] == "FROZEN_AFTER_V1_ZERO_THRESHOLD_STRUCTURAL_FINDING_BEFORE_V2_METRICS"
    assert protocol["v2_subsets"]["minimum_positive_fit_rows"] == 100
    assert "strictly positive" in protocol["v2_subsets"]["high_force"]
    assert protocol["required_parity"]["overall_metrics_exact_v1"] is True
    assert protocol["required_parity"]["boundary_metrics_exact_v1"] is True
    assert protocol["route_constraints"]["X_route_not_finalized_here"] is True


def test_remediation_preserves_v1_and_writes_versioned_outputs() -> None:
    source = (ROOT / "scripts/simulation/remediate_s4_3_pi2n_policy_domain_subsets.py").read_text()
    assert 'policy_domain_probe_results_v2.json' in source
    assert 'diagnostic_labels_manifest_v2.json' in source
    assert 'policy_domain_probe_v1_structural_audit.json' in source
    assert "refusing to overwrite" in source
    assert 'candidate_or_formal_outcomes_used": False' in source
