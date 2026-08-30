import json
from pathlib import Path

import pytest

from gr00t.tactile_unit.c6_runtime_router import C6RuntimeMode, route_c6_availability


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("action", "contact", "vision", "expected"),
    [
        (True, True, True, C6RuntimeMode.FULL_AH),
        (True, True, False, C6RuntimeMode.FULL_AH),
        (True, False, True, C6RuntimeMode.FALLBACK_A),
        (True, False, False, C6RuntimeMode.FALLBACK_A),
        (False, True, True, C6RuntimeMode.ABSTAIN_NO_ACTION),
        (False, False, False, C6RuntimeMode.ABSTAIN_NO_ACTION),
    ],
)
def test_c6_router_has_exactly_the_canonical_routes(action, contact, vision, expected):
    assert route_c6_availability(action_available=action, contact_context_available=contact, vision_available=vision) is expected


def test_c6_protocol_is_freeze_only_and_manifest_has_no_local_paths():
    protocol = json.loads((ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json").read_text())
    manifest = json.loads((ROOT / "configs/tactile_unit/m3_system_manifest.json").read_text())
    assert protocol["training_allowed"] is False
    assert protocol["model_selection_allowed"] is False
    assert protocol["test_loaded"] is False
    assert protocol["canonical_runtime_modes"] == ["FULL_AH", "FALLBACK_A", "ABSTAIN_NO_ACTION"]
    assert all(component["runtime_routable"] is False for component in manifest["components"] if component["name"] in {"F_VA", "C5 causal visual"})
    assert "/home/" not in json.dumps(manifest)


def test_c6_final_evaluation_is_frozen_and_has_only_allowed_warnings():
    path = ROOT / ".local/artifacts/tactile_unit/vac_c6/final_decision.json"
    if not path.is_file():
        pytest.skip("run evaluate_c6_m3_system.py first")
    value = json.loads(path.read_text())
    assert value["decision"] in {"C6_M3_ESTABLISHED_WITH_WARNINGS", "C6_M3_ESTABLISHED", "C6_M3_NOT_ESTABLISHED_INTEGRITY_FAIL"}
    assert value["training_performed"] is False and value["model_selection_performed"] is False
    assert value["locked_benchmark_rows"] == 17504
    assert set(value["warnings"]) <= {"POLICY_PLAN_DOMAIN_WARNING", "RANK_CONTRACTION_WARNING", "CAUSAL_VISUAL_SUBSTITUTION_NOT_PROMOTED", "PUBLICATION_EXTERNAL_CONFIRMATION_PENDING"}
