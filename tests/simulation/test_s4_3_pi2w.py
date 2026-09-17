import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pi2b_recommendation_is_nonexecuting_and_mechanism_first():
    recommendation = json.loads(
        (ROOT / "configs/simulation/s4_3_pi2b_recommendation.json").read_text()
    )
    assert recommendation["status"] == "DRAFT_NOT_EXECUTED"
    assert recommendation["training_authorized"] is False
    assert recommendation["decision"] == "PI2B_MECHANISM_DIAGNOSIS_FIRST"
    assert recommendation["immediate_new_runs"] == 0
    conditional = recommendation["conditional_protocol_after_claim_resolution"]
    assert conditional["models"] == ["B0", "BVA", "B1", "B2"]
    assert conditional["additional_training_seeds"] == [43, 44]
    assert conditional["new_30k_runs"] == 8
    assert conditional["fresh_evaluator_seed"] == 7
    assert conditional["episodes_per_checkpoint"] == 200


def test_pi2w_diagnostic_entrypoints_construct_no_optimizer():
    for name in (
        "diagnose_s4_3_pi2w_unit_gate.py",
        "diagnose_s4_3_pi2w_native_unit.py",
    ):
        source = (ROOT / "scripts/simulation" / name).read_text()
        tree = ast.parse(source)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not ({"Adam", "AdamW", "SGD"} & called_names)
        assert not ({"backward", "step", "zero_grad"} & attributes)
        assert "@torch.inference_mode()" in source


def test_pi2w_outputs_are_isolated_from_pi2v():
    diagnostic = (
        ROOT / "scripts/simulation/diagnose_s4_3_pi2w_unit_gate.py"
    ).read_text()
    native = (
        ROOT / "scripts/simulation/diagnose_s4_3_pi2w_native_unit.py"
    ).read_text()
    assert '.local/artifacts/simulation/s4_3_pi2w' in diagnostic
    assert '.local/cache/simulation/s4_3_pi2w' in diagnostic
    assert '.local/artifacts/simulation/s4_3_pi2w/native_unit_gate_sanity.json' in native
    assert "refusing to overwrite" in diagnostic
    assert "refusing to overwrite" in native


def test_pi2w_document_preserves_claim_boundary_and_three_decisions():
    report = (
        ROOT
        / "docs/research/s4_3_pi2w_unit_gate_diagnosis_and_mainline_closure.md"
    ).read_text()
    assert "S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL" in report
    assert "PI2V_FAIL_VISION_DOMAIN_TRANSFER" in report
    assert "BUNIT_APPENDIX_FAILED_TRANSFER_ONLY" in report
    assert "PI2B_MECHANISM_DIAGNOSIS_FIRST" in report
    assert "does not establish that Original UniT fundamentally fails" in report
    assert "does not establish that RQ is inferior" in report


def test_pi2b_draft_uses_training_seed_as_higher_level_unit():
    draft = (
        ROOT / "docs/research/s4_3_pi2b_preregistration_draft.md"
    ).read_text()
    assert "DRAFT — NOT EXECUTED — TRAINING NOT AUTHORIZED" in draft
    assert "Training seed is the higher-level source of randomness" in draft
    assert "Do not pool `3 × N` rollout outcomes" in draft
    assert "Seed7 is therefore the smallest unexposed evaluator seed" in draft
