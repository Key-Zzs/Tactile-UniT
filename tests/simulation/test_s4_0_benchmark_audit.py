import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/simulation/s4_0_benchmark_audit.json"
CANDIDATES = ("RoboCasa", "DexJoCo", "DexMimicGen", "IsaacLab")
SCORE_KEYS = {
    "task_coverage",
    "dexterous",
    "bimanual",
    "contact_rich",
    "tactile_proxy",
    "headless",
    "data",
    "act",
    "diffusion_policy",
    "gr00t",
    "pi0_5",
    "randomization",
    "scale",
    "icra_relevance",
    "iclr_relevance",
}


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def test_audit_schema_candidate_names_and_score_scale_are_valid():
    value = load_config()
    assert value["schema"] == "tactile3d-unit.s4-0-simulation-benchmark-audit.v1"
    assert tuple(value["candidates"]) == CANDIDATES
    for candidate in value["candidates"].values():
        assert set(candidate["scores"]) == SCORE_KEYS
        assert all(
            isinstance(score, int) and 0 <= score <= 4
            for score in candidate["scores"].values()
        )


def test_exactly_one_primary_matches_recommendation():
    value = load_config()
    primary = [name for name, row in value["candidates"].items() if row["primary"]]
    assert primary == [value["recommendations"]["primary"]] == ["DexJoCo"]
    assert value["recommendations"]["secondary_regression"] == "RoboCasa/robosuite/GR1"
    assert value["recommendations"]["optional_scale_up"] == "Isaac Lab/Isaac Sim"


def test_submodule_is_pinned_under_third_party_and_external_code_is_not_vendored():
    value = load_config()
    submodules = [row for row in value["candidates"].values() if row["submodule"]]
    assert len(submodules) == 1
    row = submodules[0]
    assert row["submodule_path"] == "third_party/dexjoco"
    assert len(row["audited_commit"]) == 40
    assert "path = third_party/dexjoco" in (ROOT / ".gitmodules").read_text()
    stage = git("ls-files", "--stage", row["submodule_path"])
    mode, commit, _, path = stage.split()
    assert mode == "160000" and commit == row["audited_commit"] and path == row["submodule_path"]
    assert git("ls-files", "third_party").splitlines() == ["third_party/dexjoco"]


def test_license_headless_contact_and_policy_fields_are_explicit():
    value = load_config()
    for row in value["candidates"].values():
        license_row = row["license"]
        assert license_row["name"]
        assert all(
            key in license_row
            for key in ("research_use", "modification", "redistribution", "restrictions")
        )
        assert row["headless_status"]
        assert row["contact_api_status"]
        assert row["policy_adapter_status"]
        assert row["engineering_cost"] in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}


def test_tactile_contract_is_design_only_and_uses_available_fields():
    value = load_config()
    contract = value["simulated_tactile_contract"]
    assert contract["status"] == "DESIGN_ONLY_NOT_IMPLEMENTED"
    assert contract["rh56dftp_asset_required"] is False
    assert set(contract["fields"]) == {
        "contact_occupancy",
        "normal_force",
        "tangential_force",
        "contact_position_or_center_of_pressure",
        "optional_contact_impulse_or_force_integral",
    }


def test_paper_tasks_are_real_primary_tasks_and_cover_distinct_contact_regimes():
    value = load_config()
    tasks = value["paper_tasks"]
    assert 4 <= len(tasks) <= 6
    assert all(row["benchmark"] == "DexJoCo" for row in tasks)
    assert {row["exact_name"] for row in tasks} == {
        "bimanual_assembly",
        "hammer_nail",
        "pinch_tongs",
        "bimanual_unlock_ipad",
        "bimanual_microwave_cook",
        "bimanual_hanoi",
    }
    assert len({row["contact_regime"] for row in tasks}) == len(tasks)


def test_all_paper_questions_and_s4_1_plan_are_covered_without_starting_s4_1():
    value = load_config()
    assert len(value["paper_questions"]) == 9
    assert all(status.startswith("SUPPORTED") for status in value["paper_questions"].values())
    assert value["s4_1"]["status"] == "RECOMMENDED_NOT_STARTED"
    assert len(value["s4_1"]["ordered_steps"]) >= 10
    assert value["prohibitions"]["s4_1_started"] is False
    assert value["implementation_started"] is False


def test_m3_contract_files_match_starting_main_identity():
    value = load_config()["m3_integrity"]
    assert sha256(ROOT / "configs/tactile_unit/m3_system_manifest.json") == value["manifest_sha256"]
    assert (
        sha256(ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json")
        == value["c6_config_sha256"]
    )
    assert sha256(ROOT / "configs/tactile_unit/m3_limitations.json") == value["limitations_sha256"]


def test_tracked_s4_0_files_have_no_private_paths_or_credentials():
    paths = [
        CONFIG_PATH,
        ROOT / "scripts/simulation/audit_s4_0_benchmarks.py",
        ROOT / "tests/simulation/test_s4_0_benchmark_audit.py",
        ROOT / "docs/research/s4_0_sim_benchmark_audit.md",
    ]
    forbidden = (
        "/" + "home/",
        "/" + "mnt/",
        "Authorization" + ":",
        "Bear" + "er ",
        "github" + "_pat_",
        "gh" + "p_",
        "HF" + "_TOKEN",
        "WANDB" + "_API_KEY",
    )
    for path in paths:
        text = path.read_text(errors="replace")
        assert not any(term in text for term in forbidden), path
