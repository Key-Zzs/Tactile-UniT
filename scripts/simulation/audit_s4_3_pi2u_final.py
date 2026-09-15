#!/usr/bin/env python3
"""Fail-closed completion audit for the full S4.3-PI2U objective."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
PI1 = ROOT / ".local/artifacts/simulation/s4_3_pi1"
PI2A = ROOT / ".local/artifacts/simulation/s4_3_pi2a"
TMP_FAILURE = ROOT / ".local/tmp/s43u6/final_audit_failure.json"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
STARTING_HEAD = "5e88dd3eaac708aed572f867d5df8ef489e54017"
DEXJOCO_COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
UNIT_COMMIT = "0d762e32180bddd765694ef3846a3a5053f9d37f"
EVALUATOR_SEED = 6

CHECKPOINTS = {
    "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "BVA": ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42/29999",
    "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}
EXPECTED_CHECKPOINTS = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "BVA": "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}
OFFICIAL_AUDIT_ARTIFACTS = (
    "starting_integrity.json",
    "official_unit_source.json",
    "official_unit_checkpoint_inventory.json",
    "official_unit_architecture.json",
    "official_unit_tokenizer_contract.json",
    "official_unit_embodiment_contract.json",
    "dexjoco_unit_compatibility.json",
    "native_unit_smoke.json",
    "unit_adaptation_classification.json",
    "future_bunit_protocol.json",
    "unit_resource_estimate.json",
)
BVA_PREEXISTING_ARTIFACTS = (
    "va_dataset_manifest.json",
    "va_bridge_training.json",
    "va_bridge_metrics.json",
    "va_bridge_checkpoint_manifest.json",
    "contact_leakage_audit.json",
    "bva_target_manifest.json",
    "bva_mode_contract.json",
    "bva_none_parity.json",
    "bva_lambda_calibration.json",
    "bva_training_protocol.json",
    "bva_launch.json",
    "bva_training_completion.json",
    "bva_checkpoint_manifest.json",
    "fresh_seed_audit.json",
    "fresh_seed_retry_seed6.json",
    "pre_eval_freeze.json",
    "pre_eval_retry_seed6.json",
    "b0_eval.json",
    "bva_eval.json",
    "b1_eval.json",
    "b2_eval.json",
    "paired_ablation_statistics.json",
    "mechanism_interpretation.json",
    "visualization_manifest.json",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    total = 0
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix()
        size = candidate.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256(candidate).encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\n")
        files += 1
        total += size
    return digest.hexdigest(), files, total


def source_tree_hash(path: Path) -> tuple[str, int]:
    rows = []
    for candidate in path.rglob("*"):
        if not candidate.is_file() or "__pycache__" in candidate.parts or candidate.suffix == ".pyc":
            continue
        rows.append((candidate.relative_to(path).as_posix(), sha256(candidate), candidate.stat().st_size))
    digest = hashlib.sha256()
    for relative, value, size in sorted(rows):
        digest.update(f"{relative}\0{value}\0{size}\n".encode())
    return digest.hexdigest(), len(rows)


def run(*args: str, cwd: Path = ROOT, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def atomic(path: Path, value: dict[str, Any] | str) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, str):
        temporary.write_text(value)
    else:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def environment_snapshot(python: Path) -> dict[str, Any]:
    script = r"""
import hashlib, importlib, json, pathlib, subprocess, sys
names = ['jax','jaxlib','mujoco','numpy','openpi','openpi_client','torch']
mods = {}
for name in names:
    try:
        module = importlib.import_module(name)
        mods[name] = {'version': getattr(module, '__version__', None), 'file': str(pathlib.Path(module.__file__).resolve())}
    except Exception as error:
        mods[name] = {'error': type(error).__name__}
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
history = pathlib.Path(sys.prefix) / 'conda-meta/history'
print(json.dumps({'python': sys.version, 'modules': mods, 'pip_freeze': freeze, 'pip_freeze_sha256': hashlib.sha256(freeze.encode()).hexdigest(), 'conda_history_sha256': hashlib.sha256(history.read_bytes()).hexdigest()}))
"""
    return json.loads(subprocess.check_output([str(python), "-c", script], text=True))


def environment_audit(current_head: str) -> dict[str, Any]:
    baseline = load(PI2A / "environment_integrity.json")["before"]
    pythons = {
        "openpi": CONDA_ROOT / "envs/openpi/bin/python",
        "tactile-unit-dexjoco": CONDA_ROOT / "envs/tactile-unit-dexjoco/bin/python",
        "unit": CONDA_ROOT / "envs/unit/bin/python",
    }
    snapshots = {name: environment_snapshot(path) for name, path in pythons.items()}
    gates: dict[str, bool] = {}
    identities: dict[str, Any] = {}
    baseline_editable_head = load(PI2A / "environment_integrity.json")["editable_source_normalization"]["current_editable_head"]
    for name, current in snapshots.items():
        expected = baseline[name]
        reconstructed = current["pip_freeze"]
        if name == "unit":
            reconstructed = reconstructed.replace(current_head, baseline_editable_head)
        reconstructed_sha = hashlib.sha256(reconstructed.encode()).hexdigest()
        versions = {module: value.get("version", value.get("error")) for module, value in current["modules"].items()}
        expected_versions = {module: value.get("version", value.get("error")) for module, value in expected["modules"].items()}
        gates[f"{name}_pip_freeze"] = reconstructed_sha == expected["pip_freeze_sha256"]
        gates[f"{name}_conda_history"] = current["conda_history_sha256"] == expected["conda_history_sha256"]
        gates[f"{name}_module_versions"] = versions == expected_versions
        identities[name] = {
            "raw_pip_freeze_sha256": current["pip_freeze_sha256"],
            "reconstructed_at_pi2a_head_sha256": reconstructed_sha,
            "expected_pi2a_sha256": expected["pip_freeze_sha256"],
            "conda_history_sha256": current["conda_history_sha256"],
            "module_versions": versions,
        }
    overlay_script = "import json,mujoco,numpy,openpi_client; print(json.dumps({'mujoco':mujoco.__version__,'numpy':numpy.__version__,'openpi_client':openpi_client.__version__}))"
    overlay = json.loads(output(str(ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"), "-c", overlay_script))
    expected_overlay = load(ROOT / ".local/artifacts/simulation/s4_3_pi0/evaluation_environment.json")["package_versions"]
    gates["pi0_evaluator_overlay_versions"] = overlay == {"mujoco": expected_overlay["mujoco"], "numpy": expected_overlay["numpy"], "openpi_client": expected_overlay["openpi-client"]}
    dex_root = ROOT / "third_party/dexjoco"
    gates["dexjoco_commit"] = output("git", "rev-parse", "HEAD", cwd=dex_root) == DEXJOCO_COMMIT
    gates["dexjoco_checkout_clean"] = output("git", "status", "--short", cwd=dex_root) == ""
    official_root = ROOT / ".local/external/s4_3_pi2u/unit_official"
    gates["official_unit_commit"] = output("git", "rev-parse", "HEAD", cwd=official_root) == UNIT_COMMIT
    gates["official_unit_checkout_clean"] = output("git", "status", "--short", cwd=official_root) == ""
    pi1_openpi = ROOT / ".local/external/simulation/s4_3_pi1/openpi/src"
    openpi_hash, openpi_files = source_tree_hash(pi1_openpi)
    expected_openpi_hash = load(PI2A / "checkpoint_immutability.json")["official_openpi_source_tree_sha256"]
    gates["frozen_openpi_source_tree"] = openpi_hash == expected_openpi_hash
    return {
        "schema": "tactile3d-unit.s4-3-pi2u-environment-integrity.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "package_install_or_update_performed": False,
        "normalization": "pip freeze --all reconstructed to the PI2A editable-source head, conda history, and imported module versions",
        "environments": identities,
        "pi0_evaluator_overlay": overlay,
        "dexjoco": {"commit": output("git", "rev-parse", "HEAD", cwd=dex_root), "expected": DEXJOCO_COMMIT},
        "official_unit": {"commit": output("git", "rev-parse", "HEAD", cwd=official_root), "expected": UNIT_COMMIT},
        "frozen_openpi_source": {"sha256": openpi_hash, "expected": expected_openpi_hash, "files": openpi_files},
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
    }


def s4_2_immutability_audit() -> dict[str, Any]:
    baseline = load(PI1 / "s4_2_immutability.json")["before"]
    current_checkpoints = {name: sha256(ROOT / relative) for name, relative in baseline["checkpoint_paths"].items()}
    current_configs = {relative: sha256(ROOT / relative) for relative in baseline["tracked_configs"]}
    vision_root = ROOT / ".local/external/s4_3_pi2u/unit_fulldata/VLA-UniT-3B-fulldata/tokenizer"
    expected_vision = load(ROOT / ".local/artifacts/simulation/s4_3/s4_2_immutability.json")["vision_checkpoint_hashes"]
    current_vision = {name: sha256(vision_root / name) for name in expected_vision}
    m3_paths = {
        "c6_contract": ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json",
        "limitations": ROOT / "configs/tactile_unit/m3_limitations.json",
        "manifest": ROOT / "configs/tactile_unit/m3_system_manifest.json",
    }
    expected_m3 = load(ROOT / ".local/artifacts/simulation/s4_3/s4_2_immutability.json")["m3_expected_hashes"]
    current_m3 = {name: sha256(path) for name, path in m3_paths.items()}
    gates = {
        "all_s4_2_checkpoints_byte_identical": current_checkpoints == baseline["checkpoints"],
        "all_s4_2_configs_byte_identical": current_configs == baseline["tracked_configs"],
        "vision_checkpoint_byte_identical": current_vision == expected_vision,
        "m3_contracts_byte_identical": current_m3 == expected_m3,
    }
    return {
        "schema": "tactile3d-unit.s4-3-pi2u-s4-2-immutability.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mutation": "NO" if all(gates.values()) else "YES",
        "checkpoint_paths": baseline["checkpoint_paths"],
        "checkpoint_hashes_expected": baseline["checkpoints"],
        "checkpoint_hashes_current": current_checkpoints,
        "tracked_config_hashes_expected": baseline["tracked_configs"],
        "tracked_config_hashes_current": current_configs,
        "vision_checkpoint_hashes_expected": expected_vision,
        "vision_checkpoint_hashes_current": current_vision,
        "m3_hashes_expected": expected_m3,
        "m3_hashes_current": current_m3,
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
    }


def checkpoint_audit() -> dict[str, Any]:
    rows = {}
    for model, path in CHECKPOINTS.items():
        value, files, size = tree_hash(path)
        rows[model] = {
            "path": "$REPO_ROOT/" + path.relative_to(ROOT).as_posix(),
            "tree_sha256": value,
            "expected_tree_sha256": EXPECTED_CHECKPOINTS[model],
            "files": files,
            "bytes": size,
            "byte_identical": value == EXPECTED_CHECKPOINTS[model],
        }
    return {
        "status": "PASS" if all(row["byte_identical"] for row in rows.values()) else "FAIL",
        "checkpoints": rows,
        "B0_B1_B2_training_performed_in_pi2u": False,
    }


def regression_audit() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    test = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    combined = test.stdout + "\n" + test.stderr
    def count(label: str) -> int:
        match = re.search(rf"(\d+) {label}", combined)
        return int(match.group(1)) if match else 0
    return {
        "schema": "tactile3d-unit.s4-3-pi2u-regressions.v1",
        "status": "PASS" if test.returncode == 0 else "FAIL",
        "pytest": {
            "command": "$UNIT_PYTHON -m pytest -q tests",
            "returncode": test.returncode,
            "passed": count("passed"),
            "skipped": count("skipped"),
            "failed": count("failed"),
            "stdout": test.stdout,
            "stderr": test.stderr,
        },
        "coverage": {
            "project": "tests/",
            "PI1": "tests/simulation/test_s4_3_pi1.py",
            "PI2A": "frozen PI2A runtime/statistics dependencies plus tests/simulation/test_s4_3_pi0.py and test_dexjoco_runtime.py",
            "BVA": "tests/simulation/test_s4_3_pi2u.py",
        },
    }


def repository_audit(current_head: str) -> dict[str, Any]:
    branch = output("git", "branch", "--show-current")
    status = output("git", "status", "--short")
    local_tracked = output("git", "ls-files", ".local")
    prior_changes = output(
        "git", "diff", "--name-only", STARTING_HEAD, current_head, "--",
        "configs/simulation/s4_3_pi2a*", "docs/research/s4_3_pi2a*",
    )
    identity_patterns = ["/" + "home/", "/" + "mnt/", "deep" + "cybo", "wbcd" + "@"]
    privacy = run("git", "grep", "-n", "-I", "-E", "|".join(identity_patterns))
    changed = output("git", "diff", "--name-only", STARTING_HEAD, current_head).splitlines()
    sensitive_findings = []
    credential_pattern = re.compile(r"(?i)(authorization\s*:\s*(bearer|basic)|api[_-]?key\s*[:=]\s*['\"][^$]|token\s*[:=]\s*['\"][A-Za-z0-9_-]{16,})")
    for relative in changed:
        path = ROOT / relative
        if path.is_file():
            try:
                for number, line in enumerate(path.read_text(errors="strict").splitlines(), start=1):
                    if credential_pattern.search(line):
                        sensitive_findings.append(f"{relative}:{number}")
            except UnicodeError:
                pass
    diff_check = run("git", "diff", "--check")
    gates = {
        "branch": branch == "develop/sim-benchmark",
        "working_tree_clean": status == "",
        "local_artifacts_untracked": local_tracked == "",
        "pi2a_tracked_configs_and_results_unchanged": prior_changes == "",
        "tracked_identity_path_privacy": privacy.returncode == 1 and privacy.stdout == "",
        "changed_files_no_embedded_credentials": not sensitive_findings,
        "git_diff_check": diff_check.returncode == 0,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "branch": branch,
        "starting_head": STARTING_HEAD,
        "current_head": current_head,
        "working_tree": status or "CLEAN",
        "pi2a_protected_path_changes": prior_changes.splitlines(),
        "privacy_findings": privacy.stdout.splitlines() + sensitive_findings,
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
    }


def main() -> None:
    final_paths = [
        ARTIFACTS / name for name in (
            "va_bridge_protocol.json", "s4_2_immutability.json", "environment_integrity.json",
            "regression_tests.json", "final_repository_audit.json", "final_decision.json", "HUMAN_ACCEPTANCE.md",
        )
    ]
    if any(path.exists() for path in final_paths):
        raise SystemExit("refusing to overwrite final PI2U evidence")
    missing = [name for name in OFFICIAL_AUDIT_ARTIFACTS + BVA_PREEXISTING_ARTIFACTS if not (ARTIFACTS / name).is_file()]
    if missing:
        raise SystemExit("PI2U final audit called before evidence exists: " + ", ".join(missing))

    current_head = output("git", "rev-parse", "HEAD")
    protocol_config = ROOT / "configs/simulation/s4_3_pi2u_va_bridge.json"
    bridge_protocol = {
        "schema": "tactile3d-unit.s4-3-pi2u-va-bridge-protocol-evidence.v1",
        "status": "PASS",
        "frozen_before_training": True,
        "tracked_config": "$REPO_ROOT/configs/simulation/s4_3_pi2u_va_bridge.json",
        "tracked_config_sha256": sha256(protocol_config),
        "protocol": load(protocol_config),
    }
    completion = load(ARTIFACTS / "bva_training_completion.json")
    bva_manifest = load(ARTIFACTS / "bva_checkpoint_manifest.json")
    freeze = load(ARTIFACTS / "pre_eval_retry_seed6.json")
    fresh = load(ARTIFACTS / "fresh_seed_retry_seed6.json")
    canonical_freeze = load(ARTIFACTS / "pre_eval_freeze.json")
    canonical_fresh = load(ARTIFACTS / "fresh_seed_audit.json")
    statistics = load(ARTIFACTS / "paired_ablation_statistics.json")
    mechanism = load(ARTIFACTS / "mechanism_interpretation.json")
    visual = load(ARTIFACTS / "visualization_manifest.json")
    compatibility = load(ARTIFACTS / "unit_adaptation_classification.json")
    bva_protocol = load(ROOT / "configs/simulation/s4_3_pi2u_bva_protocol.json")
    remediation = load(ROOT / "configs/simulation/s4_3_pi2u_bva_temporal_remediation.json")
    target_manifest = load(ARTIFACTS / "bva_target_manifest.json")
    native_smoke = load(ARTIFACTS / "native_unit_smoke.json")

    temporal = bva_protocol["auxiliary_target"]["temporal_alignment"]
    corrected = remediation["corrected_temporal_contract"]
    temporal_gates = {
        "canonical_control_steps_27": temporal.get("canonical_offset_steps") == corrected.get("canonical_offset_steps") == target_manifest.get("canonical_future_offset_steps") == 27,
        "canonical_horizon_0p54_seconds": all(math.isclose(float(value), 0.54, abs_tol=1e-12) for value in (temporal.get("canonical_horizon_seconds"), corrected.get("canonical_horizon_seconds"), target_manifest.get("canonical_future_offset_seconds"))),
        "source_dataset_30hz": math.isclose(float(temporal.get("source_dataset_fps")), 30.0, abs_tol=1e-12) and math.isclose(float(corrected.get("policy_dataset_fps")), 30.0, abs_tol=1e-12),
        "nearest_source_frame_16": temporal.get("source_offset_frames") == corrected.get("source_offset_frames") == target_manifest.get("source_future_offset_frames") == 16,
        "source_horizon_16_over_30": all(math.isclose(float(value), 16 / 30, abs_tol=1e-12) for value in (temporal.get("source_horizon_seconds"), corrected.get("source_horizon_seconds"), target_manifest.get("source_future_offset_seconds"))),
        "rounding_error_below_half_frame": float(target_manifest.get("absolute_timing_error_seconds")) < 1 / 60,
        "no_rgb_interpolation": temporal.get("selection_rule") == "nearest native source frame; no RGB interpolation" and corrected.get("rgb_interpolation") is False and target_manifest.get("source_frame_selection") == "nearest native source frame; no RGB interpolation",
        "target_rows_and_tail_exact": target_manifest.get("rows") == 40065 and target_manifest.get("valid_rows") == 38465 and target_manifest.get("invalid_tail_rows") == 1600,
        "corrected_target_hash": target_manifest.get("sidecar_sha256") == "c596b2f56880a969148f7cf06268ecfa9ad23bac01014be6c73ad20afd0d0612",
        "native_vision_smoke_bound_to_corrected_target": native_smoke.get("status") == "PASS_VISION_TRANSITION_ONLY_DURING_BVA_TARGET_BUILD" and native_smoke["native_forward"].get("samples") == 38465 and native_smoke["native_forward"].get("bva_target_sidecar_sha256") == target_manifest.get("sidecar_sha256"),
        "remediation_frozen_before_retraining": remediation.get("status") == "FROZEN_BEFORE_REMEDIATION_TRAINING" and remediation["frozen_remediation"].get("fresh_evaluator_seed_after_retraining") == EVALUATOR_SEED,
        "bva_checkpoint_bound_to_corrected_target": bva_manifest.get("checkpoint_tree_sha256") == EXPECTED_CHECKPOINTS["BVA"] and completion.get("checkpoint_tree_sha256") == EXPECTED_CHECKPOINTS["BVA"],
    }

    source_gates = {}
    for symbolic, expected in freeze["sources_sha256"].items():
        relative = symbolic.removeprefix("$REPO_ROOT/")
        source_gates[relative] = sha256(ROOT / relative) == expected

    raw = {model: load(ARTIFACTS / f"{model.lower()}_raw_rollouts.json") for model in EXPECTED_CHECKPOINTS}
    runtime_gates: dict[str, bool] = {}
    reset_hashes = set()
    for model, payload in raw.items():
        rows = payload.get("episode_results", [])
        runtime_gates[f"{model}_complete_200"] = payload.get("status") == "PASS" and len(rows) == 200
        reset_hashes.add(hashlib.sha256("\n".join(str(row.get("reset_identity")) for row in rows).encode()).hexdigest())
        if model in ("B0", "BVA"):
            runtime_gates[f"{model}_no_contact_or_training_field_delivery"] = all(
                not event.get("contact_state_sent", False) and not event.get("training_only_fields_sent", [])
                for row in rows for event in row.get("action_chunks", [])
            )
    runtime_gates["same_ordered_reset_sequence"] = len(reset_hashes) == 1
    quarantine_counts = {}
    quarantine_records_valid = {}
    malformed_diagnostic_rows = {}
    for model in ("b0", "bva", "b1", "b2"):
        path = ROOT / f".local/tmp/s43u6/{model}_inference.jsonl"
        count = 0
        valid = True
        malformed = 0
        if path.is_file():
            for line in path.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if event.get("type") == "stale_cross_episode_action_discarded":
                    count += 1
                    try:
                        valid &= (
                            event.get("model") == model.upper()
                            and int(event.get("action_timestamp", -1)) > int(event.get("current_timestamp", -1))
                            and event.get("reason") == "future timestamp is impossible within one causal episode"
                        )
                    except (TypeError, ValueError):
                        valid = False
        quarantine_counts[model.upper()] = count
        quarantine_records_valid[model.upper()] = valid
        malformed_diagnostic_rows[model.upper()] = malformed
    runtime_gates["all_cross_episode_action_quarantines_valid_and_discarded"] = all(quarantine_records_valid.values())
    runtime_gates["diagnostic_jsonl_well_formed"] = not any(malformed_diagnostic_rows.values())

    checkpoints = checkpoint_audit()
    s42 = s4_2_immutability_audit()
    environment = environment_audit(current_head)
    regressions = regression_audit()
    repository = repository_audit(current_head)
    protected = load(ARTIFACTS / "starting_integrity.json")["protected_inputs"]
    protected_rows = {relative: {"expected": expected, "current": sha256(ROOT / relative)} for relative, expected in protected.items()}

    gates = {
        "official_unit_audit_complete": compatibility.get("status") == "PASS" and compatibility.get("classification") == "UNIT_DEXJOCO_ADAPTER_ONLY_COMPATIBLE",
        "all_named_preexisting_artifacts_present": not missing,
        "va_bridge_protocol_frozen": bridge_protocol["protocol"].get("status") == "FROZEN_BEFORE_VA_BRIDGE_TRAINING",
        "bva_training_complete": completion.get("status") == "PASS" and bva_manifest.get("optimizer_steps") == 30000 and bva_manifest.get("seed") == 42,
        "corrected_temporal_contract": all(temporal_gates.values()),
        "bva_checkpoint_and_all_controls_byte_identical": checkpoints["status"] == "PASS",
        "fresh_seed6_freeze": freeze.get("status") == "PASS" and fresh.get("selected_seed") == EVALUATOR_SEED and canonical_freeze == freeze and canonical_fresh == fresh,
        "formal_analysis_seed6": statistics.get("status") == "PASS" and statistics.get("evaluator_seed") == EVALUATOR_SEED and mechanism.get("evaluator_seed") == EVALUATOR_SEED,
        "frozen_evaluator_sources_unchanged": all(source_gates.values()),
        "formal_runtime_integrity": all(runtime_gates.values()),
        "required_visuals": visual.get("status") == "PASS" and len(visual.get("required_visuals", [])) >= 17,
        "s4_2_and_m3_immutable": s42["status"] == "PASS",
        "pi2a_protected_inputs_immutable": all(row["expected"] == row["current"] for row in protected_rows.values()),
        "environments_immutable": environment["status"] == "PASS",
        "all_project_regressions": regressions["status"] == "PASS" and regressions["pytest"]["failed"] == 0,
        "repository_and_privacy": repository["status"] == "PASS",
        "no_bunit_or_pi2b_training": not any(token in output("ps", "-eo", "cmd").lower() for token in ("train_bunit", "pi2b")),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    bva_outcome = statistics["BVA_outcome"]
    bva_decision = "S4_3_PI2U_" + bva_outcome
    decision = {
        "schema": "tactile3d-unit.s4-3-pi2u-final-decision.v2",
        "status": status,
        "official_unit_compatibility": compatibility["classification"],
        "bva_mechanism": bva_decision,
        "pi2b_readiness": statistics["PI2B_readiness"],
        "evaluator_seed": EVALUATOR_SEED,
        "episodes_per_model": 200,
        "total_rollouts": 800,
        "summaries": statistics["summaries"],
        "contrasts": statistics["contrasts"],
        "mechanism_pattern": mechanism["pattern"],
        "claim_boundary": mechanism["interpretation_boundary"],
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "stop": {"BUniT_training": "NOT_STARTED", "PI2B": "NOT_STARTED", "next_action_requires_new_user_instruction": True},
    }
    audit = {
        "schema": "tactile3d-unit.s4-3-pi2u-final-repository-audit.v2",
        "status": status,
        "required_official_audit_artifacts": list(OFFICIAL_AUDIT_ARTIFACTS),
        "required_bva_artifacts": ["va_bridge_protocol.json", *BVA_PREEXISTING_ARTIFACTS, "s4_2_immutability.json", "environment_integrity.json", "regression_tests.json", "final_decision.json", "HUMAN_ACCEPTANCE.md"],
        "checkpoint_audit": checkpoints,
        "frozen_source_gates": {key: "PASS" if value else "FAIL" for key, value in source_gates.items()},
        "runtime_gates": {key: "PASS" if value else "FAIL" for key, value in runtime_gates.items()},
        "quarantine_event_counts": quarantine_counts,
        "quarantine_records_valid": quarantine_records_valid,
        "malformed_diagnostic_rows": malformed_diagnostic_rows,
        "temporal_contract_gates": {key: "PASS" if value else "FAIL" for key, value in temporal_gates.items()},
        "pi2a_protected_inputs": protected_rows,
        "repository": repository,
        "gates": decision["gates"],
    }
    if status != "PASS":
        TMP_FAILURE.parent.mkdir(parents=True, exist_ok=True)
        TMP_FAILURE.write_text(json.dumps({"status": status, "decision": decision, "audit": audit, "s4_2": s42, "environment": environment, "regressions": regressions}, indent=2, sort_keys=True) + "\n")
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("PI2U_FINAL_AUDIT_FAIL: " + ", ".join(failed))

    acceptance_rows = [
        ("Official UniT source/compatibility", "PASS", compatibility["classification"]),
        ("VA bridge structural gates", load(ARTIFACTS / "va_bridge_metrics.json")["status"], "va_bridge_metrics.json"),
        ("Contact leakage", load(ARTIFACTS / "contact_leakage_audit.json")["status"], "contact_leakage_audit.json"),
        ("BVA seed42 30k", completion["status"], EXPECTED_CHECKPOINTS["BVA"]),
        ("Fresh seed6 / 800 rollouts", "PASS", next(iter(reset_hashes))),
        ("S4.2/B0/B1/B2 immutability", "PASS", "s4_2_immutability.json"),
        ("Environment integrity", environment["status"], "environment_integrity.json"),
        ("Full regression suite", regressions["status"], f"{regressions['pytest']['passed']} passed; {regressions['pytest']['skipped']} skipped; 0 failed"),
        ("Visual evidence", visual["status"], f"{len(visual['files'])} source-backed figures"),
        ("Final mechanism", "PASS", bva_decision),
        ("PI2B readiness", "DECISION_ONLY", statistics["PI2B_readiness"]),
    ]
    acceptance = "# S4.3-PI2U Human Acceptance\n\n| Gate | Status | Evidence / decision |\n|---|---|---|\n" + "".join(f"| {name} | {gate} | `{evidence}` |\n" for name, gate, evidence in acceptance_rows) + "\nBUniT and PI2B were not started. A new user instruction is required for any follow-up.\n"

    atomic(ARTIFACTS / "va_bridge_protocol.json", bridge_protocol)
    atomic(ARTIFACTS / "s4_2_immutability.json", s42)
    atomic(ARTIFACTS / "environment_integrity.json", environment)
    atomic(ARTIFACTS / "regression_tests.json", regressions)
    atomic(ARTIFACTS / "final_repository_audit.json", audit)
    atomic(ARTIFACTS / "final_decision.json", decision)
    atomic(ARTIFACTS / "HUMAN_ACCEPTANCE.md", acceptance)
    print(json.dumps({"status": status, "official_unit": compatibility["classification"], "BVA": bva_decision, "PI2B": statistics["PI2B_readiness"]}, sort_keys=True))


if __name__ == "__main__":
    main()
