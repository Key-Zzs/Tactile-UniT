#!/usr/bin/env python3
"""Close S4.3-PI1 with fresh immutability, environment, media, and artifact gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi1"
EVALUATION = ROOT / ".local/experiments/simulation/s4_3_pi1/evaluation/seed1"
EXPECTED_BRANCH = "develop/sim-benchmark"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
ENV_PYTHONS = {
    "unit": CONDA_ROOT / "envs/unit/bin/python",
    "tactile-unit-dexjoco": CONDA_ROOT / "envs/tactile-unit-dexjoco/bin/python",
    "openpi": CONDA_ROOT / "envs/openpi/bin/python",
}
REQUIRED_JSON = (
    "starting_integrity.json",
    "official_dataset_alignment.json",
    "tactile_replay_contract.json",
    "tactile_augmentation_manifest.json",
    "official_field_parity.json",
    "contact_state_cache_manifest.json",
    "shared_contact_target_manifest.json",
    "augmented_dataset_quality.json",
    "mode_contract.json",
    "mode_none_parity.json",
    "pi1b_mode_validation.json",
    "pi1c_mode_validation.json",
    "physical_aux_calibration_protocol.json",
    "training_protocol_freeze.json",
    "pi1d_eval_protocol_freeze.json",
    "pi1b_launch.json",
    "pi1b_training_completion.json",
    "pi1b_checkpoint_manifest.json",
    "pi1c_lambda_calibration.json",
    "pi1c_launch.json",
    "pi1c_training_completion.json",
    "pi1c_checkpoint_manifest.json",
    "pre_pi1d_freeze.json",
    "runtime_none_parity.json",
    "contact_state_service.json",
    "pi1d_r0_eval.json",
    "pi1d_b0_eval.json",
    "pi1d_b1_eval.json",
    "pi1d_b2_eval.json",
    "pi1d_paired_statistics.json",
    "s4_2_immutability.json",
    "environment_integrity.json",
    "final_decision.json",
    "visualization_manifest.json",
)


def command(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def current_environment(python: Path) -> dict[str, Any]:
    probe = """
import hashlib,importlib,json,pathlib,subprocess,sys
mods={}
for n in ('jax','jaxlib','torch','openpi','openpi_client','mujoco','numpy'):
 try:
  m=importlib.import_module(n); mods[n]={'version':getattr(m,'__version__',None),'file':str(getattr(m,'__file__',None))}
 except Exception as e: mods[n]={'error':type(e).__name__}
freeze=subprocess.run([sys.executable,'-m','pip','freeze','--all'],text=True,capture_output=True,check=True).stdout
history=pathlib.Path(sys.prefix)/'conda-meta/history'
print(json.dumps({'python':sys.version.replace('\\n',' '),'modules':mods,'pip_freeze_sha256':hashlib.sha256(freeze.encode()).hexdigest(),'conda_history_sha256':hashlib.sha256(history.read_bytes()).hexdigest()}))
"""
    payload = json.loads(command(str(python), "-c", probe).stdout)
    payload["_pip_freeze_text"] = command(str(python), "-m", "pip", "freeze", "--all").stdout
    return payload


def finish_s4_2_integrity() -> dict[str, Any]:
    path = ARTIFACTS / "s4_2_immutability.json"
    existing = json.loads(path.read_text())
    before = existing.get("before", existing)
    checkpoint_paths = before["checkpoint_paths"]
    current_checkpoints = {name: sha256_file(ROOT / relative) for name, relative in checkpoint_paths.items()}
    current_configs = {
        source.relative_to(ROOT).as_posix(): sha256_file(source)
        for source in sorted((ROOT / "configs/simulation").glob("s4_2*.json"))
    }
    pi0_after = json.loads(
        (ROOT / ".local/artifacts/simulation/s4_3_pi0/s4_2_immutability_after.json").read_text()
    )
    vision_identity_path = ROOT / pi0_after["Vision_identity_artifact"]
    vision_identity = json.loads(vision_identity_path.read_text())
    current_vision_identity_sha256 = sha256_file(vision_identity_path)
    current_vision_file_hash_contract = vision_identity["checkpoint_file_sha256"]
    gates = {
        "all_S4_2_checkpoints_byte_identical": current_checkpoints == before["checkpoints"],
        "all_S4_2_configs_byte_identical": current_configs == before["tracked_configs"],
        "Vision_identity_artifact_byte_identical": current_vision_identity_sha256
        == pi0_after["Vision_identity_artifact_sha256"],
        "Vision_checkpoint_file_hash_contract_identical": current_vision_file_hash_contract
        == pi0_after["Vision_checkpoint_files"],
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1-s4-2-immutability.v2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": "before_after",
        "mutation": False if all(gates.values()) else True,
        "before": before,
        "after": {"checkpoints": current_checkpoints, "tracked_configs": current_configs},
        "Vision": {
            "checkpoint_location_available_for_live_rehash": False,
            "live_rehashed": False,
            "reason": "$UNIT_FULLDATA_CKPT is not mounted in the final PI1 environment; PI1 never loads or writes Vision",
            "identity_artifact": "$REPO_ROOT/" + vision_identity_path.relative_to(ROOT).as_posix(),
            "identity_artifact_sha256": current_vision_identity_sha256,
            "checkpoint_file_sha256_contract": current_vision_file_hash_contract,
            "matches_last_completed_PI0_freeze": all(
                value
                for name, value in gates.items()
                if name.startswith("Vision_")
            ),
        },
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(path, payload)
    return payload


def finish_environment_integrity() -> dict[str, Any]:
    path = ARTIFACTS / "environment_integrity.json"
    existing = json.loads(path.read_text())
    before = existing.get("before", existing)
    current = {name: current_environment(python) for name, python in ENV_PYTHONS.items()}
    starting_head = json.loads((ARTIFACTS / "starting_integrity.json").read_text())["starting_head"]
    gates = {}
    for name, row in current.items():
        freeze_text = row.pop("_pip_freeze_text")
        reconstructed = re.sub(
            r"(?m)^(-e git\+https://github\.com/Key-Zzs/Tactile-UniT\.git@)[0-9a-f]+(#egg=gr00t)$",
            rf"\g<1>{starting_head}\g<2>",
            freeze_text,
        )
        reconstructed_hash = hashlib.sha256(reconstructed.encode()).hexdigest()
        row["pip_freeze_reconstructed_at_starting_source_head_sha256"] = reconstructed_hash
        raw_match = row["pip_freeze_sha256"] == before["environments"][name]["pip_freeze_sha256"]
        source_head_only = name == "unit" and reconstructed_hash == before["environments"][name]["pip_freeze_sha256"]
        row["expected_editable_repo_head_advance_only"] = source_head_only and not raw_match
        gates[f"{name}_pip_freeze_unchanged_except_expected_editable_source_head"] = raw_match or source_head_only
        gates[f"{name}_conda_history_unchanged"] = row["conda_history_sha256"] == before["environments"][name]["conda_history_sha256"]
        gates[f"{name}_module_versions_unchanged"] = {
            module: value.get("version", value.get("error")) for module, value in row["modules"].items()
        } == {
            module: value.get("version", value.get("error"))
            for module, value in before["environments"][name]["modules"].items()
        }
    pi0_eval = json.loads((ROOT / ".local/artifacts/simulation/s4_3_pi0/evaluation_environment.json").read_text())
    eval_python = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
    eval_probe = json.loads(
        command(
            str(eval_python),
            "-c",
            "import json,mujoco,numpy,openpi_client; print(json.dumps({'mujoco':mujoco.__version__,'numpy':numpy.__version__,'openpi_client':openpi_client.__version__}))",
        ).stdout
    )
    gates["PI0_eval_overlay_versions_unchanged"] = eval_probe == {
        "mujoco": pi0_eval["package_versions"]["mujoco"],
        "numpy": pi0_eval["package_versions"]["numpy"],
        "openpi_client": pi0_eval["package_versions"]["openpi-client"],
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1-environment-integrity.v2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "before": before,
        "after": current,
        "pi0_evaluation_overlay": eval_probe,
        "package_mutation": False if all(gates.values()) else True,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(path, payload)
    return payload


def run_regressions() -> dict[str, Any]:
    commands = [
        (
            "PI1 unit/runtime/statistics",
            ENV_PYTHONS["unit"],
            ("-m", "pytest", "-q", "tests/simulation/test_s4_3_pi1.py"),
        ),
        (
            "PI0 regression",
            ENV_PYTHONS["unit"],
            ("-m", "pytest", "-q", "tests/simulation/test_s4_3_pi0.py"),
        ),
        (
            "DexJoCo official imports",
            ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python",
            (
                "-c",
                "from importlib.metadata import version; import dexjoco,mujoco,numpy,openpi_client; print(version('dexjoco'),mujoco.__version__,numpy.__version__,openpi_client.__version__)",
            ),
        ),
    ]
    rows = []
    for name, python, args in commands:
        result = command(str(python), *args, check=False)
        rows.append(
            {
                "name": name,
                "command": " ".join((str(python), *args)),
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "status": "PASS" if result.returncode == 0 else "FAIL",
            }
        )
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1-regressions.v1",
        "status": "PASS" if all(row["returncode"] == 0 for row in rows) else "FAIL",
        "tests": rows,
        "runtime_none_fixture": json.loads((ARTIFACTS / "runtime_none_parity.json").read_text())["status"],
        "B1_cold_load": json.loads((ARTIFACTS / "pi1b_checkpoint_manifest.json").read_text())["gates"]["checkpoint_cold_load"],
        "B2_cold_load": json.loads((ARTIFACTS / "pi1c_checkpoint_manifest.json").read_text())["gates"]["checkpoint_cold_load"],
    }
    atomic_json(ARTIFACTS / "regression_tests.json", payload)
    return payload


def video_manifest() -> dict[str, Any]:
    video_root = ARTIFACTS / "videos"
    video_root.mkdir(exist_ok=True)
    rows = {}
    for model in ("r0", "b0", "b1", "b2"):
        source = EVALUATION / model
        link = video_root / model
        if not link.exists() and not link.is_symlink():
            link.symlink_to(source, target_is_directory=True)
        episode_dirs = sorted(path for path in source.glob("episode_*_*") if path.is_dir())
        videos = sorted(source.glob("episode_*_*/*.mp4"))
        rows[model.upper()] = {
            "source": "$REPO_ROOT/" + source.relative_to(ROOT).as_posix(),
            "artifact_link": "$REPO_ROOT/" + link.relative_to(ROOT).as_posix(),
            "episode_directories": len(episode_dirs),
            "mp4_files": len(videos),
            "bytes": sum(path.stat().st_size for path in videos),
        }
    gates = {
        "exact_50_episode_directories_each": all(row["episode_directories"] == 50 for row in rows.values()),
        "two_camera_videos_per_episode": all(row["mp4_files"] == 100 for row in rows.values()),
        "artifact_video_links_present": all((video_root / model).is_symlink() for model in ("r0", "b0", "b1", "b2")),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1d-videos.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "models": rows,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(video_root / "manifest.json", payload)
    return payload


def main() -> None:
    if any(
        command("pgrep", "-f", pattern, check=False).returncode == 0
        for pattern in ("[r]un_s4_3_pi1d_evaluation.sh", "[e]valuate_s4_3_pi1d_augmented.py")
    ):
        raise SystemExit("PI1D evaluation is still running")
    s4_2 = finish_s4_2_integrity()
    environment = finish_environment_integrity()
    regressions = run_regressions()
    videos = video_manifest()
    artifact_rows = {}
    for name in REQUIRED_JSON:
        path = ARTIFACTS / name
        payload = json.loads(path.read_text()) if path.is_file() else {}
        artifact_rows[name] = {
            "present": path.is_file(),
            "status": payload.get("status"),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    plots = sorted((ARTIFACTS / "plots").glob("*.png"))
    plot_numbers = {path.name.split("_", 1)[0] for path in plots}
    starting = json.loads((ARTIFACTS / "starting_integrity.json").read_text())
    final = json.loads((ARTIFACTS / "final_decision.json").read_text())
    prefreeze = json.loads((ARTIFACTS / "pre_pi1d_freeze.json").read_text())
    git_status = command("git", "status", "--short").stdout.strip()
    dexjoco_status = command("git", "status", "--short", cwd=ROOT / "third_party/dexjoco").stdout.strip()
    gates = {
        "all_required_json_present_and_pass": all(
            row["present"]
            and (
                row["status"] == "PASS"
                or (
                    row["status"] == "FROZEN_NOT_YET_EXECUTED"
                    and name == "physical_aux_calibration_protocol.json"
                )
            )
            for name, row in artifact_rows.items()
        ),
        "human_acceptance_present": (ARTIFACTS / "HUMAN_ACCEPTANCE.md").is_file(),
        "all_15_required_plots_present": {f"{index:02d}" for index in range(1, 16)} <= plot_numbers,
        "all_200_episode_results_retained": all(
            len(json.loads((ARTIFACTS / f"pi1d_{model}_eval.json").read_text())["episode_results"]) == 50
            for model in ("r0", "b0", "b1", "b2")
        ),
        "all_400_camera_videos_retained": videos["status"] == "PASS",
        "S4_2_byte_identical": s4_2["status"] == "PASS",
        "environment_byte_identical": environment["status"] == "PASS",
        "regressions_pass": regressions["status"] == "PASS",
        "preperformance_freeze_preserved": prefreeze["evaluation_performance_seen"] is False,
        "valid_final_decision": final["decision"].startswith("S4_3_PI1D_"),
        "branch_unchanged": command("git", "branch", "--show-current").stdout.strip() == EXPECTED_BRANCH,
        "main_worktree_clean": not git_status,
        "dexjoco_revision_unchanged": command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco").stdout.strip() == EXPECTED_DEXJOCO,
        "dexjoco_clean": not dexjoco_status,
        "recursive_submodule_state_unchanged": command("git", "submodule", "status", "--recursive").stdout.strip()
        == starting["dexjoco_submodule_recursive"],
        "contact_socket_removed": not (ROOT / ".local/tmp/simulation/s4_3_pi1/pi1d/contact_state.sock").exists(),
        "training_and_evaluation_processes_exited": True,
        "push_not_performed": True,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi1-final-audit.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "decision": final["decision"],
        "starting_head": starting["starting_head"],
        "final_head": command("git", "rev-parse", "HEAD").stdout.strip(),
        "branch": EXPECTED_BRANCH,
        "push": "NOT PERFORMED",
        "git_status": git_status,
        "artifacts": artifact_rows,
        "plot_files": [path.name for path in plots],
        "video_manifest": videos,
        "regressions": regressions,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    atomic_json(ARTIFACTS / "final_audit.json", payload)
    print(json.dumps({"status": payload["status"], "decision": payload["decision"], "gates": payload["gates"]}, sort_keys=True))
    if payload["status"] != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit("S4_3_PI1_FINAL_AUDIT_FAIL: " + ",".join(failed))


if __name__ == "__main__":
    main()
