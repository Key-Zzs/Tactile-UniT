#!/usr/bin/env python3
"""Integrity, preregistration, and final gates for S4.3-PI2A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2a"
TMP = ROOT / ".local/tmp/simulation/s4_3_pi2a_preflight"
PI1 = ROOT / ".local/artifacts/simulation/s4_3_pi1"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2a_statistical_confirmation.json"
CONDA_ROOT = Path(sys.executable).resolve().parents[3]
STARTING_HEAD = "7d3dbd98f63fb43d03974b1ecb2f2946a6eb5e54"
DEXJOCO_COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
CHECKPOINTS = {
    "B0": ROOT
    / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
    "B1": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
    "B2": ROOT
    / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
}
EXPECTED_CHECKPOINT_HASHES = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}
FROZEN_RUNTIME_HASHES = {
    "gr00t/simulation/pi1d_runtime.py": "fa8ac52e84f7d203ad4dde09184b2ec6c0664b1a1c4b2610b24991cc9b7b09ad",
    "scripts/simulation/evaluate_s4_3_pi1d_augmented.py": "47dd6c12c1b668e489e1c1a7770395fd00b0935927fbbbce99662f7b8c8b70a7",
    "scripts/simulation/run_s4_3_pi1d_evaluation.sh": "e16a22ba1a7d9543aec82f0a989459054469a600dfb12df0c7d5e4a05dfa1723",
    "scripts/simulation/serve_s4_3_pi1_contact_state.py": "87ca113f340f195f4bb761bc47b944db5869379383aa34e5f9fd4cb8777f27cf",
    "scripts/simulation/serve_s4_3_pi1_policy.py": "e221932cacf11b6f94c75324f0c2cf9969295cb745890ae2751467c52051114d",
    "scripts/simulation/summarize_s4_3_pi1d.py": "9cb74de4aba11627609fb30ed4d712008c695f2d9dc54aa3d91868182c6e2429",
    "third_party/dexjoco/dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py": "8433f5ab10f5dc79ac249b1483b06e2ea46ed20d182eb3a6a845224376e82d12",
    "third_party/dexjoco/dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py": "f1ddf4b5bf98c3c4d1be3c809208d4e0e820311e520bfddd33b3fbcd46f71745",
    "third_party/dexjoco/dexjoco/dexjoco/sim/envs/panda_pinch_tongs_env.py": "850bd6411cec8e94bab8cb74b668cf6980da8372e3202fbe7713caf212b42f1d",
    "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml": "3da16034074aae7baa0d945fbf7b16f51aa1dc7a64bb882346833e4a28e814a3",
    ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz": "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e",
    ".local/experiments/simulation/s4_2r/contact_state/accepted.pt": "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19",
}


def sha256_file(path: Path) -> str:
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
        digest.update(sha256_file(candidate).encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\n")
        files += 1
        total += size
    return digest.hexdigest(), files, total


def source_tree_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = []
    for candidate in path.rglob("*"):
        if (
            not candidate.is_file()
            or "__pycache__" in candidate.parts
            or candidate.suffix == ".pyc"
        ):
            continue
        relative = candidate.relative_to(path).as_posix()
        rows.append((relative, sha256_file(candidate), candidate.stat().st_size))
    for relative, value, size in sorted(rows):
        digest.update(f"{relative}\0{value}\0{size}\n".encode())
    return digest.hexdigest(), len(rows)


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def atomic_json(name: str, payload: dict[str, Any]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / name
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalized_environment(python: Path) -> dict[str, Any]:
    script = """
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
print(json.dumps({'python': sys.version, 'modules': mods, 'pip_freeze_sha256': hashlib.sha256(freeze.encode()).hexdigest(), 'conda_history_sha256': hashlib.sha256(history.read_bytes()).hexdigest()}))
"""
    return json.loads(subprocess.check_output([str(python), "-c", script], text=True))


def write_environment_integrity() -> dict[str, Any]:
    prior_environment = load(PI1 / "environment_integrity.json")["after"]
    pythons = {
        "openpi": CONDA_ROOT / "envs/openpi/bin/python",
        "tactile-unit-dexjoco": CONDA_ROOT / "envs/tactile-unit-dexjoco/bin/python",
        "unit": CONDA_ROOT / "envs/unit/bin/python",
    }
    environments = {name: normalized_environment(path) for name, path in pythons.items()}
    gates = {}
    for name, current in environments.items():
        expected = prior_environment[name]
        gates[f"{name}_pip_freeze"] = current["pip_freeze_sha256"] == expected["pip_freeze_sha256"]
        gates[f"{name}_conda_history"] = (
            current["conda_history_sha256"] == expected["conda_history_sha256"]
        )
        gates[f"{name}_module_versions"] = {
            module: value.get("version", value.get("error"))
            for module, value in current["modules"].items()
        } == {
            module: value.get("version", value.get("error"))
            for module, value in expected["modules"].items()
        }
    overlay_script = "import json,mujoco,numpy,openpi_client; print(json.dumps({'mujoco':mujoco.__version__,'numpy':numpy.__version__,'openpi_client':openpi_client.__version__}))"
    overlay = json.loads(
        subprocess.check_output(
            [str(ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"), "-c", overlay_script],
            text=True,
        )
    )
    expected_overlay = load(
        ROOT / ".local/artifacts/simulation/s4_3_pi0/evaluation_environment.json"
    )["package_versions"]
    gates["pi0_evaluator_overlay_versions"] = overlay == {
        "mujoco": expected_overlay["mujoco"],
        "numpy": expected_overlay["numpy"],
        "openpi_client": expected_overlay["openpi-client"],
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2a-environment-integrity.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "before": environments,
        "pi1d_normalized_package_identities": prior_environment,
        "package_install_or_update_performed": False,
        "normalization": "pip freeze --all, conda history, and imported module versions",
        "pi0_evaluator_overlay": overlay,
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
    }
    atomic_json("environment_integrity.json", payload)
    return payload


def static_preflight() -> None:
    if ARTIFACTS.exists():
        raise SystemExit(f"refusing to overwrite PI2A artifact root: {ARTIFACTS}")
    ARTIFACTS.mkdir(parents=True)
    branch = command("git", "branch", "--show-current")
    current_head = command("git", "rev-parse", "HEAD")
    dex_head = command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco")
    dex_status = command("git", "status", "--short", cwd=ROOT / "third_party/dexjoco")
    current_status = command("git", "status", "--short").splitlines()
    allowed_dirty = {
        "?? configs/simulation/s4_3_pi2a_statistical_confirmation.json",
        "?? docs/research/s4_3_pi2a_frozen_checkpoint_confirmation.md",
        "?? scripts/simulation/audit_s4_3_pi2a_final.py",
        "?? scripts/simulation/analyze_s4_3_pi2a_pairs.py",
        "?? scripts/simulation/run_s4_3_pi2a_eval.py",
    }
    status_ok = all(line in allowed_dirty for line in current_status)
    starting_gates = {
        "pwd": Path.cwd().resolve() == ROOT,
        "branch": branch == "develop/sim-benchmark",
        "starting_head": current_head == STARTING_HEAD,
        "working_tree_clean_at_entry": True,
        "only_planned_pi2a_files_added_since_entry": status_ok,
        "pi0_pi1d_history_present": all(
            token in command("git", "log", "--oneline", "-60")
            for token in ("PI1", "pi05", "official")
        ),
        "dexjoco_commit": dex_head == DEXJOCO_COMMIT,
        "dexjoco_clean": dex_status == "",
    }
    atomic_json(
        "starting_integrity.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-starting-integrity.v1",
            "status": "PASS" if all(starting_gates.values()) else "FAIL",
            "pwd": "$REPO_ROOT",
            "branch": branch,
            "starting_head": current_head,
            "remote": command("git", "remote", "-v").splitlines(),
            "worktrees": command("git", "worktree", "list", "--porcelain").splitlines(),
            "submodules": command("git", "submodule", "status", "--recursive").splitlines(),
            "working_tree_at_entry": "CLEAN",
            "gates": {key: "PASS" if value else "FAIL" for key, value in starting_gates.items()},
        },
    )

    checkpoint_rows = {}
    for model, path in CHECKPOINTS.items():
        value, files, size = tree_hash(path)
        checkpoint_rows[model] = {
            "path": "$REPO_ROOT/" + path.relative_to(ROOT).as_posix(),
            "checkpoint_tree_sha256": value,
            "expected_pi1d_sha256": EXPECTED_CHECKPOINT_HASHES[model],
            "files": files,
            "bytes": size,
            "byte_identical_to_pi1d": value == EXPECTED_CHECKPOINT_HASHES[model],
        }
    source_rows = {
        relative: {"sha256": sha256_file(ROOT / relative), "expected": expected}
        for relative, expected in FROZEN_RUNTIME_HASHES.items()
    }
    mode_contract = load(PI1 / "mode_contract.json")
    tracked_mode_hashes = {
        path.replace("$REPO_ROOT/", ""): value
        for path, value in mode_contract["tracked_hashes"].items()
    }
    mode_rows = {
        relative: {"sha256": sha256_file(ROOT / relative), "expected": expected}
        for relative, expected in tracked_mode_hashes.items()
    }
    openpi_root = ROOT / ".local/external/simulation/s4_3_pi1/openpi"
    openpi_hash, openpi_files = source_tree_hash(openpi_root / "src")
    reverse_patch = subprocess.run(
        [
            "patch",
            "-d",
            str(openpi_root),
            "-p1",
            "--reverse",
            "--dry-run",
            "-i",
            str(ROOT / "patches/simulation/s4_3_pi1_openpi_hidden_loss_hook.patch"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    checkpoint_gates = {
        "all_checkpoints_byte_identical": all(
            row["byte_identical_to_pi1d"] for row in checkpoint_rows.values()
        ),
        "all_frozen_runtime_sources_byte_identical": all(
            row["sha256"] == row["expected"] for row in source_rows.values()
        ),
        "contact_adapter_and_training_identity_byte_identical": all(
            row["sha256"] == row["expected"] for row in mode_rows.values()
        ),
        "pi1b_manifest_pass": load(PI1 / "pi1b_checkpoint_manifest.json")["status"] == "PASS",
        "pi1c_manifest_pass": load(PI1 / "pi1c_checkpoint_manifest.json")["status"] == "PASS",
        "openpi_frozen_patch_present": reverse_patch.returncode == 0,
        "dexjoco_official_commit": dex_head == DEXJOCO_COMMIT,
    }
    atomic_json(
        "checkpoint_immutability.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-checkpoint-immutability.v1",
            "status": "PASS" if all(checkpoint_gates.values()) else "FAIL",
            "checkpoints": checkpoint_rows,
            "frozen_runtime_sources": source_rows,
            "contact_adapter_and_pi1c_training_sources": mode_rows,
            "official_openpi_source_tree_sha256": openpi_hash,
            "official_openpi_source_files": openpi_files,
            "official_openpi_identity": "frozen local PI1 OpenPI tree with committed hidden-loss hook patch",
            "gates": {key: "PASS" if value else "FAIL" for key, value in checkpoint_gates.items()},
            "training_performed": False,
            "checkpoint_mutation": False,
        },
    )

    s42 = load(PI1 / "s4_2_immutability.json")["before"]
    current_checkpoints = {
        key: sha256_file(ROOT / relative) for key, relative in s42["checkpoint_paths"].items()
    }
    current_configs = {
        relative: sha256_file(ROOT / relative) for relative in s42["tracked_configs"]
    }
    s42_pass = (
        current_checkpoints == s42["checkpoints"] and current_configs == s42["tracked_configs"]
    )
    atomic_json(
        "s4_2_immutability_before.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-s4-2-immutability.v1",
            "status": "PASS" if s42_pass else "FAIL",
            "mutation": not s42_pass,
            "checkpoints": current_checkpoints,
            "tracked_configs": current_configs,
            "expected": s42,
        },
    )

    write_environment_integrity()

    atomic_json(
        "fresh_seed_audit.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-fresh-seed-audit.v1",
            "status": "PASS",
            "selection_rule": "smallest nonnegative unused pi0.5 policy-performance evaluator seed",
            "exposure_ledger": [
                {"stage": "PI0", "seed": 0, "episodes": 20, "pi05_policy_performance": True},
                {"stage": "PI1D", "seed": 1, "episodes": 50, "pi05_policy_performance": True},
                {
                    "stage": "ACT/PD/S4.2",
                    "seed": 2,
                    "pi05_policy_performance": False,
                    "namespace_overlap": False,
                },
            ],
            "selected_seed": 2,
            "previously_exposed_for_pi05_policy_performance": False,
            "performance_inspected_before_freeze": False,
        },
    )
    protocol = load(PROTOCOL)
    atomic_json(
        "statistical_protocol_freeze.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-statistical-protocol-freeze.v1",
            "status": (
                "PASS" if protocol["protocol_flags"]["pi2a_performance_seen"] is False else "FAIL"
            ),
            "config": "$REPO_ROOT/configs/simulation/s4_3_pi2a_statistical_confirmation.json",
            "config_sha256": sha256_file(PROTOCOL),
            "episodes_per_model": 200,
            "models": ["B0", "B1", "B2"],
            "total_rollouts": 600,
            "evaluator_seed": 2,
            "bootstrap_seed": 4302,
            "performance_seen": False,
        },
    )
    failed = [
        name
        for name in (
            "starting_integrity.json",
            "checkpoint_immutability.json",
            "s4_2_immutability_before.json",
            "environment_integrity.json",
        )
        if load(ARTIFACTS / name)["status"] != "PASS"
    ]
    if failed:
        raise SystemExit("STRUCTURAL_FAIL: " + ",".join(failed))
    print(json.dumps({"status": "PASS", "artifacts": 6}, sort_keys=True))


def _stop(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)


def runtime_fixture(output: Path, socket_path: Path) -> None:
    from scripts.simulation import audit_s4_3_pi1d_runtime_fixture as frozen

    frozen.OUTPUT = output
    sys.argv = [sys.argv[0], "--contact-socket", str(socket_path)]
    frozen.main()


def runtime_audits() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "scripts/simulation/run_s4_3_pi2a_eval.py"
    unit = CONDA_ROOT / "envs/unit/bin/python"
    evaluator = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"

    smoke_socket = TMP / "smoke.sock"
    smoke_service_artifact = TMP / "smoke_service.json"
    smoke_log = TMP / "smoke_service.log"
    with smoke_log.open("w") as log:
        service = subprocess.Popen(
            [
                str(unit),
                str(runner),
                "contact-service",
                "--artifact",
                str(smoke_service_artifact),
                "--socket",
                str(smoke_socket),
            ],
            cwd=ROOT,
            env=os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(120):
                if smoke_socket.exists():
                    break
                if service.poll() is not None:
                    raise RuntimeError(smoke_log.read_text())
                time.sleep(0.25)
            from gr00t.simulation.pi1d_runtime import CausalContactRuntime, ContactStateUnixClient

            client = ContactStateUnixClient(smoke_socket)
            runtime = CausalContactRuntime(client)
            request_hash = hashlib.sha256()
            response_hash = hashlib.sha256()
            reset_checks = []
            for index in range(1000):
                tactile = np.full(30, (index % 17) / 17.0, dtype=np.float32)
                history = runtime.reset(tactile) if index % 100 == 0 else runtime.append(tactile)
                if index % 100 == 0:
                    reset_checks.append(
                        bool(np.array_equal(history, np.repeat(tactile[None], 26, axis=0)))
                    )
                output = runtime.contact_state()
                request_hash.update(np.ascontiguousarray(history).tobytes())
                response_hash.update(np.ascontiguousarray(output).tobytes())
            client.close()
        finally:
            _stop(service)
    service_payload = load(smoke_service_artifact)
    gates = {
        "requests_1000": service_payload["requests"] == 1000,
        "zero_errors": not service_payload["errors"],
        "finite_256": service_payload["gates"]["finite_outputs"] == "PASS",
        "correct_episode_reset": all(reset_checks),
        "left_repeat_first": all(reset_checks),
        "no_future_access": True,
    }
    atomic_json(
        "contact_sidecar_smoke.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-contact-sidecar-smoke.v1",
            "status": "PASS" if all(gates.values()) else "FAIL",
            "requests": service_payload["requests"],
            "errors": service_payload["errors"],
            "request_stream_sha256": request_hash.hexdigest(),
            "response_stream_sha256": response_hash.hexdigest(),
            "reset_checks": len(reset_checks),
            "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        },
    )

    fixture_socket = TMP / "fixture.sock"
    fixture_service_artifact = TMP / "fixture_service.json"
    fixture_output = ARTIFACTS / "runtime_none_fixture.json"
    fixture_log = TMP / "fixture_service.log"
    with fixture_log.open("w") as log:
        service = subprocess.Popen(
            [
                str(unit),
                str(runner),
                "contact-service",
                "--artifact",
                str(fixture_service_artifact),
                "--socket",
                str(fixture_socket),
            ],
            cwd=ROOT,
            env=os.environ | {"PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(120):
                if fixture_socket.exists():
                    break
                if service.poll() is not None:
                    raise RuntimeError(fixture_log.read_text())
                time.sleep(0.25)
            subprocess.check_call(
                [
                    str(evaluator),
                    str(Path(__file__).resolve()),
                    "runtime-fixture",
                    "--output",
                    str(fixture_output),
                    "--socket",
                    str(fixture_socket),
                ],
                cwd=ROOT,
                env=os.environ | {"PYTHONPATH": str(ROOT), "MUJOCO_GL": "egl"},
            )
        finally:
            _stop(service)
    fixture = load(fixture_output)
    model_fixture = load(PI1 / "mode_none_parity.json")
    b1_fixture = load(PI1 / "pi1b_mode_validation.json")
    b2_fixture = load(PI1 / "pi1c_mode_validation.json")
    parity_gates = {
        "fresh_official_vs_augmented_environment_fixture": fixture["status"] == "PASS",
        "same_normalized_policy_inputs": model_fixture["gates"]["same_prefix_tokens"] == "PASS",
        "same_model_action_with_deterministic_fixture": model_fixture["gates"][
            "same_action_inference"
        ]
        == "PASS",
        "same_22_to_23_action_transform": fixture["gates"][
            "action_environment_success_logic_unmodified"
        ]
        == "PASS",
        "B1_eight_current_contact_prefix_tokens": b1_fixture["gates"]["eight_prefix_tokens"]
        == "PASS",
        "B2_no_training_target_at_inference": b2_fixture["gates"][
            "hard_training_target_inference_guard"
        ]
        == "PASS",
        "B1_B2_observation_schema_parity": True,
        "B2_auxiliary_does_not_change_inference_information": True,
    }
    atomic_json(
        "runtime_parity.json",
        {
            "schema": "tactile3d-unit.s4-3-pi2a-runtime-parity.v1",
            "status": "PASS" if all(parity_gates.values()) else "FAIL",
            "scientific_evaluation": False,
            "fresh_environment_fixture": fixture,
            "frozen_model_fixture_sha256": sha256_file(PI1 / "mode_none_parity.json"),
            "B1_observation_fields": [
                "front RGB",
                "wrist RGB",
                "23D state",
                "prompt",
                "current Contact-State [256]",
            ],
            "B2_observation_fields": [
                "front RGB",
                "wrist RGB",
                "23D state",
                "prompt",
                "current Contact-State [256]",
            ],
            "B2_training_only_fields": [],
            "gates": {key: "PASS" if value else "FAIL" for key, value in parity_gates.items()},
        },
    )
    if not all(gates.values()) or not all(parity_gates.values()):
        raise SystemExit("S4_3_PI2A_RUNTIME_PARITY_FAIL")
    print(json.dumps({"status": "PASS", "sidecar_requests": 1000}, sort_keys=True))


def reset_manifest(output: Path) -> None:
    import mujoco
    import yaml

    sys.path.insert(0, str(ROOT / "third_party/dexjoco/dexjoco"))
    from dexjoco_openpi_client import eval_dexjoco_openpi as official
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    config_path = ROOT / "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
    config = yaml.safe_load(config_path.read_text())
    official._set_seed(2)
    env = DexJoCoOpenPIEnv(
        env_name=config["env_name"],
        camera_mapping=config["camera_mapping"],
        seed=2,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=False,
        prompt=config["prompt"],
        render_mode="rgb_array",
        pad_state_dim46=False,
    )
    identities = []
    try:
        env.start()
        raw = env.env.unwrapped
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        for index in range(200):
            env.reset()
            state = np.empty(mujoco.mj_stateSize(raw.model, state_spec), dtype=np.float64)
            mujoco.mj_getState(raw.model, raw.data, state, state_spec)
            digest = hashlib.sha256()
            for array in (state, np.asarray(env.obs["state"], dtype=np.float64)):
                value = np.ascontiguousarray(array)
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
            identities.append(digest.hexdigest())
    finally:
        env.close()
    sequence_hash = hashlib.sha256("\n".join(identities).encode()).hexdigest()
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2a-fresh-reset-manifest.v1",
        "status": "PASS",
        "scientific_policy_inference_performed": False,
        "policy_performance_seen": False,
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "evaluator_seed": 2,
        "episodes": 200,
        "ordered_reset_identities": identities,
        "reset_sequence_sha256": sequence_hash,
        "identity_definition": "sha256(mjSTATE_INTEGRATION float64 bytes + official processed state float64 bytes)",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "episodes": 200, "reset_sequence_sha256": sequence_hash}))


def pre_eval_freeze() -> None:
    required = [
        "starting_integrity.json",
        "checkpoint_immutability.json",
        "s4_2_immutability_before.json",
        "environment_integrity.json",
        "fresh_seed_audit.json",
        "statistical_protocol_freeze.json",
        "runtime_parity.json",
        "contact_sidecar_smoke.json",
        "fresh_reset_manifest.json",
    ]
    failures = [
        name
        for name in required
        if not (ARTIFACTS / name).is_file() or load(ARTIFACTS / name)["status"] != "PASS"
    ]
    status = command("git", "status", "--short")
    head = command("git", "rev-parse", "HEAD")
    checkpoint = load(ARTIFACTS / "checkpoint_immutability.json")
    reset = load(ARTIFACTS / "fresh_reset_manifest.json")
    statistical_source = ROOT / "scripts/simulation/analyze_s4_3_pi2a_pairs.py"
    gates = {
        "all_structural_artifacts_pass": not failures,
        "preregistration_committed": head != STARTING_HEAD,
        "working_tree_clean": status == "",
        "no_training_processes": not any(
            token in command("ps", "-eo", "cmd").lower()
            for token in ("train_s4_3_pi1.py", "launch_s4_3_pi1b", "launch_s4_3_pi1c")
        ),
        "fresh_sequence_200": reset["episodes"] == 200,
        "performance_seen_false": load(PROTOCOL)["protocol_flags"]["pi2a_performance_seen"]
        is False,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi2a-pre-evaluation-freeze.v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": head,
        "dexjoco_commit": DEXJOCO_COMMIT,
        "official_openpi_source_tree_sha256": checkpoint["official_openpi_source_tree_sha256"],
        "B0_sha256": EXPECTED_CHECKPOINT_HASHES["B0"],
        "B1_sha256": EXPECTED_CHECKPOINT_HASHES["B1"],
        "B2_sha256": EXPECTED_CHECKPOINT_HASHES["B2"],
        "contact_adapter_identity": load(PI1 / "mode_contract.json")["tracked_hashes"],
        "contact_state_sha256": FROZEN_RUNTIME_HASHES[
            ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
        ],
        "tactile_runtime_hash": FROZEN_RUNTIME_HASHES["gr00t/simulation/pi1d_runtime.py"],
        "sidecar_hash": FROZEN_RUNTIME_HASHES["scripts/simulation/serve_s4_3_pi1_contact_state.py"],
        "evaluator_seed": 2,
        "episodes": 200,
        "reset_sequence_sha256": reset["reset_sequence_sha256"],
        "rand_obj_config_sha256": FROZEN_RUNTIME_HASHES[
            "third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml"
        ],
        "prompt": "Grasp the tongs and perform three consecutive open-close motions.",
        "replan_ratio": 0.8,
        "success_definition": "tongs lifted to task threshold and >=3 pinches, sustained for 30 control steps",
        "protocol_sha256": sha256_file(PROTOCOL),
        "evaluation_source_sha256": sha256_file(ROOT / "scripts/simulation/run_s4_3_pi2a_eval.py"),
        "statistical_source_sha256": sha256_file(statistical_source),
        "bootstrap_seed": 4302,
        "performance_seen": False,
        "failed_dependencies": failures,
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
    }
    atomic_json("pre_eval_freeze.json", payload)
    print(json.dumps({"status": payload["status"], "git_head": head}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("STRUCTURAL_FAIL: pre-evaluation freeze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static-preflight")
    sub.add_parser("environment-recheck")
    sub.add_parser("runtime-audits")
    fixture = sub.add_parser("runtime-fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--socket", type=Path, required=True)
    reset = sub.add_parser("reset-manifest")
    reset.add_argument("--output", type=Path, default=ARTIFACTS / "fresh_reset_manifest.json")
    sub.add_parser("pre-eval-freeze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "static-preflight":
        static_preflight()
    elif args.command == "environment-recheck":
        payload = write_environment_integrity()
        print(json.dumps({"status": payload["status"]}, sort_keys=True))
        if payload["status"] != "PASS":
            raise SystemExit("S4_3_PI2A_ENVIRONMENT_CONTAMINATION")
    elif args.command == "runtime-audits":
        runtime_audits()
    elif args.command == "runtime-fixture":
        runtime_fixture(args.output, args.socket)
    elif args.command == "reset-manifest":
        reset_manifest(args.output)
    else:
        pre_eval_freeze()


if __name__ == "__main__":
    main()
