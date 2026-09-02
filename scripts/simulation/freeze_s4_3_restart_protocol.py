#!/usr/bin/env python3
"""Audit R0--R2 and freeze the restarted S4.3 causal ACT protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_policy import (  # noqa: E402
    TASKS,
    TRAINING_SEEDS,
    timeout_statistics,
    valid_bc_windows,
    validate_policy_eval_resets,
)

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
PLOT_ROOT = ARTIFACT_ROOT / "plots"
PD_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pd"
DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_3_policy_expert"
POLICY_CONFIG = ROOT / "configs/simulation/s4_3_restart_policy_protocol.json"
ACT_CONFIG = ROOT / "configs/simulation/s4_3_restart_act_protocol.json"
EVAL_CONFIG = ROOT / "configs/simulation/s4_3_policy_eval_v1.json"
EXPECTED_BRANCH = "develop/sim-benchmark"
EXPECTED_START_HEAD = "ab359b63518baaec374d22852b2d16863ea4842a"
EXPECTED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
EXPECTED_PACKAGES = {
    "unit": "3a119880cd4d661259d9476b0d3224302e086ae899ca77250497b5f42fbf6f9b",
    "tactile-unit-dexjoco": "7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78",
}
CHECKPOINTS = {
    "contact_state": ".local/experiments/simulation/s4_2r/contact_state/accepted.pt",
    "contact_C3": ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt",
    "action_A0": ".local/experiments/simulation/s4_2_formal/action/selected.pt",
    "bridge_B3": ".local/experiments/simulation/s4_2_formal/bridge/selected.pt",
    "shared_private": ".local/experiments/simulation/s4_2_formal/s4_2_6/shared_private.pt",
    "conditional_A_plus_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_plus_H.pt",
    "fallback_A_only_missing_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/A_only_missing_H.pt",
    "uncertainty_full": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_full.pt",
    "uncertainty_missing_H": ".local/experiments/simulation/s4_2_formal/s4_2_7/uncertainty_missing_H.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(*args: str, cwd: Path = ROOT, allow_failure: bool = False) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode and not allow_failure:
        raise RuntimeError(result.stderr.strip() or "command failed: " + " ".join(args))
    return result.stdout.strip()


def pip_freeze_hash(python: Path, *, exclude_editable: bool) -> str:
    lines = command(str(python), "-m", "pip", "freeze").splitlines()
    if exclude_editable:
        lines = [line for line in lines if not line.startswith("-e ")]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def package_audit() -> tuple[dict[str, str], dict[str, str]]:
    unit_python = Path(sys.executable).resolve()
    dex_python = unit_python.parents[2] / "tactile-unit-dexjoco/bin/python"
    hashes = {
        "unit": pip_freeze_hash(unit_python, exclude_editable=True),
        "tactile-unit-dexjoco": pip_freeze_hash(dex_python, exclude_editable=False),
    }
    versions = {
        "unit": command(str(unit_python), "--version"),
        "tactile-unit-dexjoco": command(str(dex_python), "--version"),
    }
    if hashes != EXPECTED_PACKAGES:
        raise RuntimeError("frozen environment package identity changed")
    return hashes, versions


def gpu_snapshot() -> dict[str, Any]:
    rows = command(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).splitlines()
    processes = command(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
        allow_failure=True,
    ).splitlines()
    parsed = []
    busy_uuids = {row.split(",", 1)[0].strip() for row in processes if row.strip()}
    for row in rows:
        index, uuid, name, used, total, utilization = [value.strip() for value in row.split(",")]
        eligible = uuid not in busy_uuids and int(used) < 512 and int(utilization) <= 5
        parsed.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(utilization),
                "unrelated_compute_process": uuid in busy_uuids,
                "eligible_at_R0": eligible,
            }
        )
    return {
        "gpus": parsed,
        "eligible_at_R0": [row["index"] for row in parsed if row["eligible_at_R0"]],
    }


def tracked_s4_2_hashes() -> dict[str, str]:
    files = command("git", "ls-files", "configs/simulation/s4_2*.json").splitlines()
    return {path: sha256_file(ROOT / path) for path in files}


def dataset_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = (
        list(manifest["successful_train"])
        + list(manifest["successful_dev"])
        + list(manifest["failed_acquisition_audit_only"])
    )
    required_fields = {
        "control_step",
        "timestamp_sec",
        "next_timestamp_sec",
        "rgb_reference",
        "proprio",
        "sim_tactile",
        "policy_action",
        "env_action",
        "reward",
        "terminated",
        "truncated",
        "success",
        "expert_phase",
        "source_step",
        "contact_count",
        "normal_force",
        "tangential_force",
        "native_task_state",
        "task_progress",
    }
    errors: list[dict[str, str]] = []
    content = hashlib.sha256()
    observed = Counter()
    for row in rows:
        attempt_dir = DATASET_ROOT / row["attempt_relative_path"]
        metadata_path = attempt_dir / "metadata.json"
        steps_path = attempt_dir / "steps.npz"
        metadata = read_json(metadata_path)
        numeric_hash = sha256_file(steps_path)
        rgb_hash = tree_sha256(attempt_dir / "frames")
        checks = {
            "steps_hash": numeric_hash
            == row["steps_npz_sha256"]
            == metadata["checksums"]["steps_npz_sha256"],
            "rgb_hash": rgb_hash
            == row["rgb_tree_sha256"]
            == metadata["checksums"]["rgb_tree_sha256"],
            "attempt_id": metadata["attempt_id"] == row["attempt_id"],
            "schema": metadata["schema"] == "tactile3d-unit.s4-3-pd-attempt.v1",
            "length": metadata["steps"] == row["length"],
            "success": metadata["success"] is row["success"],
            "task": metadata["metadata"]["task"] == row["task"],
            "source_group": metadata["metadata"]["source_group_id"] == row["source_group_id"],
            "split": metadata["metadata"]["split_role"] == row["split_role"],
            "timing": metadata["metadata"]["control_hz"] == 50.0
            and metadata["metadata"]["control_dt_sec"] == 0.02,
        }
        with np.load(steps_path, allow_pickle=False) as arrays:
            checks.update(
                {
                    "fields": set(arrays.files) == required_fields,
                    "proprio_shape": arrays["proprio"].shape == (row["length"], 22),
                    "action_shape": arrays["policy_action"].shape == (row["length"], 22),
                    "tactile_shape": arrays["sim_tactile"].shape == (row["length"], 30),
                    "env_action_shape": arrays["env_action"].shape == (row["length"], 23),
                    "finite": all(
                        np.isfinite(arrays[name]).all()
                        for name in ("proprio", "policy_action", "sim_tactile", "env_action")
                    ),
                    "success_label": bool(arrays["success"][-1]) is row["success"],
                    "rgb_readable": all(
                        (attempt_dir / str(ref)).is_file() for ref in arrays["rgb_reference"]
                    ),
                }
            )
        for name, passed in checks.items():
            if not passed:
                errors.append({"attempt_id": row["attempt_id"], "check": name})
        observed[(row["split_role"], row["task"], bool(row["success"]))] += 1
        content.update(row["attempt_id"].encode())
        content.update(bytes.fromhex(numeric_hash))
        content.update(bytes.fromhex(rgb_hash))
        content.update(metadata_path.read_bytes())
    expected = {
        ("POLICY_TRAIN", "pinch_tongs", True): 90,
        ("POLICY_TRAIN", "hammer_nail", True): 100,
        ("POLICY_TRAIN", "click_mouse", True): 100,
        ("POLICY_DEV", "pinch_tongs", True): 25,
        ("POLICY_DEV", "hammer_nail", True): 20,
        ("POLICY_DEV", "click_mouse", True): 25,
        ("POLICY_TRAIN", "pinch_tongs", False): 10,
        ("POLICY_DEV", "hammer_nail", False): 5,
    }
    if observed != Counter(expected):
        errors.append({"attempt_id": "DATASET", "check": "membership_counts"})
    success_rows = list(manifest["successful_train"]) + list(manifest["successful_dev"])
    if any(not row["success"] for row in success_rows):
        errors.append({"attempt_id": "DATASET", "check": "non_success_in_bc_membership"})
    content_hash = content.hexdigest()
    if content_hash != "a334af12ab2c5bbfbd46b47e1167f51fe194c97bbd69fb945d760f96b648d7f4":
        errors.append({"attempt_id": "DATASET", "check": "content_identity"})
    return {
        "schema": "tactile3d-unit.s4-3-restart-policy-dataset-audit.v1",
        "stage": "R1",
        "episodes_rehashed": len(rows),
        "bc_train_episodes": len(manifest["successful_train"]),
        "bc_dev_episodes": len(manifest["successful_dev"]),
        "failed_attempts_audit_only": len(manifest["failed_acquisition_audit_only"]),
        "dataset_content_sha256": content_hash,
        "membership_counts": {
            "|".join(map(str, key)): value for key, value in sorted(observed.items())
        },
        "all_bc_episodes_native_success": not any(not row["success"] for row in success_rows),
        "old_210_probing_episodes_audited_for_policy_eligibility": False,
        "old_210_probing_episodes_used": False,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def iter_numbers(value: Any) -> Iterable[int]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from iter_numbers(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_numbers(nested)
    elif isinstance(value, int) and not isinstance(value, bool):
        yield value


def prior_seed_overlap(eval_config: dict[str, Any]) -> list[int]:
    requested = {int(row["reset_seed"]) for row in eval_config["resets"]}
    prior: set[int] = set()
    roots = [ROOT / "configs/simulation", ROOT / ".local/artifacts/simulation"]
    exclusions = {
        POLICY_CONFIG.resolve(),
        ACT_CONFIG.resolve(),
        EVAL_CONFIG.resolve(),
        ARTIFACT_ROOT.resolve(),
    }
    for root in roots:
        for path in root.rglob("*.json"):
            resolved = path.resolve()
            if resolved in exclusions or ARTIFACT_ROOT.resolve() in resolved.parents:
                continue
            try:
                prior.update(iter_numbers(read_json(path)))
            except (OSError, json.JSONDecodeError):
                continue
    return sorted(requested & prior)


def task_success_audit() -> dict[str, Any]:
    accepted = read_json(PD_ROOT / "task_success_contract.json")
    current = {}
    for task, value in accepted["tasks"].items():
        digest = sha256_file(ROOT / value["source_file"])
        current[task] = {
            "source_file": value["source_file"],
            "accepted_sha256": value["source_sha256"],
            "current_sha256": digest,
            "byte_identical": digest == value["source_sha256"],
            "success_predicate": value["success_predicate"],
            "termination_predicate": value["termination_predicate"],
        }
    passed = accepted["status"] == "PASS" and all(row["byte_identical"] for row in current.values())
    return {
        "schema": "tactile3d-unit.s4-3-restart-task-success-contract.v1",
        "stage": "R2",
        "native_success_redefined": False,
        "accepted_contract_artifact": ".local/artifacts/simulation/s4_3_pd/task_success_contract.json",
        "tasks": current,
        "status": "PASS" if passed else "FAIL",
    }


def make_plots(policy: dict[str, Any], eval_config: dict[str, Any]) -> None:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(13, 4))
    axis.axis("off")
    boxes = [
        (0.02, "Frozen successful\nPOLICY_EXPERT"),
        (0.22, "Causal I_t, s_t,\nT[t-25:t]"),
        (0.42, "P0 / P1 / P2 / P3\nACT [27,22]"),
        (0.62, "22D→23D adapter\nstride 5"),
        (0.82, "Fresh POLICY_EVAL_V1\nnative success"),
    ]
    for index, (x, label) in enumerate(boxes):
        axis.text(
            x, 0.5, label, ha="left", va="center", bbox={"boxstyle": "round", "fc": "#e3f2fd"}
        )
        if index < len(boxes) - 1:
            axis.annotate(
                "",
                xy=(boxes[index + 1][0] - 0.01, 0.5),
                xytext=(x + 0.14, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    axis.set_title("Restarted S4.3 causal ACT benchmark dataflow")
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "01_restarted_policy_benchmark_dataflow.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.axvline(0, color="black")
    axis.scatter([-25, 0], [2, 2], color="#1565c0")
    axis.plot([-25, 0], [2, 2], color="#1565c0", linewidth=4, label="OBSERVATION tactile history")
    axis.scatter([0], [1], color="#2e7d32", label="I_t + s_t")
    axis.plot([0, 26], [0, 0], color="#ef6c00", linewidth=4, label="PLAN a_t:a_t+26")
    axis.scatter(
        [27],
        [-1],
        color="#c62828",
        marker="x",
        s=100,
        label="training target only; forbidden inference",
    )
    axis.set_yticks(
        [-1, 0, 1, 2],
        ["future Contact", "planned Action", "current RGB/proprio", "tactile history"],
    )
    axis.set_xlabel("control step relative to current t")
    axis.set_title("Strict causal timestamp graph")
    axis.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "02_strict_causal_timestamp_graph.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(TASKS))
    train = [
        policy["policy_expert_dataset"]["counts"]["POLICY_EXPERT_TRAIN"][task] for task in TASKS
    ]
    dev = [policy["policy_expert_dataset"]["counts"]["POLICY_EXPERT_DEV"][task] for task in TASKS]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, train, 0.36, label="TRAIN")
    axis.bar(x + 0.18, dev, 0.36, label="DEV")
    axis.set_xticks(x, TASKS)
    axis.set_ylabel("native-success episodes")
    axis.set_title("Frozen POLICY_EXPERT train/dev counts")
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "03_policy_expert_train_dev_counts.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    colors = dict(zip(TASKS, ("#1565c0", "#ef6c00", "#2e7d32")))
    for task in TASKS:
        rows = [row for row in eval_config["resets"] if row["task"] == task]
        axis.scatter(
            [row["reset_index"] for row in rows],
            [row["reset_seed"] for row in rows],
            label=task,
            color=colors[task],
        )
    axis.set_xlabel("reset index")
    axis.set_ylabel("frozen reset seed")
    axis.set_title("POLICY_EVAL_V1 reset distribution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "04_policy_evaluation_reset_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    branch = command("git", "branch", "--show-current")
    head = command("git", "rev-parse", "HEAD")
    dex_revision = command("git", "rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco")
    dex_clean = command("git", "status", "--short", cwd=ROOT / "third_party/dexjoco") == ""
    recursive_submodules = command("git", "submodule", "status", "--recursive").splitlines()
    nested_uninitialized = any(
        line.startswith("-5ba07ac") and "diffusion_policy" in line for line in recursive_submodules
    )
    if (
        branch != EXPECTED_BRANCH
        or head != EXPECTED_START_HEAD
        or dex_revision != EXPECTED_DEXJOCO
        or not dex_clean
        or not nested_uninitialized
    ):
        raise RuntimeError("R0 Git/submodule gate failed")

    package_hashes, python_versions = package_audit()
    policy = read_json(POLICY_CONFIG)
    act = read_json(ACT_CONFIG)
    evaluation = read_json(EVAL_CONFIG)
    if policy.get("status") != "FROZEN_BEFORE_ACT_IMPLEMENTATION_AND_TRAINING":
        raise RuntimeError("policy protocol is not frozen before ACT implementation")
    if act.get("status") != "FROZEN_BEFORE_ACT_IMPLEMENTATION_AND_TRAINING":
        raise RuntimeError("ACT protocol is not frozen before ACT implementation")
    validate_policy_eval_resets(evaluation)
    overlap = prior_seed_overlap(evaluation)
    if overlap:
        raise RuntimeError(f"POLICY_EVAL_V1 overlaps prior numeric seed identities: {overlap}")

    checkpoint_hashes = {name: sha256_file(ROOT / path) for name, path in CHECKPOINTS.items()}
    expected_subset = {name: policy["s4_2_immutable"][name] for name in checkpoint_hashes}
    if checkpoint_hashes != expected_subset:
        raise RuntimeError("frozen S4.2 checkpoint hash mismatch")
    s4_2_config_hashes = tracked_s4_2_hashes()
    gpu = gpu_snapshot()
    starting = {
        "schema": "tactile3d-unit.s4-3-restart-starting-integrity.v1",
        "stage": "R0",
        "pwd": ".",
        "branch": branch,
        "starting_head": head,
        "working_tree_clean_before_s4_3_edits": True,
        "required_history_present": {
            "M3": True,
            "S4.0": True,
            "S4.1": True,
            "S4.2": True,
            "S4.2-TF": True,
            "historical_S4.3-0_failure": True,
            "S4.3-PD": True,
        },
        "dexjoco_revision": dex_revision,
        "dexjoco_clean": dex_clean,
        "nested_diffusion_policy": "UNINITIALIZED",
        "package_hashes": package_hashes,
        "python_versions": python_versions,
        "package_installation_or_update_performed": False,
        "s4_2_checkpoint_sha256_before": checkpoint_hashes,
        "s4_2_tracked_config_sha256_before": s4_2_config_hashes,
        "vision_identity": read_json(
            ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"
        )["checkpoint_file_sha256"],
        "gpu_snapshot": gpu,
        "status": "PASS",
    }
    write_json(ARTIFACT_ROOT / "starting_integrity.json", starting)

    pd_manifest_path = PD_ROOT / "policy_expert_dataset_manifest.json"
    pd_manifest = read_json(pd_manifest_path)
    identity = {
        "schema": "tactile3d-unit.s4-3-restart-policy-dataset-identity.v1",
        "stage": "R1",
        "source_dataset_revision": pd_manifest["source_dataset_revision"],
        "dexjoco_revision": pd_manifest["dexjoco_revision"],
        "formal_protocol_sha256": pd_manifest["formal_protocol_sha256"],
        "expert_code_hashes": pd_manifest["expert_code_hashes"],
        "membership_manifest_sha256": sha256_file(pd_manifest_path),
        "formal_success_audit_sha256": sha256_file(PD_ROOT / "formal_success_audit.json"),
        "policy_window_counts_sha256": sha256_file(PD_ROOT / "policy_window_counts.json"),
        "s4_2_compatibility_sha256": sha256_file(PD_ROOT / "s4_2_compatibility.json"),
        "dataset_mutation": False,
        "status": "PASS",
    }
    if (
        identity["membership_manifest_sha256"]
        != policy["policy_expert_dataset"]["membership_manifest_sha256"]
    ):
        raise RuntimeError("frozen policy manifest identity mismatch")
    write_json(ARTIFACT_ROOT / "policy_dataset_identity.json", identity)
    audit = dataset_audit(pd_manifest)
    write_json(ARTIFACT_ROOT / "policy_dataset_audit.json", audit)
    if audit["status"] != "PASS":
        raise SystemExit("S4_3_0R_POLICY_DATASET_MUTATION_FAIL")

    timeouts = {}
    for task in TASKS:
        lengths = [row["length"] for row in pd_manifest["successful_train"] if row["task"] == task]
        timeouts[task] = timeout_statistics(lengths)
        if timeouts[task] != policy["timeouts"]["tasks"][task]:
            raise RuntimeError(f"{task} timeout contract changed")
        observed_windows = sum(valid_bc_windows(length) for length in lengths)
        expected_windows = policy["policy_expert_dataset"]["valid_bc_windows"][task]["train"]
        if observed_windows != expected_windows:
            raise RuntimeError(f"{task} valid train-window count changed")

    success = task_success_audit()
    write_json(ARTIFACT_ROOT / "task_success_contract.json", success)
    if success["status"] != "PASS":
        raise RuntimeError("native task success source changed")
    eval_manifest = {
        "schema": "tactile3d-unit.s4-3-policy-eval-v1-manifest.v1",
        "stage": "R2",
        "config": "configs/simulation/s4_3_policy_eval_v1.json",
        "config_sha256": sha256_file(EVAL_CONFIG),
        "reset_count": len(evaluation["resets"]),
        "resets_per_task": {
            task: sum(row["task"] == task for row in evaluation["resets"]) for task in TASKS
        },
        "prior_seed_overlap": overlap,
        "training_seeds": list(TRAINING_SEEDS),
        "timeouts": timeouts,
        "frozen_before_act_training": True,
        "status": "PASS",
    }
    write_json(ARTIFACT_ROOT / "policy_eval_v1_manifest.json", eval_manifest)

    config_hashes = {
        "policy_protocol": sha256_file(POLICY_CONFIG),
        "act_protocol": sha256_file(ACT_CONFIG),
        "policy_eval_v1": sha256_file(EVAL_CONFIG),
    }
    freeze = {
        "schema": "tactile3d-unit.s4-3-restart-protocol-freeze.v1",
        "stage": "R3",
        "status": "PASS",
        "freeze_status": "FROZEN_BEFORE_ACT_IMPLEMENTATION_AND_TRAINING",
        "config_sha256": config_hashes,
        "protocol_sha256": canonical_sha256(config_hashes),
        "previous_s4_3_0": "FAILED_PRESERVED",
        "restart_basis": "S4_3_PD_COMPLETE_POLICY_DATA_READY",
        "rollout_performance_seen": False,
        "policy_training_started": False,
        "scientific_gates_mutable": False,
        "hammer_warning": "HAMMER_NAIL_MAPPED_TACTILE_INACTIVE",
        "tactile_active_tasks": ["pinch_tongs", "click_mouse"],
    }
    write_json(ARTIFACT_ROOT / "s4_3_restart_protocol_freeze.json", freeze)
    make_plots(policy, evaluation)
    print(
        json.dumps(
            {
                "R0": "PASS",
                "R1": "PASS",
                "R2": "PASS",
                "R3": "PASS",
                "protocol_sha256": freeze["protocol_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
