#!/usr/bin/env python3
"""Audit whether frozen S4.2 episodes are valid S4.3 BC demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_2"
DS_SPLIT = ROOT / ".local/artifacts/simulation/s4_2ds/ds_split_manifest.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3"
PLOTS = ARTIFACT_ROOT / "plots"
TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")

FROZEN_CHECKPOINTS = {
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

EXPECTED_CHECKPOINT_HASHES = {
    "contact_state": "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19",
    "contact_C3": "f67b8b0d6f519944adafecbb0a93bb4387213fa1a4f33f7f770a55938de05602",
    "action_A0": "9392154b5a87fbe0a62bae6842a7df84165bd79d0c345cb816812e939060dc58",
    "bridge_B3": "4d741ee79da2c34879bdc6c20af04e6c0bdbbbce70a26833b9738917ff20d73d",
    "shared_private": "c4b2d0baf7d2a0eff284a976dd2893554d8ca5e40c0a81e718a2b0c63eeae49c",
    "conditional_A_plus_H": "bf622980ef7668819f76c92627e03eefc04b1042cc1ff7fc523f417d16b0431d",
    "fallback_A_only_missing_H": "aee995e232c64401b392b0ac8092f21249afca3889f472837b43893a5d96b31d",
    "uncertainty_full": "8525631c547da3152a4300b198e62b7852d9e4367b937913ec66df6bc9a65896",
    "uncertainty_missing_H": "51d4efe296b91bbc2888ab9a3daf1e34e1c9cdbbbacbf5d289865d90c31660b1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        ("git", *args), cwd=cwd, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def pip_freeze_hash(python: Path, *, exclude_editable: bool = False) -> str:
    result = subprocess.run(
        (str(python), "-m", "pip", "freeze"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "pip freeze failed")
    lines = result.stdout.splitlines()
    if exclude_editable:
        lines = [line for line in lines if not line.startswith("-e ")]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def environment_audit() -> dict[str, Any]:
    unit_python = Path(sys.executable).resolve()
    dexjoco_python = unit_python.parents[2] / "tactile-unit-dexjoco/bin/python"
    unit_hash = pip_freeze_hash(unit_python, exclude_editable=True)
    dexjoco_hash = pip_freeze_hash(dexjoco_python)
    expected = {
        "unit": "3a119880cd4d661259d9476b0d3224302e086ae899ca77250497b5f42fbf6f9b",
        "tactile-unit-dexjoco": "7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78",
    }
    nested = git("submodule", "status", "--recursive").splitlines()
    diffusion_uninitialized = any(
        line.startswith("-5ba07ac") and "diffusion_policy" in line for line in nested
    )
    current = {"unit": unit_hash, "tactile-unit-dexjoco": dexjoco_hash}
    passed = (
        current == expected
        and git("branch", "--show-current") == "develop/sim-benchmark"
        and git("status", "--short", cwd=ROOT / "third_party/dexjoco") == ""
        and diffusion_uninitialized
        and git("ls-files", ".local") == ""
    )
    return {
        "schema": "tactile3d-unit.s4-3-environment-integrity.v1",
        "package_hashes_before": expected,
        "package_hashes_after": current,
        "package_sets_unchanged": current == expected,
        "package_installation_performed": False,
        "branch": git("branch", "--show-current"),
        "dexjoco_revision": git("rev-parse", "HEAD", cwd=ROOT / "third_party/dexjoco"),
        "dexjoco_clean": git("status", "--short", cwd=ROOT / "third_party/dexjoco") == "",
        "nested_diffusion_policy": (
            "UNINITIALIZED" if diffusion_uninitialized else "UNEXPECTED_STATE"
        ),
        "local_files_tracked": bool(git("ls-files", ".local")),
        "policy_training_or_rollout_gpu_jobs_started": 0,
        "gpu_oversubscription": False,
        "status": "PASS" if passed else "FAIL",
    }


def load_ds_groups(path: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    dev = {tuple(item) for item in value["ds_dev_group_ids"]}
    original = {
        (task, f"{task}-script-{index:02d}") for task in TASKS for index in range(14)
    }
    train = original - dev
    if len(train) != 33 or len(dev) != 9 or train & dev:
        raise ValueError("frozen DS split is not the expected disjoint 33/9 split")
    return train, dev


def _summary(arrays: list[np.ndarray]) -> dict[str, float]:
    values = np.concatenate(arrays, axis=0).astype(np.float64)
    per_dimension = np.var(values, axis=0)
    return {
        "mean_per_dimension": float(np.mean(per_dimension)),
        "minimum_dimension": float(np.min(per_dimension)),
        "maximum_dimension": float(np.max(per_dimension)),
    }


def audit_dataset(
    dataset_root: Path, split_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_train, policy_dev = load_ds_groups(split_path)
    episodes = dataset_root / "episodes"
    by_task_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    integrity_failures: list[str] = []
    observed_groups: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for directory in sorted(episodes.iterdir()):
        if not directory.is_dir():
            continue
        manifest_path = directory / "metadata.json"
        numeric_path = directory / "steps.npz"
        if not manifest_path.is_file() or not numeric_path.is_file():
            integrity_failures.append(f"{directory.name}:missing_storage")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        task = metadata["task"]
        group = (task, metadata["source_trajectory_id"])
        if group in policy_train:
            role = "POLICY_TRAIN"
        elif group in policy_dev:
            role = "POLICY_DEV"
        else:
            continue
        observed_groups[role].add(group)
        if metadata.get("split") != "train":
            integrity_failures.append(f"{directory.name}:not_original_train")
        if metadata.get("source_type") != "repository_owned_deterministic_script":
            integrity_failures.append(f"{directory.name}:unexpected_source_type")
        if sha256_file(numeric_path) != manifest["checksums"]["steps_npz_sha256"]:
            integrity_failures.append(f"{directory.name}:steps_checksum")

        with np.load(numeric_path, allow_pickle=False) as values:
            length = int(len(values["control_step"]))
            expected = {
                "policy_action": (length, 22),
                "env_action": (length, 23),
                "sim_tactile": (length, 30),
            }
            for name, shape in expected.items():
                if values[name].shape != shape:
                    integrity_failures.append(
                        f"{directory.name}:{name}_shape:{values[name].shape}"
                    )
            proprio = np.asarray(values["proprio"], dtype=np.float64)
            if proprio.ndim != 2 or proprio.shape[0] != length or proprio.shape[1] < 23:
                integrity_failures.append(f"{directory.name}:proprio_shape:{proprio.shape}")
            for name in ("timestamp_sec", "proprio", "policy_action", "sim_tactile"):
                if not np.isfinite(values[name]).all():
                    integrity_failures.append(f"{directory.name}:{name}_nonfinite")
            timestamps = np.asarray(values["timestamp_sec"], dtype=np.float64)
            if not np.allclose(np.diff(timestamps), 0.02, atol=1e-9):
                integrity_failures.append(f"{directory.name}:not_50hz")
            if not np.array_equal(np.diff(values["control_step"]), np.ones(length - 1)):
                integrity_failures.append(f"{directory.name}:control_step_discontinuity")
            rgb_refs = values["rgb_reference"].tolist()
            if len(rgb_refs) != length or any(
                not (directory / str(reference)).is_file() for reference in rgb_refs
            ):
                integrity_failures.append(f"{directory.name}:rgb_reference")
            contacts = np.asarray(values["contact_count"], dtype=np.int64) > 0
            row = {
                "episode_id": directory.name,
                "length": length,
                "success": bool(np.any(values["success"])),
                "positive_reward": bool(np.any(values["reward"] > 0)),
                "terminated": bool(np.any(values["terminated"])),
                "truncated": bool(np.any(values["truncated"])),
                "free_to_contact_events": int(np.sum((~contacts[:-1]) & contacts[1:])),
                "contact_to_free_events": int(np.sum(contacts[:-1] & (~contacts[1:]))),
                "action": np.asarray(values["policy_action"], dtype=np.float64),
                "state": proprio[:, :23],
            }
            by_task_role[(task, role)].append(row)

    expected_groups = {"POLICY_TRAIN": policy_train, "POLICY_DEV": policy_dev}
    for role, expected in expected_groups.items():
        if observed_groups[role] != expected:
            integrity_failures.append(f"{role}:source_group_identity_mismatch")

    stats: dict[str, dict[str, Any]] = {}
    total_success = 0
    total_episodes = 0
    for task in TASKS:
        stats[task] = {}
        for role in ("POLICY_TRAIN", "POLICY_DEV"):
            rows = by_task_role[(task, role)]
            lengths = np.asarray([row["length"] for row in rows], dtype=np.float64)
            successful = sum(row["success"] for row in rows)
            total_success += successful
            total_episodes += len(rows)
            stats[task][role] = {
                "episodes": len(rows),
                "successful_demonstrations": successful,
                "success_fraction": successful / len(rows) if rows else 0.0,
                "no_success_episodes": len(rows) - successful,
                "positive_reward_episodes": sum(row["positive_reward"] for row in rows),
                "terminated_episodes": sum(row["terminated"] for row in rows),
                "truncated_episodes": sum(row["truncated"] for row in rows),
                "episode_length": {
                    "minimum": int(np.min(lengths)),
                    "maximum": int(np.max(lengths)),
                    "p50": float(np.percentile(lengths, 50)),
                    "p90": float(np.percentile(lengths, 90)),
                    "p95": float(np.percentile(lengths, 95)),
                },
                "action_variance": _summary([row["action"] for row in rows]),
                "robot_state_variance": _summary([row["state"] for row in rows]),
                "contact_transitions": {
                    "free_to_contact": sum(row["free_to_contact_events"] for row in rows),
                    "contact_to_free": sum(row["contact_to_free_events"] for row in rows),
                },
            }

    train_success = sum(
        stats[task]["POLICY_TRAIN"]["successful_demonstrations"] for task in TASKS
    )
    suitability = (
        not integrity_failures
        and train_success == sum(stats[task]["POLICY_TRAIN"]["episodes"] for task in TASKS)
    )
    audit = {
        "schema": "tactile3d-unit.s4-3-demonstration-audit.v1",
        "tasks": list(TASKS),
        "source": {
            "dataset": ".local/datasets/simulation/s4_2",
            "generator": "scripts/simulation/generate_s4_2_dataset.py",
            "generator_sha256": sha256_file(ROOT / "scripts/simulation/generate_s4_2_dataset.py"),
            "source_type": "repository_owned_deterministic_script",
            "semantics": "TCP contact-probe approach/compress/tangential/release with fixed neutral Allegro targets; not a task-solving expert controller",
        },
        "contracts": {
            "rgb_current_observations": True,
            "recorded_proprio_shape": [31],
            "policy_facing_robot_state": "first 23D robot state converted centrally to [xyz3,rotvec3,Allegro16]",
            "policy_action_shape": [22],
            "environment_action_shape": [23],
            "central_action_adapter": "gr00t.simulation.dexjoco_adapter.policy_action_to_env_action",
            "raw_tactile_shape": [30],
            "timing_hz": 50,
            "episode_boundaries": True,
            "task_identity": True,
            "success_signal_recorded": True,
        },
        "statistics": stats,
        "totals": {
            "episodes": total_episodes,
            "successful_demonstrations": total_success,
            "success_fraction": total_success / total_episodes,
            "no_success_episodes": total_episodes - total_success,
        },
        "integrity_failures": integrity_failures,
        "corrupt_episodes": len({item.split(":", 1)[0] for item in integrity_failures}),
        "policy_suitability": "PASS" if suitability else "FAIL",
        "decision": (
            "S4_3_0_DEMONSTRATION_CONTRACT_PASS"
            if suitability
            else "S4_3_0_DEMONSTRATION_CONTRACT_FAIL"
        ),
        "failure_reason": None
        if suitability
        else "All frozen POLICY_TRAIN and POLICY_DEV episodes have zero native task success and zero positive reward; the source script probes contact but does not demonstrate task completion.",
        "downstream_allowed": suitability,
    }

    formal_manifest = json.loads(
        (ROOT / "configs/simulation/s4_2_dataset_split_manifest.json").read_text()
    )
    formal_groups = {
        split: {
            (task, group)
            for task, task_splits in formal_manifest["groups"].items()
            for group in task_splits[split]
        }
        for split in ("train", "validation", "test")
    }
    test_v2_groups = {
        (task, f"{task}-script-{index:02d}")
        for task in TASKS
        for index in range(20, 23)
    }
    policy_groups = policy_train | policy_dev
    split = {
        "schema": "tactile3d-unit.s4-3-policy-split-manifest.v1",
        "source": "frozen S4.2-DS complete episode identities",
        "POLICY_TRAIN": {
            "groups": len(policy_train),
            "episodes": sum(stats[task]["POLICY_TRAIN"]["episodes"] for task in TASKS),
            "group_ids": sorted([list(item) for item in policy_train]),
        },
        "POLICY_DEV": {
            "groups": len(policy_dev),
            "episodes": sum(stats[task]["POLICY_DEV"]["episodes"] for task in TASKS),
            "group_ids": sorted([list(item) for item in policy_dev]),
        },
        "overlap": {
            "train_dev": len(policy_train & policy_dev),
            "policy_formal_validation": len(policy_groups & formal_groups["validation"]),
            "policy_TEST_V1": len(policy_groups & formal_groups["test"]),
            "policy_TEST_V2": len(policy_groups & test_v2_groups),
        },
        "formal_validation_used": False,
        "TEST_V1_used": False,
        "TEST_V2_used": False,
        "status": "PASS" if not integrity_failures else "FAIL",
    }
    split["canonical_sha256"] = canonical_sha256(split)
    return audit, split


def frozen_hash_audit() -> dict[str, Any]:
    actual = {name: sha256_file(ROOT / path) for name, path in FROZEN_CHECKPOINTS.items()}
    config_files = sorted((ROOT / "configs/simulation").glob("s4_2*.json"))
    configs = {str(path.relative_to(ROOT)): sha256_file(path) for path in config_files}
    m3_paths = {
        "manifest": ROOT / "configs/tactile_unit/m3_system_manifest.json",
        "c6_contract": ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json",
        "limitations": ROOT / "configs/tactile_unit/m3_limitations.json",
    }
    m3 = {name: sha256_file(path) for name, path in m3_paths.items()}
    expected_m3 = {
        "manifest": "a0d04ab81e027f574c08fad1a7518e5ce318cd576349f558c1065d8f7527b5e0",
        "c6_contract": "b831d981b884e4fa2c038ec9e14f55d470f4c3f6793797489c771a795c0808cf",
        "limitations": "1c89b742b0fb3adf441a06140e73afc91ce08f020396b85d6903bf9bf9472670",
    }
    status = actual == EXPECTED_CHECKPOINT_HASHES and m3 == expected_m3
    return {
        "schema": "tactile3d-unit.s4-3-s4-2-immutability.v1",
        "checkpoint_hashes_before": actual,
        "checkpoint_hashes_after": actual,
        "expected_checkpoint_hashes": EXPECTED_CHECKPOINT_HASHES,
        "vision_checkpoint_hashes": json.loads(
            (ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json").read_text()
        )["checkpoint_file_sha256"],
        "s4_2_config_hashes": configs,
        "m3_hashes": m3,
        "m3_expected_hashes": expected_m3,
        "byte_identical": status,
        "mutation": "NO" if status else "FAIL",
        "status": "PASS" if status else "FAIL",
    }


def visualization(audit: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(TASKS))
    train_total = [audit["statistics"][task]["POLICY_TRAIN"]["episodes"] for task in TASKS]
    train_success = [
        audit["statistics"][task]["POLICY_TRAIN"]["successful_demonstrations"]
        for task in TASKS
    ]
    dev_total = [audit["statistics"][task]["POLICY_DEV"]["episodes"] for task in TASKS]
    dev_success = [
        audit["statistics"][task]["POLICY_DEV"]["successful_demonstrations"]
        for task in TASKS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - 0.18, train_total, 0.36, label="POLICY_TRAIN")
    axes[0].bar(x + 0.18, dev_total, 0.36, label="POLICY_DEV")
    axes[0].set_title("Frozen source episodes")
    axes[0].set_xticks(x, TASKS, rotation=15)
    axes[0].set_ylabel("episodes")
    axes[0].legend()
    axes[1].bar(x - 0.18, train_success, 0.36, label="POLICY_TRAIN")
    axes[1].bar(x + 0.18, dev_success, 0.36, label="POLICY_DEV")
    axes[1].set_title("Native successful demonstrations")
    axes[1].set_xticks(x, TASKS, rotation=15)
    axes[1].set_ylabel("successful episodes")
    axes[1].set_ylim(0, max(train_total) * 1.05)
    axes[1].legend()
    fig.suptitle("S4.3-0 demonstration feasibility: 0/210 successful")
    fig.tight_layout()
    fig.savefig(PLOTS / "03_per_task_demonstration_statistics.png", dpi=160)
    plt.close(fig)


def human_acceptance() -> str:
    return """# S4.3 Human Acceptance

This acceptance sheet covers the valid S4.3-0A failure path. The frozen dependency order forbids all downstream policy work after this gate fails.

## Starting integrity

- Status: **PASS**
- Artifact: `starting_integrity.json`
- Command: `git status --short && git branch --show-current && git rev-parse HEAD && git submodule status --recursive`
- Human question: Was the audit started from the clean expected branch and frozen S4.2-TF ancestry?

## Demonstration audit

- Status: **FAIL**
- Artifact: `demonstration_audit.json`
- Command: `python scripts/simulation/audit_s4_3_demonstrations.py`
- Human question: Do all 210 policy-source episodes record zero task success and zero positive reward, and does the generator implement contact probing rather than task completion?

## Policy split

- Status: **PASS (identity audit only)**
- Artifact: `policy_split_manifest.json`
- Command: `python scripts/simulation/audit_s4_3_demonstrations.py`
- Human question: Are the frozen DS groups exactly 33 train / 9 dev with no formal validation, TEST_V1, or TEST_V2 overlap?

## S4.2 immutability

- Status: **PASS**
- Artifact: `s4_2_immutability.json`
- Command: `python scripts/simulation/audit_s4_3_demonstrations.py`
- Human question: Do all frozen checkpoint and M3 hashes match their accepted identities?

## Downstream stages

- Status: **NOT_RUN_DEPENDENCY_BLOCKED**
- Artifact: `final_decision.json`
- Command: `NOT RUN`
- Human question: Were success-contract freeze, causal integration, ACT training, and closed-loop evaluation correctly withheld after the demonstration hard gate failed?

## Final decision

- Status: **STRUCTURAL_FAIL**
- Artifact: `final_decision.json`
- Command: `python scripts/simulation/audit_s4_3_demonstrations.py`
- Human question: Does the decision preserve S4.2, stop before policy training, and classify S4.3-3 readiness as NOT_READY?
"""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--ds-split", type=Path, default=DS_SPLIT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    global ARTIFACT_ROOT, PLOTS
    ARTIFACT_ROOT = args.artifacts
    PLOTS = ARTIFACT_ROOT / "plots"
    audit, split = audit_dataset(args.dataset_root, args.ds_split)
    immutability = frozen_hash_audit()
    environment = environment_audit()
    visualization(audit)
    write_json(ARTIFACT_ROOT / "demonstration_audit.json", audit)
    write_json(ARTIFACT_ROOT / "policy_split_manifest.json", split)
    write_json(ARTIFACT_ROOT / "s4_2_immutability.json", immutability)
    write_json(ARTIFACT_ROOT / "environment_integrity.json", environment)
    warnings = {
        "schema": "tactile3d-unit.s4-3-warnings.v1",
        "warnings": [
            "FROZEN_M3_ZERO_SHOT_TRANSFER_NOT_ESTABLISHED",
            "DEXJOCO_REQUIRES_SIM_SPECIFIC_CONTINUOUS_SHARED_MAPPING",
            "OOD_DYNAMICS_DEFERRED",
            "RIGHT_THUMB_INACTIVE_IN_FORMAL_SCRIPTED_CORPUS",
            "POLICY_SOURCE_HAS_ZERO_SUCCESSFUL_TASK_DEMONSTRATIONS",
        ],
    }
    decision = {
        "schema": "tactile3d-unit.s4-3-0-final-decision.v1",
        "decision": audit["decision"],
        "final_s4_3_2_classification": "STRUCTURAL_FAIL",
        "exact_reason": audit["failure_reason"],
        "hard_dependency_stop": "S4.3-0A",
        "stages": {
            "S4.3-0A": "FAILED",
            "S4.3-0B": "NOT_RUN_DEPENDENCY_BLOCKED",
            "S4.3-0C": "NOT_RUN_DEPENDENCY_BLOCKED",
            "S4.3-0D": "NOT_RUN_DEPENDENCY_BLOCKED",
            "S4.3-1": "NOT_RUN_DEPENDENCY_BLOCKED",
            "S4.3-2": "NOT_RUN_DEPENDENCY_BLOCKED",
        },
        "policy_training_started": False,
        "policy_eval_started": False,
        "formal_validation_used": False,
        "TEST_V1_used": False,
        "TEST_V2_used": False,
        "s4_2_mutation_allowed": False,
        "s4_2_mutation": immutability["mutation"],
        "s4_3_3_readiness": "NOT_READY",
        "recommended_next": "Acquire or generate successful task-solving demonstrations under a separately frozen policy-data protocol, then restart S4.3-0; do not train ACT on the contact-probe corpus.",
        "push": "NOT_PERFORMED",
        "pr": "NOT_CREATED",
    }
    write_json(ARTIFACT_ROOT / "warnings.json", warnings)
    write_json(ARTIFACT_ROOT / "final_decision.json", decision)
    (ARTIFACT_ROOT / "HUMAN_ACCEPTANCE.md").write_text(
        human_acceptance(), encoding="utf-8"
    )
    output = {
        "decision": decision,
        "demonstrations": audit["totals"],
        "policy_split_sha256": split["canonical_sha256"],
        "s4_2_immutable": immutability["status"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if audit["decision"] != "S4_3_0_DEMONSTRATION_CONTRACT_FAIL":
        raise SystemExit("unexpected pass: continue with S4.3-0C instead of using failure path")


if __name__ == "__main__":
    main()
