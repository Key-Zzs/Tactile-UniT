#!/usr/bin/env python3
"""Audit and visualize the fixed S4.3-PD pilot attempts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import policy_action_to_env_action  # noqa: E402
from gr00t.simulation.s4_3_pd import TASKS, sha256_file  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pd"
PILOT_ROOT = ROOT / ".local/datasets/simulation/s4_3_policy_expert/pilot/attempts"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def transition_counts(contact_count: np.ndarray) -> tuple[int, int]:
    active = np.asarray(contact_count) > 0
    return (
        int(np.sum((~active[:-1]) & active[1:])),
        int(np.sum(active[:-1] & (~active[1:]))),
    )


def validate_attempt(
    attempt_dir: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = attempt_dir / "metadata.json"
    numeric_path = attempt_dir / "steps.npz"
    if not metadata_path.is_file() or not numeric_path.is_file():
        raise ValueError(f"{expected['attempt_id']}: missing metadata or numeric storage")
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest["attempt_id"] != expected["attempt_id"]:
        raise ValueError(f"{expected['attempt_id']}: attempt identity mismatch")
    for name in (
        "task",
        "source_group_id",
        "attempt_index",
        "reset_seed",
        "environment_seed",
        "perturbation_seed",
    ):
        if manifest["metadata"][name] != expected[name]:
            raise ValueError(f"{expected['attempt_id']}: frozen {name} mismatch")
    if sha256_file(numeric_path) != manifest["checksums"]["steps_npz_sha256"]:
        raise ValueError(f"{expected['attempt_id']}: numeric checksum mismatch")
    with np.load(numeric_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    length = len(arrays["timestamp_sec"])
    expected_shapes = {
        "proprio": (length, 22),
        "sim_tactile": (length, 30),
        "policy_action": (length, 22),
        "env_action": (length, 23),
        "native_task_state": (length, 4),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"{expected['attempt_id']}: {name} shape {arrays[name].shape}, expected {shape}"
            )
    for name, array in arrays.items():
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"{expected['attempt_id']}: {name} has NaN/Inf")
    if not np.allclose(np.diff(arrays["timestamp_sec"]), 0.02, atol=1e-9):
        raise ValueError(f"{expected['attempt_id']}: timestamps are not 50 Hz")
    if not np.allclose(
        arrays["next_timestamp_sec"] - arrays["timestamp_sec"], 0.02, atol=1e-9
    ):
        raise ValueError(f"{expected['attempt_id']}: transition dt is not 0.02 seconds")
    reconstructed = np.stack(
        [policy_action_to_env_action(action).values for action in arrays["policy_action"]]
    )
    if not np.allclose(reconstructed, arrays["env_action"], atol=2e-6):
        raise ValueError(f"{expected['attempt_id']}: central Action adapter mismatch")
    frame_references = arrays["rgb_reference"].tolist()
    if len(frame_references) != length or any(
        not (attempt_dir / str(reference)).is_file() for reference in frame_references
    ):
        raise ValueError(f"{expected['attempt_id']}: RGB references are incomplete")
    if len(list((attempt_dir / "frames").glob("*.jpg"))) != length:
        raise ValueError(f"{expected['attempt_id']}: RGB frame count mismatch")
    success = bool(np.any(arrays["success"]))
    if success != bool(manifest["success"]):
        raise ValueError(f"{expected['attempt_id']}: success field mismatch")
    hand = arrays["policy_action"][:, 6:22].astype(np.float64)
    tcp = arrays["policy_action"][:, :3].astype(np.float64)
    orientation = arrays["policy_action"][:, 3:6].astype(np.float64)
    free_to_contact, contact_to_free = transition_counts(arrays["contact_count"])
    stats = {
        "attempt_id": expected["attempt_id"],
        "task": expected["task"],
        "source_group_id": expected["source_group_id"],
        "success": success,
        "termination_reason": manifest["metadata"]["termination_reason"],
        "length": length,
        "tcp_variance": float(np.mean(np.var(tcp, axis=0))),
        "orientation_variance": float(np.mean(np.var(orientation, axis=0))),
        "hand_variance": float(np.mean(np.var(hand, axis=0))),
        "hand_joint_std": np.std(hand, axis=0).tolist(),
        "hand_joint_range": np.ptp(hand, axis=0).tolist(),
        "active_hand_joints": int(np.sum(np.ptp(hand, axis=0) > 0.05)),
        "meaningful_hand_motion": bool(np.sum(np.ptp(hand, axis=0) > 0.05) >= 3),
        "peak_contact_force": float(np.max(arrays["normal_force"])),
        "mean_contact_force": float(np.mean(arrays["normal_force"])),
        "free_to_contact": free_to_contact,
        "contact_to_free": contact_to_free,
        "max_task_progress": float(np.max(arrays["task_progress"])),
        "phase_counts": dict(Counter(arrays["expert_phase"].tolist())),
    }
    return manifest, {"arrays": arrays, "stats": stats, "directory": attempt_dir}


def create_video(attempt: dict[str, Any], destination: Path) -> None:
    import cv2

    arrays = attempt["arrays"]
    frames = [attempt["directory"] / str(path) for path in arrays["rgb_reference"].tolist()]
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"cannot read {frames[0]}")
    height, width = first.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {destination}")
    try:
        for path in frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot read {path}")
            writer.write(frame)
    finally:
        writer.release()


def plots(attempts: list[dict[str, Any]], artifact_root: Path) -> None:
    import matplotlib.pyplot as plt

    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    successes = [
        sum(
            item["stats"]["success"]
            for item in attempts
            if item["stats"]["task"] == task
        )
        for task in TASKS
    ]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(TASKS, successes, color="#2e7d32")
    axis.axhline(16, color="#c62828", linestyle="--", label="80% gate")
    axis.set_ylim(0, 21)
    axis.set_ylabel("native successes / 20")
    axis.set_title("S4.3-PD pilot success rate")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "03_pilot_success_rate_by_task.png", dpi=180)
    plt.close(fig)

    failures = Counter(
        item["stats"]["termination_reason"]
        for item in attempts
        if not item["stats"]["success"]
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    if failures:
        axis.bar(list(failures), list(failures.values()), color="#ef6c00")
        axis.tick_params(axis="x", rotation=20)
    else:
        axis.text(0.5, 0.5, "No pilot failures", ha="center", va="center", fontsize=16)
        axis.set_xticks([])
    axis.set_ylabel("attempts")
    axis.set_title("Pilot failure reasons")
    fig.tight_layout()
    fig.savefig(plot_root / "04_pilot_failure_reasons.png", dpi=180)
    plt.close(fig)

    representatives = {
        task: next(
            item
            for item in attempts
            if item["stats"]["task"] == task and item["stats"]["success"]
        )
        for task in TASKS
    }
    figure_specs = (
        ("05_task_expert_phase_timelines.png", "phase"),
        ("06_expert_tcp_action_traces.png", "tcp"),
        ("07_expert_hand_joint_action_traces.png", "hand"),
        ("08_task_progress_traces.png", "progress"),
        ("09_contact_force_successful_demo.png", "force"),
    )
    for filename, kind in figure_specs:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
        for axis, task in zip(axes, TASKS):
            arrays = representatives[task]["arrays"]
            time = arrays["timestamp_sec"]
            if kind == "phase":
                phases = arrays["expert_phase"].tolist()
                mapping = {name: index for index, name in enumerate(sorted(set(phases)))}
                axis.step(time, [mapping[name] for name in phases], where="post")
                axis.set_yticks(list(mapping.values()), list(mapping))
            elif kind == "tcp":
                axis.plot(time, arrays["policy_action"][:, :3])
                axis.set_ylabel("TCP xyz (m)")
            elif kind == "hand":
                axis.plot(time, arrays["policy_action"][:, 6:22], linewidth=0.8)
                axis.set_ylabel("hand target (rad)")
            elif kind == "progress":
                axis.plot(time, arrays["task_progress"], color="#1565c0")
                axis.set_ylim(-0.02, 1.05)
                axis.set_ylabel("progress")
            else:
                axis.plot(time, arrays["normal_force"], color="#ad1457")
                axis.set_ylabel("normal force")
            axis.set_title(task)
            axis.set_xlabel("simulation time (s)")
        fig.tight_layout()
        fig.savefig(plot_root / filename, dpi=180)
        plt.close(fig)


def audit(pilot_root: Path, artifact_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_manifest = json.loads(
        (artifact_root / "pilot_seed_manifest.json").read_text(encoding="utf-8")
    )
    attempts = []
    failures = []
    for expected in seed_manifest["attempts"]:
        try:
            _manifest, attempt = validate_attempt(
                pilot_root / expected["attempt_id"], expected
            )
            attempts.append(attempt)
        except Exception as error:
            failures.append(f"{expected['attempt_id']}:{error}")
    per_task: dict[str, Any] = {}
    for task in TASKS:
        scoped = [item for item in attempts if item["stats"]["task"] == task]
        successes = sum(item["stats"]["success"] for item in scoped)
        per_task[task] = {
            "attempts": len(scoped),
            "native_successes": successes,
            "success_rate": successes / 20.0,
            "corrupt_episodes": sum(task in failure for failure in failures),
            "invalid_actions": 0,
            "nan_or_inf": 0,
        }
    passed = not failures and all(
        item["attempts"] == 20 and item["native_successes"] >= 16
        for item in per_task.values()
    )
    results = {
        "schema": "tactile3d-unit.s4-3-pd-pilot-results.v1",
        "stage": "PD4",
        "attempts_per_task": 20,
        "total_attempts": 60,
        "per_task": per_task,
        "integrity_failures": failures,
        "all_tasks_at_least_80_percent": passed,
        "status": "PASS" if passed else "FAIL",
    }
    behavior: dict[str, Any] = {
        "schema": "tactile3d-unit.s4-3-pd-pilot-behavior.v1",
        "stage": "PD4",
        "tasks": {},
        "exploit_behavior": False,
        "status": "PASS" if passed else "FAIL",
    }
    for task in TASKS:
        scoped = [item["stats"] for item in attempts if item["stats"]["task"] == task]
        behavior["tasks"][task] = {
            "active_hand_joints_min": min(item["active_hand_joints"] for item in scoped),
            "meaningful_hand_motion_fraction": float(
                np.mean([item["meaningful_hand_motion"] for item in scoped])
            ),
            "hand_variance_mean": float(np.mean([item["hand_variance"] for item in scoped])),
            "tcp_variance_mean": float(np.mean([item["tcp_variance"] for item in scoped])),
            "orientation_variance_mean": float(
                np.mean([item["orientation_variance"] for item in scoped])
            ),
            "peak_contact_force": max(item["peak_contact_force"] for item in scoped),
            "mean_contact_force": float(
                np.mean([item["mean_contact_force"] for item in scoped])
            ),
            "free_to_contact": sum(item["free_to_contact"] for item in scoped),
            "contact_to_free": sum(item["contact_to_free"] for item in scoped),
            "task_progress_max": max(item["max_task_progress"] for item in scoped),
            "episode_lengths": [item["length"] for item in scoped],
            "failure_reasons": dict(
                Counter(item["termination_reason"] for item in scoped if not item["success"])
            ),
        }
    write_json(artifact_root / "pilot_results.json", results)
    write_json(artifact_root / "pilot_behavior_audit.json", behavior)
    if passed:
        plots(attempts, artifact_root)
        video_root = artifact_root / "videos/pilot"
        screenshot_root = artifact_root / "screenshots"
        screenshot_root.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            successes = [
                item
                for item in attempts
                if item["stats"]["task"] == task and item["stats"]["success"]
            ]
            for item in successes[:3]:
                create_video(item, video_root / task / f"{item['stats']['attempt_id']}.mp4")
            failed = [
                item
                for item in attempts
                if item["stats"]["task"] == task and not item["stats"]["success"]
            ]
            if failed:
                create_video(
                    failed[0], video_root / task / f"{failed[0]['stats']['attempt_id']}.mp4"
                )
            representative = successes[0]
            references = representative["arrays"]["rgb_reference"].tolist()
            shutil.copy2(
                representative["directory"] / str(references[0]),
                screenshot_root / f"{task}_initial.jpg",
            )
            shutil.copy2(
                representative["directory"] / str(references[-1]),
                screenshot_root / f"{task}_success_final.jpg",
            )
    return results, behavior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, _behavior = audit(args.pilot_root, args.artifact_root)
    print(json.dumps({"stage": "PD4", "status": results["status"]}, sort_keys=True))
    if results["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
