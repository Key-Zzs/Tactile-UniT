#!/usr/bin/env python3
"""Execute one checkpoint on its 30 frozen POLICY_EVAL_V1 DexJoCo resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import DexJoCoRuntimeAdapter  # noqa: E402
from gr00t.simulation.s4_3_runtime import (  # noqa: E402
    ActionChunkQueue,
    CausalHistoryBuffer,
    mapped_force_metrics,
)

EVAL_CONFIG = ROOT / "configs/simulation/s4_3_policy_eval_v1.json"
PROTOCOL = ROOT / "configs/simulation/s4_3_restart_policy_protocol.json"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/closed_loop"
RAW_VIDEO_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/raw_videos"
WARMUP_ACTION_STEPS = 25
POLICY_RGB_JPEG_QUALITY = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def existing_rollout(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if any(value.get(name) != expected for name, expected in identity.items()):
        return None
    trace = ROOT / value["trace"]
    if value.get("logging_status") != "PASS" or sha256_file(trace) != value["trace_sha256"]:
        return None
    return value


def jpeg(rgb: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, POLICY_RGB_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("failed to encode causal current RGB")
    return encoded.tobytes()


def collect(
    connection: Any,
    reset: dict[str, Any],
    args: argparse.Namespace,
    timeout: int,
) -> dict[str, Any]:
    rollout_id = f"{reset['evaluation_reset_id']}-{args.variant}-trainseed{args.seed}"
    destination = LOG_ROOT / args.task / args.variant / f"seed_{args.seed}" / rollout_id
    metadata_path = destination / "metadata.json"
    identity = {
        "rollout_id": rollout_id,
        "task": args.task,
        "variant": args.variant,
        "training_seed": args.seed,
        "evaluation_reset_id": reset["evaluation_reset_id"],
        "reset_index": reset["reset_index"],
        "seed_namespace": reset["seed_namespace"],
        "reset_seed": reset["reset_seed"],
        "environment_seed": reset["environment_seed"],
        "visual_randomization_seed": reset["visual_randomization_seed"],
        "dynamics_randomization": reset["dynamics_randomization"],
        "perturbation_seed": reset["perturbation_seed"],
        "overlap_with_prior_sources": reset["overlap_with_prior_sources"],
        "checkpoint_sha256": args.checkpoint_sha256,
    }
    existing = existing_rollout(metadata_path, identity)
    if existing is not None:
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / "trace.npz"
    video_path = (
        RAW_VIDEO_ROOT / args.task / args.variant / f"seed_{args.seed}" / f"{rollout_id}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (320, 320))
    if not writer.isOpened():
        raise RuntimeError("failed to open raw rollout video")
    adapter = DexJoCoRuntimeAdapter(
        task_name=args.task,
        seed=int(reset["environment_seed"]),
        episode_id=rollout_id,
        camera_name="random_camera",
        randomize=True,
        randomize_dynamics=False,
    )
    histories = CausalHistoryBuffer(rollout_id)
    queue = ActionChunkQueue(rollout_id)
    rows: dict[str, list[Any]] = {
        "policy_step": [],
        "control_step": [],
        "sim_timestamp_sec": [],
        "wall_timestamp_sec": [],
        "proprio": [],
        "sim_tactile": [],
        "policy_action": [],
        "total_normal_force": [],
        "total_tangential_force": [],
        "success": [],
        "terminated": [],
        "truncated": [],
        "replan": [],
        "history_source_min": [],
        "history_source_max": [],
    }
    plan_steps: list[int] = []
    p3_contact: list[np.ndarray] = []
    runtime_exceptions: list[str] = []
    termination_reason = "TIMEOUT"
    success = False
    region_audit = None
    started_at = time.time()
    try:
        region_audit = adapter.start()
        observation = adapter.reset()
        writer.write(cv2.cvtColor(cv2.resize(observation.rgb, (320, 320)), cv2.COLOR_RGB2BGR))
        histories.append(observation.control_step, observation.sim_tactile)
        for _ in range(WARMUP_ACTION_STEPS):
            observation, _, _, _ = adapter.step(adapter.neutral_policy_action())
            histories.append(observation.control_step, observation.sim_tactile)
            writer.write(cv2.cvtColor(cv2.resize(observation.rgb, (320, 320)), cv2.COLOR_RGB2BGR))
            if observation.terminated or observation.truncated or observation.success:
                raise RuntimeError("environment ended during common causal warm-up")
        if not histories.ready or histories.step_interval != (0, 25):
            raise RuntimeError("warm-up did not produce exact step-0..25 tactile history")
        for policy_step in range(timeout):
            causal = histories.observation(observation.rgb, adapter.neutral_policy_action().values)
            replan = queue.needs_replan
            if replan:
                connection.send(
                    {
                        "command": "infer",
                        "rgb_jpeg": jpeg(causal.rgb),
                        "proprio": causal.proprio,
                        "tactile_history": causal.tactile_history,
                    }
                )
                response = connection.recv()
                action_chunk = np.asarray(response["action_chunk"], dtype=np.float32)
                provenance = queue.set_plan(action_chunk, observation.control_step)
                provenance.validate(inference=True)
                plan_steps.append(policy_step)
                if args.variant == "P3":
                    predicted = np.asarray(response["p3_predicted_contact"], dtype=np.float32)
                    if predicted.shape != (8, 32) or not np.isfinite(predicted).all():
                        raise RuntimeError("invalid P3 causal predicted Contact diagnostic")
                    p3_contact.append(predicted)
                if response["uncertainty"]["invoked"] or response["uncertainty"]["intervention"]:
                    raise RuntimeError("illegal uncertainty invocation/intervention")
            action = queue.pop()
            if not np.isfinite(action).all():
                termination_reason = "NUMERIC_FAILURE"
                break
            next_observation, _, _, _ = adapter.step(action)
            tactile_matrix = causal.tactile_history[-1].reshape(5, 6)
            rows["policy_step"].append(policy_step)
            rows["control_step"].append(observation.control_step)
            rows["sim_timestamp_sec"].append(observation.timestamp_sec)
            rows["wall_timestamp_sec"].append(time.time())
            rows["proprio"].append(causal.proprio)
            rows["sim_tactile"].append(causal.tactile_history[-1])
            rows["policy_action"].append(action)
            rows["total_normal_force"].append(float(np.maximum(tactile_matrix[:, 1], 0).sum()))
            rows["total_tangential_force"].append(float(np.maximum(tactile_matrix[:, 2], 0).sum()))
            rows["success"].append(bool(next_observation.success))
            rows["terminated"].append(bool(next_observation.terminated))
            rows["truncated"].append(bool(next_observation.truncated))
            rows["replan"].append(replan)
            rows["history_source_min"].append(causal.provenance[2].source_min_step)
            rows["history_source_max"].append(causal.provenance[2].source_max_step)
            writer.write(
                cv2.cvtColor(cv2.resize(next_observation.rgb, (320, 320)), cv2.COLOR_RGB2BGR)
            )
            histories.append(next_observation.control_step, next_observation.sim_tactile)
            observation = next_observation
            if observation.success:
                success = True
                termination_reason = "SUCCESS"
                break
            if observation.terminated or observation.truncated:
                termination_reason = "ENV_TERMINATION_FAILURE"
                break
    except Exception as error:
        termination_reason = "SIMULATION_EXCEPTION"
        runtime_exceptions.append(
            f"{type(error).__name__}:{error}\n{traceback.format_exc(limit=8)}"
        )
    finally:
        adapter.close()
        writer.release()

    arrays = {name: np.asarray(value) for name, value in rows.items()}
    arrays["plan_policy_step"] = np.asarray(plan_steps, dtype=np.int32)
    arrays["p3_predicted_contact"] = np.asarray(p3_contact, dtype=np.float32).reshape(-1, 8, 32)
    np.savez_compressed(trace_path, **arrays)
    tactile = arrays["sim_tactile"].astype(np.float32).reshape(-1, 30)
    forces = mapped_force_metrics(tactile)
    actions = arrays["policy_action"].astype(np.float32).reshape(-1, 22)
    action_step_norm = (
        float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean()) if len(actions) > 1 else 0.0
    )
    action_acceleration = (
        float(np.linalg.norm(np.diff(actions, n=2, axis=0), axis=1).mean())
        if len(actions) > 2
        else 0.0
    )
    tcp_jerk = (
        float(np.linalg.norm(np.diff(actions[:, :3], n=3, axis=0), axis=1).mean())
        if len(actions) > 3
        else 0.0
    )
    hand_total_variation = (
        float(np.abs(np.diff(actions[:, 6:22], axis=0)).sum()) if len(actions) > 1 else 0.0
    )
    metadata = {
        "schema": "tactile3d-unit.s4-3-closed-loop-rollout.v1",
        **identity,
        "success": success,
        "termination_reason": termination_reason,
        "control_steps": len(actions),
        "timeout_steps": timeout,
        "time_to_success_sec": len(actions) * 0.02 if success else None,
        "replan_count": queue.replan_count,
        "warmup_samples": 26,
        "warmup_action_steps": WARMUP_ACTION_STEPS,
        "warmup_duration_sec": 0.5,
        "warmup_counted_in_timeout": False,
        "replan_stride": 5,
        "action_chunk_shape": [27, 22],
        "vision_transport": {
            "source": "current RGB I_t only",
            "codec": "JPEG",
            "quality": POLICY_RGB_JPEG_QUALITY,
            "matches_policy_expert_storage": True,
        },
        "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha256_file(trace_path),
        "raw_video": str(video_path.relative_to(ROOT)),
        "raw_video_sha256": sha256_file(video_path),
        "force_metrics": forces,
        "action_metrics": {
            "action_step_norm": action_step_norm,
            "action_acceleration": action_acceleration,
            "tcp_jerk": tcp_jerk,
            "hand_total_variation": hand_total_variation,
        },
        "p3_predicted_contact_replans": len(p3_contact),
        "uncertainty": {
            "status": "UNAVAILABLE_CAUSAL_INPUT_MISMATCH",
            "invoked": False,
            "intervention": False,
        },
        "runtime_exceptions": runtime_exceptions,
        "region_resolution": region_audit,
        "wall_duration_sec": time.time() - started_at,
        "future_observation_read": False,
        "expert_action_read": False,
        "actual_future_contact_read": False,
        "logging_status": "PASS",
    }
    atomic_json(metadata_path, metadata)
    return metadata


def main() -> None:
    args = parse_args()
    evaluation = json.loads(EVAL_CONFIG.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    resets = [row for row in evaluation["resets"] if row["task"] == args.task]
    if len(resets) != 30:
        raise RuntimeError("frozen task reset count is not 30")
    timeout = int(protocol["timeouts"]["tasks"][args.task]["timeout_steps"])
    connection = Client(str(args.socket), family="AF_UNIX", authkey=b"s4_3_local_v1")
    try:
        connection.send({"command": "ping"})
        if connection.recv().get("status") != "READY":
            raise RuntimeError("policy server did not pass readiness ping")
        results = []
        for reset in resets:
            result = collect(connection, reset, args, timeout)
            results.append(result)
            print(
                json.dumps(
                    {
                        "rollout_id": result["rollout_id"],
                        "success": result["success"],
                        "termination_reason": result["termination_reason"],
                        "control_steps": result["control_steps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if len(results) != 30:
            raise RuntimeError("rollout job did not produce exactly 30 frozen resets")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
