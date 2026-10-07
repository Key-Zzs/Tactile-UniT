"""Shadow-only versioned stepwise telemetry for PI2S-R."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Mapping

import numpy as np


SCHEMA = "tactile3d-unit.pi2sr-stepwise-telemetry.v1"
REGION_COUNT = 5
REGION_WIDTH = 6
TACTILE_DIM = REGION_COUNT * REGION_WIDTH
CONTACT_STATE_DIM = 256


def _finite_vector(value: np.ndarray, role: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or (size is not None and array.shape != (size,)):
        expected = f"[{size}]" if size is not None else "one-dimensional"
        raise ValueError(f"{role} must be {expected}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{role} must be finite numeric data")
    return array.copy()


def _optional_vector(value: np.ndarray | None, role: str) -> list[float] | None:
    return None if value is None else _finite_vector(value, role).astype(float).tolist()


def tactile_summary(tactile: np.ndarray) -> dict[str, Any]:
    value = _finite_vector(tactile, "raw canonical tactile", TACTILE_DIM).astype(np.float32)
    regions = value.reshape(REGION_COUNT, REGION_WIDTH)
    occupied = regions[:, 0] > 0.5
    cop: list[list[float] | None] = []
    for index in range(REGION_COUNT):
        cop.append(regions[index, 3:6].astype(float).tolist() if occupied[index] else None)
    return {
        "raw_canonical_tactile": value.astype(float).tolist(),
        "occupancy_count_by_region": occupied.astype(int).tolist(),
        "normal_force_by_region": np.maximum(regions[:, 1], 0.0).astype(float).tolist(),
        "tangential_magnitude_by_region": np.maximum(regions[:, 2], 0.0)
        .astype(float)
        .tolist(),
        "cop_by_region": cop,
    }


class StableGraspTracker:
    """Prospective-only five-step stable-grasp diagnostic."""

    def __init__(self, required_steps: int = 5) -> None:
        self.required_steps = int(required_steps)
        self.consecutive = 0

    def update(self, tactile: np.ndarray) -> bool:
        regions = _finite_vector(tactile, "raw canonical tactile", TACTILE_DIM).reshape(
            REGION_COUNT, REGION_WIDTH
        )
        occupied = int(np.sum(regions[:, 0] > 0.5))
        normal_force = float(np.maximum(regions[:, 1], 0.0).sum())
        self.consecutive = self.consecutive + 1 if occupied >= 2 and normal_force >= 0.5 else 0
        return self.consecutive >= self.required_steps


def build_step_record(
    *,
    run_id: str,
    episode_id: str,
    task: str,
    reset_identity: str,
    control_step_index: int,
    simulation_time_sec: float,
    observation_timestamp_ns: int,
    policy_request_id: str,
    policy_request_index: int,
    action_chunk_generated_ns: int,
    action_applied_ns: int,
    policy_facing_state: np.ndarray,
    environment_facing_state: np.ndarray,
    commanded_action: np.ndarray,
    environment_applied_action: np.ndarray,
    raw_canonical_tactile: np.ndarray,
    canonical_h: np.ndarray | None,
    canonical_h_reference: str | None,
    h_runtime_identity: str,
    history_valid_samples: int,
    history_bootstrap_status: str,
    replan_required: bool,
    replan_stride: int,
    action_queue_index: int,
    action_queue_remaining: int,
    terminated: bool,
    truncated: bool,
    stage_predicates: Mapping[str, bool | None],
    evaluation_only_fields: Mapping[str, Any],
    wall_monotonic_ns: int | None = None,
    tcp_target: np.ndarray | None = None,
    tcp_actual: np.ndarray | None = None,
    hand_target: np.ndarray | None = None,
    hand_actual: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a validated record that is never returned as a policy input."""

    if bool(canonical_h is None) == bool(canonical_h_reference is None):
        raise ValueError("exactly one of canonical_h or canonical_h_reference is required")
    h_value = (
        None
        if canonical_h is None
        else _finite_vector(canonical_h, "canonical H", CONTACT_STATE_DIM).astype(np.float32)
    )
    required_stages = {"CONTACT", "LIFT", "NATIVE_TRIGGER", "SUCCESS", "STABLE_GRASP"}
    if set(stage_predicates) != required_stages:
        raise ValueError("stage predicates must contain exactly the frozen PI2S-R stage keys")
    tactile = tactile_summary(raw_canonical_tactile)
    record = {
        "schema": SCHEMA,
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "task": str(task),
        "reset_identity": str(reset_identity),
        "control_step_index": int(control_step_index),
        "simulation_time_sec": float(simulation_time_sec),
        "wall_monotonic_ns": int(
            time.monotonic_ns() if wall_monotonic_ns is None else wall_monotonic_ns
        ),
        "observation_timestamp_ns": int(observation_timestamp_ns),
        "policy_request_id": str(policy_request_id),
        "policy_request_index": int(policy_request_index),
        "action_chunk_generated_ns": int(action_chunk_generated_ns),
        "action_applied_ns": int(action_applied_ns),
        "policy_facing_state": _finite_vector(
            policy_facing_state, "policy-facing state"
        ).astype(float).tolist(),
        "environment_facing_state": _finite_vector(
            environment_facing_state, "environment-facing state"
        ).astype(float).tolist(),
        "commanded_action": _finite_vector(commanded_action, "commanded action")
        .astype(float)
        .tolist(),
        "environment_applied_action": _finite_vector(
            environment_applied_action, "environment-applied action"
        ).astype(float).tolist(),
        "tcp_target": _optional_vector(tcp_target, "TCP target"),
        "tcp_actual": _optional_vector(tcp_actual, "TCP actual"),
        "hand_target": _optional_vector(hand_target, "hand target"),
        "hand_actual": _optional_vector(hand_actual, "hand actual"),
        **tactile,
        "history_valid_samples": int(history_valid_samples),
        "history_bootstrap_status": str(history_bootstrap_status),
        "canonical_h": None if h_value is None else h_value.astype(float).tolist(),
        "canonical_h_reference": canonical_h_reference,
        "h_norm": None if h_value is None else float(np.linalg.norm(h_value.astype(np.float64))),
        "h_finite": True if h_value is None else bool(np.isfinite(h_value).all()),
        "h_runtime_identity": str(h_runtime_identity),
        "replan_required": bool(replan_required),
        "replan_stride": int(replan_stride),
        "action_queue_index": int(action_queue_index),
        "action_queue_remaining": int(action_queue_remaining),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "stage_predicates": dict(stage_predicates),
        "evaluation_only": {
            "evaluation_only": True,
            "policy_input": False,
            "fields": dict(evaluation_only_fields),
        },
        "policy_input": False,
    }
    json.dumps(record, allow_nan=False, sort_keys=True)
    return record


class AsyncTelemetryWriter:
    """Bounded non-blocking gzip JSONL writer with failure isolation."""

    _SENTINEL = object()

    def __init__(self, destination: Path, queue_capacity: int = 1024) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.destination = Path(destination)
        self.temporary = self.destination.with_name(
            f".{self.destination.name}.tmp-{os.getpid()}-{id(self)}"
        )
        self.queue: queue.Queue[str | object] = queue.Queue(maxsize=queue_capacity)
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self.max_queue_backlog = 0
        self.writer_error: str | None = None
        self.submit_latency_ns: list[int] = []
        self._accepting = True
        self._thread = threading.Thread(target=self._worker, name="pi2sr-telemetry", daemon=True)
        self._thread.start()

    def submit(self, record: Mapping[str, Any]) -> bool:
        started = time.perf_counter_ns()
        accepted = False
        try:
            if not self._accepting or self.writer_error is not None:
                self.dropped += 1
                return False
            payload = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
            self.submitted += 1
            try:
                self.queue.put_nowait(payload)
                self.max_queue_backlog = max(self.max_queue_backlog, self.queue.qsize())
                accepted = True
            except queue.Full:
                self.dropped += 1
            return accepted
        except Exception as error:
            self.dropped += 1
            self.writer_error = f"submit:{type(error).__name__}:{error}"
            return False
        finally:
            self.submit_latency_ns.append(time.perf_counter_ns() - started)

    def _worker(self) -> None:
        try:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            with self.temporary.open("xb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    while True:
                        item = self.queue.get()
                        try:
                            if item is self._SENTINEL:
                                break
                            assert isinstance(item, str)
                            compressed.write(item.encode("utf-8") + b"\n")
                            self.written += 1
                        finally:
                            self.queue.task_done()
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(self.temporary, self.destination)
            directory = os.open(self.destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception as error:  # pragma: no cover - integration failure is reported
            self.writer_error = f"worker:{type(error).__name__}:{error}"

    def close(self, timeout_seconds: float = 30.0) -> dict[str, Any]:
        self._accepting = False
        try:
            self.queue.put(self._SENTINEL, timeout=min(timeout_seconds, 1.0))
        except queue.Full:
            self.writer_error = self.writer_error or "close:queue remained full"
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            self.writer_error = self.writer_error or "close:writer thread timeout"
        unwritten = max(0, self.submitted - self.written - self.dropped)
        self.dropped += unwritten
        latencies = np.asarray(self.submit_latency_ns, dtype=np.float64) / 1_000_000.0

        def percentile(level: float) -> float:
            return float(np.quantile(latencies, level)) if len(latencies) else 0.0

        return {
            "status": "PASS" if self.writer_error is None and self.written == self.submitted else "FAIL",
            "destination": str(self.destination),
            "submitted_records": self.submitted,
            "written_records": self.written,
            "dropped_records": self.dropped,
            "max_queue_backlog": self.max_queue_backlog,
            "writer_error": self.writer_error,
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
        }


def storage_probe(root: Path) -> dict[str, Any]:
    """Verify same-directory create/fsync/rename/readback semantics."""

    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    identity = f"probe-{os.getpid()}-{time.monotonic_ns()}"
    temporary = destination_root / f".{identity}.tmp"
    destination = destination_root / identity
    payload = b"PI2SR_STORAGE_PROBE_V1\n"
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        readback = destination.read_bytes()
        return {
            "status": "PASS" if readback == payload else "FAIL",
            "create": True,
            "fsync": True,
            "same_directory_rename": True,
            "readback": readback == payload,
        }
    finally:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
