#!/usr/bin/env python3
"""Run the frozen PI2S CPU-only offline/direct/live Contact-State parity audit.

The audit sends one reconstructed float32 history through all three frozen paths:

* A: the Contact-State values cached in the PI1 training sidecar;
* B: the accepted S4.2 teacher with ``TorchTactileNormalization`` in-process; and
* C: the same teacher behind the PI1 Unix-domain-socket service.

Only the two JSON artifacts are persistent outputs.  The service socket and its own
service report live in a temporary directory and are removed after a graceful stop.
``--check`` recomputes the complete audit but never replaces either JSON artifact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNNER_IMPLEMENTATION = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from gr00t.simulation.pi1d_runtime import ContactStateUnixClient  # noqa: E402
from gr00t.simulation.s4_3_pi1 import (  # noqa: E402
    CONTACT_STATE_DIM,
    HISTORY_STEPS,
    TACTILE_DIM,
    OnlineTactileHistory,
)

PROTOCOL = ROOT / "configs/simulation/pi2s/diagnostic_protocol.json"
PI2S_ROOT = ROOT / ".local/experiments/simulation/s4_3_pi2s"
SIDECAR = PI2S_ROOT / "source_snapshots/data/pi1_contact_sidecar.npz"
ALIGNMENT = (
    PI2S_ROOT / "source_snapshots/pi2s_protocol_v3/source_manifests/official_dataset_alignment.json"
)
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
SERVICE_SCRIPT = ROOT / "scripts/simulation/serve_s4_3_pi1_contact_state.py"
DIRECT_IMPLEMENTATION = ROOT / "gr00t/simulation/s4_3_act.py"
RUNTIME_IMPLEMENTATION = ROOT / "gr00t/simulation/pi1d_runtime.py"
HISTORY_IMPLEMENTATION = ROOT / "gr00t/simulation/s4_3_pi1.py"
TEACHER_LOADER_IMPLEMENTATION = ROOT / "gr00t/simulation/sim_contact_models.py"
TEACHER_ARCHITECTURE_IMPLEMENTATION = ROOT / "gr00t/tactile_teacher/models.py"
OUTPUTS = {
    "parity": PI2S_ROOT / "artifacts/offline_online_h_parity.json",
    "physical_time": PI2S_ROOT / "artifacts/h_physical_time_audit.json",
}

PAIR_ORDER = (
    ("cached_vs_direct", "cached_sidecar", "direct_et"),
    ("cached_vs_live", "cached_sidecar", "live_unix_service"),
    ("direct_vs_live", "direct_et", "live_unix_service"),
)
RAW_CONTROL_HZ = 50


class AuditError(RuntimeError):
    """Raised when a frozen input or an audit invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash an array together with the dtype and shape that give its bytes meaning."""

    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


class InputRegistry:
    """Bind every persistent audit input to the bytes actually consumed."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def record(self, path: Path, role: str, expected_sha256: str | None = None) -> str:
        require(path.is_file(), f"missing {role}: {path}")
        resolved = str(path.resolve())
        observed = sha256_file(path)
        if expected_sha256 is not None:
            require(observed == expected_sha256, f"{role} SHA mismatch: {observed}")
        if resolved in self._rows:
            row = self._rows[resolved]
            require(row["sha256"] == observed, f"input changed during audit: {path}")
            if role not in row["roles"]:
                row["roles"].append(role)
                row["roles"].sort()
            return observed
        self._rows[resolved] = {
            "path": resolved,
            "bytes": path.stat().st_size,
            "sha256": observed,
            "roles": [role],
            "expected_sha256": expected_sha256,
            "expected_sha256_status": "MATCH" if expected_sha256 is not None else "NOT_PROVIDED",
        }
        return observed

    def rows(self) -> list[dict[str, Any]]:
        return [json_copy(self._rows[key]) for key in sorted(self._rows)]

    def manifest_sha256(self) -> str:
        return canonical_sha(self.rows())

    def verify_unchanged(self) -> None:
        for row in self._rows.values():
            path = Path(row["path"])
            require(path.is_file(), f"input disappeared during audit: {path}")
            require(path.stat().st_size == row["bytes"], f"input size changed during audit: {path}")
            require(sha256_file(path) == row["sha256"], f"input bytes changed during audit: {path}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid {label} JSON: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def validate_protocol(protocol: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    require(
        protocol.get("schema") == "tactile3d-unit.s4-3-pi2s-diagnostic-protocol.v1",
        "unexpected PI2S protocol schema",
    )
    rows = protocol.get("samples", {}).get("h_parity_rows")
    require(isinstance(rows, list), "protocol h_parity_rows is missing")
    require(len(rows) == 64, f"protocol must contain 64 h_parity_rows, got {len(rows)}")
    require(
        all(isinstance(row, int) and not isinstance(row, bool) for row in rows),
        "invalid parity row",
    )
    require(len(set(rows)) == len(rows), "h_parity_rows contains duplicates")
    expected_rows_sha = protocol["samples"].get("h_parity_rows_sha256")
    require(canonical_sha(rows) == expected_rows_sha, "h_parity_rows canonical SHA mismatch")

    h_metrics = protocol.get("metrics", {}).get("h_parity", {})
    levels = h_metrics.get("absolute_error_quantile_levels")
    require(levels == [0.0, 0.5, 0.9, 0.95, 0.99, 1.0], "unexpected H parity quantiles")
    require(h_metrics.get("comparison_dtype") == "float32", "comparison dtype is not float32")
    require(h_metrics.get("pairs") == [row[0] for row in PAIR_ORDER], "H parity pair order drifted")
    encoder = protocol.get("encoder_contract", {})
    require(encoder.get("input_shape") == [HISTORY_STEPS, TACTILE_DIM], "E_T input shape drifted")
    require(encoder.get("output_shape") == [CONTACT_STATE_DIM], "E_T output shape drifted")
    require(
        encoder.get("history_policy")
        == "LEFT_REPEAT_FIRST within episode; reset before every episode",
        "E_T history/reset policy drifted",
    )
    data = protocol.get("data_contract", {})
    require(data.get("history_ticks") == HISTORY_STEPS, "protocol history tick count drifted")
    require(data.get("raw_control_hz") == RAW_CONTROL_HZ, "protocol raw control rate drifted")
    require(data.get("official_converter_fps_label") == 30, "converter fps label drifted")
    return list(rows), [float(level) for level in levels]


def _expect_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    require(name in arrays, f"sidecar is missing {name}")
    value = np.asarray(arrays[name])
    require(value.shape == shape, f"sidecar {name} shape is {value.shape}, expected {shape}")
    require(
        value.dtype == np.dtype(dtype), f"sidecar {name} dtype is {value.dtype}, expected {dtype}"
    )
    return value


def rebuild_online_histories(
    arrays: Mapping[str, np.ndarray],
    alignment: Mapping[str, Any],
    selected_rows: Sequence[int],
    *,
    expected_rows: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild every episode in order and retain the selected canonical histories."""

    require("tactile_sim" in arrays, "sidecar is missing tactile_sim")
    tactile = np.asarray(arrays["tactile_sim"])
    require(tactile.ndim == 2 and tactile.shape[1] == TACTILE_DIM, "invalid tactile_sim shape")
    require(tactile.dtype == np.float32, "tactile_sim must be float32")
    row_count = len(tactile)
    if expected_rows is not None:
        require(
            row_count == expected_rows,
            f"sidecar row count is {row_count}, expected {expected_rows}",
        )

    index = _expect_array(arrays, "index", (row_count,), np.int64)
    episode_index = _expect_array(arrays, "episode_index", (row_count,), np.int64)
    frame_index = _expect_array(arrays, "frame_index", (row_count,), np.int64)
    control_tick = _expect_array(arrays, "control_tick_end", (row_count,), np.int64)
    bootstrap = _expect_array(arrays, "history_bootstrap_count", (row_count,), np.int16)
    cached_history = _expect_array(
        arrays, "tactile_history", (row_count, HISTORY_STEPS, TACTILE_DIM), np.float32
    )
    require(np.isfinite(tactile).all(), "tactile_sim contains non-finite values")
    require(np.isfinite(cached_history).all(), "tactile_history contains non-finite values")
    require(
        np.array_equal(index, np.arange(row_count, dtype=np.int64)), "global index is not identity"
    )

    episodes = np.unique(episode_index)
    require(len(episodes) > 0, "sidecar contains no episodes")
    require(
        np.array_equal(episodes, np.arange(len(episodes), dtype=np.int64)),
        "episode indices are not contiguous from zero",
    )
    lengths = alignment.get("episode_lengths")
    trims = alignment.get("raw_leading_static_frames_excluded_by_official_conversion")
    raw_lengths = alignment.get("raw_episode_lengths")
    require(
        isinstance(lengths, list) and len(lengths) == len(episodes),
        "alignment episode lengths drifted",
    )
    require(isinstance(trims, list) and len(trims) == len(episodes), "alignment trim list drifted")
    require(
        isinstance(raw_lengths, list) and len(raw_lengths) == len(episodes), "raw lengths drifted"
    )
    require(sum(int(value) for value in lengths) == row_count, "alignment length total drifted")

    selected = [int(row) for row in selected_rows]
    require(len(set(selected)) == len(selected), "selected history rows are not unique")
    require(all(0 <= row < row_count for row in selected), "selected history row is out of bounds")
    selected_positions = {row: position for position, row in enumerate(selected)}
    selected_histories = np.empty((len(selected), HISTORY_STEPS, TACTILE_DIM), dtype=np.float32)
    selected_records: list[dict[str, Any] | None] = [None] * len(selected)

    reset_count = 0
    append_count = 0
    bootstrap_rows = 0
    runtime = OnlineTactileHistory()
    episode_records: list[dict[str, Any]] = []
    offset = 0
    for episode in episodes.tolist():
        length = int(lengths[episode])
        trim = int(trims[episode])
        raw_length = int(raw_lengths[episode])
        require(length > 0, f"episode {episode} is empty")
        require(trim >= 0, f"episode {episode} has a negative leading trim")
        stop = offset + length
        rows = np.arange(offset, stop, dtype=np.int64)
        require(
            np.all(episode_index[rows] == episode), f"episode {episode} is not one contiguous block"
        )
        require(
            np.array_equal(frame_index[rows], np.arange(length, dtype=np.int64)),
            f"episode {episode} frame order is not contiguous",
        )
        require(
            raw_length - trim == length, f"episode {episode} raw/trim/official lengths disagree"
        )
        expected_ticks = trim + np.arange(length, dtype=np.int64)
        require(
            np.array_equal(control_tick[rows], expected_ticks),
            f"episode {episode} current control ticks do not match trim + frame",
        )

        for local_frame, row_value in enumerate(rows.tolist()):
            if local_frame == 0:
                rebuilt = runtime.reset(tactile[row_value])
                reset_count += 1
            else:
                rebuilt = runtime.append(tactile[row_value])
                append_count += 1
            expected_bootstrap = max(0, HISTORY_STEPS - 1 - local_frame)
            bootstrap_rows += int(expected_bootstrap > 0)
            require(
                int(bootstrap[row_value]) == expected_bootstrap,
                f"history bootstrap count drifted at row {row_value}",
            )
            require(
                np.array_equal(rebuilt, cached_history[row_value]),
                f"online/cached history mismatch at row {row_value}",
            )
            require(
                np.array_equal(rebuilt[-1], tactile[row_value]),
                f"history does not end at current tactile row {row_value}",
            )
            if local_frame == 0:
                require(
                    np.array_equal(
                        rebuilt,
                        np.repeat(tactile[row_value : row_value + 1], HISTORY_STEPS, axis=0),
                    ),
                    f"episode {episode} did not reset with LEFT_REPEAT_FIRST",
                )
            if row_value in selected_positions:
                position = selected_positions[row_value]
                selected_histories[position] = rebuilt
                selected_records[position] = {
                    "row": row_value,
                    "episode_index": episode,
                    "frame_index": local_frame,
                    "current_control_tick_raw50hz": int(control_tick[row_value]),
                    "post_trim_episode_start_control_tick_raw50hz": trim,
                    "oldest_history_source_control_tick_raw50hz": max(
                        trim, int(control_tick[row_value]) - (HISTORY_STEPS - 1)
                    ),
                    "history_bootstrap_count": expected_bootstrap,
                    "history_sha256": sha256_array(rebuilt),
                }

        episode_records.append(
            {
                "episode_index": episode,
                "global_start_row": offset,
                "global_end_row_inclusive": stop - 1,
                "frames": length,
                "raw_leading_static_trim_ticks": trim,
                "first_current_control_tick_raw50hz": int(control_tick[offset]),
                "last_current_control_tick_raw50hz": int(control_tick[stop - 1]),
                "reset_history_sha256": sha256_array(cached_history[offset]),
                "gates": {
                    "frame_order_contiguous": "PASS",
                    "current_tick_equals_trim_plus_frame": "PASS",
                    "left_repeat_first_at_reset": "PASS",
                    "history_never_precedes_post_trim_episode_start": "PASS",
                },
            }
        )
        offset = stop

    require(offset == row_count, "episode traversal did not consume every sidecar row")
    require(
        all(record is not None for record in selected_records),
        "not every selected history was rebuilt",
    )
    expected_bootstrap_rows = sum(min(HISTORY_STEPS - 1, int(length)) for length in lengths)
    require(bootstrap_rows == expected_bootstrap_rows, "bootstrap row count drifted")

    gates = {
        "global_index_identity": True,
        "episodes_contiguous": True,
        "frames_contiguous_within_episode": True,
        "exactly_one_reset_per_episode": reset_count == len(episodes),
        "all_noninitial_rows_appended": append_count == row_count - len(episodes),
        "left_repeat_first": True,
        "cached_history_matches_online_rebuild_bitwise": True,
        "history_ends_at_current_tactile_sample": True,
        "current_control_tick_equals_trim_plus_frame": True,
        "control_ticks_advance_by_one_raw_tick": True,
        "no_history_crosses_episode_boundary": True,
        "all_selected_histories_captured": True,
    }
    require(all(gates.values()), f"physical-time/history gates failed: {gates}")
    audit = {
        "status": "PASS",
        "rows_checked": row_count,
        "episodes_checked": len(episodes),
        "resets": reset_count,
        "appends": append_count,
        "bootstrap_rows": bootstrap_rows,
        "history_contract": {
            "shape": [HISTORY_STEPS, TACTILE_DIM],
            "dtype": "float32",
            "policy": "LEFT_REPEAT_FIRST within each post-trim episode",
            "reset_before_every_episode": True,
            "history_samples": HISTORY_STEPS,
            "full_history_interval_count": HISTORY_STEPS - 1,
            "full_history_physical_seconds_at_raw_control_rate": (HISTORY_STEPS - 1)
            / RAW_CONTROL_HZ,
        },
        "physical_time_contract": {
            "authoritative_clock": "control_tick_end at raw 50 Hz",
            "raw_control_hz": RAW_CONTROL_HZ,
            "official_converter_fps_label": int(alignment.get("fps_from_info", -1)),
            "official_converter_fps_label_is_physical_resampling_evidence": False,
            "current_tick_formula": "raw_leading_static_trim_ticks + frame_index",
            "bootstrap_repeats_first_post_trim_sample_without_fabricating_earlier_time": True,
        },
        "selected_rows": [record for record in selected_records if record is not None],
        "episodes": episode_records,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
    }
    return selected_histories, audit


def encode_direct(histories: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    """Run exactly the accepted teacher and exact Torch normalization on CPU."""

    import torch

    from gr00t.simulation.s4_3_act import TorchTactileNormalization
    from gr00t.simulation.sim_contact_models import load_teacher_checkpoint

    value = np.ascontiguousarray(histories, dtype=np.float32)
    require(value.shape[1:] == (HISTORY_STEPS, TACTILE_DIM), "invalid direct E_T history shape")
    teacher, checkpoint = load_teacher_checkpoint(checkpoint_path, "cpu")
    require(
        checkpoint.get("schema") == "tactile3d-unit.s4-2-sim-contact-teacher.v1",
        "unexpected accepted E_T checkpoint schema",
    )
    normalization = TorchTactileNormalization(checkpoint["normalization"])
    teacher.cpu().eval().requires_grad_(False)
    normalization.cpu().eval().requires_grad_(False)
    with torch.inference_mode():
        tensor = torch.from_numpy(value)
        require(tensor.device.type == "cpu", "direct E_T input unexpectedly left CPU")
        output = teacher(normalization(tensor))["latent"].to(dtype=torch.float32, device="cpu")
    result = np.ascontiguousarray(output.numpy(), dtype=np.float32)
    require(result.shape == (len(value), CONTACT_STATE_DIM), "direct E_T output shape drifted")
    require(np.isfinite(result).all(), "direct E_T output is not finite")
    return result


def _service_wrapper_code() -> str:
    # Patch only the service's two filesystem constants.  Its model, normalization,
    # request loop, and transport remain the frozen implementation under audit.
    return (
        "import sys\n"
        "from pathlib import Path\n"
        "import scripts.simulation.serve_s4_3_pi1_contact_state as service\n"
        "service.ARTIFACT = Path(sys.argv[1])\n"
        "service.CHECKPOINT = Path(sys.argv[2])\n"
        "sys.argv = [service.__file__, '--socket', sys.argv[3]]\n"
        "service.main()\n"
    )


def _normalized_service_stdout(stdout: str) -> tuple[list[str], dict[str, Any] | None]:
    lines: list[str] = []
    final: dict[str, Any] | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("CONTACT_STATE_SERVICE_READY socket="):
            lines.append("CONTACT_STATE_SERVICE_READY socket=$TEMP/contact_state.sock")
            continue
        lines.append(line)
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "status" in parsed:
            final = parsed
    return lines, final


def encode_live(
    histories: np.ndarray,
    checkpoint_path: Path,
    *,
    readiness_timeout_seconds: float = 60.0,
    shutdown_timeout_seconds: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the frozen Unix service in an isolated temporary directory."""

    value = np.ascontiguousarray(histories, dtype=np.float32)
    require(value.shape[1:] == (HISTORY_STEPS, TACTILE_DIM), "invalid live E_T history shape")
    client: ContactStateUnixClient | None = None
    process: subprocess.Popen[str] | None = None
    outputs: list[np.ndarray] = []
    client_errors: list[str] = []
    stdout = ""
    stderr = ""
    graceful_shutdown = False
    forced_kill = False
    ready = False

    with tempfile.TemporaryDirectory(prefix="pi2s-h-parity-") as temporary_name:
        temporary = Path(temporary_name)
        socket_path = temporary / "contact_state.sock"
        service_artifact_path = temporary / "service.json"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        command = [
            sys.executable,
            "-B",
            "-c",
            _service_wrapper_code(),
            str(service_artifact_path),
            str(checkpoint_path),
            str(socket_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + readiness_timeout_seconds
            while time.monotonic() < deadline:
                if socket_path.exists():
                    ready = True
                    break
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(0.02)
            require(ready, "live Contact-State service did not create its socket before timeout")
            client = ContactStateUnixClient(socket_path, timeout_seconds=30.0)
            for history in value:
                try:
                    outputs.append(client.encode(history))
                except Exception as error:
                    client_errors.append(f"{type(error).__name__}: {error}")
                    raise
        finally:
            if client is not None:
                client.close()
            if process is not None:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        stdout, stderr = process.communicate(timeout=shutdown_timeout_seconds)
                        graceful_shutdown = True
                    except subprocess.TimeoutExpired:
                        forced_kill = True
                        process.kill()
                        stdout, stderr = process.communicate(timeout=5.0)
                else:
                    stdout, stderr = process.communicate(timeout=5.0)

        require(process is not None, "live service process was not started")
        normalized_stdout, final_stdout = _normalized_service_stdout(stdout)
        service_artifact: dict[str, Any] = {}
        if service_artifact_path.is_file():
            service_artifact = load_json(service_artifact_path, "temporary service report")
        stable_service_report = {
            key: service_artifact[key]
            for key in (
                "schema",
                "status",
                "environment",
                "transport",
                "checkpoint",
                "checkpoint_sha256",
                "history_bootstrap",
                "requests",
                "errors",
                "gates",
            )
            if key in service_artifact
        }
        evidence = {
            "launch": {
                "python_executable": str(Path(sys.executable).resolve()),
                "implementation": str(SERVICE_SCRIPT.resolve()),
                "isolated_temporary_directory": True,
                "socket_path": "$TEMP/contact_state.sock",
                "service_artifact_path": "$TEMP/service.json",
                "cuda_visible_devices": "",
            },
            "ready_socket_observed": ready,
            "termination_signal": "SIGTERM",
            "graceful_shutdown": graceful_shutdown,
            "forced_kill": forced_kill,
            "returncode": process.returncode,
            "stdout_lines": normalized_stdout,
            "stderr_lines": [line for line in stderr.splitlines() if line.strip()],
            "stdout_final_report": final_stdout,
            "client_requests": 0 if client is None else client.requests,
            "client_errors": (
                client_errors if client is None else list(client.errors) + client_errors
            ),
            "service_report": stable_service_report,
            "socket_removed_before_temp_cleanup": not socket_path.exists(),
        }

    live = np.ascontiguousarray(np.stack(outputs, axis=0), dtype=np.float32)
    require(live.shape == (len(value), CONTACT_STATE_DIM), "live E_T output shape drifted")
    require(np.isfinite(live).all(), "live E_T output is not finite")
    service_gates = evidence["service_report"].get("gates", {})
    process_gates = {
        "service_ready": evidence["ready_socket_observed"],
        "all_requests_completed": evidence["client_requests"] == len(value),
        "client_no_errors": not evidence["client_errors"],
        "service_no_errors": not evidence["stderr_lines"]
        and not evidence["service_report"].get("errors", []),
        "service_status_pass": evidence["service_report"].get("status") == "PASS",
        "service_report_requests_match": evidence["service_report"].get("requests") == len(value),
        "service_report_gates_pass": bool(service_gates)
        and all(result == "PASS" for result in service_gates.values()),
        "graceful_sigterm": evidence["graceful_shutdown"]
        and not evidence["forced_kill"]
        and evidence["returncode"] == 0,
        "socket_removed": evidence["socket_removed_before_temp_cleanup"],
    }
    evidence["gates"] = {
        name: "PASS" if result else "FAIL" for name, result in process_gates.items()
    }
    require(all(process_gates.values()), f"live service lifecycle gates failed: {process_gates}")
    return live, evidence


def _finite_fraction(value: np.ndarray) -> float:
    return float(np.count_nonzero(np.isfinite(value)) / value.size)


def _quantile_records(error: np.ndarray, levels: Sequence[float]) -> list[dict[str, float]]:
    flattened = np.asarray(error, dtype=np.float32).reshape(-1)
    require(flattened.size > 0, "cannot compute quantiles of an empty comparison")
    require(np.isfinite(flattened).all(), "cannot compute quantiles of non-finite errors")
    values = np.quantile(flattened.astype(np.float64), np.asarray(levels), method="linear")
    return [
        {"level": float(level), "absolute_error": float(value)}
        for level, value in zip(levels, values.tolist(), strict=True)
    ]


def path_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_fraction": _finite_fraction(array),
        "array_sha256": sha256_array(array),
    }


def parent_runtime_environment() -> dict[str, Any]:
    """Record the interpreter and numeric stack used by the direct CPU path."""

    import torch

    numpy_module_path = getattr(np, "__file__", None)
    torch_module_path = getattr(torch, "__file__", None)
    require(isinstance(numpy_module_path, str), "NumPy module path is unavailable")
    require(isinstance(torch_module_path, str), "Torch module path is unavailable")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_implementation": sys.implementation.name,
        "numpy_version": str(np.__version__),
        "numpy_module_path": str(Path(numpy_module_path).resolve()),
        "torch_version": str(torch.__version__),
        "torch_module_path": str(Path(torch_module_path).resolve()),
        "direct_torch_device": "cpu",
        "parent_os_environ_unchanged": True,
    }


def pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
    levels: Sequence[float],
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], np.ndarray]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    require(lhs.shape == rhs.shape, f"pair shapes differ: {lhs.shape} != {rhs.shape}")
    require(lhs.dtype == np.float32 and rhs.dtype == np.float32, "pair inputs must both be float32")
    require(np.isfinite(lhs).all() and np.isfinite(rhs).all(), "pair contains non-finite values")
    error = np.abs(lhs - rhs).astype(np.float32, copy=False)
    bitwise_equal = bool(
        np.array_equal(
            np.ascontiguousarray(lhs).view(np.uint32),
            np.ascontiguousarray(rhs).view(np.uint32),
        )
    )
    metrics = {
        "shape": list(lhs.shape),
        "comparison_dtype": "float32",
        "left_finite_fraction": _finite_fraction(lhs),
        "right_finite_fraction": _finite_fraction(rhs),
        "array_equal": bool(np.array_equal(lhs, rhs)),
        "bitwise_equal": bitwise_equal,
        "array_equal_bitwise": bitwise_equal,
        "allclose_at_protocol_tolerance": bool(np.allclose(lhs, rhs, atol=atol, rtol=rtol)),
        "max_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error, dtype=np.float64)),
        "absolute_error_quantiles": _quantile_records(error, levels),
    }
    return metrics, error


def parity_metrics(
    paths: Mapping[str, np.ndarray],
    levels: Sequence[float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    require(
        set(paths) == {row[1] for row in PAIR_ORDER} | {row[2] for row in PAIR_ORDER},
        "paths drifted",
    )
    summaries = {name: path_summary(paths[name]) for name in sorted(paths)}
    pairs: dict[str, Any] = {}
    all_errors: list[np.ndarray] = []
    for pair_name, left_name, right_name in PAIR_ORDER:
        metrics, error = pair_metrics(
            paths[left_name], paths[right_name], levels, atol=atol, rtol=rtol
        )
        metrics["left"] = left_name
        metrics["right"] = right_name
        pairs[pair_name] = metrics
        all_errors.append(error.reshape(-1))
    aggregate_error = np.concatenate(all_errors).astype(np.float32, copy=False)
    return {
        "paths": summaries,
        "pairs": pairs,
        "aggregate": {
            "all_pairs_array_equal": all(row["array_equal"] for row in pairs.values()),
            "all_pairs_bitwise_equal": all(row["bitwise_equal"] for row in pairs.values()),
            "all_pairs_array_equal_bitwise": all(
                row["array_equal_bitwise"] for row in pairs.values()
            ),
            "all_pairs_allclose_at_protocol_tolerance": all(
                row["allclose_at_protocol_tolerance"] for row in pairs.values()
            ),
            "compared_pair_count": len(pairs),
            "absolute_error_element_count": int(aggregate_error.size),
            "max_absolute_error": float(np.max(aggregate_error)),
            "mean_absolute_error": float(np.mean(aggregate_error, dtype=np.float64)),
            "absolute_error_quantiles": _quantile_records(aggregate_error, levels),
        },
    }


def output_artifact(
    schema: str,
    body: Mapping[str, Any],
    inputs: InputRegistry,
) -> dict[str, Any]:
    semantic = {
        "schema": schema,
        **json_copy(body),
        "producer": {
            "path": str(RUNNER_IMPLEMENTATION),
            "sha256": sha256_file(RUNNER_IMPLEMENTATION),
            "execution": (
                "CPU_ONLY_READ_ONLY_INPUTS_TEMPORARY_UNIX_SERVICE_"
                "ATOMIC_FSYNC_ARTIFACTS_NO_TRAINING"
            ),
        },
        "input_manifest": inputs.rows(),
        "input_manifest_sha256": inputs.manifest_sha256(),
    }
    return {
        **semantic,
        "created_at_utc": now_utc(),
        "semantic_sha256": canonical_sha(semantic),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically persist JSON, fsync the file and directory, then read it back."""

    path.parent.mkdir(parents=True, exist_ok=True)
    require(
        not os.path.lexists(path),
        f"refusing to overwrite existing artifact; use --check: {path}",
    )
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A same-directory hard link publishes the complete, fsynced inode
            # atomically and, unlike os.replace(), can never clobber a target
            # created by a concurrent audit after the lexists preflight.
            try:
                os.link(temporary, path, follow_symlinks=False)
            except PermissionError:
                # NFS identity mapping can expose a different local owner and
                # trigger protected_hardlinks.  Permit the link briefly, then
                # seal and revalidate the published inode below.
                os.chmod(temporary, 0o666)
                os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise AuditError(
                f"refusing to overwrite existing artifact; use --check: {path}"
            ) from error
        try:
            os.chmod(path, 0o444)
            require(path.read_bytes() == data, f"atomic write readback mismatch: {path}")
        except BaseException:
            try:
                source_info = temporary.stat()
                published_info = path.stat()
                if (
                    source_info.st_dev == published_info.st_dev
                    and source_info.st_ino == published_info.st_ino
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
            raise
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    observed = path.read_bytes()
    require(observed == data, f"atomic write readback mismatch: {path}")
    return hashlib.sha256(observed).hexdigest()


def check_artifact(path: Path, expected: Mapping[str, Any]) -> str:
    """Compare deterministic semantics while retaining the original creation time."""

    require(path.is_file(), f"missing output for --check: {path}")
    try:
        current = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid output JSON {path}: {error}") from error
    require(isinstance(current, dict), f"output is not an object: {path}")
    created_at = current.pop("created_at_utc", None)
    observed_semantic_sha = current.pop("semantic_sha256", None)
    require(isinstance(created_at, str) and created_at, f"missing created_at_utc: {path}")
    require(observed_semantic_sha == canonical_sha(current), f"bad semantic SHA: {path}")

    expected_semantic = json_copy(expected)
    expected_semantic.pop("created_at_utc", None)
    internal_semantic_sha = expected_semantic.pop("semantic_sha256", None)
    require(
        internal_semantic_sha == canonical_sha(expected_semantic), "internal semantic SHA error"
    )
    require(current == expected_semantic, f"output is stale or drifted: {path}")
    return sha256_file(path)


def _protocol_expected_sha(protocol: Mapping[str, Any], *keys: str) -> str:
    value: Any = protocol
    for key in keys:
        require(
            isinstance(value, Mapping) and key in value, f"protocol is missing {'.'.join(keys)}"
        )
        value = value[key]
    require(isinstance(value, str) and len(value) == 64, f"invalid SHA at {'.'.join(keys)}")
    return value


def build_all(
    *,
    protocol_path: Path = PROTOCOL,
    sidecar_path: Path = SIDECAR,
    alignment_path: Path = ALIGNMENT,
    checkpoint_path: Path = CHECKPOINT,
) -> dict[str, dict[str, Any]]:
    parent_environment_before = dict(os.environ)
    inputs = InputRegistry()
    inputs.record(protocol_path, "frozen PI2S diagnostic protocol")
    protocol = load_json(protocol_path, "PI2S diagnostic protocol")
    runner_expected = _protocol_expected_sha(
        protocol,
        "diagnostic_implementation_sha256",
        "run_pi2s_h_parity",
    )
    inputs.record(
        RUNNER_IMPLEMENTATION,
        "H parity runner implementation",
        runner_expected,
    )
    rows, levels = validate_protocol(protocol)

    sidecar_expected = _protocol_expected_sha(protocol, "data_contract", "contact_sidecar_sha256")
    checkpoint_expected = _protocol_expected_sha(
        protocol, "encoder_contract", "accepted_et_checkpoint_sha256"
    )
    alignment_expected = _protocol_expected_sha(
        protocol, "source_manifests", "official_raw_alignment", "sha256"
    )
    direct_expected = _protocol_expected_sha(
        protocol, "encoder_contract", "paths", "direct_et", "implementation_sha256"
    )
    runtime_expected = _protocol_expected_sha(
        protocol, "encoder_contract", "paths", "live_sidecar", "runtime_implementation_sha256"
    )
    history_expected = _protocol_expected_sha(
        protocol, "encoder_contract", "paths", "live_sidecar", "history_implementation_sha256"
    )
    service_expected = _protocol_expected_sha(
        protocol, "encoder_contract", "paths", "live_sidecar", "service_implementation_sha256"
    )
    teacher_loader_expected = _protocol_expected_sha(
        protocol,
        "encoder_contract",
        "paths",
        "shared_teacher_loader",
        "implementation_sha256",
    )
    teacher_architecture_expected = _protocol_expected_sha(
        protocol,
        "encoder_contract",
        "paths",
        "shared_teacher_loader",
        "architecture_implementation_sha256",
    )
    inputs.record(sidecar_path, "cached PI1 tactile and H sidecar", sidecar_expected)
    inputs.record(checkpoint_path, "accepted frozen E_T checkpoint", checkpoint_expected)
    inputs.record(alignment_path, "official raw/converted alignment", alignment_expected)
    inputs.record(
        DIRECT_IMPLEMENTATION, "exact Torch normalization implementation", direct_expected
    )
    inputs.record(
        RUNTIME_IMPLEMENTATION, "online history and Unix client implementation", runtime_expected
    )
    inputs.record(
        HISTORY_IMPLEMENTATION,
        "online LEFT_REPEAT_FIRST history implementation and constants",
        history_expected,
    )
    inputs.record(SERVICE_SCRIPT, "live Unix service implementation", service_expected)
    inputs.record(
        TEACHER_LOADER_IMPLEMENTATION,
        "shared accepted E_T checkpoint loader and architecture dispatcher",
        teacher_loader_expected,
    )
    inputs.record(
        TEACHER_ARCHITECTURE_IMPLEMENTATION,
        "accepted E_T neural architecture implementation",
        teacher_architecture_expected,
    )

    alignment = load_json(alignment_path, "official raw/converted alignment")
    require(alignment.get("status") == "PASS", "official raw/converted alignment is not PASS")
    expected_rows = int(protocol.get("data_contract", {}).get("rows", -1))
    required_sidecar_arrays = (
        "index",
        "episode_index",
        "frame_index",
        "tactile_sim",
        "tactile_history",
        "history_bootstrap_count",
        "control_tick_end",
        "contact_state",
    )
    with np.load(sidecar_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in required_sidecar_arrays if name in source.files}
    selected_histories, physical_body = rebuild_online_histories(
        arrays, alignment, rows, expected_rows=expected_rows
    )
    cached = _expect_array(arrays, "contact_state", (expected_rows, CONTACT_STATE_DIM), np.float32)[
        np.asarray(rows, dtype=np.int64)
    ].copy()
    require(np.isfinite(cached).all(), "selected cached H contains non-finite values")

    direct = encode_direct(selected_histories, checkpoint_path)
    live, service_evidence = encode_live(selected_histories, checkpoint_path)
    numeric = protocol.get("numeric_tolerances", {})
    atol = float(numeric.get("float32_same_path_atol"))
    rtol = float(numeric.get("float32_same_path_rtol"))
    metrics = parity_metrics(
        {
            "cached_sidecar": cached,
            "direct_et": direct,
            "live_unix_service": live,
        },
        levels,
        atol=atol,
        rtol=rtol,
    )
    history_sha = sha256_array(selected_histories)
    parity_matches = bool(metrics["aggregate"]["all_pairs_allclose_at_protocol_tolerance"])
    parity_body = {
        "status": (
            "COMPLETE_MATCH_WITHIN_FROZEN_TOLERANCE"
            if parity_matches
            else "COMPLETE_MISMATCH_EXCEEDS_FROZEN_TOLERANCE"
        ),
        "scientific_role": "POST_HOC_DESCRIPTIVE_H_PATH_PARITY_NO_CAUSAL_CLAIM",
        "cpu_only": True,
        "gpu_used": False,
        "training_or_optimizer_updates": 0,
        "selected_rows": rows,
        "selected_rows_sha256": canonical_sha(rows),
        "selected_row_count": len(rows),
        "comparison_input_history": {
            "shape": list(selected_histories.shape),
            "dtype": str(selected_histories.dtype),
            "array_sha256": history_sha,
            "cached_history_matches_full_order_online_rebuild_bitwise": True,
            "routes": {
                "cached_sidecar": history_sha,
                "direct_et": history_sha,
                "live_unix_service": history_sha,
            },
        },
        "protocol_tolerance": {"atol": atol, "rtol": rtol},
        "parent_runtime_environment": parent_runtime_environment(),
        "metrics": metrics,
        "live_service_execution": service_evidence,
        "interpretation": (
            "Descriptive parity only; bitwise/allclose flags and errors are reported without "
            "introducing a post-hoc scientific acceptance threshold."
        ),
        "gates": {
            "protocol_64_rows_exact": "PASS",
            "same_float32_history_for_all_paths": "PASS",
            "all_outputs_finite_64x256_float32": "PASS",
            "live_service_lifecycle": "PASS",
            "source_and_model_inputs_read_only": "PASS",
            "all_pairs_within_frozen_tolerance": "PASS" if parity_matches else "FAIL",
        },
    }
    physical_body = {
        **physical_body,
        "selected_history_array_sha256": history_sha,
        "selected_history_shape": list(selected_histories.shape),
        "converter_label_warning": protocol.get("data_contract", {}).get("converter_label_warning"),
    }
    inputs.verify_unchanged()
    require(
        dict(os.environ) == parent_environment_before,
        "parent process environment mutated during the audit",
    )
    return {
        "parity": output_artifact(
            "tactile3d-unit.s4-3-pi2s-offline-online-h-parity.v1", parity_body, inputs
        ),
        "physical_time": output_artifact(
            "tactile3d-unit.s4-3-pi2s-h-physical-time-audit.v1", physical_body, inputs
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute all CPU parity evidence and verify existing JSON without replacing it",
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PI2S_ROOT / "artifacts",
        help="artifact directory; defaults to the NAS-backed PI2S experiment root",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = {
        "parity": args.output_root / "offline_online_h_parity.json",
        "physical_time": args.output_root / "h_physical_time_audit.json",
    }
    artifacts = build_all(
        protocol_path=args.protocol,
        sidecar_path=args.sidecar,
        alignment_path=args.alignment,
        checkpoint_path=args.checkpoint,
    )
    hashes: dict[str, str] = {}
    if args.check:
        for name, path in outputs.items():
            hashes[name] = check_artifact(path, artifacts[name])
        operation = "CHECK_PASS"
    else:
        existing = [str(path) for path in outputs.values() if os.path.lexists(path)]
        require(not existing, f"refusing to overwrite existing artifacts; use --check: {existing}")
        for name, path in outputs.items():
            hashes[name] = atomic_json(path, artifacts[name])
        for name, path in outputs.items():
            check_artifact(path, artifacts[name])
        operation = "WRITE_AND_READBACK_PASS"
    print(
        json.dumps(
            {
                "status": operation,
                "cpu_only": True,
                "gpu_used": False,
                "selected_rows": 64,
                "outputs": {
                    name: {"path": str(outputs[name]), "sha256": hashes[name]} for name in outputs
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except AuditError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from error
