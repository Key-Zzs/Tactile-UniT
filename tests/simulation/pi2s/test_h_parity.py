from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from gr00t.simulation.s4_3_pi1 import HISTORY_STEPS, TACTILE_DIM, OnlineTactileHistory
from scripts.simulation import run_pi2s_h_parity as parity

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs/simulation/pi2s/diagnostic_protocol.json"


def test_defaults_use_persistent_pi2s_input_snapshots() -> None:
    assert parity.SIDECAR == parity.PI2S_ROOT / "source_snapshots/data/pi1_contact_sidecar.npz"
    assert parity.ALIGNMENT == (
        parity.PI2S_ROOT
        / "source_snapshots/pi2s_protocol_v3/source_manifests/official_dataset_alignment.json"
    )


def test_build_all_hard_gates_runner_implementation_sha(tmp_path: Path) -> None:
    protocol = tmp_path / "diagnostic_protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "diagnostic_implementation_sha256": {
                    "run_pi2s_h_parity": "0" * 64,
                }
            }
        )
    )

    with pytest.raises(parity.AuditError, match="H parity runner implementation SHA mismatch"):
        parity.build_all(protocol_path=protocol)


def _synthetic_sidecar() -> tuple[dict[str, np.ndarray], dict]:
    lengths = [3, 2]
    trims = [7, 11]
    tactile = np.arange(sum(lengths) * TACTILE_DIM, dtype=np.float32).reshape(-1, TACTILE_DIM)
    tactile[3:] += 10_000.0
    episode_index = np.repeat(np.arange(2, dtype=np.int64), lengths)
    frame_index = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    control_tick = np.concatenate(
        [trim + np.arange(length, dtype=np.int64) for trim, length in zip(trims, lengths)]
    )
    histories = np.empty((len(tactile), HISTORY_STEPS, TACTILE_DIM), dtype=np.float32)
    bootstrap = np.empty(len(tactile), dtype=np.int16)
    runtime = OnlineTactileHistory()
    offset = 0
    for length in lengths:
        for frame in range(length):
            row = offset + frame
            histories[row] = (
                runtime.reset(tactile[row]) if frame == 0 else runtime.append(tactile[row])
            )
            bootstrap[row] = HISTORY_STEPS - 1 - frame
        offset += length
    arrays = {
        "index": np.arange(len(tactile), dtype=np.int64),
        "episode_index": episode_index,
        "frame_index": frame_index,
        "tactile_sim": tactile,
        "tactile_history": histories,
        "history_bootstrap_count": bootstrap,
        "control_tick_end": control_tick,
    }
    alignment = {
        "episode_lengths": lengths,
        "raw_leading_static_frames_excluded_by_official_conversion": trims,
        "raw_episode_lengths": [trim + length for trim, length in zip(trims, lengths)],
        "fps_from_info": 30,
    }
    return arrays, alignment


def test_frozen_protocol_has_exact_64_canonical_h_rows() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    rows, levels = parity.validate_protocol(protocol)
    assert len(rows) == 64
    assert len(set(rows)) == 64
    assert parity.canonical_sha(rows) == protocol["samples"]["h_parity_rows_sha256"]
    assert levels == [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    shared = protocol["encoder_contract"]["paths"]["shared_teacher_loader"]
    assert shared["implementation_sha256"] == parity.sha256_file(
        parity.TEACHER_LOADER_IMPLEMENTATION
    )
    assert shared["architecture_implementation_sha256"] == parity.sha256_file(
        parity.TEACHER_ARCHITECTURE_IMPLEMENTATION
    )
    live = protocol["encoder_contract"]["paths"]["live_sidecar"]
    assert live["history_implementation_sha256"] == parity.sha256_file(
        parity.HISTORY_IMPLEMENTATION
    )


def test_parent_runtime_evidence_is_cpu_only_without_environment_mutation() -> None:
    environment_before = dict(os.environ)
    evidence = parity.parent_runtime_environment()
    assert evidence["python_executable"] == str(Path(sys.executable).resolve())
    assert evidence["numpy_module_path"] == str(Path(np.__file__).resolve())
    assert Path(evidence["torch_module_path"]).is_file()
    assert evidence["torch_version"]
    assert evidence["direct_torch_device"] == "cpu"
    assert evidence["parent_os_environ_unchanged"] is True
    assert dict(os.environ) == environment_before


def test_full_order_online_rebuild_resets_and_never_crosses_episode() -> None:
    arrays, alignment = _synthetic_sidecar()
    selected, audit = parity.rebuild_online_histories(
        arrays, alignment, [0, 2, 3, 4], expected_rows=5
    )
    assert selected.dtype == np.float32
    assert selected.shape == (4, HISTORY_STEPS, TACTILE_DIM)
    assert np.array_equal(selected[0], np.repeat(arrays["tactile_sim"][0:1], HISTORY_STEPS, axis=0))
    assert np.array_equal(selected[2], np.repeat(arrays["tactile_sim"][3:4], HISTORY_STEPS, axis=0))
    assert audit["resets"] == 2
    assert audit["appends"] == 3
    assert audit["gates"]["no_history_crosses_episode_boundary"] == "PASS"
    assert audit["gates"]["history_ends_at_current_tactile_sample"] == "PASS"
    assert audit["physical_time_contract"]["raw_control_hz"] == 50
    assert (
        audit["physical_time_contract"][
            "official_converter_fps_label_is_physical_resampling_evidence"
        ]
        is False
    )


def test_full_order_rebuild_rejects_cross_episode_cached_history() -> None:
    arrays, alignment = _synthetic_sidecar()
    arrays["tactile_history"] = arrays["tactile_history"].copy()
    arrays["tactile_history"][3, 0] = arrays["tactile_sim"][2]
    with pytest.raises(parity.AuditError, match="online/cached history mismatch at row 3"):
        parity.rebuild_online_histories(arrays, alignment, [3], expected_rows=5)


def test_pair_metrics_report_value_and_bitwise_equality_separately() -> None:
    positive_zero = np.zeros((1, 2), dtype=np.float32)
    negative_zero = positive_zero.copy()
    negative_zero[0, 0] = np.float32(-0.0)
    metrics, error = parity.pair_metrics(
        positive_zero,
        negative_zero,
        [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
        atol=1e-6,
        rtol=1e-5,
    )
    assert metrics["array_equal"] is True
    assert metrics["bitwise_equal"] is False
    assert metrics["max_absolute_error"] == 0.0
    assert metrics["mean_absolute_error"] == 0.0
    assert [row["level"] for row in metrics["absolute_error_quantiles"]] == [
        0.0,
        0.5,
        0.9,
        0.95,
        0.99,
        1.0,
    ]
    assert error.dtype == np.float32


def test_parity_metrics_cover_all_frozen_pairs_and_aggregate() -> None:
    cached = np.zeros((2, 256), dtype=np.float32)
    direct = cached.copy()
    live = cached.copy()
    live[1, 255] = np.float32(2.0)
    metrics = parity.parity_metrics(
        {
            "cached_sidecar": cached,
            "direct_et": direct,
            "live_unix_service": live,
        },
        [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
        atol=1e-6,
        rtol=1e-5,
    )
    assert list(metrics["pairs"]) == [
        "cached_vs_direct",
        "cached_vs_live",
        "direct_vs_live",
    ]
    assert metrics["pairs"]["cached_vs_direct"]["bitwise_equal"] is True
    assert metrics["pairs"]["cached_vs_live"]["max_absolute_error"] == 2.0
    assert metrics["aggregate"]["all_pairs_bitwise_equal"] is False
    assert metrics["aggregate"]["absolute_error_element_count"] == 3 * 2 * 256


def test_normalized_service_stdout_removes_ephemeral_socket_path() -> None:
    lines, final = parity._normalized_service_stdout(
        "CONTACT_STATE_SERVICE_READY socket=/tmp/random/contact_state.sock\n"
        '{"status": "PASS", "requests": 64, "errors": 0}\n'
    )
    assert lines[0] == "CONTACT_STATE_SERVICE_READY socket=$TEMP/contact_state.sock"
    assert final == {"status": "PASS", "requests": 64, "errors": 0}


def test_atomic_fsync_write_and_check_is_read_only(tmp_path: Path) -> None:
    inputs = parity.InputRegistry()
    source = tmp_path / "input.json"
    source.write_text("{}\n")
    inputs.record(source, "fixture")
    artifact = parity.output_artifact("fixture.v1", {"status": "PASS"}, inputs)
    output = tmp_path / "artifact.json"
    persisted_sha = parity.atomic_json(output, artifact)
    before = output.read_bytes()
    assert persisted_sha == hashlib.sha256(before).hexdigest()
    assert parity.check_artifact(output, artifact) == persisted_sha
    assert output.read_bytes() == before
    with pytest.raises(parity.AuditError, match="refusing to overwrite"):
        parity.atomic_json(output, artifact)
    assert output.read_bytes() == before


def test_atomic_write_cannot_clobber_concurrent_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact.json"
    competing_bytes = b'{"publisher":"other"}\n'
    real_link = parity.os.link

    def concurrent_link(source: Path, destination: Path, **kwargs: object) -> None:
        Path(destination).write_bytes(competing_bytes)
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(parity.os, "link", concurrent_link)
    with pytest.raises(parity.AuditError, match="refusing to overwrite"):
        parity.atomic_json(output, {"publisher": "audit"})
    assert output.read_bytes() == competing_bytes
    assert not list(tmp_path.glob(".artifact.json.tmp.*"))


def test_check_detects_semantic_drift(tmp_path: Path) -> None:
    inputs = parity.InputRegistry()
    source = tmp_path / "input.json"
    source.write_text("{}\n")
    inputs.record(source, "fixture")
    artifact = parity.output_artifact("fixture.v1", {"status": "PASS"}, inputs)
    output = tmp_path / "artifact.json"
    parity.atomic_json(output, artifact)
    changed = json.loads(output.read_text())
    changed["status"] = "FAIL"
    output.chmod(0o644)
    output.write_text(json.dumps(changed))
    with pytest.raises(parity.AuditError, match="bad semantic SHA"):
        parity.check_artifact(output, artifact)
