from __future__ import annotations

import json
from pathlib import Path
import socket
import threading

import numpy as np
import pytest

from gr00t.simulation.pi2sr_runtime import (
    DirectContactEncoder,
    numeric_metrics,
    receive_float32_frame,
    repeatability_metrics,
    send_float32_frame,
)


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs/simulation/pi2sr/runtime_contract_v2.json"
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"


def test_runtime_contract_preserves_historical_pi2s_and_freezes_scope() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "FROZEN_BEFORE_QUALIFICATION"
    assert contract["historical_pi2s"] == {
        "decision": "COMPLETE_VALID_DIAGNOSIS_INCONCLUSIVE",
        "live_direct_status": "HISTORICAL_PI2S_LIVE_DIRECT_NONPARITY",
        "max_absolute_error": 2.7418136596679688e-06,
        "old_tolerance_result_modified": False,
        "formal_scores_modified": False,
    }
    assert contract["canonical_candidate"]["runtime"] == "DIRECT_IN_PROCESS"
    assert contract["canonical_candidate"]["batch_shape"] == [1, 26, 30]
    assert contract["compute_only"]["strata_counts"] == {
        "free_space": 5,
        "contact": 6,
        "dynamic_boundary": 5,
    }


def test_float32_framing_is_exact_for_h_transport() -> None:
    client, server = socket.socketpair()
    value = np.linspace(-4.0, 4.0, 256, dtype=np.float32)

    def echo() -> None:
        with server:
            received = receive_float32_frame(server, (256,))
            send_float32_frame(server, received)

    worker = threading.Thread(target=echo)
    worker.start()
    with client:
        send_float32_frame(client, value)
        returned = receive_float32_frame(client, (256,))
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert returned.dtype == np.float32
    assert returned.shape == (256,)
    assert returned.tobytes() == value.tobytes()


def test_numeric_metrics_separate_value_and_byte_equality() -> None:
    left = np.zeros((2, 3), dtype=np.float32)
    right = left.copy()
    right[0, 0] = np.float32(-0.0)
    metrics = numeric_metrics(left, right)
    assert metrics["array_equal"] is True
    assert metrics["byte_equal"] is False
    assert metrics["max_absolute_error"] == 0.0
    assert metrics["cosine_similarity"] == 1.0


def test_repeatability_reports_exact_and_nonexact_runs() -> None:
    exact = np.zeros((3, 2, 4), dtype=np.float32)
    assert repeatability_metrics(exact)["all_byte_equal"] is True
    changed = exact.copy()
    changed[2, 1, 3] = 1.0
    metrics = repeatability_metrics(changed)
    assert metrics["all_byte_equal"] is False
    assert metrics["max_absolute_error"] == 1.0


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="accepted E_T checkpoint is not mounted")
def test_direct_canonical_encoder_is_bitwise_repeatable() -> None:
    encoder = DirectContactEncoder(CHECKPOINT)
    history = np.zeros((26, 30), dtype=np.float32)
    first = encoder.encode(history)
    second = encoder.encode(history)
    assert first.shape == (256,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert first.tobytes() == second.tobytes()
    assert encoder.settings["torch_num_threads"] == 1
    assert encoder.settings["mkldnn"] is False
