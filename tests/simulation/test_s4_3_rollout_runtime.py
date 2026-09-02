from pathlib import Path

import numpy as np

from scripts.simulation.run_s4_3_policy_rollouts_dex import (
    POLICY_RGB_JPEG_QUALITY,
    existing_rollout,
    jpeg,
)


def test_rollout_rgb_ipc_encoding_is_valid_jpeg() -> None:
    import cv2

    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    payload = jpeg(rgb)
    expected_ok, expected = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    assert expected_ok
    assert POLICY_RGB_JPEG_QUALITY == 80
    assert payload == expected.tobytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == rgb.shape
    assert decoded.dtype == np.uint8


def test_existing_rollout_requires_identity_and_trace_hash(tmp_path: Path) -> None:
    import hashlib
    import json

    trace = tmp_path / "trace.npz"
    np.savez_compressed(trace, action=np.zeros((2, 22), dtype=np.float32))
    identity = {
        "rollout_id": "rollout",
        "task": "pinch_tongs",
        "variant": "P0",
        "training_seed": 0,
        "evaluation_reset_id": "reset",
        "checkpoint_sha256": "a" * 64,
    }
    metadata = {
        **identity,
        "trace": str(trace),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "logging_status": "PASS",
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert existing_rollout(path, identity) == metadata
    assert existing_rollout(path, {**identity, "training_seed": 1}) is None
