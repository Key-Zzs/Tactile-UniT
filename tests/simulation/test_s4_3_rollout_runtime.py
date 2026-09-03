import json
import os
import pickle
import shutil
import tempfile
import threading
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np
import pytest

from gr00t.simulation.s4_3_transport import (
    MAX_ENDPOINT_BYTES,
    EndpointOwnershipError,
    build_runtime_endpoint,
    cleanup_server_endpoint,
    load_endpoint_manifest,
    platform_af_unix_payload_limit,
    prepare_server_endpoint,
    register_server_endpoint,
    write_endpoint_manifest,
)

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


def short_test_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="tu3d-t-"))


def endpoint(root: Path, index: int = 0):
    return build_runtime_endpoint(
        experiment_identity="S4.3-RR-ACT-rollout-v2",
        task="click_mouse",
        variant="P0",
        training_seed=0,
        worker_identity=f"worker-{index}",
        root=root,
        launcher_pid=4242 + index,
        nonce=f"{index:06x}",
    )


def test_historical_endpoint_regression_and_platform_limit() -> None:
    root = Path(__file__).resolve().parents[2]
    old = (
        root
        / ".local/tmp/simulation/s4_3_restart/rollout_sockets/click_mouse_P0_seed0.sock"
    )
    assert len(os.fsencode(old)) == 118
    assert platform_af_unix_payload_limit() == 107
    runtime = short_test_root()
    try:
        current = endpoint(runtime)
        assert current.encoded_length <= MAX_ENDPOINT_BYTES
        assert not any(value in current.socket_path.name for value in ("click_mouse", "P0"))
    finally:
        shutil.rmtree(runtime)


def test_all_36_job_endpoint_identities_are_short_unique_and_manifest_centralized() -> None:
    runtime = short_test_root()
    try:
        rows = []
        index = 0
        for task in ("pinch_tongs", "hammer_nail", "click_mouse"):
            for variant in ("P0", "P1", "P2", "P3"):
                for training_seed in range(3):
                    item = build_runtime_endpoint(
                        experiment_identity="S4.3-RR-ACT-rollout-v2",
                        task=task,
                        variant=variant,
                        training_seed=training_seed,
                        worker_identity=f"simultaneous-worker-{index}",
                        root=runtime,
                        launcher_pid=5000 + index,
                        nonce=f"{index:06x}",
                    )
                    manifest = runtime / f"manifest-{index}.json"
                    write_endpoint_manifest(
                        manifest,
                        item,
                        {"task": task, "variant": variant, "training_seed": training_seed},
                    )
                    loaded, raw = load_endpoint_manifest(manifest)
                    assert loaded == item
                    assert raw["scientific_seed_influence"] is False
                    rows.append(item)
                    index += 1
        assert len(rows) == 36
        assert len({row.path for row in rows}) == 36
        assert len({row.job_hash for row in rows}) == 36
        assert max(row.encoded_length for row in rows) <= MAX_ENDPOINT_BYTES
    finally:
        shutil.rmtree(runtime)


def test_four_worker_bind_connect_has_no_cross_talk_and_cleans_up() -> None:
    runtime = short_test_root()
    endpoints = [endpoint(runtime, index) for index in range(4)]
    ready = [threading.Event() for _ in endpoints]
    errors: list[BaseException] = []

    def serve(index: int) -> None:
        item = endpoints[index]
        listener = None
        try:
            prepare_server_endpoint(item)
            listener = Listener(item.path, family="AF_UNIX", authkey=b"transport-test")
            register_server_endpoint(item)
            ready[index].set()
            connection = listener.accept()
            try:
                request = connection.recv()
                connection.send({"worker": index, "echo": request})
            finally:
                connection.close()
        except BaseException as error:  # pragma: no cover - asserted in parent
            errors.append(error)
            ready[index].set()
        finally:
            if listener is not None:
                listener.close()
            try:
                cleanup_server_endpoint(item, allow_unregistered_own_socket=True)
            except BaseException as error:  # pragma: no cover - asserted in parent
                errors.append(error)

    threads = [threading.Thread(target=serve, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for event in ready:
        assert event.wait(5)
    for index, item in enumerate(endpoints):
        connection = Client(item.path, family="AF_UNIX", authkey=b"transport-test")
        connection.send({"target": index})
        assert connection.recv() == {"worker": index, "echo": {"target": index}}
        connection.close()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    try:
        assert not errors
        assert all(not item.socket_path.exists() for item in endpoints)
        assert all(not item.lease_path.exists() for item in endpoints)
    finally:
        shutil.rmtree(runtime)


def test_stale_active_arbitrary_and_failed_startup_cleanup_contracts() -> None:
    runtime = short_test_root()
    item = endpoint(runtime)
    try:
        listener = Listener(item.path, family="AF_UNIX")
        register_server_endpoint(item, owner_pid=99_999_999)
        listener.close()
        assert prepare_server_endpoint(item) is True
        assert not item.socket_path.exists()
        assert not item.lease_path.exists()

        listener = Listener(item.path, family="AF_UNIX")
        register_server_endpoint(item, owner_pid=os.getpid())
        listener.close()
        with pytest.raises(EndpointOwnershipError, match="active"):
            prepare_server_endpoint(item)
        assert cleanup_server_endpoint(item, owner_pid=os.getpid())

        item.socket_path.write_text("not a socket", encoding="utf-8")
        with pytest.raises(EndpointOwnershipError, match="unregistered"):
            prepare_server_endpoint(item)
        item.socket_path.unlink()

        with pytest.raises((FileNotFoundError, ConnectionRefusedError, OSError)):
            Client(item.path, family="AF_UNIX")
    finally:
        shutil.rmtree(runtime)


def test_transport_change_does_not_change_rpc_payload_bytes() -> None:
    request = {
        "command": "infer",
        "rgb_jpeg": b"fixed-jpeg",
        "proprio": np.zeros(22, dtype=np.float32),
        "tactile_history": np.zeros((26, 30), dtype=np.float32),
    }
    response = {
        "action_chunk": np.zeros((27, 22), dtype=np.float32),
        "p3_predicted_contact": None,
        "vision_shape": [8, 32],
        "uncertainty": {
            "status": "UNAVAILABLE_CAUSAL_INPUT_MISMATCH",
            "invoked": False,
            "intervention": False,
        },
    }
    before = pickle.dumps((request, response), protocol=pickle.DEFAULT_PROTOCOL)
    after = pickle.dumps((request, response), protocol=pickle.DEFAULT_PROTOCOL)
    assert before == after
    assert b"tu3d_" not in after
