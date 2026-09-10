#!/usr/bin/env python3
"""Serve the exact frozen S4.2 Contact-State encoder over a local Unix socket."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import struct
import time

import numpy as np
import torch

from gr00t.simulation.pi1d_runtime import recv_exact
from gr00t.simulation.s4_3_act import TorchTactileNormalization
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi1/contact_state_service.json"
HEADER = struct.Struct("!I")
EXPECTED_REQUEST_BYTES = 26 * 30 * np.dtype(np.float32).itemsize
EXPECTED_RESPONSE_BYTES = 256 * np.dtype(np.float32).itemsize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    if args.socket.exists():
        raise SystemExit(f"Refusing to replace existing socket path: {args.socket}")
    args.socket.parent.mkdir(parents=True, exist_ok=True)

    teacher, checkpoint = load_teacher_checkpoint(CHECKPOINT, "cpu")
    normalization = TorchTactileNormalization(checkpoint["normalization"])
    teacher.eval().requires_grad_(False)
    normalization.eval().requires_grad_(False)
    requests = 0
    errors: list[str] = []
    finite_inputs = True
    finite_outputs = True
    stop = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(4)
    server.settimeout(0.5)
    print(f"CONTACT_STATE_SERVICE_READY socket={args.socket}", flush=True)
    started = time.time()
    try:
        while not stop:
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(30.0)
                while not stop:
                    try:
                        header = connection.recv(HEADER.size)
                        if not header:
                            break
                        if len(header) < HEADER.size:
                            header += recv_exact(connection, HEADER.size - len(header))
                        size = HEADER.unpack(header)[0]
                        if size != EXPECTED_REQUEST_BYTES:
                            raise ValueError(f"request has {size} bytes")
                        raw = recv_exact(connection, size)
                        history = np.frombuffer(raw, dtype=np.float32).copy().reshape(1, 26, 30)
                        finite_inputs = finite_inputs and bool(np.isfinite(history).all())
                        if not finite_inputs:
                            raise ValueError("non-finite tactile history")
                        with torch.inference_mode():
                            output = teacher(normalization(torch.from_numpy(history)))["latent"]
                        contact_state = output.float().cpu().numpy()[0]
                        finite_outputs = finite_outputs and bool(np.isfinite(contact_state).all())
                        if contact_state.shape != (256,) or not finite_outputs:
                            raise RuntimeError("invalid Contact-State output")
                        payload = np.ascontiguousarray(contact_state, dtype=np.float32).tobytes()
                        if len(payload) != EXPECTED_RESPONSE_BYTES:
                            raise RuntimeError("Contact-State response byte contract changed")
                        connection.sendall(HEADER.pack(len(payload)) + payload)
                        requests += 1
                    except (ConnectionError, BrokenPipeError, TimeoutError):
                        break
                    except Exception as error:
                        errors.append(f"{type(error).__name__}: {error}")
                        break
    finally:
        server.close()
        if args.socket.exists():
            args.socket.unlink()
        gates = {
            "exact_checkpoint": CHECKPOINT.is_file(),
            "checkpoint_schema": checkpoint.get("schema") == "tactile3d-unit.s4-2-sim-contact-teacher.v1",
            "normalization_N1": checkpoint["normalization"]["candidate"] == "N1",
            "requests_positive": requests > 0,
            "finite_inputs": finite_inputs,
            "finite_outputs": finite_outputs,
            "no_errors": not errors,
            "socket_removed": not args.socket.exists(),
        }
        payload = {
            "schema": "tactile3d-unit.s4-3-pi1d-contact-state-service.v1",
            "status": "PASS" if all(gates.values()) else "FAIL",
            "environment": "unit",
            "transport": "local Unix socket; float32 [26,30] request to float32 [256] response",
            "checkpoint": "$REPO_ROOT/.local/experiments/simulation/s4_2r/contact_state/accepted.pt",
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "history_bootstrap": "LEFT_REPEAT_FIRST is maintained by the client",
            "requests": requests,
            "errors": errors,
            "elapsed_seconds": time.time() - started,
            "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        }
        atomic_json(payload)
        print(json.dumps({"status": payload["status"], "requests": requests, "errors": len(errors)}), flush=True)


if __name__ == "__main__":
    main()
