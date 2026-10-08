#!/usr/bin/env python3
"""Serve PI2S-R transport echo or canonical CPU Contact-State compute."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2sr_runtime import (  # noqa: E402
    DirectContactEncoder,
    receive_float32_frame,
    send_float32_frame,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("echo", "compute"), required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.socket.exists():
        raise SystemExit(f"refusing to replace socket: {args.socket}")
    if args.mode == "compute" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required in compute mode")

    encoder = DirectContactEncoder(args.checkpoint) if args.mode == "compute" else None
    request_shape = (26, 30) if encoder is not None else (256,)
    stop = False
    requests = 0
    errors: list[str] = []

    def request_stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(4)
    server.settimeout(0.25)
    started = time.monotonic()
    print(f"PI2SR_SERVICE_READY mode={args.mode} socket={args.socket}", flush=True)
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
                        request = receive_float32_frame(connection, request_shape)
                        response = request if encoder is None else encoder.encode(request)
                        send_float32_frame(connection, np.ascontiguousarray(response))
                        requests += 1
                    except (ConnectionError, BrokenPipeError, TimeoutError):
                        break
                    except Exception as error:  # pragma: no cover - persisted for integration audit
                        errors.append(f"{type(error).__name__}: {error}")
                        break
    finally:
        server.close()
        if args.socket.exists():
            args.socket.unlink()
        payload = {
            "schema": "tactile3d-unit.pi2sr-contact-state-service.v1",
            "mode": args.mode,
            "status": "PASS" if requests > 0 and not errors else "FAIL",
            "requests": requests,
            "errors": errors,
            "elapsed_seconds": time.monotonic() - started,
            "socket_removed": not args.socket.exists(),
            "runtime_settings": encoder.settings if encoder is not None else None,
        }
        atomic_json(args.report, payload)
        print(json.dumps({"status": payload["status"], "requests": requests}), flush=True)


if __name__ == "__main__":
    main()
