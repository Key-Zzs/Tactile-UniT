#!/usr/bin/env python3
"""Audit the completed local-checkpoint official evaluator/server lifecycle."""

from __future__ import annotations

import json
import re
import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi0"
EVAL_ROOT = ROOT / ".local/experiments/simulation/s4_3_pi0/reproduced_checkpoint_eval/seed0"
CLIENT_LOG = ROOT / ".local/logs/simulation/s4_3_pi0/reproduced_eval.log"
SERVER_LOG = ROOT / ".local/logs/simulation/s4_3_pi0/reproduced_server.log"
OUTPUT = ARTIFACT_ROOT / "reproduced_eval_runtime.json"


def port_is_closed(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        return sock.connect_ex(("127.0.0.1", port)) != 0
    finally:
        sock.close()


def main() -> None:
    client = CLIENT_LOG.read_text(encoding="utf-8", errors="replace")
    server = SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    server_closed_at = server.rfind("INFO:websockets.server:server closed")
    runtime_prefix = server[:server_closed_at] if server_closed_at >= 0 else server
    episode_numbers = [int(value) for value in re.findall(r"^Episode (\d+)/20$", client, re.MULTILINE)]
    outcome_lines = re.findall(r"^(Success!|Failed)$", client, re.MULTILINE)
    marker = EVAL_ROOT / "success_rate_2_20.txt"
    lock_file = (
        Path(
            subprocess.run(
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        / "tactile3d_unit_gpu1.lock"
    )
    lock_released = subprocess.run(["flock", "-n", str(lock_file), "true"], check=False).returncode == 0
    server_process_gone = subprocess.run(
        ["pgrep", "-f", "scripts/serve_policy.py.*--port 8125"],
        check=False,
        capture_output=True,
    ).returncode
    server_process_gone = server_process_gone != 0
    gates = {
        "official_client_episode_sequence_1_to_20": episode_numbers == list(range(1, 21)),
        "one_terminal_outcome_per_episode": len(outcome_lines) == 20,
        "client_reported_2_of_20": "Success rate: 2/20 (10.0%)" in client,
        "client_no_traceback": "Traceback (most recent call last)" not in client,
        "canonical_marker": marker.is_file(),
        "official_server_loaded_final_params": "Finished restoring checkpoint" in server
        and "/29999/params" in server,
        "server_listened": "server listening on 0.0.0.0:8125" in server,
        "client_connections_closed": server.count("connection open") >= 2
        and server.count("connection closed") >= 2,
        "server_no_runtime_traceback": "Traceback (most recent call last)" not in runtime_prefix,
        "server_graceful_close_before_interrupt": server_closed_at >= 0,
        "shutdown_interrupt_only": "asyncio.exceptions.CancelledError" in server[server_closed_at:]
        and server.rstrip().endswith("KeyboardInterrupt"),
        "server_process_gone": server_process_gone,
        "port_released": port_is_closed(8125),
        "gpu_lock_released": lock_released,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-reproduced-eval-runtime.v1",
        "stage": "PI0-10",
        "entrypoints": {
            "server": "third_party/dexjoco/openpi/scripts/serve_policy.py",
            "client": "third_party/dexjoco/dexjoco/dexjoco_openpi_client/cli/evaluate.py",
        },
        "checkpoint": "$REPRODUCED_TRAIN_ROOT/29999",
        "port": 8125,
        "gpu": 1,
        "client_episode_numbers": episode_numbers,
        "client_outcomes": [value.lower().rstrip("!") for value in outcome_lines],
        "server_shutdown": (
            "operator SIGINT after evaluation completion; websocket closed before the "
            "expected asyncio CancelledError/KeyboardInterrupt traceback"
        ),
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_REPRODUCED_EVAL_RUNTIME_FAIL")


if __name__ == "__main__":
    main()
