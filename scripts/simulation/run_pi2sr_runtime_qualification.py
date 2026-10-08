#!/usr/bin/env python3
"""Run bounded CPU transport/compute qualification for PI2S-R."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2sr_runtime import (  # noqa: E402
    CANONICAL_CHECKPOINT_SHA256,
    DirectContactEncoder,
    FramedFloat32UnixClient,
    latency_percentiles,
    measure_requests,
    numeric_metrics,
    repeatability_metrics,
    sha256_file,
)

CONTRACT = ROOT / "configs/simulation/pi2sr/runtime_contract_v2.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi2sr"
SERVICE = ROOT / "scripts/simulation/serve_pi2sr_contact_state.py"


def experiment_root() -> Path:
    value = os.environ.get("UNIT_EXPERIMENT_ROOT")
    return Path(value) if value else ROOT / ".local/experiments"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextmanager
def running_service(
    mode: str, checkpoint: Path | None = None
) -> Iterator[tuple[Path, subprocess.Popen[str], Path]]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    with tempfile.TemporaryDirectory(prefix=f"pi2sr-{mode}-") as directory:
        temporary = Path(directory)
        socket_path = temporary / "service.sock"
        report = temporary / "service.json"
        command = [
            sys.executable,
            str(SERVICE),
            "--mode",
            mode,
            "--socket",
            str(socket_path),
            "--report",
            str(report),
        ]
        if checkpoint is not None:
            command.extend(("--checkpoint", str(checkpoint)))
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not socket_path.exists() and process.poll() is None:
            time.sleep(0.01)
        if not socket_path.exists():
            stdout, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"{mode} service did not become ready: return={process.returncode} "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
        try:
            yield socket_path, process, report
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=10)
            if process.returncode != 0:
                raise RuntimeError(
                    f"{mode} service failed: return={process.returncode} "
                    f"stdout={stdout!r} stderr={stderr!r}"
                )
            if not report.is_file():
                raise RuntimeError(f"{mode} service did not write its report")
            service_report = json.loads(report.read_text(encoding="utf-8"))
            if service_report["status"] != "PASS" or service_report["errors"]:
                raise RuntimeError(f"{mode} service report failed: {service_report}")
            running_service.last_report = service_report
            running_service.last_stdout = stdout.splitlines()
            running_service.last_stderr = stderr.splitlines()


running_service.last_report = {}
running_service.last_stdout = []
running_service.last_stderr = []


def qualify_transport(contract: dict[str, Any]) -> dict[str, Any]:
    spec = contract["transport_only"]
    generator = np.random.default_rng(4327001)
    inputs = generator.standard_normal((spec["requests"], 256)).astype(np.float32)
    with running_service("echo") as (socket_path, _process, _report):
        with FramedFloat32UnixClient(socket_path, (256,), (256,)) as client:
            outputs, latencies = measure_requests(client.request, inputs)
    received = np.stack(outputs)
    metrics = numeric_metrics(inputs, received)
    status = "TRANSPORT_EXACT" if metrics["byte_equal"] else "TRANSPORT_MISMATCH"
    return {
        "schema": "tactile3d-unit.pi2sr-transport-parity.v1",
        "status": "PASS" if status == "TRANSPORT_EXACT" else "FAIL",
        "classification": status,
        "method": "already-computed H echo over protocol-compatible AF_UNIX framing; E_T not loaded",
        "requests": len(inputs),
        "metrics": metrics,
        "latency": latency_percentiles(latencies),
        "framing": spec["framing"],
        "byte_order": sys.byteorder,
        "service": running_service.last_report,
        "training_or_policy_execution": False,
    }


def qualify_compute(
    contract: dict[str, Any], checkpoint: Path, sidecar: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = contract["compute_only"]
    if sha256_file(sidecar) != spec["source_sha256"]:
        raise RuntimeError("frozen PI2S sidecar SHA mismatch")
    with np.load(sidecar, allow_pickle=False) as loaded:
        histories = loaded["tactile_history"][spec["sample_rows"]].astype(np.float32, copy=True)
        episode_indices = loaded["episode_index"][spec["sample_rows"]].astype(int).tolist()
        frame_indices = loaded["frame_index"][spec["sample_rows"]].astype(int).tolist()
    direct = DirectContactEncoder(checkpoint)
    warmup = [histories[index % len(histories)] for index in range(8)]
    measure_requests(direct.encode, warmup)

    direct_repeats = []
    for _ in range(spec["direct_repeats"]):
        direct_repeats.append(np.stack([direct.encode(value) for value in histories]))
    direct_array = np.stack(direct_repeats)

    with running_service("compute", checkpoint) as (socket_path, _process, _report):
        with FramedFloat32UnixClient(socket_path, (26, 30), (256,)) as client:
            for value in warmup:
                client.request(value)
            service_repeats = []
            for _ in range(spec["service_repeats"]):
                service_repeats.append(np.stack([client.request(value) for value in histories]))
            service_array = np.stack(service_repeats)
            _, service_latency_ns = measure_requests(
                client.request,
                [histories[index % len(histories)] for index in range(64)],
            )
    _, direct_latency_ns = measure_requests(
        direct.encode, [histories[index % len(histories)] for index in range(64)]
    )

    direct_repeatability = repeatability_metrics(direct_array)
    service_repeatability = repeatability_metrics(service_array)
    cross = numeric_metrics(direct_array[0], service_array[0])
    maximum_output = float(
        max(np.abs(direct_array[0]).max(initial=0.0), np.abs(service_array[0]).max(initial=0.0))
    )
    epsilon = float(np.finfo(np.float32).eps)
    absolute_tolerance = max(
        1e-7,
        8.0 * direct_repeatability["max_absolute_error"],
        8.0 * service_repeatability["max_absolute_error"],
        8.0 * epsilon * maximum_output,
    )
    relative_tolerance = 8.0 * epsilon
    cross_within = bool(
        np.allclose(
            direct_array[0],
            service_array[0],
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    if not direct_repeatability["all_byte_equal"]:
        classification = "COMPUTE_NOT_QUALIFIED"
    elif cross["byte_equal"]:
        classification = "COMPUTE_REPEATABLE_CANONICAL"
    elif cross_within:
        classification = "COMPUTE_CROSS_PATH_NUMERICALLY_STABLE_WITH_SCOPE"
    else:
        classification = "COMPUTE_CROSS_PATH_MISMATCH"

    difference = np.abs(direct_array[0].astype(np.float64) - service_array[0].astype(np.float64))
    compute = {
        "schema": "tactile3d-unit.pi2sr-compute-parity.v1",
        "status": (
            "PASS"
            if classification
            in {"COMPUTE_REPEATABLE_CANONICAL", "COMPUTE_CROSS_PATH_NUMERICALLY_STABLE_WITH_SCOPE"}
            else "FAIL"
        ),
        "classification": classification,
        "conditions": direct.settings,
        "checkpoint_sha256": sha256_file(checkpoint),
        "normalization": "N1 embedded in accepted checkpoint",
        "sample_rows": spec["sample_rows"],
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "strata_counts": spec["strata_counts"],
        "direct_repeatability": direct_repeatability,
        "service_repeatability": service_repeatability,
        "cross_path": cross,
        "cross_path_within_prospective_v2_tolerance": cross_within,
        "prospective_v2_tolerance": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
            "calibration": spec["prospective_v2_tolerance_calibration"],
            "historical_pi2s_result_modified": False,
        },
        "per_dimension": {
            "max_absolute_error": difference.max(axis=0).tolist(),
            "mean_absolute_error": difference.mean(axis=0).tolist(),
        },
        "latency": {
            "direct_in_process": latency_percentiles(direct_latency_ns),
            "unix_service": latency_percentiles(service_latency_ns),
        },
        "service": running_service.last_report,
        "training_or_optimizer_updates": 0,
        "closed_loop_causality_claim": False,
    }
    decision = {
        "schema": "tactile3d-unit.pi2sr-canonical-runtime-decision.v1",
        "status": (
            "PI2SR_CANONICAL_RUNTIME_READY_WITH_SCOPE"
            if compute["status"] == "PASS"
            else "PI2SR_BLOCKED_RUNTIME_MISMATCH"
        ),
        "selected_runtime": "DIRECT_IN_PROCESS",
        "selection_uses_policy_success": False,
        "checkpoint": "$UNIT_EXPERIMENT_ROOT/simulation/s4_2r/contact_state/accepted.pt",
        "checkpoint_sha256": CANONICAL_CHECKPOINT_SHA256,
        "normalization": "N1",
        "runtime_settings": direct.settings,
        "input_schema": "causal float32 [26,30], oldest-to-current, LEFT_REPEAT_FIRST per episode",
        "output_schema": "finite float32 [256] Contact-State H",
        "latency": compute["latency"]["direct_in_process"],
        "supported_scope": [
            "CPU float32 batch-one direct in-process inference",
            "accepted E_T checkpoint with embedded N1 normalization",
            "26-step causal LEFT_REPEAT_FIRST history reset per episode",
        ],
        "unsupported_scope": [
            "GPU and accelerator kernels",
            "float16/bfloat16/autocast",
            "batch sizes other than one",
            "MKLDNN-enabled CPU execution",
            "cross-host or network transport",
        ],
        "compute_classification": classification,
        "historical_pi2s_result_modified": False,
        "policy_effect_claim": False,
    }
    return compute, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["status"] != "FROZEN_BEFORE_QUALIFICATION":
        raise RuntimeError("PI2S-R runtime protocol is not frozen")
    root = experiment_root()
    checkpoint = root / "simulation/s4_2r/contact_state/accepted.pt"
    sidecar = root / "simulation/s4_3_pi2s/source_snapshots/data/pi1_contact_sidecar.npz"
    if sha256_file(checkpoint) != CANONICAL_CHECKPOINT_SHA256:
        raise RuntimeError("canonical checkpoint integrity failed")

    freeze = {
        "schema": "tactile3d-unit.pi2sr-runtime-protocol-freeze.v1",
        "status": "PASS",
        "contract": "configs/simulation/pi2sr/runtime_contract_v2.json",
        "contract_sha256": sha256_file(CONTRACT),
        "freeze_commit": git("rev-parse", "d26e670"),
        "measurement_head": git("rev-parse", "HEAD"),
        "historical_pi2s_result_modified": False,
    }
    atomic_json(args.artifact_root / "runtime_protocol_freeze.json", freeze)
    transport = qualify_transport(contract)
    atomic_json(args.artifact_root / "transport_parity.json", transport)
    compute, decision = qualify_compute(contract, checkpoint, sidecar)
    atomic_json(args.artifact_root / "compute_parity.json", compute)
    atomic_json(args.artifact_root / "canonical_runtime_decision.json", decision)

    environment = {
        "schema": "tactile3d-unit.pi2sr-environment-integrity.v1",
        "status": "PASS" if transport["status"] == compute["status"] == "PASS" else "FAIL",
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("numpy", "torch", "jax", "jaxlib", "scipy", "pytest")
        },
        "device": "cpu",
        "cuda_visible_devices": "",
        "deterministic_environment": contract["canonical_candidate"]["environment"],
        "source": {
            "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"),
            "dexjoco_commit": git("-C", "third_party/dexjoco", "rev-parse", "HEAD"),
            "openpi_tree": git("-C", "third_party/dexjoco", "rev-parse", "HEAD:openpi"),
            "runtime_module_sha256": sha256_file(ROOT / "gr00t/simulation/pi2sr_runtime.py"),
            "service_sha256": sha256_file(SERVICE),
        },
        "training": False,
        "real_hardware": False,
    }
    atomic_json(args.artifact_root / "environment_integrity.json", environment)
    print(
        json.dumps(
            {
                "transport": transport["classification"],
                "compute": compute["classification"],
                "runtime": decision["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
