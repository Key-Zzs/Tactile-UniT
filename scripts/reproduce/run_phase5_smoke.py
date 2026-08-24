#!/usr/bin/env python3
"""Run the official UniT/RoboCasa closed-loop smoke with local artifacts only.

This wrapper keeps the official inference and simulation services unchanged. It
only supplies process isolation, local log/video paths, retries, and machine
checks required by the reproduction protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "examples" / "run_eval.sh"
SIM_SCRIPT = ROOT / "scripts" / "simulation" / "simulation_service_unit.py"
SERVER_SCRIPT = ROOT / "scripts" / "inference_service_unit.py"
RATE_SCRIPT = ROOT / "scripts" / "compute_success_rate.py"


def parse_id_tasks() -> list[str]:
    text = EVAL_SCRIPT.read_text()
    section = re.search(r"(?ms)^  id\)\n(?P<body>.*?)^    \) ;;", text)
    if section is None:
        raise RuntimeError(f"Could not locate id task list in {EVAL_SCRIPT}")
    tasks = re.findall(r"^\s+(gr1_unified/\S+_Env)\s*$", section.group("body"), re.MULTILINE)
    if len(tasks) != 24:
        raise RuntimeError(f"Expected 24 official id tasks, found {len(tasks)}")
    return tasks


def build_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env.pop("DISPLAY", None)
    return env


def check_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already occupied") from exc


def run_preflight(checkpoint: Path, data_config: str, port: int, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checks = [
        [sys.executable, "-c", "import mujoco, robosuite, robocasa; print('simulation imports: PASS')"],
        [sys.executable, "-c", "import OpenGL; print('EGL python import: PASS')"],
    ]
    with log_path.open("w") as log:
        log.write(f"checkpoint={checkpoint}\ndata_config={data_config}\nport={port}\n")
        for command in checks:
            log.write(f"$ {shlex.join(command)}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(f"Preflight command failed: {shlex.join(command)}")
        log.write("preflight: PASS\n")


def wait_for_server(server: subprocess.Popen[bytes], log_path: Path, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            tail = log_path.read_text(errors="replace")[-8000:]
            raise RuntimeError(f"Inference server exited with code {server.returncode}\n{tail}")
        if "Server is ready and listening" in log_path.read_text(errors="replace"):
            return
        time.sleep(1)
    tail = log_path.read_text(errors="replace")[-8000:]
    raise RuntimeError(f"Inference server did not become ready within {timeout}s\n{tail}")


def start_server(checkpoint: Path, data_config: str, port: int, env: dict[str, str], log_path: Path, timeout: int) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-u",
        str(SERVER_SCRIPT.relative_to(ROOT)),
        "--server",
        "--model_path",
        str(checkpoint),
        "--port",
        str(port),
        "--data_config",
        data_config,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    log.write(f"$ {shlex.join(command)}\n")
    log.flush()
    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    wait_for_server(server, log_path, timeout)
    return server


def stop_server(server: subprocess.Popen[bytes] | None) -> None:
    if server is None or server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=10)


def run_task(
    task: str,
    episodes: int,
    output_dir: Path,
    port: int,
    env: dict[str, str],
    client_log: Path,
    max_retries: int,
) -> dict:
    video_dir = output_dir / "videos" / task
    video_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(SIM_SCRIPT.relative_to(ROOT)),
        "--client",
        "--env_name",
        task,
        "--video_dir",
        str(video_dir),
        "--max_episode_steps",
        "720",
        "--n_envs",
        "1",
        "--n_episodes",
        str(episodes),
        "--port",
        str(port),
    ]
    attempts = []
    for attempt in range(1, max_retries + 1):
        with client_log.open("a") as log:
            log.write(f"\nExecuting command for: {task} attempt={attempt}\n")
            log.write(f"$ {shlex.join(command)}\n")
            log.flush()
            started = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - started
            log.write(f"exit_code={result.returncode} elapsed_seconds={elapsed:.3f}\n")
        attempts.append({"attempt": attempt, "returncode": result.returncode, "elapsed_seconds": elapsed})
        if result.returncode == 0:
            return {"task": task, "status": "completed", "attempts": attempts}
        if attempt < max_retries:
            time.sleep(5)
    return {"task": task, "status": "failed", "attempts": attempts}


def video_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "videos").rglob("*.mp4")) if (output_dir / "videos").exists() else []


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,width,height,duration,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"ffprobe_error": result.stderr.strip()}
    try:
        stream = json.loads(result.stdout).get("streams", [{}])[0]
    except (json.JSONDecodeError, IndexError):
        return {"ffprobe_error": "invalid ffprobe JSON"}
    frames = int(stream.get("nb_read_frames") or 0)
    duration = float(stream.get("duration") or 0.0)
    return {
        "frame_count": frames,
        "duration_seconds": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frame_rate": stream.get("r_frame_rate"),
    }


def make_video_manifest(output_dir: Path) -> dict:
    files = video_files(output_dir)
    records = []
    for path in files:
        relative = path.relative_to(output_dir / "videos")
        task = relative.parent.as_posix()
        match = re.search(r"success([01])_", path.name)
        success = None if match is None else bool(int(match.group(1)))
        record = {
            "task": task,
            "episode": len([item for item in records if item["task"] == task]) + 1,
            "success": success,
            "video_path": str(path),
        }
        record.update(ffprobe(path))
        records.append(record)

    selected = []
    selected_tasks = set()

    def is_pnpclose(item: dict) -> bool:
        task_name = item["task"].split("/", 1)[-1]
        return task_name.startswith("PnP") and "Close_" in task_name

    def add_one(predicate) -> None:
        for item in records:
            if item["task"] in selected_tasks:
                continue
            if predicate(item):
                selected.append(item)
                selected_tasks.add(item["task"])
                return

    # Prefer a balanced, human-useful sample when both outcomes are present:
    # three successes and three policy failures, covering PnPClose and
    # PosttrainPnPNovel task families.  The simulator's success flag is only
    # used for artifact selection; it is not a Phase 5 numeric gate.
    if sum(item["success"] is True for item in records) >= 3 and sum(item["success"] is False for item in records) >= 3:
        for want_success in (True, False):
            add_one(lambda item, want_success=want_success: item["success"] is want_success and is_pnpclose(item))
            add_one(lambda item, want_success=want_success: item["success"] is want_success and "PosttrainPnPNovel" in item["task"])
            add_one(lambda item, want_success=want_success: item["success"] is want_success)
    for item in records:
        if len(selected) >= 6:
            break
        if item["task"] not in selected_tasks:
            selected.append(item)
            selected_tasks.add(item["task"])
    selected = selected[:6]
    manifest = {
        "video_count": len(records),
        "selected_count": len(selected),
        "all_videos": records,
        "selected_videos": selected,
    }
    return manifest


def run_success_rate(output_dir: Path, client_log: Path, env: dict[str, str]) -> Path | None:
    result_path = output_dir / "results.json"
    result = subprocess.run(
        [sys.executable, str(RATE_SCRIPT.relative_to(ROOT)), "-i", str(client_log), "-o", str(result_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    with client_log.open("a") as log:
        log.write("\ncompute_success_rate stdout:\n" + result.stdout)
        log.write("compute_success_rate stderr:\n" + result.stderr)
        log.write(f"compute_success_rate exit_code={result.returncode}\n")
    return result_path if result.returncode == 0 and result_path.exists() else None


def run_stage(
    name: str,
    tasks: list[str],
    episodes: int,
    output_dir: Path,
    port: int,
    env: dict[str, str],
    max_retries: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    client_log = output_dir / "client.log"
    client_log.write_text("")
    started = time.monotonic()
    results = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{name}] {index}/{len(tasks)} starting {task}", flush=True)
        task_result = run_task(task, episodes, output_dir, port, env, client_log, max_retries)
        results.append(task_result)
        print(
            f"[{name}] {index}/{len(tasks)} {task_result['status']} "
            f"attempts={len(task_result['attempts'])}",
            flush=True,
        )
    result_path = run_success_rate(output_dir, client_log, env)
    manifest = make_video_manifest(output_dir)
    summary = {
        "stage": name,
        "expected_tasks": len(tasks),
        "expected_episodes_per_task": episodes,
        "expected_rollouts": len(tasks) * episodes,
        "task_results": results,
        "completed_tasks": sum(item["status"] == "completed" for item in results),
        "failed_tasks": [item["task"] for item in results if item["status"] != "completed"],
        "video_count": manifest["video_count"],
        "elapsed_seconds": time.monotonic() - started,
        "results_json": str(result_path) if result_path else None,
    }
    (output_dir / "video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".local/artifacts/reproduction/phase5"))
    parser.add_argument("--data-config", default="fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--server-ready-timeout", type=int, default=600)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint}")
    if args.episodes != 2:
        raise SystemExit("Phase 5 official ID smoke requires --episodes 2")
    gpu_id = args.cuda_visible_devices.split(",", 1)[0]
    port = args.port if args.port is not None else 5800 + int(gpu_id)
    output_dir = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    log_dir = ROOT / ".local" / "logs" / "reproduction" / "phase5"
    env = build_env(args.cuda_visible_devices)
    tasks = parse_id_tasks()
    pre_task = tasks[0]
    full_dir = output_dir / "full"
    pre_dir = output_dir / "pre"
    server_log = log_dir / "inference_server.log"
    pid_path = ROOT / ".local" / "tmp" / "phase5_inference.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    check_port(port)
    run_preflight(checkpoint, args.data_config, port, env, log_dir / "preflight.log")
    server = None
    run_summary = {
        "checkpoint": str(checkpoint),
        "data_config": args.data_config,
        "cuda_visible_devices": args.cuda_visible_devices,
        "port": port,
        "stages": [],
        "status": "RUNNING",
    }
    try:
        server = start_server(checkpoint, args.data_config, port, env, server_log, args.server_ready_timeout)
        pid_path.write_text(str(server.pid) + "\n")
        pre_summary = run_stage("P5-A pre-smoke", [pre_task], 1, pre_dir, port, env, args.max_retries)
        run_summary["stages"].append(pre_summary)
        if pre_summary["completed_tasks"] != 1 or pre_summary["video_count"] < 1:
            raise RuntimeError("P5-A did not complete one task and produce one rollout video")
        full_summary = run_stage("P5-B official id smoke", tasks, args.episodes, full_dir, port, env, args.max_retries)
        run_summary["stages"].append(full_summary)
        manifest = make_video_manifest(full_dir)
        (output_dir / "video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if full_summary["completed_tasks"] != 24 or manifest["video_count"] < 48:
            raise RuntimeError("P5-B did not complete 24 tasks with 2 videos per task")
        run_summary["status"] = "PASS"
    finally:
        stop_server(server)
        if pid_path.exists():
            pid_path.unlink()
        run_summary["server_exit_code"] = None if server is None else server.returncode
        (output_dir / "phase5_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE5 ERROR: {exc}", file=sys.stderr)
        raise
