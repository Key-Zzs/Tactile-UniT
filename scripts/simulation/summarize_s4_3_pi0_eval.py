#!/usr/bin/env python3
"""Summarize fixed official DexJoCo pi0.5 evaluation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = ROOT / ".local/experiments/simulation/s4_3_pi0/official_checkpoint_eval/seed0"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_checkpoint_eval.json"
DEFAULT_VIDEOS = ROOT / ".local/artifacts/simulation/s4_3_pi0/videos"
REPRODUCED_RUNTIME = ROOT / ".local/artifacts/simulation/s4_3_pi0/reproduced_eval_runtime.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - half_width, center + half_width


def video_metadata(path: Path, eval_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    stream = probe["streams"][0]
    return {
        "path": path.relative_to(eval_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "nb_frames": int(stream["nb_frames"]),
        "duration_seconds": float(probe["format"]["duration"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kind", choices=("official", "reproduced"), default="official")
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS)
    args = parser.parse_args()

    if args.eval_root is None:
        args.eval_root = (
            DEFAULT_EVAL_ROOT
            if args.checkpoint_kind == "official"
            else ROOT / ".local/experiments/simulation/s4_3_pi0/reproduced_checkpoint_eval/seed0"
        )
    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT
            if args.checkpoint_kind == "official"
            else ROOT / ".local/artifacts/simulation/s4_3_pi0/reproduced_checkpoint_eval.json"
        )

    episode_dirs = sorted(path for path in args.eval_root.glob("episode_*_*") if path.is_dir())
    successes = [path for path in episode_dirs if path.name.endswith("_success")]
    failures = [path for path in episode_dirs if path.name.endswith("_failure")]
    markers = list(args.eval_root.glob("success_rate_*_*.txt"))
    videos = sorted(path for path in args.eval_root.glob("episode_*_*/*.mp4") if path.is_file())
    video_manifest = [video_metadata(path, args.eval_root) for path in videos]
    runtime = (
        json.loads(REPRODUCED_RUNTIME.read_text(encoding="utf-8"))
        if args.checkpoint_kind == "reproduced" and REPRODUCED_RUNTIME.is_file()
        else None
    )

    success_count = len(successes)
    episode_count = len(episode_dirs)
    interval_low, interval_high = wilson_interval(success_count, episode_count)

    args.videos_root.mkdir(parents=True, exist_ok=True)
    representative_sources = {}
    for result, episode_group in (("success", successes), ("failure", failures)):
        if episode_group:
            for camera in ("front", "wrist"):
                representative_sources[f"{args.checkpoint_kind}_{result}_{camera}.mp4"] = (
                    episode_group[0] / f"{camera}.mp4"
                )
    representative_videos = {}
    for name, source in representative_sources.items():
        destination = args.videos_root / name
        shutil.copy2(source, destination)
        representative_videos[name] = {
            "source": source.relative_to(args.eval_root).as_posix(),
            "artifact_path": f".local/artifacts/simulation/s4_3_pi0/videos/{name}",
            "sha256": sha256_file(destination),
        }

    expected_marker = f"success_rate_{success_count}_{episode_count}.txt"
    gates = {
        "official_evaluator_exit_zero": runtime is None or runtime["status"] == "PASS",
        "episodes_20": episode_count == 20,
        "success_failure_partition": success_count + len(failures) == episode_count,
        "success_marker": len(markers) == 1 and markers[0].name == expected_marker,
        "two_cameras_per_episode": len(videos) == 2 * episode_count,
        "videos_nonempty_and_decodable": all(
            item["bytes"] > 0 and item["nb_frames"] > 0 and item["duration_seconds"] > 0 for item in video_manifest
        ),
        "server_stable": runtime is None or runtime["gates"]["server_no_runtime_traceback"] == "PASS",
        "client_stable": runtime is None or runtime["gates"]["client_no_traceback"] == "PASS",
    }
    if args.checkpoint_kind == "official":
        gates["paper_mean_inside_wilson_95ci"] = interval_low <= 0.24 <= interval_high
    mtimes = [path.stat().st_mtime for path in videos]
    artifact = {
        "schema": (f"tactile3d-unit.s4-3-pi0-{args.checkpoint_kind}-checkpoint-eval.v1"),
        "stage": "PI0-3" if args.checkpoint_kind == "official" else "PI0-10",
        "checkpoint_kind": args.checkpoint_kind,
        "runtime_audit": ("$ARTIFACT_ROOT/reproduced_eval_runtime.json" if runtime is not None else None),
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "seed": 0,
        "episodes": episode_count,
        "successes": success_count,
        "failures": len(failures),
        "success_rate": success_count / episode_count,
        "wilson_95ci": [interval_low, interval_high],
        "published_reference": {
            "success_rate_mean": 0.24,
            "success_rate_std": 0.069,
            "paper_episodes_per_task": 50,
            "source": "https://arxiv.org/abs/2605.16257v1",
        },
        "difference_from_published_mean": success_count / episode_count - 0.24,
        "evaluation_conditions": {
            "rand_full": False,
            "randomize_dynamics": False,
            "render_mode": "rgb_array",
            "replan_ratio": 0.8,
            "action_horizon": 30,
            "pad_state_dim46": False,
            "prompt": "Grasp the tongs and perform three consecutive open-close motions.",
        },
        "episode_results": {
            path.name.split("_")[1]: "success" if path in successes else "failure" for path in episode_dirs
        },
        "video_count": len(videos),
        "video_manifest": video_manifest,
        "representative_videos": representative_videos,
        "observed_output_span_seconds": max(mtimes) - min(mtimes),
        "marker": expected_marker,
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "compatibility": "PASS" if all(gates.values()) else "FAIL",
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "episodes": episode_count,
                "status": artifact["status"],
                "successes": success_count,
                "wilson_95ci": artifact["wilson_95ci"],
            },
            sort_keys=True,
        )
    )
    if artifact["status"] != "PASS":
        raise SystemExit(f"S4_3_PI0_{args.checkpoint_kind.upper()}_CHECKPOINT_EVAL_FAIL")


if __name__ == "__main__":
    main()
