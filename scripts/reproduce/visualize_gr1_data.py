#!/usr/bin/env python3
"""Create standalone interactive HTML viewers for local GR1 LeRobot episodes.

The S0 GR1 data is a LeRobot v2.0 layout.  The LeRobot viewer pinned by UniT
expects a ``frame_index`` field that this layout does not provide, and the
joints-only data does not contain enough environment state for faithful
simulator replay.  This tool is therefore deliberately a data viewer, not a
simulation replay or a physical-3D trajectory renderer.

Each generated HTML file embeds selected episode videos and the matching raw
44-D state/action arrays.  It needs no web server or network connection once
written.  The displayed goal image is the frame at ``goal_offset`` steps in the
future, matching the active UniT GR1 loader's appended action-horizon video
frame (16 by default).

Examples:

    python scripts/reproduce/visualize_gr1_data.py \
        --dataset-root "$GR1_DATASET_DIR" \
        --task gr1_unified.PnPWineToCabinetClose \
        --episodes 0 500 999

    python scripts/reproduce/visualize_gr1_data.py \
        --dataset-root "$GR1_DATASET_DIR" \
        --task gr1_unified.PnPWineToCabinetClose \
        --task gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA \
        --task gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / ".local/artifacts/visualization"
DEFAULT_EPISODES = (0, 500, 999)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_tasks_jsonl(path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        tasks[int(value["task_index"])] = str(value["task"])
    return tasks


def dimension_labels(modality: dict[str, Any], group: str, expected_dim: int) -> list[str]:
    """Create stable labels from LeRobot modality ranges."""
    groups = modality.get(group, {})
    entries = sorted(groups.items(), key=lambda item: int(item[1]["start"]))
    labels = [f"{group}[{index}]" for index in range(expected_dim)]
    for name, value in entries:
        start = int(value["start"])
        end = int(value["end"])
        for index in range(start, min(end, expected_dim)):
            labels[index] = f"{name}[{index - start}]"
    return labels


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def as_matrix(column: Any, name: str) -> np.ndarray:
    value = np.asarray(column.to_pylist(), dtype=np.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{name} must be a non-empty 2-D array; got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def make_episode_payload(
    dataset_path: Path,
    info: dict[str, Any],
    task_names: dict[int, str],
    episode_index: int,
    goal_offset: int,
) -> dict[str, Any]:
    chunk_size = int(info["chunks_size"])
    episode_chunk = episode_index // chunk_size
    parquet_path = dataset_path / str(info["data_path"]).format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")

    columns = ["observation.state", "action", "timestamp", "task_index"]
    table = pq.read_table(parquet_path, columns=columns)
    state = as_matrix(table["observation.state"], "observation.state")
    action = as_matrix(table["action"], "action")
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    task_index = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    if timestamp.ndim != 1 or len(timestamp) != len(state) or len(action) != len(state):
        raise ValueError(f"Inconsistent arrays in {parquet_path}")
    if not np.isfinite(timestamp).all():
        raise ValueError(f"timestamp contains NaN or Inf in {parquet_path}")

    video_keys = [
        name for name, feature in info["features"].items() if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if len(video_keys) != 1:
        raise ValueError(f"Expected exactly one video stream, found {video_keys}")
    video_path = dataset_path / str(info["video_path"]).format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_keys[0],
    )
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing episode video: {video_path}")

    task_id = int(task_index[0])
    return {
        "episodeIndex": episode_index,
        "frameCount": int(len(state)),
        "timestamps": timestamp.tolist(),
        "state": state.tolist(),
        "action": action.tolist(),
        "task": task_names.get(task_id, f"task_index={task_id}"),
        "taskIndex": task_id,
        "video": data_uri(video_path),
        "videoSha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "videoBytes": int(video_path.stat().st_size),
        "sourceVideoKey": video_keys[0],
        "goalOffset": goal_offset,
    }


def html_document(payload: dict[str, Any]) -> str:
    """Return an offline HTML document with no external JavaScript dependency."""
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    document = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GR1 data viewer</title>
<style>
:root { color-scheme: dark; --bg:#10151f; --panel:#182132; --line:#37465f; --ink:#eef4ff; --muted:#9bacc8; --accent:#6ec8ff; --goal:#ffcc70; }
* { box-sizing:border-box; }
body { margin:0; padding:22px; font:14px/1.45 system-ui,sans-serif; background:var(--bg); color:var(--ink); }
h1,h2,p { margin:0; } h1 { font-size:1.35rem; } h2 { font-size:1rem; margin:12px 0 6px; }
.shell { max-width:1500px; margin:auto; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; margin-top:14px; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
button,select,input { font:inherit; color:inherit; background:#202d43; border:1px solid var(--line); border-radius:6px; padding:6px 8px; }
button:hover { border-color:var(--accent); cursor:pointer; }
input[type=range] { flex:1 1 420px; padding:0; accent-color:var(--accent); }
.meta,.hint { color:var(--muted); } .warning { color:var(--goal); }
.video-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }
.video-card { background:#0c111a; border-radius:8px; padding:10px; }
video { width:100%; display:block; background:#000; aspect-ratio:1/1; object-fit:contain; }
.chart-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(450px,1fr)); gap:14px; }
.chart { width:100%; height:275px; display:block; background:#0c111a; border:1px solid var(--line); border-radius:8px; cursor:crosshair; }
.chart-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:4px 0 7px; }
.chart-head select { max-width:290px; }
code { background:#0c111a; padding:1px 4px; border-radius:4px; } .status { min-height:22px; color:var(--accent); }
</style>
</head>
<body>
<main class="shell">
  <h1>GR1 standalone data viewer</h1>
  <p class="hint">Offline artifact. It embeds RGB MP4, raw LeRobot joint state/action arrays, and a goal frame <code>+16</code> steps ahead, matching the active UniT GR1 transform's goal-frame offset.</p>
  <section class="panel controls">
    <label>Episode <select id="episode-select"></select></label>
    <button id="previous">◀ previous frame</button><button id="play">▶ play</button><button id="next">next frame ▶</button>
    <input id="frame" type="range" min="0" value="0" step="1" aria-label="Frame">
    <strong id="frame-label"></strong>
  </section>
  <section class="panel">
    <p id="episode-meta" class="meta"></p>
    <p class="warning">This is RGB + 44-D joint state/action temporal data. It is not physical XYZ geometry and does not claim simulator replay.</p>
    <div class="video-grid">
      <div class="video-card"><h2>Observation RGB</h2><video id="observation" controls muted playsinline preload="metadata"></video></div>
      <div class="video-card"><h2>UniT goal image: +16 steps</h2><video id="goal" muted playsinline preload="metadata"></video></div>
    </div>
  </section>
  <section class="panel">
    <div class="chart-grid">
      <div><div class="chart-head"><h2>Raw state trajectory</h2><label>dimension <select id="state-dimension"></select></label></div><canvas id="state-chart" class="chart"></canvas></div>
      <div><div class="chart-head"><h2>Raw action trajectory</h2><label>dimension <select id="action-dimension"></select></label></div><canvas id="action-chart" class="chart"></canvas></div>
    </div>
    <p id="hover-status" class="status">Hover a chart for frame/time/value; click a chart to seek.</p>
  </section>
</main>
<script>
const DATA = __PAYLOAD__;
const episodeSelect = document.getElementById('episode-select');
const frameInput = document.getElementById('frame');
const frameLabel = document.getElementById('frame-label');
const observation = document.getElementById('observation');
const goal = document.getElementById('goal');
const stateSelect = document.getElementById('state-dimension');
const actionSelect = document.getElementById('action-dimension');
const stateCanvas = document.getElementById('state-chart');
const actionCanvas = document.getElementById('action-chart');
const hoverStatus = document.getElementById('hover-status');
let episode = null, currentFrame = 0, hoverFrame = null, playing = false;

function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function option(text, value) { const node = document.createElement('option'); node.textContent = text; node.value = value; return node; }
function populateDimensionSelect(select, labels) { select.replaceChildren(); labels.forEach((label, index) => select.append(option(`${index}: ${label}`, index))); }

function loadEpisode(index) {
  episode = DATA.episodes[index]; currentFrame = 0; hoverFrame = null;
  frameInput.max = episode.frameCount - 1; frameInput.value = 0;
  document.getElementById('episode-meta').textContent = `Episode ${episode.episodeIndex} · ${episode.frameCount} frames · ${episode.task} · embedded video ${Math.round(episode.videoBytes / 1024)} KiB`;
  observation.src = episode.video; goal.src = episode.video;
  observation.load(); goal.load();
  updateFrame(0, false); drawAll();
}

function seek(video, seconds) {
  if (Number.isFinite(seconds)) { try { video.currentTime = seconds; } catch (_) {} }
}
function updateFrame(index, updateVideo = true) {
  if (!episode) return;
  currentFrame = clamp(Math.round(index), 0, episode.frameCount - 1);
  frameInput.value = currentFrame;
  const time = episode.timestamps[currentFrame];
  const goalFrame = clamp(currentFrame + DATA.goalOffset, 0, episode.frameCount - 1);
  frameLabel.textContent = `frame ${currentFrame}/${episode.frameCount - 1} · t=${time.toFixed(3)} s · goal frame ${goalFrame}`;
  if (updateVideo) { seek(observation, time); seek(goal, episode.timestamps[goalFrame]); }
  drawAll();
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  return { width, height, ratio };
}
function series(kind, dimension) { return episode[kind].map(row => row[dimension]); }
function drawChart(canvas, kind, dimension) {
  if (!episode) return;
  const ctx = canvas.getContext('2d'); const {width, height, ratio} = resizeCanvas(canvas);
  ctx.save();
  ctx.clearRect(0, 0, width, height); ctx.scale(ratio, ratio);
  const w = width / ratio, h = height / ratio, left = 48, right = 12, top = 14, bottom = 28;
  const values = series(kind, dimension); let low = Math.min(...values), high = Math.max(...values);
  if (low === high) { low -= 1; high += 1; } const margin = (high - low) * 0.08; low -= margin; high += margin;
  const x = frame => left + frame / Math.max(1, values.length - 1) * (w - left - right);
  const y = value => top + (high - value) / (high - low) * (h - top - bottom);
  ctx.strokeStyle = '#37465f'; ctx.lineWidth = 1; ctx.beginPath();
  for (let i = 0; i < 5; i++) { const yy = top + i / 4 * (h - top - bottom); ctx.moveTo(left, yy); ctx.lineTo(w-right, yy); } ctx.stroke();
  ctx.fillStyle = '#9bacc8'; ctx.font = '11px system-ui'; ctx.fillText(high.toFixed(3), 2, top + 4); ctx.fillText(low.toFixed(3), 2, h-bottom + 4); ctx.fillText('0', left, h-8); ctx.fillText(String(values.length-1), w-right-34, h-8);
  ctx.strokeStyle = kind === 'state' ? '#6ec8ff' : '#ffcc70'; ctx.lineWidth = 1.6; ctx.beginPath();
  values.forEach((value, frame) => { if (frame === 0) ctx.moveTo(x(frame), y(value)); else ctx.lineTo(x(frame), y(value)); }); ctx.stroke();
  [currentFrame, hoverFrame].filter((frame, position, all) => frame !== null && all.indexOf(frame) === position).forEach((frame, position) => {
    ctx.strokeStyle = position === 0 ? '#ffffff' : '#f29bff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(frame), top); ctx.lineTo(x(frame), h-bottom); ctx.stroke();
  });
  ctx.restore();
}
function drawAll() { if (episode) { drawChart(stateCanvas, 'state', Number(stateSelect.value)); drawChart(actionCanvas, 'action', Number(actionSelect.value)); } }
function updateHover(canvas, event) {
  if (!episode) return;
  const rect = canvas.getBoundingClientRect(); const normalized = clamp((event.clientX - rect.left - 48) / Math.max(1, rect.width - 60), 0, 1);
  hoverFrame = Math.round(normalized * (episode.frameCount - 1));
  const stateDimension = Number(stateSelect.value), actionDimension = Number(actionSelect.value);
  hoverStatus.textContent = `hover frame ${hoverFrame} · t=${episode.timestamps[hoverFrame].toFixed(3)} s · state[${stateDimension}]=${episode.state[hoverFrame][stateDimension].toFixed(5)} · action[${actionDimension}]=${episode.action[hoverFrame][actionDimension].toFixed(5)}`;
  drawAll();
}
function installChartInteraction(canvas) {
  canvas.addEventListener('mousemove', event => updateHover(canvas, event));
  canvas.addEventListener('mouseleave', () => { hoverFrame = null; hoverStatus.textContent = 'Hover a chart for frame/time/value; click a chart to seek.'; drawAll(); });
  canvas.addEventListener('click', event => { updateHover(canvas, event); updateFrame(hoverFrame); });
}
function setPlaying(next) { playing = next; document.getElementById('play').textContent = playing ? '⏸ pause' : '▶ play'; if (playing) { observation.play(); } else { observation.pause(); } }

DATA.episodes.forEach((item, index) => episodeSelect.append(option(`episode ${item.episodeIndex}`, index)));
populateDimensionSelect(stateSelect, DATA.stateLabels); populateDimensionSelect(actionSelect, DATA.actionLabels);
episodeSelect.addEventListener('change', () => loadEpisode(Number(episodeSelect.value)));
frameInput.addEventListener('input', () => updateFrame(Number(frameInput.value)));
document.getElementById('previous').addEventListener('click', () => updateFrame(currentFrame - 1));
document.getElementById('next').addEventListener('click', () => updateFrame(currentFrame + 1));
document.getElementById('play').addEventListener('click', () => setPlaying(!playing));
observation.addEventListener('timeupdate', () => { if (playing && episode) updateFrame(Math.round(observation.currentTime * DATA.fps), false); });
observation.addEventListener('ended', () => setPlaying(false));
stateSelect.addEventListener('change', drawAll); actionSelect.addEventListener('change', drawAll);
window.addEventListener('resize', drawAll); installChartInteraction(stateCanvas); installChartInteraction(actionCanvas);
loadEpisode(0);
</script>
</body>
</html>
"""
    return document.replace("__PAYLOAD__", payload_json)


def safe_output_stem(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task)


def write_task_viewer(dataset_root: Path, task: str, episodes: list[int], output_dir: Path, goal_offset: int) -> Path:
    dataset_path = dataset_root / task
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset task directory does not exist: {dataset_path}")
    info = read_json(dataset_path / "meta/info.json")
    modality = read_json(dataset_path / "meta/modality.json")
    task_names = parse_tasks_jsonl(dataset_path / "meta/tasks.jsonl")
    total_episodes = int(info["total_episodes"])
    selected = sorted(set(episodes))
    if not selected:
        raise ValueError("At least one episode is required")
    invalid = [episode for episode in selected if episode < 0 or episode >= total_episodes]
    if invalid:
        raise ValueError(f"Invalid episodes for {task} (valid range: 0..{total_episodes - 1}): {invalid}")

    payload_episodes = [make_episode_payload(dataset_path, info, task_names, episode, goal_offset) for episode in selected]
    state_size = len(payload_episodes[0]["state"][0])
    action_size = len(payload_episodes[0]["action"][0])
    if any(len(item["state"][0]) != state_size or len(item["action"][0]) != action_size for item in payload_episodes):
        raise ValueError(f"Inconsistent state/action dimensions across selected episodes in {task}")
    payload = {
        "schema": "tactile3d-unit.gr1-html-v1",
        "task": task,
        "fps": float(info["fps"]),
        "goalOffset": goal_offset,
        "stateLabels": dimension_labels(modality, "state", state_size),
        "actionLabels": dimension_labels(modality, "action", action_size),
        "episodes": payload_episodes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{safe_output_stem(task)}_episodes_{'-'.join(str(item) for item in selected)}.html"
    output.write_text(html_document(payload), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default=os.environ.get("GR1_DATASET_DIR"), help="LeRobot root containing gr1_unified.* datasets")
    parser.add_argument("--task", action="append", required=True, help="Task directory to visualize; repeat for multiple tasks")
    parser.add_argument("--episodes", type=int, nargs="+", default=list(DEFAULT_EPISODES), help="Episode indices embedded in every generated viewer")
    parser.add_argument("--goal-offset", type=int, default=16, help="Future frame offset used for the displayed UniT goal image")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Ignored directory for standalone HTML outputs")
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path; defaults under output-dir")
    args = parser.parse_args()
    if not args.dataset_root:
        parser.error("--dataset-root or GR1_DATASET_DIR is required")
    if args.goal_offset < 1:
        parser.error("--goal-offset must be positive")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset root is not accessible: {dataset_root}")
    output_dir = args.output_dir.resolve()
    results: list[dict[str, Any]] = []
    for task in args.task:
        output = write_task_viewer(dataset_root, task, args.episodes, output_dir, args.goal_offset)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        results.append({"task": task, "html": output.name, "bytes": output.stat().st_size, "sha256": digest})
        print(f"{task:<66} PASS  {output}")
    manifest = args.manifest.resolve() if args.manifest else output_dir / "gr1_visualization_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "schema": "tactile3d-unit.gr1-html-v1",
                "episodes": sorted(set(args.episodes)),
                "goal_offset": args.goal_offset,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
