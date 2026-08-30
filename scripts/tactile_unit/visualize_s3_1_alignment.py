#!/usr/bin/env python3
"""Render representative S3.1 RGB/action/contact synchronization examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    decode_rgb_frame,
    sha256_file,
    sha256_json,
)


DEFAULT_MANIFEST = ROOT / ".local/artifacts/tactile_unit/s3_1/paired_eval_manifest.json"
DEFAULT_OUTPUT = ROOT / ".local/artifacts/tactile_unit/s3_1/representative_vac_sync.png"
DEFAULT_SUMMARY = ROOT / ".local/artifacts/tactile_unit/s3_1/representative_vac_sync.json"
PRIMITIVES = ("reach", "grasp_and_lifting", "lift_and_place", "shake", "wrap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def choose_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = []
    for primitive in PRIMITIVES:
        candidates = [row for row in rows if row["metadata"]["motor_primitive"] == primitive]
        if not candidates:
            raise ValueError(f"canonical 960 manifest has no {primitive!r} example")
        candidates.sort(key=lambda row: (not row["contact"]["dynamic"], row["pair_id"]))
        chosen.append(candidates[0])
    return chosen


def load_episode_signals(dataset_root: Path, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        dataset_root / row["state"]["relative_path"],
        columns=["episode_index", "frame_index", "observation.tactile_force", "action"],
        filters=[("episode_index", "=", row["episode_id"])],
    ).sort_by([("frame_index", "ascending")])
    frames = np.asarray(table["frame_index"])
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError(f"episode {row['episode_id']} frames are not contiguous")
    wrench = np.asarray(table["observation.tactile_force"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    fingertip_force = np.linalg.norm(wrench.reshape(-1, 10, 6)[..., :3], axis=-1).max(axis=1)
    action_change = np.zeros(len(actions), dtype=np.float32)
    action_change[1:] = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    return frames, fingertip_force, action_change


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    if len(manifest["rows"]) != 960:
        raise ValueError("visualization must derive from the canonical 960-pair manifest")
    selected = choose_rows(manifest["rows"])
    figure, axes = plt.subplots(
        len(selected), 3, figsize=(16, 3.15 * len(selected)), constrained_layout=True
    )
    summary_rows = []
    for row_index, row in enumerate(selected):
        video_path = args.dataset_root / row["vision"]["relative_path"]
        current = decode_rgb_frame(video_path, row["vision"]["current"]["packed_timestamp"])
        future = decode_rgb_frame(video_path, row["vision"]["future"]["packed_timestamp"])
        axes[row_index, 0].imshow(current)
        axes[row_index, 1].imshow(future)
        axes[row_index, 0].set_title(f"I_t · frame {row['anchor']['frame']}")
        axes[row_index, 1].set_title(f"I_t+16 · frame {row['anchor']['frame'] + 16}")
        for column in (0, 1):
            axes[row_index, column].axis("off")

        frames, force, action_change = load_episode_signals(args.dataset_root, row)
        anchor = int(row["anchor"]["frame"])
        start = max(0, anchor - 30)
        stop = min(len(frames), anchor + 32)
        relative = frames[start:stop] - anchor
        signal_axis = axes[row_index, 2]
        action_axis = signal_axis.twinx()
        force_line = signal_axis.plot(relative, force[start:stop], color="#1864ab", label="max fingertip force")
        action_line = action_axis.plot(
            relative,
            action_change[start:stop],
            color="#e67700",
            alpha=0.8,
            label="||Δ action||₂",
        )
        signal_axis.axvline(0, color="#c92a2a", linestyle="--", linewidth=1.5, label="t")
        signal_axis.axvline(16, color="#7048e8", linestyle="--", linewidth=1.5, label="t+16")
        signal_axis.axvspan(0, 15, color="#ffd43b", alpha=0.12, label="a_t:t+15")
        signal_axis.set_xlabel("episode frame relative to t")
        signal_axis.set_ylabel("max fingertip force", color="#1864ab")
        action_axis.set_ylabel("action-change magnitude", color="#e67700")
        handles = force_line + action_line + signal_axis.get_lines()[-2:]
        labels = [handle.get_label() for handle in handles]
        signal_axis.legend(handles, labels, loc="upper right", fontsize=7)
        title = (
            f"{row['metadata']['motor_primitive']} · {row['metadata']['object']}\n"
            f"contact={row['contact']['transition_class']} · dynamic={row['contact']['dynamic']}"
        )
        signal_axis.set_title(title, fontsize=10)
        signal_axis.grid(alpha=0.2)
        summary_rows.append(
            {
                "pair_id": row["pair_id"],
                "episode_id": row["episode_id"],
                "anchor_frame": anchor,
                "future_frame": anchor + 16,
                "action_frames_inclusive": [anchor, anchor + 15],
                "primitive": row["metadata"]["motor_primitive"],
                "object": row["metadata"]["object"],
                "contact_transition": row["contact"]["transition_class"],
                "dynamic": row["contact"]["dynamic"],
                "max_fingertip_force_current_window": float(force[anchor - 15 : anchor + 1].max()),
                "max_fingertip_force_future_window": float(force[anchor + 1 : anchor + 17].max()),
                "canonical_action_change_l2": float(
                    np.linalg.norm(
                        np.asarray(action_change[anchor : anchor + 16], dtype=np.float64)
                    )
                ),
            }
        )
    figure.suptitle(
        "S3.1 canonical V+A+C synchronization · red=t · purple=t+16 · yellow=a_t:t+15",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    summary = {
        "schema": "tactile3d-unit.s3-1-human-sync.v1",
        "source_manifest_sha256": sha256_file(args.manifest),
        "selection": "prefer dynamic, then lexical pair_id, for fixed primitive list",
        "representative_primitives": list(PRIMITIVES),
        "rows": summary_rows,
        "image_file": args.output.name,
    }
    summary["summary_sha256"] = sha256_json(summary)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
