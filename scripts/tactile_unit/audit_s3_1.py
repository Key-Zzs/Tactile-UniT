#!/usr/bin/env python3
"""Assemble the final S3.1 acceptance gates from authoritative artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.paired_contract import sha256_file, sha256_json  # noqa: E402


DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/s3_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    video = read(args.artifacts / "video_completeness_summary.json")
    paired = read(args.artifacts / "paired_contract_summary.json")
    manifest_path = args.artifacts / "paired_eval_manifest.json"
    manifest = read(manifest_path)
    frozen = read(args.artifacts / "frozen_branch_smoke.json")
    normalization = read(args.artifacts / "state_action_normalization.json")
    synchronization = read(args.artifacts / "representative_vac_sync.json")
    sync_image = args.artifacts / synchronization["image_file"]

    gates = {
        "A_video_completeness": (
            video["status"] == "PASS"
            and video["inventory"]["missing_referenced"] == []
            and video["inventory"]["zero_size_referenced"] == []
            and video["container_failure_count"] == 0
            and video["decode_failure_count"] == 0
            and video["all_file_probe_count"] == video["inventory"]["unique_referenced_mp4s"]
        ),
        "B_temporal_alignment": (
            paired["status"] == "PASS"
            and paired["off_by_one_contract"]["status"] == "PASS"
            and paired["timestamp_decode_max_abs_error_sec"]
            <= paired["timestamp_decode_tolerance_sec"]
        ),
        "C_split_pair_integrity": (
            not any(paired["split_leakage"].values())
            and all(value == 1.0 for value in paired["pair_coverage_fraction"].values())
            and paired["pair_summaries"]["train"]["pairs"] == 279680
            and paired["pair_summaries"]["val"]["pairs"] == 17504
            and paired["pair_summaries"]["test"]["pairs"] == 17504
        ),
        "D_vision_contract": (
            paired["processed_vision_shape"] == [3, 224, 224]
            and frozen["vision"]["status"] == "PASS"
            and frozen["vision"]["canonical_pair_count"] == 960
            and frozen["vision"]["l2_shape"] == [8, 32]
        ),
        "E_action_state_contract": (
            paired["raw_state_shapes_960"] == {"(58,)": 960}
            and paired["raw_action_shapes_960"] == {"(16, 58)": 960}
            and paired["action_transform_failure_count"] == 0
            and normalization["fit_split"] == "frozen S1 train episodes only"
            and normalization["canonical_sha256"]
            == "a441b4c287c8bf8dc4c48088553d4488beef59709061784ca534e3cfdba3e3d9"
            and paired["embodiment"]["aliases_gr1"] is False
            and frozen["action"]["semantic_status"]
            == "ACTION_BRANCH_TRAINING_REQUIRED_IN_LATER_STAGE"
        ),
        "F_contact_contract": (
            paired["contact"]["s1_teacher_identity"] == "PASS"
            and paired["contact"]["s2_encoder_identity"] == "PASS"
            and paired["contact"]["h_current_shape"] == [256]
            and paired["contact"]["h_future_shape"] == [256]
            and paired["contact"]["z_c_shape"] == [8, 32]
            and paired["contact"]["z_c_finite"] is True
            and paired["contact"]["adaptor_applied"] is False
        ),
        "G_paired_eval": (
            paired["paired_eval"]["expected"] == 960
            and paired["paired_eval"]["resolved"] == 960
            and len(manifest["rows"]) == 960
            and len({row["pair_id"] for row in manifest["rows"]}) == 960
            and paired["paired_eval"]["manifest_file_sha256"] == sha256_file(manifest_path)
            and paired["paired_eval"]["deterministic_rebuild_matches_existing"] is True
        ),
        "H_human_sync_artifact": (
            sync_image.is_file()
            and sync_image.stat().st_size > 0
            and synchronization["representative_primitives"]
            == ["reach", "grasp_and_lifting", "lift_and_place", "shake", "wrap"]
        ),
        "I_no_training": (
            frozen["optimizer_created"] is False
            and frozen["trainable_parameter_count"] == 0
            and paired["contact"]["adaptor_applied"] is False
        ),
    }
    tests = {"run": False, "returncode": None, "status": "NOT_RUN"}
    if args.run_tests:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/tactile_unit",
            "tests/contact_dynamics/test_contract.py",
            "tests/contact_dynamics/test_cache.py",
            "tests/tactile_teacher/test_split.py",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        tests = {
            "run": True,
            "command": "python -m pytest -q tests/tactile_unit tests/contact_dynamics/test_contract.py tests/contact_dynamics/test_cache.py tests/tactile_teacher/test_split.py",
            "returncode": completed.returncode,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "last_output_lines": (completed.stdout + completed.stderr).splitlines()[-20:],
        }
        gates["J_tests"] = completed.returncode == 0

    action_warning = (
        frozen["action"]["category_expansion_required"] is True
        and frozen["action"]["released_tokenizer_max_num_embodiments"] == 30
        and frozen["action"]["embodiment_id"] == 31
    )
    all_pass = all(gates.values())
    final = "PASS WITH ACTION-BRANCH WARNING" if all_pass and action_warning else "FAIL"
    acceptance: dict[str, Any] = {
        "schema": "tactile3d-unit.s3-1-acceptance.v1",
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "tests": tests,
        "action_branch_warning": {
            "required": action_warning,
            "reason": (
                "generic new_embodiment ID 31 is outside the released tokenizer action encoder's "
                "30 category slots; later category expansion and T-Rex action-branch training are required"
            ),
        },
        "s3_1_final": final,
        "s3_2_started": False,
        "model_training_performed": False,
    }
    acceptance["acceptance_sha256"] = sha256_json(acceptance)
    (args.artifacts / "s3_1_acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n"
    )
    guide = """# S3.1 Human Acceptance Guide

Run from the repository root after sourcing `.local/config/tactile_unit.env`.

```bash
jq . .local/artifacts/tactile_unit/s3_1/video_completeness_summary.json
jq . .local/artifacts/tactile_unit/s3_1/paired_contract_summary.json
xdg-open .local/artifacts/tactile_unit/s3_1/representative_vac_sync.png
jq . .local/artifacts/tactile_unit/s3_1/representative_vac_sync.json
jq '.distribution' .local/artifacts/tactile_unit/s3_1/paired_eval_manifest.json
jq '.rows[0] | {pair_id, anchor, vision, state, action, contact}' \
  .local/artifacts/tactile_unit/s3_1/paired_eval_manifest.json
sha256sum -c .local/artifacts/tactile_unit/s3_1/paired_eval_manifest.sha256
```

Inspect whether `I_t`, `I_t+16`, the yellow `a_t:t+15` interval, and the
fingertip-force/contact transition describe the same physical event. The red
line is `t`; the purple line is `t+16`.
"""
    (args.artifacts / "HUMAN_ACCEPTANCE.md").write_text(guide)
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    if final == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
