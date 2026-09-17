#!/usr/bin/env python3
"""Freeze S4.3-PI2W starting integrity and protected evidence hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2w"
UNIT = ROOT / ".local/external/s4_3_pi2u/unit_official"
UNIT_CKPT = ROOT / ".local/external/s4_3_pi2u/unit_fulldata/VLA-UniT-3B-fulldata/tokenizer"
STARTING_HEAD = "5d02dfc7df7aa545d66db55e9c955dc9c5694d4e"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path, *, exclude_git: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    total = 0
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = candidate.relative_to(path)
        if exclude_git and (".git" in relative_path.parts or "__pycache__" in relative_path.parts or candidate.suffix == ".pyc"):
            continue
        relative = relative_path.as_posix()
        size = candidate.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256(candidate).encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\n")
        files += 1
        total += size
    return {"tree_sha256": digest.hexdigest(), "files": files, "bytes": total}


def config_set(pattern: str) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(ROOT.glob(pattern))
        if path.is_file()
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if (ARTIFACTS / "protected_hashes_before.json").exists():
        raise SystemExit("refusing to overwrite PI2W protected hashes")
    branch = output("git", "branch", "--show-current")
    head = output("git", "rev-parse", "HEAD")
    if branch != "develop/sim-benchmark" or head != STARTING_HEAD:
        raise RuntimeError("PI2W starting Git source-of-truth changed")
    starting = {
        "schema": "tactile3d-unit.s4-3-pi2w-starting-integrity.v1",
        "status": "PASS",
        "pwd": "$REPO_ROOT",
        "branch": branch,
        "starting_head": head,
        "initial_worktree_clean": True,
        "initial_status_short": [],
        "note": "Clean status was captured before the new PI2W entrypoints were added.",
        "remote": output("git", "remote", "-v"),
        "worktrees": output("git", "worktree", "list", "--porcelain"),
        "submodules": output("git", "submodule", "status", "--recursive"),
        "training_authorized": False,
    }
    atomic_json(ARTIFACTS / "starting_integrity.json", starting)

    checkpoints = {
        "B0": ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
        "B1": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
        "B2": ROOT / ".local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
        "BVA": ROOT / ".local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42/29999",
    }
    expected = {
        "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
        "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
        "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
        "BVA": "04b609d7cc89e5fffdfab8219bf34da362117a15d0c7e3d5cd4ee20a9ee4770d",
    }
    checkpoint_hashes = {name: tree_hash(path) for name, path in checkpoints.items()}
    for name in checkpoint_hashes:
        checkpoint_hashes[name]["expected_tree_sha256"] = expected[name]
        checkpoint_hashes[name]["matches_expected"] = checkpoint_hashes[name]["tree_sha256"] == expected[name]
        checkpoint_hashes[name]["path"] = f"$REPO_ROOT/{checkpoints[name].relative_to(ROOT).as_posix()}"

    unit_tree = output("git", "ls-tree", "-r", "--full-tree", "HEAD", cwd=UNIT)
    unit_tree_sha = hashlib.sha256((unit_tree + "\n").encode()).hexdigest()
    official_checkpoint_files = {
        name: sha256(UNIT_CKPT / name)
        for name in (
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
        )
    }
    protected = {
        "schema": "tactile3d-unit.s4-3-pi2w-protected-hashes.v1",
        "status": "PASS",
        "phase": "BEFORE_READ_ONLY_DIAGNOSIS",
        "hash_algorithm": "sha256(sorted(relative_path NUL file_sha256 NUL bytes newline))",
        "s4_2": {
            "configs": config_set("configs/simulation/s4_2*.json"),
            "checkpoint_trees": {
                name: tree_hash(ROOT / ".local/experiments/simulation" / name)
                for name in ("s4_2", "s4_2_formal", "s4_2dr", "s4_2ds", "s4_2r")
            },
        },
        "policy_checkpoints": checkpoint_hashes,
        "pi2a": {
            "configs": config_set("configs/simulation/s4_3_pi2a*.json"),
            "artifacts": tree_hash(ROOT / ".local/artifacts/simulation/s4_3_pi2a"),
        },
        "pi2u": {
            "configs": config_set("configs/simulation/s4_3_pi2u*.json"),
            "artifacts": tree_hash(ROOT / ".local/artifacts/simulation/s4_3_pi2u"),
        },
        "pi2v": {
            "protocol_sha256": sha256(ROOT / "configs/simulation/s4_3_pi2v_unit_adapter_protocol.json"),
            "tracked_final_decision_sha256": sha256(ROOT / "configs/simulation/s4_3_pi2v_final_decision.json"),
            "validator_sha256": sha256(ROOT / "scripts/simulation/validate_s4_3_pi2v_unit_adapter.py"),
            "artifact_tree": tree_hash(ROOT / ".local/artifacts/simulation/s4_3_pi2v"),
            "adapter_checkpoint_sha256": sha256(ROOT / ".local/experiments/simulation/s4_3_pi2v/unit_adapter/checkpoint-step80000.pt"),
        },
        "official_unit": {
            "repository": "https://github.com/xpeng-robotics/UniT.git",
            "commit": output("git", "rev-parse", "HEAD", cwd=UNIT),
            "status_short": output("git", "status", "--short", cwd=UNIT).splitlines(),
            "git_tree_listing_sha256": unit_tree_sha,
            "checkpoint_files_sha256": official_checkpoint_files,
        },
    }
    if not all(row["matches_expected"] for row in checkpoint_hashes.values()):
        protected["status"] = "FAIL"
    atomic_json(ARTIFACTS / "protected_hashes_before.json", protected)
    if protected["status"] != "PASS":
        raise RuntimeError("protected checkpoint hash mismatch")

    closure = {
        "schema": "tactile3d-unit.s4-3-pi2w-pi2v-historical-closure.v1",
        "status": "CLOSED_IMMUTABLY",
        "historical_status": "S4_3_PI2V_UNIT_ADAPTER_REPRESENTATION_FAIL",
        "adapter_training": "COMPLETE",
        "representation_gate": "FAIL",
        "failure_class": "REPRESENTATION_TRAINING_FAILURE",
        "adapter_checkpoint": "step80000",
        "adapter_checkpoint_sha256": protected["pi2v"]["adapter_checkpoint_sha256"],
        "BUniT_target_generation": "NOT_RUN",
        "BUniT_policy_training": "NOT_RUN",
        "BUniT_policy_evaluation": "NOT_RUN",
        "PI2B": "NOT_RUN",
        "retuning_authorized": False,
        "checkpoint_shopping_authorized": False,
        "historical_artifacts_modified": False,
        "frozen_statement": "PI2V failed its preregistered future-vision reconstruction gate.",
    }
    atomic_json(ARTIFACTS / "pi2v_historical_closure.json", closure)

    contract = {
        "schema": "tactile3d-unit.s4-3-pi2w-official-unit-visual-contract.v1",
        "status": "PASS",
        "official_commit": protected["official_unit"]["commit"],
        "source_clean": not protected["official_unit"]["status_short"],
        "current_condition": "f_current = DINOv2-large hidden_states[-2][:,1:,:] from ImageNet-normalized 224x224 current RGB",
        "future_target": "f_future = DINOv2-large hidden_states[-2][:,1:,:] from identically preprocessed future RGB",
        "dino_layer_index": -2,
        "cls_excluded": True,
        "patch_shape": [256, 1024],
        "vision_query_features_shape": ["B", 8, 1024],
        "action_query_features_shape": ["B", 8, 1024],
        "unit_tokens_pre_vq_shape": ["B", 8, 1024],
        "vq_input_shape": ["B", 8, 32],
        "unit_tokens_post_vq_shape": ["B", 8, 32],
        "bridge_projected_shape": ["B", 8, 1024],
        "reconstructed_future_feature_shape": ["B", 256, 1024],
        "dataflow": "(f_current,f_future)->VisionBranch M-Former queries; action/state->ActionBranch queries; pv/pa Fusion->down-project->2-stage RVQ->bridge-project; VisionDecoder(f_current,projected quantized tokens)->reconstructed f_future",
        "loss_equation": "L = (1/B) sum_i [1 - (1/256) sum_j cosine(reconstructed_future[i,j,:], f_future[i,j,:])]",
        "cosine_dimension": -1,
        "patch_reduction": "mean over all 256 patches",
        "batch_reduction": "mean over samples",
        "explicit_l2_normalization_before_loss": False,
        "cosine_similarity_internal_normalization": True,
        "target_projection": None,
        "current_future_preprocessing_identical": True,
        "source_locations": {
            "dino_wrapper": "$UNIT_SOURCE/gr00t/model/gr00t_n1_tokenizer_unit.py:33-81",
            "vision_branch": "$UNIT_SOURCE/gr00t/model/tokenizer/vision_branch_encoder.py:92-130",
            "fusion_vq_bridge": "$UNIT_SOURCE/gr00t/model/gr00t_n1_tokenizer_unit.py:578-605",
            "vision_loss": "$UNIT_SOURCE/gr00t/model/gr00t_n1_tokenizer_unit.py:613-639",
        },
    }
    atomic_json(ARTIFACTS / "official_unit_visual_contract.json", contract)

    no_motion = {
        "schema": "tactile3d-unit.s4-3-pi2w-no-motion-gate-audit.v1",
        "status": "NO_MOTION_GATE_IMPLEMENTATION_VALID",
        "static_prediction": "obs_embeddings = f_current from the same DINOv2-large layer -2 patch tensor",
        "future_target": "goal_embeddings = f_future from the same DINOv2-large layer -2 patch tensor",
        "same_representation_space": True,
        "same_dino_layer": True,
        "same_patch_ordering": True,
        "same_normalization": True,
        "same_temporal_frame_pair": True,
        "same_cosine_and_reduction": True,
        "asymmetric_projection": False,
        "equation": "L_static = (1/B) sum_i [1 - (1/256) sum_j cosine(f_current[i,j,:], f_future[i,j,:])]",
        "validator_location": "$REPO_ROOT/scripts/simulation/validate_s4_3_pi2v_unit_adapter.py:324-329",
        "official_unit_acceptance_criterion": False,
        "interpretation_boundary": "Valid implementation of the preregistered custom persistence diagnostic; not an official UniT acceptance metric and not by itself proof that Original UniT fundamentally fails.",
    }
    atomic_json(ARTIFACTS / "no_motion_gate_audit.json", no_motion)
    print(json.dumps({"status": "PASS", "protected": protected["status"], "closure": closure["status"]}))


if __name__ == "__main__":
    main()
