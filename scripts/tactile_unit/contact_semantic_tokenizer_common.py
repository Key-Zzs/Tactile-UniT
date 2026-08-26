"""Runtime, identity, cache, and metric helpers for Track B S3.2-Q."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/contact_dynamics"))
sys.path.insert(0, str(ROOT / "scripts/tactile_unit"))

from audit_shared_rq_compatibility import load_s2_model  # noqa: E402
from evaluate_contact_dynamics import apply_ridge, fit_ridge  # noqa: E402
from gr00t.contact_dynamics.evaluation import transition_metrics  # noqa: E402
from gr00t.tactile_teacher.evaluation import classification_metrics  # noqa: E402
from gr00t.tactile_unit.contact_semantic_tokenizer import (  # noqa: E402
    ContactSemanticTokenizer,
    WhiteningStatistics,
    assert_episode_disjoint,
    same_episode_horizon_links,
)
from gr00t.tactile_unit.s3_2_r import (  # noqa: E402
    collapse_diagnostics,
    linear_cka,
    representation_structure,
)


DEFAULT_SPEC = ROOT / "configs/tactile_unit/s3_2_q_semantic_tokenizer.json"
DEFAULT_RUNTIME = ROOT / ".local"
DEFAULT_CACHE = DEFAULT_RUNTIME / "cache/tactile_unit/s3_2_q"
DEFAULT_EXPERIMENTS = DEFAULT_RUNTIME / "experiments/tactile_unit/s3_2_q"
DEFAULT_ARTIFACTS = DEFAULT_RUNTIME / "artifacts/tactile_unit/s3_2_q"
DEFAULT_LOGS = DEFAULT_RUNTIME / "logs/tactile_unit/s3_2_q"
SPLIT_DIR = {"train": "train", "validation": "val", "test": "test"}
ARRAY_NAMES = (
    "current",
    "future",
    "episode_id",
    "anchor_frame",
    "dynamic",
    "contact_transition",
    "force_trend_class",
    "finger_change",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch")
    return actual


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_gpu() -> tuple[torch.device, int]:
    """Verify Track B's masked single-GPU contract and return physical identity."""

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Track B requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"2", "3"}:
        raise RuntimeError("Track B permits only physical GPU2 or GPU3")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    return torch.device("cuda:0"), int(visible)


def source_paths(source_root: Path) -> dict[str, Path]:
    local = source_root.resolve() / ".local"
    return {
        "local": local,
        "s1": local / "experiments/tactile_teacher/s1_teacher/best.pt",
        "s2": local / "experiments/contact_dynamics/s2_models/proposed_best.pt",
        "transition": local / "cache/contact_dynamics/s2_transition_pairs",
        "codes": local / "cache/contact_dynamics/s2_codes",
        "r0_cache": local / "cache/tactile_unit/s3_2_r/r0",
        "r0_capacity_cache": local / "cache/tactile_unit/s3_2_r/r0_capacity",
        "r0_experiments": local / "experiments/tactile_unit/s3_2_r/r0",
        "r0_capacity_experiments": local / "experiments/tactile_unit/s3_2_r/r0_capacity",
        "r0_artifacts": local / "artifacts/tactile_unit/s3_2_r/r0/r0_result.json",
        "r0_capacity_artifacts": local
        / "artifacts/tactile_unit/s3_2_r/r0_capacity/r0_result.json",
    }


def load_arrays(transition_cache: Path, split: str) -> dict[str, np.ndarray]:
    directory = transition_cache / SPLIT_DIR[split]
    return {
        name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ARRAY_NAMES
    }


def baseline_integrity(spec: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    identity = spec["frozen_identity"]
    nominal = json.loads(paths["r0_artifacts"].read_text())
    capacity = json.loads(paths["r0_capacity_artifacts"].read_text())
    if nominal.get("r0_final") != "SAME_CAPACITY_CONTACT_RQ_FAIL":
        raise RuntimeError("accepted two-stage R0 decision changed")
    if capacity.get("r0_final") != "SAME_CAPACITY_CONTACT_RQ_FAIL":
        raise RuntimeError("accepted three-stage R0 decision changed")
    if nominal["architecture"] != {
        "codes_per_stage": 128,
        "embedding_dim": 32,
        "queries": 8,
        "stages": 2,
    }:
        raise RuntimeError("accepted Q_BASE_2 architecture changed")
    if capacity["architecture"] != {
        "codes_per_stage": 128,
        "embedding_dim": 32,
        "queries": 8,
        "stages": 3,
    }:
        raise RuntimeError("accepted Q_BASE_3 architecture changed")
    nominal_training = json.loads((paths["r0_experiments"] / "training_summary.json").read_text())
    capacity_training = json.loads(
        (paths["r0_capacity_experiments"] / "training_summary.json").read_text()
    )
    nominal_checkpoint = Path(nominal_training["selected"]["checkpoint"])
    if not nominal_checkpoint.is_file():
        nominal_checkpoint = paths["r0_experiments"] / nominal_checkpoint.name
    capacity_checkpoint = Path(capacity_training["selected"]["checkpoint"])
    if not capacity_checkpoint.is_file():
        capacity_checkpoint = paths["r0_capacity_experiments"] / capacity_checkpoint.name
    verify_file(
        nominal_checkpoint, identity["q_base_2_checkpoint_sha256"], "Q_BASE_2 checkpoint"
    )
    verify_file(
        capacity_checkpoint, identity["q_base_3_checkpoint_sha256"], "Q_BASE_3 checkpoint"
    )
    for result in (nominal, capacity):
        if result["frozen_identity"]["s1_checkpoint_sha256"] != identity[
            "s1_teacher_checkpoint_sha256"
        ]:
            raise RuntimeError("baseline S1 identity mismatch")
        if result["frozen_identity"]["s2_checkpoint_sha256"] != identity[
            "s2_checkpoint_sha256"
        ]:
            raise RuntimeError("baseline S2 identity mismatch")
    return {
        "Q_BASE_2": {
            "status": "FROZEN",
            "checkpoint": nominal_checkpoint,
            "checkpoint_sha256": identity["q_base_2_checkpoint_sha256"],
            "result": nominal,
        },
        "Q_BASE_3": {
            "status": "FROZEN",
            "checkpoint": capacity_checkpoint,
            "checkpoint_sha256": identity["q_base_3_checkpoint_sha256"],
            "result": capacity,
        },
    }


def ensure_validation_codes(
    *,
    s2: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    accepted_cache: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    shape = (len(arrays["current"]), 8, 32)
    if accepted_cache.is_file():
        cached = np.load(accepted_cache, mmap_mode="r")
        if tuple(cached.shape) != shape or not np.isfinite(cached).all():
            raise RuntimeError("accepted validation z_c cache is invalid")
        with torch.inference_mode():
            regenerated = s2.encoder(
                torch.from_numpy(np.array(arrays["current"][:64], copy=True)).to(device),
                torch.from_numpy(np.array(arrays["future"][:64], copy=True)).to(device),
            ).cpu().numpy()
        if not np.allclose(regenerated, np.asarray(cached[:64]), atol=1e-5, rtol=1e-5):
            raise RuntimeError("accepted validation z_c cache does not match frozen E_c")
        return cached
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
    with torch.inference_mode():
        for start in range(0, len(result), batch_size):
            stop = min(start + batch_size, len(result))
            current = torch.from_numpy(np.array(arrays["current"][start:stop], copy=True)).to(
                device
            )
            future = torch.from_numpy(np.array(arrays["future"][start:stop], copy=True)).to(
                device
            )
            result[start:stop] = s2.encoder(current, future).float().cpu().numpy()
    result.flush()
    del result
    return np.load(output_path, mmap_mode="r")


def load_runtime(
    *, spec_path: Path, source_root: Path, device: torch.device, batch_size: int = 2048
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    paths = source_paths(source_root)
    identity_spec = spec["frozen_identity"]
    identity = {
        "s1_teacher_checkpoint_sha256": verify_file(
            paths["s1"], identity_spec["s1_teacher_checkpoint_sha256"], "S1 checkpoint"
        ),
        "s2_checkpoint_sha256": verify_file(
            paths["s2"], identity_spec["s2_checkpoint_sha256"], "S2 checkpoint"
        ),
        "s2_transition_manifest_sha256": verify_file(
            paths["transition"] / "manifest.json",
            identity_spec["s2_transition_manifest_sha256"],
            "S2 transition manifest",
        ),
        "s2_train_codes_sha256": verify_file(
            paths["codes"] / "train.npy",
            identity_spec["s2_train_codes_sha256"],
            "S2 train z_c",
        ),
        "s2_test_codes_sha256": verify_file(
            paths["codes"] / "test.npy",
            identity_spec["s2_test_codes_sha256"],
            "S2 test z_c",
        ),
    }
    manifest = json.loads((paths["transition"] / "manifest.json").read_text())
    arrays = {split: load_arrays(paths["transition"], split) for split in SPLIT_DIR}
    expected = spec["canonical_contact"]["pairs"]
    for split, cache_name in SPLIT_DIR.items():
        if int(manifest["splits"][cache_name]["pairs"]) != int(expected[split]):
            raise RuntimeError(f"{split} pair count mismatch")
    assert_episode_disjoint(
        np.unique(arrays["train"]["episode_id"]),
        np.unique(arrays["validation"]["episode_id"]),
        np.unique(arrays["test"]["episode_id"]),
    )
    s2 = load_s2_model(paths["s2"], device).eval().requires_grad_(False)
    codes = {
        "train": np.load(paths["codes"] / "train.npy", mmap_mode="r"),
        "test": np.load(paths["codes"] / "test.npy", mmap_mode="r"),
    }
    codes["validation"] = ensure_validation_codes(
        s2=s2,
        arrays=arrays["validation"],
        accepted_cache=paths["local"]
        / "cache/tactile_unit/s3_2_r/canonical_validation_z_c.npy",
        output_path=DEFAULT_CACHE / "canonical_validation_z_c.npy",
        device=device,
        batch_size=batch_size,
    )
    for split, values in codes.items():
        if tuple(values.shape) != (int(expected[split]), 8, 32):
            raise RuntimeError(f"{split} z_c shape mismatch")
    baselines = baseline_integrity(spec, paths)
    return {
        "spec": spec,
        "paths": paths,
        "identity": identity,
        "manifest": manifest,
        "arrays": arrays,
        "codes": codes,
        "s2": s2,
        "baselines": baselines,
    }


@torch.inference_mode()
def materialize_candidate(
    model: ContactSemanticTokenizer,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    split: str,
) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = {
        "semantic": tuple(values.shape),
        "semantic_native": tuple(values.shape),
        "full_native": tuple(values.shape),
        "semantic_indices": (len(values), 8, model.semantic_stages),
    }
    if model.private_quantizer is not None:
        shapes.update(
            {
                "private": tuple(values.shape),
                "private_indices": (len(values), 8, model.private_stages),
            }
        )
    arrays: dict[str, np.memmap] = {}
    for name, shape in shapes.items():
        dtype = np.int64 if name.endswith("indices") else np.float32
        arrays[name] = open_memmap(
            output_dir / f"{split}_{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
    model.eval()
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        z_c = torch.from_numpy(np.array(values[start:stop], copy=True)).to(device)
        result = model(z_c)
        for name in arrays:
            arrays[name][start:stop] = result[name].cpu().numpy()
    for values_array in arrays.values():
        values_array.flush()
    del arrays
    return {
        name: np.load(output_dir / f"{split}_{name}.npy", mmap_mode="r") for name in shapes
    }


@torch.inference_mode()
def decode_codes(
    decoder: torch.nn.Module,
    codes: np.ndarray,
    current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(codes), 256), dtype=np.float32)
    for start in range(0, len(codes), batch_size):
        stop = min(start + batch_size, len(codes))
        z_c = torch.from_numpy(np.array(codes[start:stop], copy=True)).to(device)
        h_t = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
        result[start:stop] = decoder(z_c, h_t).float().cpu().numpy()
    return result


def reconstruction_bundle(
    arrays: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, dict[str, float | int]]:
    current = np.asarray(arrays["current"])
    future = np.asarray(arrays["future"])
    dynamic = np.asarray(arrays["dynamic"], dtype=bool)
    return {
        "all": transition_metrics(current, future, prediction),
        "dynamic": transition_metrics(current, future, prediction, dynamic),
    }


def probe_bundle(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    ridge: float = 10.0,
) -> dict[str, Any]:
    definitions = {
        "contact_transition": ("contact_transition", 4),
        "force_trend": ("force_trend_class", 3),
    }
    result = {}
    for name, (label_name, classes) in definitions.items():
        probe = fit_ridge(
            train_feature,
            np.asarray(train_arrays[label_name]),
            classes,
            device,
            batch_size,
            ridge,
            classes=classes,
        )
        scores = apply_ridge(probe, test_feature, device, batch_size)
        prediction = scores.argmax(axis=1)
        target = np.asarray(test_arrays[label_name])
        majority_class = int(
            np.bincount(np.asarray(train_arrays[label_name]), minlength=classes).argmax()
        )
        majority = np.full(len(target), majority_class, dtype=np.int64)
        row = {
            **classification_metrics(target, prediction),
            "majority": classification_metrics(target, majority),
            "ridge": float(ridge),
        }
        if name == "contact_transition":
            per_class = {}
            class_names = ("free_to_free", "free_to_contact", "contact_to_contact", "contact_to_free")
            for index, class_name in enumerate(class_names):
                truth = target == index
                predicted = prediction == index
                tp = int(np.sum(truth & predicted))
                fp = int(np.sum(~truth & predicted))
                fn = int(np.sum(truth & ~predicted))
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                per_class[class_name] = {
                    "support": int(truth.sum()),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
                }
            row["per_class"] = per_class
        result[name] = row
    return result


def candidate_structure(
    values: np.ndarray, indices: np.ndarray, *, codebook_size: int = 128
) -> dict[str, Any]:
    return collapse_diagnostics(indices, values, codebook_size=codebook_size)


def representation_metrics(native: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    x = np.asarray(native, dtype=np.float64)
    y = np.asarray(candidate, dtype=np.float64)
    error = y - x
    input_energy = float(np.square(x).mean())
    return {
        "absolute_mse": float(np.square(error).mean()),
        "relative_distortion": float(np.square(error).mean() / max(input_energy, 1e-12)),
        "linear_cka": linear_cka(x.reshape(len(x), -1), y.reshape(len(y), -1)),
        "structure": representation_structure(y),
    }


def whitening_payload(statistics: WhiteningStatistics) -> dict[str, Any]:
    return {
        "kind": statistics.kind,
        "regularization": statistics.regularization,
        "mean": statistics.mean,
        "transform": statistics.transform,
        "inverse": statistics.inverse,
        "eigenvalues": statistics.eigenvalues,
    }


def whitening_from_payload(payload: dict[str, Any]) -> WhiteningStatistics:
    return WhiteningStatistics(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        transform=np.asarray(payload["transform"], dtype=np.float32),
        inverse=np.asarray(payload["inverse"], dtype=np.float32),
        eigenvalues=np.asarray(payload["eigenvalues"], dtype=np.float32),
        kind=payload["kind"],
        regularization=float(payload["regularization"]),
    )
