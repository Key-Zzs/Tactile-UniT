"""Bounded diagnostic utilities for Track C C3-R0.

Only diagnostic probes and small source-to-``u_c`` ceilings are trainable.
Every source constructor is explicit so Contact targets, future Contact state,
pair identity, and the Contact-private residual cannot enter a source by
accident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from gr00t.tactile_teacher.evaluation import classification_metrics
from gr00t.tactile_unit.continuous_vac_shared_space import geometry_diagnostics


SOURCE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "V": ("u_v",),
    "A": ("u_a",),
    "VA": ("u_v", "u_a"),
    "H": ("h_current",),
    "VH": ("u_v", "h_current"),
    "AH": ("u_a", "h_current"),
    "VAH": ("u_v", "u_a", "h_current"),
    "C": ("u_c",),
    "ZC": ("z_c",),
}
ALLOWED_SOURCE_ARRAYS = {"u_v", "u_a", "h_current", "u_c", "z_c"}
PRIVATE_ARRAYS = {"r_c_priv", "z_c_shared", "h_future", "pair_id"}
CONTACT_BOUNDARY_CLASSES = (1, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _flat(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    return array.reshape(len(array), -1).astype(np.float32, copy=False)


def source_features(source: str, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Construct one legal source set without any implicit array access."""

    if source not in SOURCE_COMPONENTS:
        raise ValueError(f"unknown C3-R0 source {source!r}")
    components = SOURCE_COMPONENTS[source]
    if any(name not in ALLOWED_SOURCE_ARRAYS for name in components):
        raise RuntimeError("STRUCTURAL_FAIL: illegal source component")
    missing = [name for name in components if name not in arrays]
    if missing:
        raise KeyError(f"missing source arrays: {missing}")
    values = [_flat(arrays[name]) for name in components]
    if any(len(value) != len(values[0]) for value in values):
        raise ValueError("unaligned C3-R0 source components")
    return values[0] if len(values) == 1 else np.concatenate(values, axis=1)


@dataclass(frozen=True)
class TrainStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray) -> "TrainStandardizer":
        value = _flat(train).astype(np.float64)
        mean = value.mean(0)
        scale = value.std(0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, value: np.ndarray) -> np.ndarray:
        return ((_flat(value) - self.mean) / self.scale).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, mean=self.mean, scale=self.scale)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TrainStandardizer":
        payload = np.load(path, allow_pickle=False)
        return cls(np.asarray(payload["mean"]), np.asarray(payload["scale"]))


def per_class_metrics(target: np.ndarray, prediction: np.ndarray, classes: int) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    result: dict[str, Any] = {}
    for label in range(classes):
        tp = int(np.sum((target == label) & (prediction == label)))
        fp = int(np.sum((target != label) & (prediction == label)))
        fn = int(np.sum((target == label) & (prediction != label)))
        support = int(np.sum(target == label))
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if support == 0 else tp / support
        denominator = 2 * tp + fp + fn
        f1 = None if support == 0 else (0.0 if denominator == 0 else 2 * tp / denominator)
        result[str(label)] = {
            "precision": None if precision is None else float(precision),
            "recall": None if recall is None else float(recall),
            "f1": None if f1 is None else float(f1),
            "support": support,
        }
    return result


def evaluate_prediction(
    target: np.ndarray,
    prediction: np.ndarray,
    majority_prediction: np.ndarray,
    classes: int,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    change_target = np.isin(target, CONTACT_BOUNDARY_CLASSES).astype(np.int64)
    change_prediction = np.isin(prediction, CONTACT_BOUNDARY_CLASSES).astype(np.int64)
    per_class = per_class_metrics(target, prediction, classes)
    result = {
        **classification_metrics(target, prediction),
        "majority": classification_metrics(target, majority_prediction),
        "per_class": per_class,
    }
    if classes == 4:
        result.update(
            {
                "free_to_contact": per_class["1"],
                "contact_to_free": per_class["3"],
                "future_change": classification_metrics(change_target, change_prediction),
            }
        )
    return result


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, *, alpha: float = 10.0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=float(alpha), class_weight="balanced")
    )
    model.fit(_flat(train_x), np.asarray(train_y, dtype=np.int64))
    return model


def probe_prediction(model: Any, value: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(_flat(value)), dtype=np.int64)


def semantic_ratio(source_f1: float, majority_f1: float, oracle_f1: float) -> float:
    return float((source_f1 - majority_f1) / max(oracle_f1 - majority_f1, 1e-12))


def bootstrap_f1_difference(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    target = np.asarray(target, dtype=np.int64)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(target), size=len(target))
        differences[index] = (
            classification_metrics(target[selected], left[selected])["macro_f1"]
            - classification_metrics(target[selected], right[selected])["macro_f1"]
        )
    return np.quantile(differences, [0.025, 0.975]).astype(float).tolist()


def normalized_label_entropy(neighbor_labels: np.ndarray, classes: int = 4) -> np.ndarray:
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    counts = np.stack([(labels == label).sum(1) for label in range(classes)], axis=1)
    probability = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    entropy = -np.sum(
        np.where(probability > 0, probability * np.log(np.maximum(probability, 1e-12)), 0.0),
        axis=1,
    )
    return entropy / np.log(classes)


def neighbor_purity(neighbor_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    counts = np.stack([(labels == label).sum(1) for label in np.unique(labels)], axis=1)
    return counts.max(1) / labels.shape[1]


def majority_neighbor_prediction(neighbor_labels: np.ndarray, classes: int = 4) -> np.ndarray:
    labels = np.asarray(neighbor_labels, dtype=np.int64)
    counts = np.stack([(labels == label).sum(1) for label in range(classes)], axis=1)
    return counts.argmax(1).astype(np.int64)


def distribution_summary(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def local_target_ambiguity(neighbor_targets: np.ndarray, global_variance: float) -> dict[str, Any]:
    targets = np.asarray(neighbor_targets, dtype=np.float32)
    flat = targets.reshape(targets.shape[0], targets.shape[1], -1)
    centroid = flat.mean(1, keepdims=True)
    local_variance = np.square(flat - centroid, dtype=np.float64).mean(axis=(1, 2))
    normalized = local_variance / max(float(global_variance), 1e-12)
    trace_covariance = local_variance * flat.shape[2]
    normalized_targets = flat / np.maximum(np.linalg.norm(flat, axis=2, keepdims=True), 1e-12)
    cosine = np.einsum("nkd,njd->nkj", normalized_targets, normalized_targets)
    off_diagonal = ~np.eye(flat.shape[1], dtype=bool)
    pairwise = (1.0 - cosine[:, off_diagonal]).mean(1) if flat.shape[1] > 1 else np.zeros(len(flat))
    return {
        "local_variance": distribution_summary(local_variance),
        "trace_covariance": distribution_summary(trace_covariance),
        "local_over_global": distribution_summary(normalized),
        "pairwise_cosine_distance": distribution_summary(pairwise),
    }


def neighborhood_audit(
    indices: np.ndarray,
    train_labels: np.ndarray,
    query_labels: np.ndarray,
    train_targets: np.ndarray,
    *,
    k: int,
    global_variance: float,
    classes: int = 4,
) -> dict[str, Any]:
    selected = np.asarray(indices[:, :k], dtype=np.int64)
    labels = np.asarray(train_labels, dtype=np.int64)[selected]
    majority = majority_neighbor_prediction(labels, classes)
    entropy = normalized_label_entropy(labels, classes)
    purity = neighbor_purity(labels)
    query = np.asarray(query_labels, dtype=np.int64)
    boundary = np.isin(query, CONTACT_BOUNDARY_CLASSES)
    result = {
        "k": int(k),
        "label_purity": distribution_summary(purity),
        "normalized_label_entropy": distribution_summary(entropy),
        "majority_neighbor": classification_metrics(query, majority),
        "target_ambiguity": local_target_ambiguity(
            np.asarray(train_targets)[selected], global_variance
        ),
        "boundary": {
            "count": int(boundary.sum()),
            "normalized_entropy": None
            if not boundary.any()
            else distribution_summary(entropy[boundary]),
        },
    }
    return result


def knn_target_predictions(indices: np.ndarray, train_targets: np.ndarray, k: int) -> dict[str, np.ndarray]:
    selected = np.asarray(indices[:, :k], dtype=np.int64)
    targets = np.asarray(train_targets)[selected]
    one = targets[:, 0].copy()
    mean = targets.mean(1, dtype=np.float64).astype(np.float32)
    flat = targets.reshape(len(targets), k, -1).astype(np.float64)
    square = np.sum(np.square(flat), axis=2)
    distance = square[:, :, None] + square[:, None, :] - 2.0 * np.einsum(
        "nkd,njd->nkj", flat, flat
    )
    medoid_index = distance.sum(2).argmin(1)
    medoid = targets[np.arange(len(targets)), medoid_index].copy()
    return {"1nn": one, "medoid": medoid, "mean": mean}


def regression_geometry(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float32)
    oracle = np.asarray(target, dtype=np.float32)
    error = np.square(predicted.astype(np.float64) - oracle.astype(np.float64)).reshape(len(oracle), -1).mean(1)
    left = predicted.reshape(len(predicted), -1).astype(np.float64)
    right = oracle.reshape(len(oracle), -1).astype(np.float64)
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return {
        "mse": float(error.mean()),
        "paired_cosine": float(np.sum(left * right, axis=1).mean()),
        "variance": float(np.var(predicted, dtype=np.float64)),
        "geometry": geometry_diagnostics(predicted),
    }


class SmallContactCeiling(nn.Module):
    """At most 100k parameters; source-only and diagnostic-only."""

    def __init__(self, source: str, architecture: str):
        super().__init__()
        if source not in {"VA", "VAH"} or architecture not in {"M0", "M1"}:
            raise ValueError("unsupported C3-R0 deterministic ceiling")
        self.source = source
        self.architecture = architecture
        input_dim = 512 if source == "VA" else 768
        if architecture == "M0":
            self.network = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 64),
                nn.GELU(),
                nn.Linear(64, 256),
            )
        else:
            self.target_slots = nn.Parameter(torch.randn(8, 32) * 0.02)
            self.h_projection = nn.Linear(256, 32) if source == "VAH" else None
            self.memory_norm = nn.LayerNorm(32)
            self.query_norm = nn.LayerNorm(32)
            self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.output = nn.Sequential(
                nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)
            )

    def forward(self, source_features_flat: torch.Tensor) -> torch.Tensor:
        if source_features_flat.ndim != 2:
            raise ValueError("C3-R0 ceiling input must be flattened source features")
        if self.architecture == "M0":
            return self.network(source_features_flat).view(-1, 8, 32)
        vision = source_features_flat[:, :256].view(-1, 8, 32)
        action = source_features_flat[:, 256:512].view(-1, 8, 32)
        memories = [vision, action]
        if self.source == "VAH":
            assert self.h_projection is not None
            memories.append(self.h_projection(source_features_flat[:, 512:]).unsqueeze(1))
        memory = self.memory_norm(torch.cat(memories, dim=1))
        query = self.target_slots.unsqueeze(0).expand(len(memory), -1, -1)
        attended, _ = self.attention(self.query_norm(query), memory, memory, need_weights=False)
        value = query + attended
        return value + self.output(value)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def save_ceiling_checkpoint(path: Path, model: SmallContactCeiling, metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "tactile3d-unit.vac-c3r0-ceiling.v1",
            "source": model.source,
            "architecture": model.architecture,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def load_ceiling_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[SmallContactCeiling, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "tactile3d-unit.vac-c3r0-ceiling.v1":
        raise ValueError("unsupported C3-R0 ceiling checkpoint")
    model = SmallContactCeiling(str(payload["source"]), str(payload["architecture"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, dict(payload.get("metadata", {}))


def root_cause_decision(evidence: Mapping[str, Any]) -> tuple[str, str]:
    """Apply the pre-registered project-level decision taxonomy."""

    if not bool(evidence.get("structural_pass", True)):
        return "STRUCTURAL_FAIL", "NO_NEXT_STAGE_DUE_TO_STRUCTURAL_FAIL"
    single = bool(evidence.get("single_source_sufficient", False))
    nn_strong = bool(evidence.get("nonparametric_strong", False))
    if (single or nn_strong) and bool(evidence.get("predictor_gap", False)):
        return "PREDICTOR_OBJECTIVE_BOTTLENECK", "C3-R1_SEMANTIC_RANK_PRESERVING_PREDICTOR"
    if bool(evidence.get("va_sufficient", False)):
        return "MULTISOURCE_COMPLEMENTARITY_REQUIRED", "C3-MS_MULTISOURCE_PREDICTION"
    if bool(evidence.get("vah_sufficient", False)):
        return "CAUSAL_CONTACT_CONTEXT_REQUIRED", "C3-MS-CC_CAUSAL_CONTEXT_PREDICTION"
    if bool(evidence.get("multimodality", False)):
        return "CONDITIONAL_MULTIMODALITY_LIKELY", "C3-DISTRIBUTIONAL_CONTACT_PREDICTION"
    if bool(evidence.get("direct_high_target_low", False)):
        return "SHARED_CONTACT_TARGET_TOO_ENTANGLED", "C3-SHARED_TARGET_REFACTOR"
    return "MIXED", "C3-SHARED_TARGET_REFACTOR"


def frozen_parameter_guard(*models: nn.Module) -> bool:
    for model in models:
        model.eval().requires_grad_(False)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            return False
    return True
