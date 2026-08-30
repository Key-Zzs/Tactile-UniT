"""Contact-only preservation remediation for the accepted continuous VAC space.

The functions in this module deliberately preserve the C2 model architecture.
Only the Contact projector and Contact recovery head can be made trainable; the
shared slots and both Vision/Action paths remain frozen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.tactile_teacher.evaluation import classification_metrics
from gr00t.tactile_unit.continuous_vac_shared_space import (
    ContinuousVACSharedSpace,
    different_episode_info_nce,
    relational_preservation,
    variance_floor,
)


ACCEPTED_C2_CHECKPOINT_SHA256 = (
    "454d7a33df20e5329e2be4804760dad211462e3eb405c16141f061f0c1ef113a"
)
CONTACT_TRAINABLE_PREFIXES = ("projectors.contact.", "recovery.contact.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_accepted_c2_checkpoint(path: Path) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError("accepted C2 checkpoint is unavailable")
    actual = sha256_file(path)
    if actual != ACCEPTED_C2_CHECKPOINT_SHA256:
        raise RuntimeError("C2R_BASELINE_CHECKPOINT_INVALID")
    return actual


def configure_contact_only_trainability(model: ContinuousVACSharedSpace) -> dict[str, Any]:
    """Freeze C2 globally, then expose only modality-specific Contact modules."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_names = []
    frozen_names = []
    for name, parameter in model.named_parameters():
        if name.startswith(CONTACT_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    if not trainable_names or any(name == "shared_slots" for name in trainable_names):
        raise RuntimeError("C2R_GRADIENT_ISOLATION_FAIL")
    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(CONTACT_TRAINABLE_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError("C2R_GRADIENT_ISOLATION_FAIL")
    return {
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "trainable_params": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_params": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def frozen_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(CONTACT_TRAINABLE_PREFIXES):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class C2RLossWeights:
    alignment: float = 1.0
    native_z: float = 5.0
    future: float = 1.0
    delta: float = 1.0
    relational: float = 0.25
    variance: float = 0.05


def contact_sample_weight(
    dynamic: torch.Tensor,
    transition: torch.Tensor,
    *,
    dynamic_weight: float,
    boundary_weight: float,
) -> torch.Tensor:
    if dynamic.ndim != 1 or transition.shape != dynamic.shape:
        raise ValueError("Contact weighting metadata must be one-dimensional and aligned")
    result = torch.where(
        dynamic.bool(),
        torch.full_like(dynamic, float(dynamic_weight), dtype=torch.float32),
        torch.ones_like(dynamic, dtype=torch.float32),
    )
    boundary = (transition == 1) | (transition == 2)
    return result * torch.where(
        boundary,
        torch.full_like(result, float(boundary_weight)),
        torch.ones_like(result),
    )


def weighted_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[0] != len(sample_weight):
        raise ValueError("weighted Contact MSE geometry mismatch")
    per_sample = torch.square(prediction - target).flatten(1).mean(dim=1)
    weight = sample_weight.to(per_sample)
    return torch.sum(per_sample * weight) / torch.clamp(weight.sum(), min=1.0)


def contact_relational_preservation(
    native: torch.Tensor,
    shared: torch.Tensor,
    maximum: int = 128,
    neighbors: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve global cosine, native neighborhoods, and distance ordering."""

    count = min(len(native), maximum)
    if count < 3:
        zero = native.new_zeros(())
        return zero, {"pairwise": zero, "neighborhood": zero, "ordering": zero}
    native_flat = F.normalize(native[:count].flatten(1), dim=-1, eps=1e-8).detach()
    shared_flat = F.normalize(shared[:count].flatten(1), dim=-1, eps=1e-8)
    native_similarity = native_flat @ native_flat.T
    shared_similarity = shared_flat @ shared_flat.T
    diagonal = torch.eye(count, dtype=torch.bool, device=native.device)
    pairwise = relational_preservation(native[:count], shared[:count], maximum=count)
    k = min(neighbors, count - 1)
    neighborhood_index = native_similarity.masked_fill(diagonal, -torch.inf).topk(k, dim=1).indices
    neighborhood = F.mse_loss(
        shared_similarity.gather(1, neighborhood_index),
        native_similarity.gather(1, neighborhood_index),
    )
    near_index = neighborhood_index[:, 0]
    far_index = native_similarity.masked_fill(diagonal, torch.inf).argmin(dim=1)
    row = torch.arange(count, device=native.device)
    native_gap = (
        native_similarity[row, near_index] - native_similarity[row, far_index]
    ).detach()
    shared_gap = shared_similarity[row, near_index] - shared_similarity[row, far_index]
    ordering = F.relu(torch.clamp(native_gap, max=0.25) - shared_gap).mean()
    total = (pairwise + neighborhood + ordering) / 3.0
    return total, {
        "pairwise": pairwise,
        "neighborhood": neighborhood,
        "ordering": ordering,
    }


def c2r_contact_loss(
    model: ContinuousVACSharedSpace,
    decoder: nn.Module,
    native: Mapping[str, torch.Tensor],
    episode_id: torch.Tensor,
    dynamic: torch.Tensor,
    transition: torch.Tensor,
    h_current: torch.Tensor,
    h_future: torch.Tensor,
    *,
    temperature: float,
    dynamic_weight: float,
    boundary_weight: float,
    weights: C2RLossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the preregistered Contact-only C2-R objective.

    Vision and Action are encoded without autograd. The frozen decoder remains
    in the graph, allowing gradients to flow through it to ``R_c`` and ``P_c``.
    """

    if set(native) != {"vision", "action", "contact"}:
        raise ValueError("C2-R loss requires exactly Vision, Action, and Contact native tensors")
    with torch.no_grad():
        shared_vision = model.encode("vision", native["vision"])
        shared_action = model.encode("action", native["action"])
    shared_contact = model.encode("contact", native["contact"])
    sample_weight = contact_sample_weight(
        dynamic,
        transition,
        dynamic_weight=dynamic_weight,
        boundary_weight=boundary_weight,
    )
    pair_losses = []
    for left, right in (
        (shared_vision, shared_contact),
        (shared_contact, shared_vision),
        (shared_action, shared_contact),
        (shared_contact, shared_action),
    ):
        pair_losses.append(
            different_episode_info_nce(
                left,
                right,
                episode_id,
                temperature=temperature,
                sample_weight=sample_weight,
            )
        )
    alignment = torch.stack(pair_losses).mean()
    recovered = model.recover("contact", shared_contact)
    native_z = weighted_sample_mse(recovered, native["contact"].detach(), sample_weight)
    predicted_future = decoder(recovered, h_current)
    future = weighted_sample_mse(predicted_future, h_future.detach(), sample_weight)
    predicted_delta = predicted_future - h_current
    target_delta = h_future - h_current
    delta = weighted_sample_mse(predicted_delta, target_delta.detach(), sample_weight)
    relational, relational_parts = contact_relational_preservation(
        native["contact"], shared_contact
    )
    variance = variance_floor(shared_contact)
    total = (
        weights.alignment * alignment
        + weights.native_z * native_z
        + weights.future * future
        + weights.delta * delta
        + weights.relational * relational
        + weights.variance * variance
    )
    return total, {
        "total": total.detach(),
        "alignment_contact": alignment.detach(),
        "native_z": native_z.detach(),
        "future": future.detach(),
        "delta": delta.detach(),
        "relational_contact": relational.detach(),
        "relational_pairwise": relational_parts["pairwise"].detach(),
        "relational_neighborhood": relational_parts["neighborhood"].detach(),
        "relational_ordering": relational_parts["ordering"].detach(),
        "variance_contact": variance.detach(),
    }


def _per_class_metrics(
    target: np.ndarray, prediction: np.ndarray, classes: int
) -> dict[str, dict[str, float | None]]:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    result: dict[str, dict[str, float | None]] = {}
    for label in range(classes):
        true_positive = int(np.sum((target == label) & (prediction == label)))
        false_positive = int(np.sum((target != label) & (prediction == label)))
        false_negative = int(np.sum((target == label) & (prediction != label)))
        support = int(np.sum(target == label))
        precision = (
            None if true_positive + false_positive == 0
            else true_positive / (true_positive + false_positive)
        )
        recall = None if support == 0 else true_positive / support
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = None if support == 0 else (0.0 if denominator == 0 else 2 * true_positive / denominator)
        result[str(label)] = {
            "precision": None if precision is None else float(precision),
            "recall": None if recall is None else float(recall),
            "f1": None if f1 is None else float(f1),
            "support": support,
        }
    return result


def canonical_contact_probe(
    train_x: np.ndarray,
    evaluation_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_y: np.ndarray,
    classes: int,
    *,
    train_order: np.ndarray | None = None,
    return_prediction: bool = False,
) -> dict[str, Any]:
    """The exact accepted C2 Contact probe, shared by C2 and C2-R."""

    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_features = np.asarray(train_x).reshape(len(train_x), -1)
    labels = np.asarray(train_y, dtype=np.int64)
    if train_order is not None:
        train_features = train_features[train_order]
        labels = labels[train_order]
    model = make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced")
    )
    model.fit(train_features, labels)
    target = np.asarray(evaluation_y, dtype=np.int64)
    prediction = model.predict(np.asarray(evaluation_x).reshape(len(evaluation_x), -1))
    majority_class = int(np.bincount(np.asarray(train_y), minlength=classes).argmax())
    majority_prediction = np.full(len(target), majority_class, dtype=np.int64)
    per_class = _per_class_metrics(target, prediction, classes)
    result: dict[str, Any] = {
        **classification_metrics(target, prediction),
        "majority": classification_metrics(target, majority_prediction),
        "per_class": per_class,
        "per_class_recall": {label: value["recall"] for label, value in per_class.items()},
        "probe": "StandardScaler + RidgeClassifier(alpha=10,class_weight=balanced)",
        "protocol": {
            "feature": "flatten [N,8,32] to [N,256]",
            "normalization": "StandardScaler fit on train rows",
            "classifier": "RidgeClassifier",
            "alpha": 10.0,
            "class_weight": "balanced",
            "optimizer": "closed-form scipy conjugate-gradient solver selected by sklearn",
            "epochs": None,
            "checkpoint_selection": None,
            "macro_f1": "unweighted mean over classes present in evaluation target",
            "majority": "most frequent train label",
        },
    }
    if return_prediction:
        result["_prediction"] = prediction
        result["_majority_prediction"] = majority_prediction
    return result


def retention(shared: Mapping[str, Any], native: Mapping[str, Any]) -> float:
    majority = float(native["majority"]["macro_f1"])
    return float(
        (float(shared["macro_f1"]) - majority)
        / max(float(native["macro_f1"]) - majority, 1e-12)
    )


def bootstrap_probe_metrics(
    target: np.ndarray,
    native_prediction: np.ndarray,
    shared_prediction: np.ndarray,
    majority_prediction: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    target = np.asarray(target, dtype=np.int64)
    native_prediction = np.asarray(native_prediction, dtype=np.int64)
    shared_prediction = np.asarray(shared_prediction, dtype=np.int64)
    majority_prediction = np.asarray(majority_prediction, dtype=np.int64)
    rng = np.random.default_rng(seed)
    native_values = np.empty(samples, dtype=np.float64)
    shared_values = np.empty(samples, dtype=np.float64)
    retention_values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(target), size=len(target))
        native_metric = classification_metrics(target[selected], native_prediction[selected])["macro_f1"]
        shared_metric = classification_metrics(target[selected], shared_prediction[selected])["macro_f1"]
        majority_metric = classification_metrics(target[selected], majority_prediction[selected])["macro_f1"]
        native_values[index] = native_metric
        shared_values[index] = shared_metric
        retention_values[index] = (
            (shared_metric - majority_metric) / max(native_metric - majority_metric, 1e-12)
        )
    return {
        "native_f1_ci95": [float(value) for value in np.quantile(native_values, [0.025, 0.975])],
        "shared_f1_ci95": [float(value) for value in np.quantile(shared_values, [0.025, 0.975])],
        "retention_ci95": [float(value) for value in np.quantile(retention_values, [0.025, 0.975])],
    }
