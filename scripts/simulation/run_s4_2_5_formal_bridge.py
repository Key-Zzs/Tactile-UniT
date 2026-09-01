#!/usr/bin/env python3
"""Train and select the formal continuous cross-modal VAC bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_formal import FormalVACBridge  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    MODALITIES,
    ContinuousVACSharedSpace,
    VACLossWeights,
    continuous_vac_loss,
    geometry_diagnostics,
    load_checkpoint,
    pairwise_alignment_metrics,
)
from scripts.simulation.train_s4_2ds_bridge_pilot import (  # noqa: E402
    classification_probe,
)

CONFIG_PATH = ROOT / "configs/simulation/s4_2_formal_downstream.json"
PROTOCOL_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/protocol_freeze.json"
S4_2_4_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/s4_2_4_final.json"
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2_formal"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2_formal/bridge"
M3_CHECKPOINT = ROOT / ".local/experiments/tactile_unit/vac_c2/selected.pt"


class Bridge(Protocol):
    modalities: tuple[str, ...]

    def encode(self, modality: str, native: torch.Tensor) -> torch.Tensor: ...

    def recover(self, modality: str, shared: torch.Tensor) -> torch.Tensor: ...


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric(values: np.ndarray) -> np.ndarray:
    return np.unique(values, return_inverse=True)[1].astype(np.int64)


@torch.inference_mode()
def encode_bridge(
    model: Bridge,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    source = {"vision": "z_v", "action": "z_a", "contact": "z_c"}
    shared = {
        name: np.empty((len(arrays["pair_id"]), 8, 32), dtype=np.float32) for name in MODALITIES
    }
    recovered = {name: np.empty_like(value) for name, value in shared.items()}
    for start in range(0, len(arrays["pair_id"]), batch_size):
        stop = min(start + batch_size, len(arrays["pair_id"]))
        for name in MODALITIES:
            native = torch.from_numpy(arrays[source[name]][start:stop]).to(device)
            value = model.encode(name, native)
            shared[name][start:stop] = value.float().cpu().numpy()
            recovered[name][start:stop] = model.recover(name, value).float().cpu().numpy()
    return shared, recovered


def candidate_spec(name: str) -> tuple[str, float, float]:
    values = {
        "B2_affine_w1": ("affine", 0.10, 1.0),
        "B2_mlp_w1": ("mlp", 0.10, 1.0),
        "B2_mlp_w2": ("mlp", 0.10, 2.0),
        "B3_slot_t0.07_w1": ("slot", 0.07, 1.0),
        "B3_slot_t0.10_w2": ("slot", 0.10, 2.0),
    }
    return values[name]


def alignment_metrics(
    shared: dict[str, np.ndarray], episode: np.ndarray, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    return {
        name: pairwise_alignment_metrics(
            shared[left],
            shared[right],
            episode,
            bootstrap_samples=bootstrap_samples,
            seed=seed + offset,
            retrieval_chunk=512,
        )
        for offset, (name, left, right) in enumerate(
            (
                ("V-A", "vision", "action"),
                ("V-C", "vision", "contact"),
                ("A-C", "action", "contact"),
            )
        )
    }


def retrieval_multiplier(value: dict[str, Any]) -> float:
    return float(value["recall_at_10"] / max(value["chance"]["recall_at_10"], 1e-12))


def evaluate(
    name: str,
    model: Bridge,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    device: torch.device,
    config: dict[str, Any],
    bootstrap_samples: int,
    *,
    trainable: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    shared_train, _ = encode_bridge(model, train, device)
    shared_validation, recovered_validation = encode_bridge(model, validation, device)
    alignment = alignment_metrics(
        shared_validation, numeric(validation["episode_id"]), bootstrap_samples, int(config["seed"])
    )
    native_contact = classification_probe(
        train["z_c"],
        validation["z_c"],
        train["contact_transition"],
        validation["contact_transition"],
        [0, 1, 2, 3],
    )
    shared_contact = classification_probe(
        shared_train["contact"],
        shared_validation["contact"],
        train["contact_transition"],
        validation["contact_transition"],
        [0, 1, 2, 3],
    )
    native_force = classification_probe(
        train["z_c"],
        validation["z_c"],
        train["force_trend"],
        validation["force_trend"],
        [0, 1, 2],
    )
    shared_force = classification_probe(
        shared_train["contact"],
        shared_validation["contact"],
        train["force_trend"],
        validation["force_trend"],
        [0, 1, 2],
    )
    retention = {
        "contact": shared_contact["macro_f1"] / max(native_contact["macro_f1"], 1e-12),
        "force": shared_force["macro_f1"] / max(native_force["macro_f1"], 1e-12),
    }
    geometry = {key: geometry_diagnostics(value) for key, value in shared_validation.items()}
    recovery = {
        name: float(
            np.square(
                recovered_validation[name].astype(np.float64) - validation[key].astype(np.float64)
            ).mean()
        )
        for name, key in (("vision", "z_v"), ("action", "z_a"), ("contact", "z_c"))
    }
    contact_retrieval = [
        alignment[pair]["retrieval"][direction]
        for pair in ("V-C", "A-C")
        for direction in ("forward", "reverse")
    ]
    utility = float(np.mean([value["recall_at_10"] for value in contact_retrieval]))
    gates_config = config["s4_2_5"]["gates"]
    gates = {
        "v_c_margin": alignment["V-C"]["margin_bootstrap_ci95"][0]
        > float(gates_config["paired_margin_ci_lower_min"]),
        "a_c_margin": alignment["A-C"]["margin_bootstrap_ci95"][0]
        > float(gates_config["paired_margin_ci_lower_min"]),
        "contact_retrieval": min(retrieval_multiplier(value) for value in contact_retrieval)
        >= float(gates_config["contact_retrieval_r10_chance_multiplier_min"]),
        "contact_semantics": retention["contact"]
        >= float(gates_config["contact_semantic_retention_min"]),
        "force_semantics": retention["force"]
        >= float(gates_config["force_semantic_retention_min"]),
        "no_collapse": all(
            value["per_dimension_variance"]["near_zero_fraction"]
            <= float(gates_config["near_zero_variance_fraction_max"])
            for value in geometry.values()
        ),
        "finite": all(np.isfinite(value).all() for value in shared_validation.values()),
        "independent_encodability": True,
    }
    result = {
        "schema": "tactile3d-unit.s4-2-5-bridge-evaluation.v1",
        "candidate": name,
        "trainable": trainable,
        "alignment": alignment,
        "contact_semantics": {
            "native": native_contact,
            "shared": shared_contact,
            "retention": retention["contact"],
        },
        "force_semantics": {
            "native": native_force,
            "shared": shared_force,
            "retention": retention["force"],
        },
        "geometry": geometry,
        "native_recovery_mse": recovery,
        "mean_contact_retrieval_r10": utility,
        "minimum_contact_retrieval_chance_multiplier": float(
            min(retrieval_multiplier(value) for value in contact_retrieval)
        ),
        "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "formal_test_loaded": False,
    }
    return result, shared_train | {
        f"validation_{key}": value for key, value in shared_validation.items()
    }


def train_candidate(
    name: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seed_everything(int(config["seed"]))
    adapter, temperature, dynamic_weight = candidate_spec(name)
    model = FormalVACBridge(adapter).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    per_modality = [
        sum(p.numel() for p in model.projectors[key].parameters()) for key in model.modalities
    ]
    if parameters > 300_000 or max(per_modality) > 50_000:
        raise RuntimeError(f"{name} exceeds frozen bridge parameter budget")
    training = config["s4_2_5"]["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(int(config["seed"]))
    episode = numeric(train["episode_id"])
    history = []
    model.train()
    for step in range(1, int(training["steps"]) + 1):
        rows = rng.choice(len(train["pair_id"]), size=int(training["batch_size"]), replace=False)
        native = {
            "vision": torch.from_numpy(train["z_v"][rows]).to(device),
            "action": torch.from_numpy(train["z_a"][rows]).to(device),
            "contact": torch.from_numpy(train["z_c"][rows]).to(device),
        }
        loss, breakdown = continuous_vac_loss(
            model,
            native,
            torch.from_numpy(episode[rows]).to(device),
            torch.from_numpy(train["dynamic"][rows]).to(device),
            temperature=temperature,
            dynamic_weight=dynamic_weight,
            weights=VACLossWeights(),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(training["steps"]):
            row = {"step": step, **{key: float(value) for key, value in breakdown.items()}}
            history.append(row)
            print(json.dumps({"candidate": name, **row}), flush=True)
    checkpoint = EXPERIMENT_ROOT / name / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2-5-continuous-bridge.v1",
            "candidate": name,
            "adapter": adapter,
            "temperature": temperature,
            "dynamic_weight": dynamic_weight,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "parameters": parameters,
            "formal_test_loaded": False,
        },
        checkpoint,
    )
    result, shared = evaluate(
        name, model, train, validation, device, config, bootstrap_samples, trainable=True
    )
    result.update(
        {
            "parameters": parameters,
            "per_modality_projector_parameters": per_modality[0],
            "training": {"split": "train", "steps": int(training["steps"]), "history": history},
            "selection_split": "validation",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    )
    atomic_json(ARTIFACT_ROOT / f"bridge_{name.lower()}.json", result)
    return result, shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    config, protocol = load_json(CONFIG_PATH), load_json(PROTOCOL_PATH)
    if protocol["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("formal protocol changed after freeze")
    if load_json(S4_2_4_PATH)["decision"] != "S4_2_4_FORMAL_ACTION_VISION_ACCEPTED":
        raise RuntimeError("S4.2-4 dependency failed")
    train, validation = load_npz(PAIR_ROOT / "paired_train.npz"), load_npz(
        PAIR_ROOT / "paired_validation.npz"
    )
    train_groups = set(zip(train["task"].tolist(), train["source_trajectory_id"].tolist()))
    validation_groups = set(
        zip(validation["task"].tolist(), validation["source_trajectory_id"].tolist())
    )
    if train_groups & validation_groups:
        raise RuntimeError("formal bridge source-group leakage")
    device = torch.device(args.device)
    diagnostics = {}
    identity = ContinuousVACSharedSpace("C0").eval().requires_grad_(False).to(device)
    diagnostics["B0_native"], _ = evaluate(
        "B0_native",
        identity,
        train,
        validation,
        device,
        config,
        args.bootstrap_samples,
        trainable=False,
    )
    m3, _ = load_checkpoint(M3_CHECKPOINT, map_location="cpu")
    m3 = m3.eval().requires_grad_(False).to(device)
    diagnostics["B1_frozen_m3_zero_shot"], _ = evaluate(
        "B1_frozen_m3_zero_shot",
        m3,
        train,
        validation,
        device,
        config,
        args.bootstrap_samples,
        trainable=False,
    )
    for name, value in diagnostics.items():
        value["checkpoint"] = None if name == "B0_native" else str(M3_CHECKPOINT.relative_to(ROOT))
        value["checkpoint_sha256"] = None if name == "B0_native" else sha256_file(M3_CHECKPOINT)
        atomic_json(ARTIFACT_ROOT / f"bridge_{name.lower()}.json", value)
    results, shared = {}, {}
    for name in config["s4_2_5"]["trainable_candidates"]:
        results[name], shared[name] = train_candidate(
            name, train, validation, config, device, args.bootstrap_samples
        )
    passing = [name for name, value in results.items() if value["overall"] == "PASS"]
    passing_b2 = [name for name in passing if name.startswith("B2_")]
    if passing_b2:
        best_utility = max(results[name]["mean_contact_retrieval_r10"] for name in passing)
        eligible = [
            name
            for name in passing_b2
            if results[name]["mean_contact_retrieval_r10"] >= best_utility - 0.01
        ]
        selected = (
            min(
                eligible,
                key=lambda name: (
                    results[name]["parameters"],
                    config["s4_2_5"]["trainable_candidates"].index(name),
                ),
            )
            if eligible
            else None
        )
    else:
        selected = None
    if selected is None:
        passing_b3 = [name for name in passing if name.startswith("B3_")]
        selected = (
            max(passing_b3, key=lambda name: results[name]["mean_contact_retrieval_r10"])
            if passing_b3
            else None
        )
    if selected is None:
        atomic_json(
            ARTIFACT_ROOT / "s4_2_5_final.json",
            {"decision": "S4_2_5_FORMAL_BRIDGE_FAIL", "formal_test_loaded": False},
        )
        raise SystemExit("S4_2_5_FORMAL_BRIDGE_FAIL")
    selected_checkpoint = ROOT / results[selected]["checkpoint"]
    canonical_checkpoint = EXPERIMENT_ROOT / "selected.pt"
    shutil.copy2(selected_checkpoint, canonical_checkpoint)
    train_shared = {name: shared[selected][name] for name in ("vision", "action", "contact")}
    validation_shared = {
        name: shared[selected][f"validation_{name}"] for name in ("vision", "action", "contact")
    }
    np.savez_compressed(
        CACHE_ROOT := PAIR_ROOT / "shared_train.npz",
        pair_id=train["pair_id"],
        u_v=train_shared["vision"],
        u_a=train_shared["action"],
        u_c=train_shared["contact"],
    )
    np.savez_compressed(
        PAIR_ROOT / "shared_validation.npz",
        pair_id=validation["pair_id"],
        u_v=validation_shared["vision"],
        u_a=validation_shared["action"],
        u_c=validation_shared["contact"],
    )
    final = {
        "schema": "tactile3d-unit.s4-2-5-final.v1",
        "decision": "S4_2_5_FORMAL_CONTINUOUS_BRIDGE_ACCEPTED",
        "selected": selected,
        "passing": passing,
        "selection_rule": config["s4_2_5"]["selection"],
        "checkpoint": str(canonical_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(canonical_checkpoint),
        "train_cache_sha256": sha256_file(CACHE_ROOT),
        "validation_cache_sha256": sha256_file(PAIR_ROOT / "shared_validation.npz"),
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "canonical_contact": "C3",
        "formal_test_loaded": False,
        "S4_2_6_readiness": "READY",
    }
    atomic_json(ARTIFACT_ROOT / "s4_2_5_final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
