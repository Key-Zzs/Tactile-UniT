#!/usr/bin/env python3
"""Evaluate Contact RQ and train the formal shared/private decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.evaluation import different_episode_permutation  # noqa: E402
from gr00t.simulation.s4_2_formal import (  # noqa: E402
    FormalVACBridge,
    ResidualVectorQuantizer,
    SharedPrivateDecomposer,
)
from scripts.simulation.train_s4_2ds_bridge_pilot import classification_probe  # noqa: E402

CONFIG_PATH = ROOT / "configs/simulation/s4_2_formal_downstream.json"
PROTOCOL_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/protocol_freeze.json"
S4_2_5_PATH = ROOT / ".local/artifacts/simulation/s4_2_formal/s4_2_5_final.json"
PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2_formal"
BRIDGE_PATH = ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt"
EXPERIMENT_ROOT = ROOT / ".local/experiments/simulation/s4_2_formal/s4_2_6"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2_formal"


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


def load_bridge(device: torch.device) -> FormalVACBridge:
    payload = torch.load(BRIDGE_PATH, map_location="cpu", weights_only=False)
    model = FormalVACBridge(str(payload["adapter"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def recover_contact(
    bridge: FormalVACBridge, shared: np.ndarray, device: torch.device
) -> np.ndarray:
    result = np.empty_like(shared)
    for start in range(0, len(shared), 1024):
        stop = min(start + 1024, len(shared))
        value = torch.from_numpy(shared[start:stop]).to(device)
        result[start:stop] = bridge.recover("contact", value).float().cpu().numpy()
    return result


@torch.inference_mode()
def quantize(
    model: ResidualVectorQuantizer, value: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty_like(value)
    indices = np.empty((len(value), 8, 2), dtype=np.int64)
    model.eval()
    for start in range(0, len(value), 2048):
        stop = min(start + 2048, len(value))
        quantized, index = model(torch.from_numpy(value[start:stop]).to(device))
        output[start:stop] = quantized.float().cpu().numpy()
        indices[start:stop] = index.cpu().numpy()
    return output, indices


def train_rq(
    learning_rate: float,
    train_shared: np.ndarray,
    validation_shared: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ResidualVectorQuantizer, dict[str, Any], np.ndarray, np.ndarray]:
    seed_everything(int(config["seed"]))
    rq_config = config["s4_2_6"]["contact_rq"]
    model = ResidualVectorQuantizer().to(device)
    rng = np.random.default_rng(int(config["seed"]))
    initial = train_shared.reshape(-1, 32)[
        rng.choice(train_shared.shape[0] * 8, size=256, replace=False)
    ].reshape(2, 128, 32)
    with torch.no_grad():
        model.codebooks.copy_(torch.from_numpy(initial).to(device))
        model.codebooks[1].mul_(0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for step in range(1, int(rq_config["steps"]) + 1):
        rows = rng.choice(len(train_shared), size=1024, replace=False)
        target = torch.from_numpy(train_shared[rows]).to(device)
        reconstructed, _ = model(target)
        loss = F.mse_loss(reconstructed, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(rq_config["steps"]):
            history.append({"step": step, "mse": float(loss.detach())})
            print(json.dumps({"rq_lr": learning_rate, **history[-1]}), flush=True)
    train_quantized, train_indices = quantize(model, train_shared, device)
    validation_quantized, validation_indices = quantize(model, validation_shared, device)
    metrics = {
        "learning_rate": learning_rate,
        "train_mse": float(np.square(train_quantized.astype(np.float64) - train_shared).mean()),
        "validation_mse": float(
            np.square(validation_quantized.astype(np.float64) - validation_shared).mean()
        ),
        "stage_utilization": [
            float(len(np.unique(validation_indices[..., stage])) / 128.0) for stage in range(2)
        ],
        "history": history,
    }
    return model, metrics, validation_quantized, validation_indices


@torch.inference_mode()
def decompose(
    model: SharedPrivateDecomposer,
    native: dict[str, np.ndarray],
    shared: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    private = {
        name: np.empty((len(next(iter(native.values()))), 8, 16), dtype=np.float32)
        for name in native
    }
    reconstruction = {name: np.empty_like(value) for name, value in native.items()}
    cross = {name: np.empty_like(value) for name, value in native.items()}
    for start in range(0, len(next(iter(native.values()))), 1024):
        stop = min(start + 1024, len(next(iter(native.values()))))
        native_t = {
            name: torch.from_numpy(value[start:stop]).to(device) for name, value in native.items()
        }
        shared_t = {
            name: torch.from_numpy(value[start:stop]).to(device) for name, value in shared.items()
        }
        for target in native:
            p = model.encode_private(target, native_t[target], shared_t[target])
            prediction = model.reconstruct(target, shared_t[target], p)
            counterparts = [shared_t[name] for name in shared_t if name != target]
            cross_prediction = torch.stack(
                [model.cross_predict(target, value) for value in counterparts]
            ).mean(0)
            private[target][start:stop] = p.float().cpu().numpy()
            reconstruction[target][start:stop] = prediction.float().cpu().numpy()
            cross[target][start:stop] = cross_prediction.float().cpu().numpy()
    return private, reconstruction, cross


def train_shared_private(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    shared_train: dict[str, np.ndarray],
    shared_validation: dict[str, np.ndarray],
    bridge: FormalVACBridge,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], SharedPrivateDecomposer]:
    seed_everything(int(config["seed"]))
    model = SharedPrivateDecomposer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(int(config["seed"]))
    native_train = {"vision": train["z_v"], "action": train["z_a"], "contact": train["z_c"]}
    native_validation = {
        "vision": validation["z_v"],
        "action": validation["z_a"],
        "contact": validation["z_c"],
    }
    history = []
    for step in range(1, 801):
        rows = rng.choice(len(train["pair_id"]), size=512, replace=False)
        native = {
            name: torch.from_numpy(value[rows]).to(device) for name, value in native_train.items()
        }
        shared = {
            name: torch.from_numpy(value[rows]).to(device) for name, value in shared_train.items()
        }
        reconstruction, cross, variance = [], [], []
        for target in native:
            private = model.encode_private(target, native[target], shared[target])
            reconstruction.append(
                F.mse_loss(model.reconstruct(target, shared[target], private), native[target])
            )
            counterparts = [shared[name] for name in shared if name != target]
            cross.extend(
                F.mse_loss(model.cross_predict(target, value), native[target])
                for value in counterparts
            )
            variance.append(
                F.relu(0.1 - torch.sqrt(private.flatten(1).var(0, unbiased=False) + 1e-4)).mean()
            )
        loss = (
            torch.stack(reconstruction).mean()
            + 0.5 * torch.stack(cross).mean()
            + 0.05 * torch.stack(variance).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == 800:
            history.append({"step": step, "loss": float(loss.detach())})
            print(json.dumps({"shared_private": history[-1]}), flush=True)
    private, reconstruction, cross = decompose(model, native_validation, shared_validation, device)
    shared_only = {
        name: (
            recover_contact(bridge, shared_validation[name], device) if name == "contact" else None
        )
        for name in native_validation
    }
    with torch.inference_mode():
        for name in ("vision", "action"):
            values = []
            for start in range(0, len(validation["pair_id"]), 1024):
                value = torch.from_numpy(shared_validation[name][start : start + 1024]).to(device)
                values.append(bridge.recover(name, value).float().cpu().numpy())
            shared_only[name] = np.concatenate(values)
    recovery = {}
    cross_metrics = {}
    permutation = different_episode_permutation(validation["episode_id"], seed=int(config["seed"]))
    for name in native_validation:
        shared_error = float(
            np.square(shared_only[name].astype(np.float64) - native_validation[name]).mean()
        )
        private_error = float(
            np.square(reconstruction[name].astype(np.float64) - native_validation[name]).mean()
        )
        correct = np.square(cross[name].astype(np.float64) - native_validation[name]).mean(
            axis=(1, 2)
        )
        shuffled = np.square(
            cross[name].astype(np.float64) - native_validation[name][permutation]
        ).mean(axis=(1, 2))
        recovery[name] = {
            "shared_only_mse": shared_error,
            "shared_plus_private_mse": private_error,
            "private_relative_improvement": (shared_error - private_error)
            / max(shared_error, 1e-12),
        }
        cross_metrics[name] = {
            "paired_mse": float(correct.mean()),
            "shuffled_target_mse": float(shuffled.mean()),
            "shuffled_minus_paired_margin": float((shuffled - correct).mean()),
        }
    minimum_increment = min(value["private_relative_improvement"] for value in recovery.values())
    minimum_margin = min(value["shuffled_minus_paired_margin"] for value in cross_metrics.values())
    gates_config = config["s4_2_6"]["gates"]
    gates = {
        "private_incremental_recovery": minimum_increment
        >= float(gates_config["private_incremental_recovery_min"]),
        "shared_cross_modal_margin": minimum_margin
        > float(gates_config["shared_cross_modal_margin_min"]),
        "finite": all(np.isfinite(value).all() for value in private.values()),
    }
    checkpoint = EXPERIMENT_ROOT / "shared_private.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2-6-shared-private.v1",
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "formal_test_loaded": False,
        },
        checkpoint,
    )
    result = {
        "schema": "tactile3d-unit.s4-2-6-shared-private-evaluation.v1",
        "recovery": recovery,
        "cross_prediction": cross_metrics,
        "minimum_private_relative_improvement": minimum_increment,
        "minimum_shared_cross_modal_margin": minimum_margin,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "training_history": history,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "formal_test_loaded": False,
    }
    return result, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config, protocol, prior = (
        load_json(CONFIG_PATH),
        load_json(PROTOCOL_PATH),
        load_json(S4_2_5_PATH),
    )
    if (
        protocol["config_sha256"] != sha256_file(CONFIG_PATH)
        or prior["decision"] != "S4_2_5_FORMAL_CONTINUOUS_BRIDGE_ACCEPTED"
    ):
        raise RuntimeError("S4.2-6 frozen dependency failure")
    train, validation = load_npz(PAIR_ROOT / "paired_train.npz"), load_npz(
        PAIR_ROOT / "paired_validation.npz"
    )
    shared_train_raw, shared_validation_raw = load_npz(PAIR_ROOT / "shared_train.npz"), load_npz(
        PAIR_ROOT / "shared_validation.npz"
    )
    if not np.array_equal(train["pair_id"], shared_train_raw["pair_id"]) or not np.array_equal(
        validation["pair_id"], shared_validation_raw["pair_id"]
    ):
        raise RuntimeError("S4.2-6 paired identity mismatch")
    shared_train = {
        "vision": shared_train_raw["u_v"],
        "action": shared_train_raw["u_a"],
        "contact": shared_train_raw["u_c"],
    }
    shared_validation = {
        "vision": shared_validation_raw["u_v"],
        "action": shared_validation_raw["u_a"],
        "contact": shared_validation_raw["u_c"],
    }
    device, bridge = torch.device(args.device), load_bridge(torch.device(args.device))
    rq_results, rq_models, quantized = [], [], []
    for learning_rate in config["s4_2_6"]["contact_rq"]["learning_rate_candidates"]:
        model, metrics, validation_quantized, validation_indices = train_rq(
            float(learning_rate),
            shared_train["contact"],
            shared_validation["contact"],
            config,
            device,
        )
        rq_results.append(metrics)
        rq_models.append(model)
        quantized.append((validation_quantized, validation_indices))
    selected_index = int(np.argmin([value["validation_mse"] for value in rq_results]))
    selected_rq, selected_metrics = rq_models[selected_index], rq_results[selected_index]
    train_quantized, train_indices = quantize(selected_rq, shared_train["contact"], device)
    validation_quantized, validation_indices = quantized[selected_index]
    continuous_native = recover_contact(bridge, shared_validation["contact"], device)
    quantized_native = recover_contact(bridge, validation_quantized, device)
    continuous_mse = float(
        np.square(continuous_native.astype(np.float64) - validation["z_c"]).mean()
    )
    quantized_mse = float(np.square(quantized_native.astype(np.float64) - validation["z_c"]).mean())
    continuous_contact = classification_probe(
        shared_train["contact"],
        shared_validation["contact"],
        train["contact_transition"],
        validation["contact_transition"],
        [0, 1, 2, 3],
    )
    quantized_contact = classification_probe(
        train_quantized,
        validation_quantized,
        train["contact_transition"],
        validation["contact_transition"],
        [0, 1, 2, 3],
    )
    continuous_force = classification_probe(
        shared_train["contact"],
        shared_validation["contact"],
        train["force_trend"],
        validation["force_trend"],
        [0, 1, 2],
    )
    quantized_force = classification_probe(
        train_quantized,
        validation_quantized,
        train["force_trend"],
        validation["force_trend"],
        [0, 1, 2],
    )
    gates_config = config["s4_2_6"]["gates"]
    ratios = {
        "native_recovery": quantized_mse / max(continuous_mse, 1e-12),
        "contact_semantic_retention": quantized_contact["macro_f1"]
        / max(continuous_contact["macro_f1"], 1e-12),
        "force_semantic_retention": quantized_force["macro_f1"]
        / max(continuous_force["macro_f1"], 1e-12),
    }
    rq_gates = {
        "recovery": ratios["native_recovery"]
        <= float(gates_config["quantized_recovery_mse_over_continuous_max"]),
        "contact_semantics": ratios["contact_semantic_retention"]
        >= float(gates_config["contact_semantic_retention_min"]),
        "force_semantics": ratios["force_semantic_retention"]
        >= float(gates_config["force_semantic_retention_min"]),
        "utilization": min(selected_metrics["stage_utilization"])
        >= float(gates_config["minimum_stage_utilization"]),
    }
    rq_checkpoint = EXPERIMENT_ROOT / "contact_rq_selected.pt"
    rq_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "tactile3d-unit.s4-2-6-contact-rq.v1",
            "state_dict": selected_rq.state_dict(),
            "learning_rate": selected_metrics["learning_rate"],
            "formal_test_loaded": False,
        },
        rq_checkpoint,
    )
    rq_result = {
        "schema": "tactile3d-unit.s4-2-6-contact-rq-evaluation.v1",
        "trials": rq_results,
        "selected_learning_rate": selected_metrics["learning_rate"],
        "continuous_native_recovery_mse": continuous_mse,
        "quantized_native_recovery_mse": quantized_mse,
        "ratios": ratios,
        "selected_stage_utilization": selected_metrics["stage_utilization"],
        "gates": {name: "PASS" if value else "FAIL" for name, value in rq_gates.items()},
        "overall": "PASS" if all(rq_gates.values()) else "FAIL",
        "checkpoint": str(rq_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(rq_checkpoint),
        "formal_test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "contact_rq_evaluation.json", rq_result)
    shared_private, _ = train_shared_private(
        train, validation, shared_train, shared_validation, bridge, config, device
    )
    atomic_json(ARTIFACT_ROOT / "shared_private_evaluation.json", shared_private)
    if shared_private["overall"] != "PASS":
        atomic_json(
            ARTIFACT_ROOT / "s4_2_6_final.json",
            {"decision": "S4_2_6_SHARED_PRIVATE_FAIL", "formal_test_loaded": False},
        )
        raise SystemExit("S4_2_6_SHARED_PRIVATE_FAIL")
    discrete = "VIABLE_AUXILIARY" if rq_result["overall"] == "PASS" else "REJECTED_RECOVERY_GATE"
    decision = (
        "S4_2_6_CONTINUOUS_PRIMARY_DISCRETE_AUXILIARY_SHARED_PRIVATE_ACCEPTED"
        if rq_result["overall"] == "PASS"
        else "S4_2_6_CONTINUOUS_SELECTED_DISCRETE_REJECTED_SHARED_PRIVATE_ACCEPTED"
    )
    final = {
        "schema": "tactile3d-unit.s4-2-6-final.v1",
        "decision": decision,
        "continuous_contact": "PRIMARY",
        "discrete_contact": discrete,
        "discrete_gates": rq_result["gates"],
        "shared_private": "PASS",
        "historical_10_percent_gate": "FAIL_UNCHANGED",
        "canonical_contact": "C3",
        "formal_test_loaded": False,
        "S4_2_7_readiness": "READY",
    }
    atomic_json(ARTIFACT_ROOT / "s4_2_6_final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
