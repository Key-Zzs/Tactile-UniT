#!/usr/bin/env python3
"""Locked C3-R0 conditional sufficiency and multimodality evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_teacher.evaluation import classification_metrics  # noqa: E402
from gr00t.tactile_unit.c3dp_shared_private import load_predictor_checkpoint  # noqa: E402
from gr00t.tactile_unit.c3r0_conditional_sufficiency import (  # noqa: E402
    CONTACT_BOUNDARY_CLASSES,
    TrainStandardizer,
    bootstrap_f1_difference,
    distribution_summary,
    evaluate_prediction,
    knn_target_predictions,
    load_ceiling_checkpoint,
    neighborhood_audit,
    normalized_label_entropy,
    probe_prediction,
    regression_geometry,
    root_cause_decision,
    semantic_ratio,
    sha256_file,
    source_features,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    different_episode_permutation,
    load_checkpoint,
    retrieval_metrics,
    state_dict_digest,
)
from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
)
from gr00t.tactile_unit.trex_action_transition import load_shared_transition_checkpoint  # noqa: E402
from scripts.tactile_unit.audit_c3r0_source_semantics import (  # noqa: E402
    legal_arrays,
    predict_ceiling,
)
from scripts.tactile_unit.c3r0_runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    atomic_json,
    identity_snapshot,
    load_aligned_split,
    load_config,
    validate_test_freeze,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.evaluate_c3dp_cross_prediction import predict_numpy  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument(
        "--unit-checkpoint",
        type=Path,
        default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None,
    )
    return parser.parse_args()


def clean_metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def majority(train_y, rows, classes):
    label = int(np.bincount(np.asarray(train_y, dtype=np.int64), minlength=classes).argmax())
    return np.full(rows, label, dtype=np.int64)


def semantic_metrics(probe, representation, train_y, target, classes, oracle):
    prediction = probe_prediction(probe, representation)
    result = evaluate_prediction(target, prediction, majority(train_y, len(target), classes), classes)
    result["semantic_ratio"] = semantic_ratio(
        result["macro_f1"], oracle["majority"]["macro_f1"], oracle["macro_f1"]
    )
    result["_prediction"] = prediction
    return result


def boundary_not_catastrophic(candidate, oracle):
    candidate_mean = np.mean([candidate["free_to_contact"]["f1"], candidate["contact_to_free"]["f1"]])
    oracle_mean = np.mean([oracle["free_to_contact"]["f1"], oracle["contact_to_free"]["f1"]])
    return bool(candidate_mean >= 0.25 * oracle_mean)


def paired_cosine_control(prediction, target, seed):
    left = np.asarray(prediction).reshape(len(prediction), -1).astype(np.float64)
    right = np.asarray(target).reshape(len(target), -1).astype(np.float64)
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    shuffled = np.random.default_rng(seed).permutation(len(right))
    paired = np.sum(left * right, axis=1)
    control = np.sum(left * right[shuffled], axis=1)
    return {
        "paired": float(paired.mean()), "shuffled": float(control.mean()),
        "margin": float((paired - control).mean()),
    }


def representation_evaluation(prediction, target, retrieval_rows, retrieval_chunk, seed):
    result = regression_geometry(prediction, target)
    result["paired_vs_shuffled_cosine"] = paired_cosine_control(prediction, target, seed)
    rows = min(int(retrieval_rows), len(target))
    result["retrieval"] = {
        "protocol": f"first {rows} identity-ordered locked rows; pre-registered label-free subset",
        **retrieval_metrics(np.asarray(prediction)[:rows], np.asarray(target)[:rows], chunk=int(retrieval_chunk)),
    }
    return result


def load_probe_records(protocol, source):
    return {
        metric: joblib.load(ROOT / protocol["records"][source][metric]["path"])
        for metric in ("contact_transition", "force_trend_class")
    }


def direct_source_audit(config, protocol, train, test):
    probes = {}
    predictions = {}
    for source in (*config["sources"], *config["references"]):
        probes[source] = load_probe_records(protocol, source)
        features = source_features(source, legal_arrays(test))
        predictions[source] = {}
        for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
            prediction = probe_prediction(probes[source][metric], features)
            predictions[source][metric] = prediction
    results = {}
    for source in (*config["sources"], *config["references"]):
        results[source] = {}
        for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
            results[source][metric] = evaluate_prediction(
                test[metric], predictions[source][metric], majority(train[metric], len(test[metric]), classes), classes
            )
    for source in config["sources"]:
        for metric in ("contact_transition", "force_trend_class"):
            results[source][metric]["semantic_ratio"] = semantic_ratio(
                results[source][metric]["macro_f1"],
                results["C"][metric]["majority"]["macro_f1"],
                results["C"][metric]["macro_f1"],
            )
        results[source]["sufficient"] = bool(
            results[source]["contact_transition"]["semantic_ratio"] >= float(config["probe"]["sufficiency_min"])
            and results[source]["force_trend_class"]["semantic_ratio"] >= float(config["probe"]["sufficiency_min"])
            and boundary_not_catastrophic(results[source]["contact_transition"], results["C"]["contact_transition"])
        )
    return probes, predictions, results


def complementarity(config, test, predictions, results):
    pairs = (
        ("VA", "V"), ("VA", "A"), ("VH", "V"), ("VH", "H"),
        ("AH", "A"), ("AH", "H"), ("VAH", "VA"), ("VAH", "VH"), ("VAH", "AH"),
    )
    output = {}
    samples = int(config["evaluation"]["bootstrap_samples"])
    for offset, (left, right) in enumerate(pairs):
        key = f"{left}_vs_{right}"
        output[key] = {}
        for metric in ("contact_transition", "force_trend_class"):
            output[key][metric] = {
                "delta_f1": float(results[left][metric]["macro_f1"] - results[right][metric]["macro_f1"]),
                "delta_semantic_ratio": float(results[left][metric]["semantic_ratio"] - results[right][metric]["semantic_ratio"]),
                "ci95": bootstrap_f1_difference(
                    test[metric], predictions[left][metric], predictions[right][metric],
                    samples=samples, seed=int(config["seed"]) + offset * 10 + (metric == "force_trend_class"),
                ),
            }
    va_best = max(results["V"]["contact_transition"]["macro_f1"], results["A"]["contact_transition"]["macro_f1"])
    va_positive = min(output["VA_vs_V"]["contact_transition"]["ci95"][0], output["VA_vs_A"]["contact_transition"]["ci95"][0]) > 0
    output["MULTISOURCE_COMPLEMENTARITY"] = bool(
        results["VA"]["sufficient"] and not results["V"]["sufficient"] and not results["A"]["sufficient"]
        and results["VA"]["contact_transition"]["macro_f1"] > va_best and va_positive
    )
    vah_gain = output["VAH_vs_VA"]["contact_transition"]["ci95"][0] > 0
    output["CAUSAL_CONTEXT_COMPLEMENTARITY"] = bool(results["VAH"]["sufficient"] and vah_gain)
    return output


def load_test_embeddings(knn_protocol, test):
    atomic = {"V": "u_v", "A": "u_a", "H": "h_current", "C": "u_c"}
    embeddings = {}
    for name, key in atomic.items():
        record = knn_protocol["pca"][name]
        if sha256_file(ROOT / record["standardizer"]) != record["standardizer_sha256"] or sha256_file(ROOT / record["pca"]) != record["pca_sha256"]:
            raise RuntimeError("STRUCTURAL_FAIL: frozen kNN projection changed")
        standardizer = TrainStandardizer.load(ROOT / record["standardizer"])
        pca = joblib.load(ROOT / record["pca"])
        embeddings[name] = pca.transform(standardizer.transform(test[key])).astype(np.float32)
    components = {
        "V": ("V",), "A": ("A",), "VA": ("V", "A"), "H": ("H",),
        "VH": ("V", "H"), "AH": ("A", "H"), "VAH": ("V", "A", "H"), "C": ("C",),
    }
    return {source: np.concatenate([embeddings[name] for name in names], axis=1) for source, names in components.items()}


def local_rank_summary(neighbor_targets, seed):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(neighbor_targets), size=min(512, len(neighbor_targets)), replace=False)
    ranks = []
    for value in np.asarray(neighbor_targets)[rows]:
        flat = value.reshape(len(value), -1).astype(np.float64)
        flat -= flat.mean(0, keepdims=True)
        singular = np.linalg.svd(flat, full_matrices=False, compute_uv=False)
        probability = np.square(singular)
        probability /= max(float(probability.sum()), 1e-12)
        ranks.append(float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))))
    return distribution_summary(np.asarray(ranks))


def neighborhood_and_nonparametric(config, knn_protocol, train, test, oracle, artifact_root):
    test_embeddings = load_test_embeddings(knn_protocol, test)
    max_k = max(map(int, config["knn"]["k"]))
    global_variance = float(np.var(np.asarray(train["u_c"], dtype=np.float64)))
    neighborhoods = {}
    ceilings = {}
    rng = np.random.default_rng(int(config["seed"]) + 900)
    for offset, source in enumerate((*config["sources"], "C")):
        print(f"[c3r0] querying locked train-reference neighborhoods: {source}", flush=True)
        record = knn_protocol["indices"][source]
        path = ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("STRUCTURAL_FAIL: frozen kNN index changed")
        index = joblib.load(path)
        indices, distances = index.query(test_embeddings[source], k=max_k, epsilon=float(config["knn"]["query_epsilon"]))
        shuffled_indices = rng.integers(0, len(train["u_c"]), size=indices.shape)
        neighborhoods[source] = {"k": {}, "shuffled_control": {}}
        for k in config["knn"]["k"]:
            audit = neighborhood_audit(
                indices, train["contact_transition"], test["contact_transition"], train["u_c"],
                k=int(k), global_variance=global_variance,
            )
            entropy = normalized_label_entropy(np.asarray(train["contact_transition"])[indices[:, : int(k)]], 4)
            audit["dynamic"] = {
                "count": int(np.asarray(test["dynamic"], dtype=bool).sum()),
                "normalized_entropy": distribution_summary(entropy[np.asarray(test["dynamic"], dtype=bool)]),
            }
            audit["class_conditional_entropy"] = {
                str(label): distribution_summary(entropy[np.asarray(test["contact_transition"]) == label])
                for label in range(4)
            }
            audit["same_primitive_neighbor_fraction"] = float(
                np.mean(np.asarray(train["primitive_id"])[indices[:, : int(k)]] == np.asarray(test["primitive_id"])[:, None])
            )
            audit["same_object_neighbor_fraction"] = float(
                np.mean(np.asarray(train["object_id"])[indices[:, : int(k)]] == np.asarray(test["object_id"])[:, None])
            )
            if int(k) > 1:
                audit["local_effective_rank"] = local_rank_summary(
                    np.asarray(train["u_c"])[indices[:, : int(k)]], int(config["seed"]) + offset + int(k)
                )
            neighborhoods[source]["k"][str(k)] = audit
            neighborhoods[source]["shuffled_control"][str(k)] = neighborhood_audit(
                shuffled_indices, train["contact_transition"], test["contact_transition"], train["u_c"],
                k=int(k), global_variance=global_variance,
            )
        neighborhoods[source]["distance"] = distribution_summary(distances)
        if source == "C":
            continue
        train_indices = np.load(ROOT / knn_protocol["nonparametric_records"][source]["train_indices"], allow_pickle=False)
        ceilings[source] = {}
        for variant, variant_record in knn_protocol["nonparametric_records"][source]["variants"].items():
            k = int(variant_record["k"])
            test_prediction = knn_target_predictions(indices, train["u_c"], k)[variant]
            result = {
                "k": k,
                "representation": representation_evaluation(
                    test_prediction, test["u_c"], config["evaluation"]["retrieval_rows"],
                    config["evaluation"]["retrieval_chunk"], int(config["seed"]) + offset * 20 + k,
                ),
            }
            for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
                probe_record = variant_record["probes"][metric]
                if sha256_file(ROOT / probe_record["path"]) != probe_record["sha256"]:
                    raise RuntimeError("STRUCTURAL_FAIL: frozen nonparametric probe changed")
                probe = joblib.load(ROOT / probe_record["path"])
                result[metric] = semantic_metrics(
                    probe, test_prediction, train[metric], test[metric], classes, oracle[metric]
                )
                result[metric] = clean_metrics(result[metric])
            ceilings[source][variant] = result
        cache_path = ROOT / config["runtime"]["cache_root"] / "test" / f"{source}_indices.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            np.save(handle, indices.astype(np.int32), allow_pickle=False)
    return neighborhoods, ceilings


def p2_audit(config, probe_protocol, train, test, oracle, device):
    predictor, metadata = load_predictor_checkpoint(ROOT / config["runtime"]["c3dp_checkpoint"], device)
    predictor.eval().requires_grad_(False).to(device)
    output = {}
    predictions = {}
    for offset, (source, source_name, key) in enumerate((("V", "vision", "u_v"), ("A", "action", "u_a"))):
        value = predict_numpy(predictor, np.asarray(test[key]), source_name, "contact", device, 512)
        predictions[source] = value
        result = {
            "representation": representation_evaluation(
                value, test["u_c"], config["evaluation"]["retrieval_rows"],
                config["evaluation"]["retrieval_chunk"], int(config["seed"]) + 1000 + offset,
            )
        }
        for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
            record = probe_protocol["p2_records"][source]["probes"][metric]
            if sha256_file(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError("STRUCTURAL_FAIL: frozen P2 probe changed")
            probe = joblib.load(ROOT / record["path"])
            result[metric] = clean_metrics(semantic_metrics(
                probe, value, train[metric], test[metric], classes, oracle[metric]
            ))
        output[source] = result
    return output, predictions, metadata


def contact_physics(shared_space, s2, split, prediction, device, batch_size):
    errors = np.empty(len(prediction), dtype=np.float64)
    shared_space.eval().requires_grad_(False)
    s2.eval().requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(prediction), batch_size):
            stop = min(start + batch_size, len(prediction))
            shared = torch.from_numpy(np.asarray(prediction[start:stop], dtype=np.float32)).to(device)
            native = shared_space.recover("contact", shared)
            current = torch.from_numpy(np.array(split["h_current"][start:stop], copy=True)).to(device)
            future = torch.from_numpy(np.array(split["h_future"][start:stop], copy=True)).to(device)
            errors[start:stop] = torch.square(s2.decoder(native, current) - future).mean(1).double().cpu().numpy()
    dynamic = np.asarray(split["dynamic"], dtype=bool)
    return {"future_mse": float(errors.mean()), "dynamic_mse": float(errors[dynamic].mean())}


def deterministic_ceilings(config, ceiling_protocol, train, test, oracle, shared_space, s2, device, batch_size):
    output = {}
    predictions = {}
    for offset, source in enumerate(("VA", "VAH")):
        selected = ceiling_protocol["selected"][source]
        if sha256_file(ROOT / selected["checkpoint"]) != selected["checkpoint_sha256"]:
            raise RuntimeError("STRUCTURAL_FAIL: frozen deterministic ceiling changed")
        if sha256_file(ROOT / selected["standardizer"]) != selected["standardizer_sha256"]:
            raise RuntimeError("STRUCTURAL_FAIL: frozen deterministic standardizer changed")
        model, metadata = load_ceiling_checkpoint(ROOT / selected["checkpoint"], device)
        standardizer = TrainStandardizer.load(ROOT / selected["standardizer"])
        features = standardizer.transform(source_features(source, legal_arrays(test)))
        value = predict_ceiling(model, features, device, batch_size)
        predictions[source] = value
        result = {
            "architecture": model.architecture, "parameters": model.parameter_count(), "metadata": metadata,
            "representation": representation_evaluation(
                value, test["u_c"], config["evaluation"]["retrieval_rows"],
                config["evaluation"]["retrieval_chunk"], int(config["seed"]) + 1100 + offset,
            ),
            "contact_physics": contact_physics(shared_space, s2, test, value, device, batch_size),
        }
        for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
            record = selected["probes"][metric]
            if sha256_file(ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError("STRUCTURAL_FAIL: deterministic ceiling probe changed")
            probe = joblib.load(ROOT / record["path"])
            result[metric] = clean_metrics(semantic_metrics(
                probe, value, train[metric], test[metric], classes, oracle[metric]
            ))
        result["gate"] = bool(
            result["contact_transition"]["semantic_ratio"] >= float(config["evaluation"]["contact_cross_retention_min"])
            and result["force_trend_class"]["semantic_ratio"] >= float(config["evaluation"]["force_cross_retention_min"])
            and boundary_not_catastrophic(result["contact_transition"], oracle["contact_transition"])
        )
        output[source] = result
    return output, predictions


def reversed_action_shared(config, test, shared_space, unit_checkpoint, device, batch_size):
    if unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required for Action temporal audit")
    source = ReleasedTokenizerSource.open(unit_checkpoint / "tokenizer")
    action_model, metadata = load_shared_transition_checkpoint(
        ROOT / config["runtime"]["action_checkpoint"], source, device
    )
    action_model.eval().requires_grad_(False).to(device)
    result = np.empty((len(test["u_a"]), 8, 32), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(result), batch_size):
            stop = min(start + batch_size, len(result))
            state = torch.from_numpy(np.array(test["state"][start:stop], copy=True)).to(device)
            action = torch.from_numpy(np.array(test["action"][start:stop], copy=True)).to(device)
            embodiment = torch.full((stop - start,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
            native, _, _ = action_model.encode(state, action.flip(1), embodiment)
            result[start:stop] = shared_space.encode("action", native).float().cpu().numpy()
    return result, metadata


def action_temporal_audit(config, direct_probes, train, test, direct_results, ceilings, unit_checkpoint, shared_space, device, batch_size):
    reversed_a, metadata = reversed_action_shared(config, test, shared_space, unit_checkpoint, device, batch_size)
    rng = np.random.default_rng(int(config["seed"]) + 1200)
    shuffled_a = np.asarray(test["u_a"])[rng.permutation(len(test["u_a"]))]
    different_a = np.asarray(test["u_a"])[different_episode_permutation(np.asarray(test["episode_id"]), int(config["seed"]) + 1201)]
    variants = {"correct": np.asarray(test["u_a"]), "reversed": reversed_a, "shuffled": shuffled_a, "different_episode": different_a}
    output = {"action_checkpoint_metadata": metadata, "direct_A": {}, "VA_ceiling": {}}
    for name, value in variants.items():
        output["direct_A"][name] = {}
        for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
            prediction = probe_prediction(direct_probes["A"][metric], value)
            output["direct_A"][name][metric] = evaluate_prediction(
                test[metric], prediction, majority(train[metric], len(test[metric]), classes), classes
            )
        selected = ceilings["VA"]["selection"]
        standardizer = TrainStandardizer.load(ROOT / selected["standardizer"])
        model, _ = load_ceiling_checkpoint(ROOT / selected["checkpoint"], device)
        features = standardizer.transform(np.concatenate([np.asarray(test["u_v"]).reshape(len(value), -1), value.reshape(len(value), -1)], axis=1))
        contact = predict_ceiling(model, features, device, batch_size)
        output["VA_ceiling"][name] = {}
        for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
            record = selected["probes"][metric]
            probe = joblib.load(ROOT / record["path"])
            prediction = probe_prediction(probe, contact)
            output["VA_ceiling"][name][metric] = evaluate_prediction(
                test[metric], prediction, majority(train[metric], len(test[metric]), classes), classes
            )
    correct = output["direct_A"]["correct"]["contact_transition"]["macro_f1"]
    reversed_score = output["direct_A"]["reversed"]["contact_transition"]["macro_f1"]
    output["diagnosis"] = (
        "ACTION_TEMPORAL_ORDER_NOT_NEEDED_FOR_CONTACT_TARGET"
        if reversed_score >= correct - 0.01
        else "ACTION_TEMPORAL_SIGNAL_LOST_BY_C3_SOURCE_USE"
    )
    return output


def current_context_confound(test, predictions, direct_results):
    target = np.asarray(test["contact_transition"], dtype=np.int64)
    current_target = np.isin(target, [2, 3]).astype(np.int64)
    change_target = np.isin(target, CONTACT_BOUNDARY_CLASSES).astype(np.int64)
    result = {}
    for source in ("H", "VH", "AH", "VAH"):
        prediction = predictions[source]["contact_transition"]
        result[source] = {
            "current_state": classification_metrics(current_target, np.isin(prediction, [2, 3]).astype(np.int64)),
            "future_change": classification_metrics(change_target, np.isin(prediction, CONTACT_BOUNDARY_CLASSES).astype(np.int64)),
            "free_to_contact": direct_results[source]["contact_transition"]["free_to_contact"],
            "contact_to_free": direct_results[source]["contact_transition"]["contact_to_free"],
        }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.bootstrap_samples is not None:
        config["evaluation"]["bootstrap_samples"] = int(args.bootstrap_samples)
    if args.device == "cuda:0":
        raise RuntimeError("use the repository GPU lock helper before enabling C3-R0 CUDA")
    device = torch.device(args.device)
    set_seed(int(config["seed"]))
    artifact_root = ROOT / config["runtime"]["artifact_root"]

    # This validation must remain before the first locked benchmark load.
    selection = validate_test_freeze(config)
    probe_protocol = json.loads((artifact_root / "probe_selection.json").read_text())
    knn_protocol = json.loads((artifact_root / "knn_protocol.json").read_text())
    ceiling_protocol = json.loads((artifact_root / "deterministic_ceiling_selection.json").read_text())
    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity mismatch before locked test")

    train = load_aligned_split(config, "train")
    test = load_aligned_split(config, "test")
    if len(test["u_v"]) != int(config["counts"]["test"]):
        raise RuntimeError("STRUCTURAL_FAIL: locked benchmark row count changed")
    print("[c3r0] locked benchmark loaded after protocol verification", flush=True)

    direct_probes, direct_predictions, direct_results = direct_source_audit(
        config, probe_protocol, train, test
    )
    oracle = {metric: direct_results["C"][metric] for metric in ("contact_transition", "force_trend_class")}
    complementarity_result = complementarity(config, test, direct_predictions, direct_results)
    confound = current_context_confound(test, direct_predictions, direct_results)
    neighborhoods, nonparametric = neighborhood_and_nonparametric(
        config, knn_protocol, train, test, oracle, artifact_root
    )
    p2, p2_predictions, p2_metadata = p2_audit(
        config, probe_protocol, train, test, oracle, device
    )
    shared_space, shared_metadata = load_checkpoint(ROOT / config["runtime"]["c2r_checkpoint"], device)
    shared_space.eval().requires_grad_(False).to(device)
    if state_dict_digest(shared_space) != config["accepted"]["shared_state_sha256"]:
        raise RuntimeError("STRUCTURAL_FAIL: shared-space state changed")
    s2 = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], device).eval().requires_grad_(False)
    deterministic, deterministic_predictions = deterministic_ceilings(
        config, ceiling_protocol, train, test, oracle, shared_space, s2, device, args.batch_size
    )
    # Preserve the frozen selection record next to test results for temporal perturbations.
    for source in deterministic:
        deterministic[source]["selection"] = ceiling_protocol["selected"][source]
    action_temporal = action_temporal_audit(
        config, direct_probes, train, test, direct_results, deterministic,
        args.unit_checkpoint, shared_space, device, args.batch_size,
    )

    gaps = {}
    for offset, source in enumerate(("V", "A")):
        gaps[source] = {}
        for metric in ("contact_transition", "force_trend_class"):
            direct_prediction = direct_predictions[source][metric]
            p2_record = probe_protocol["p2_records"][source]["probes"][metric]
            p2_probe = joblib.load(ROOT / p2_record["path"])
            p2_prediction = probe_prediction(p2_probe, p2_predictions[source])
            gaps[source][metric] = {
                "direct_f1": direct_results[source][metric]["macro_f1"],
                "p2_f1": p2[source][metric]["macro_f1"],
                "gap": float(direct_results[source][metric]["macro_f1"] - p2[source][metric]["macro_f1"]),
                "ci95": bootstrap_f1_difference(
                    test[metric], direct_prediction, p2_prediction,
                    samples=int(config["evaluation"]["bootstrap_samples"]),
                    seed=int(config["seed"]) + 1300 + offset * 10 + (metric == "force_trend_class"),
                ),
            }

    identities_after = identity_snapshot(config)
    structural_pass = bool(
        identities_after["pass"] and identities_after["actual"] == identities_before["actual"]
        and state_dict_digest(shared_space) == config["accepted"]["shared_state_sha256"]
    )
    # Predictor-objective evidence must come from the same single-source V/A
    # contract tested by P2.  A strong VAH ceiling is evidence for causal
    # context, not evidence that a V-only or A-only predictor averaged modes.
    best_nonparam_contact = max(
        nonparametric[source][variant]["contact_transition"]["semantic_ratio"]
        for source in ("V", "A") for variant in ("1nn", "medoid", "mean")
    )
    strongest_entropy = neighborhoods["VAH"]["k"]["10"]["normalized_label_entropy"]["mean"]
    oracle_entropy = neighborhoods["C"]["k"]["10"]["normalized_label_entropy"]["mean"]
    mean_rank = nonparametric["VAH"]["mean"]["representation"]["geometry"]["effective_rank"]
    medoid_rank = nonparametric["VAH"]["medoid"]["representation"]["geometry"]["effective_rank"]
    evidence = {
        "structural_pass": structural_pass,
        "single_source_sufficient": bool(direct_results["V"]["sufficient"] or direct_results["A"]["sufficient"]),
        "predictor_gap": bool(gaps["V"]["contact_transition"]["ci95"][0] > 0 or gaps["A"]["contact_transition"]["ci95"][0] > 0),
        "nonparametric_strong": best_nonparam_contact >= 0.75,
        "va_sufficient": bool(direct_results["VA"]["sufficient"] and deterministic["VA"]["gate"]),
        "vah_sufficient": bool(direct_results["VAH"]["sufficient"] and deterministic["VAH"]["gate"] and confound["VAH"]["future_change"]["macro_f1"] > 0.5),
        "multimodality": bool(
            not deterministic["VAH"]["gate"] and strongest_entropy > oracle_entropy * 1.25
            and nonparametric["VAH"]["medoid"]["contact_transition"]["macro_f1"]
            > nonparametric["VAH"]["mean"]["contact_transition"]["macro_f1"]
            and medoid_rank > mean_rank
        ),
        "direct_high_target_low": bool(direct_results["VAH"]["sufficient"] and not deterministic["VAH"]["gate"]),
    }
    primary, next_stage = root_cause_decision(evidence)
    root_decision = {
        "schema": "tactile3d-unit.vac-c3r0-root-cause.v1", "primary": primary,
        "secondary": ["PREDICTOR_OBJECTIVE_BOTTLENECK"] if evidence["predictor_gap"] and primary != "PREDICTOR_OBJECTIVE_BOTTLENECK" else [],
        "next_stage": next_stage, "evidence": evidence,
        "private_residual": "PRIVATE_RESIDUAL_LARGELY_PRIVATE", "private_changed": False,
        "C4": "NOT READY", "C5": "NOT STARTED", "C6_M3": "NOT STARTED", "M3": "NOT ESTABLISHED",
    }

    atomic_json(artifact_root / "direct_source_probes.json", {
        "schema": "tactile3d-unit.vac-c3r0-direct-source-test.v1", "rows": len(test["u_v"]),
        "results": direct_results,
    })
    atomic_json(artifact_root / "source_complementarity.json", complementarity_result)
    atomic_json(artifact_root / "knn_ambiguity.json", {
        "schema": "tactile3d-unit.vac-c3r0-knn-ambiguity-test.v1", "rows": len(test["u_v"]),
        "estimator_caveat": "empirical neighborhood conditional ambiguity; not exact information-theoretic entropy",
        "sources": neighborhoods,
    })
    atomic_json(artifact_root / "nonparametric_ceiling.json", {
        "schema": "tactile3d-unit.vac-c3r0-nonparametric-test.v1", "rows": len(test["u_v"]),
        "results": nonparametric,
    })
    atomic_json(artifact_root / "multisource_ceiling_training.json", {
        "schema": "tactile3d-unit.vac-c3r0-multisource-test.v1", "selection": ceiling_protocol["selected"],
        "results": deterministic,
    })
    atomic_json(artifact_root / "root_cause_decision.json", root_decision)
    locked = {
        "schema": "tactile3d-unit.vac-c3r0-locked-evaluation.v1",
        "evaluation": "POST-C3 DIAGNOSTIC AUDIT", "first_look_untouched": False,
        "rows": len(test["u_v"]), "selection": selection,
        "test_loaded_before_freeze": False, "direct_source": direct_results,
        "complementarity": complementarity_result, "current_context_confound": confound,
        "neighborhoods": neighborhoods, "nonparametric": nonparametric,
        "p2": p2, "p2_metadata": p2_metadata,
        "deterministic": deterministic, "action_temporal": action_temporal,
        "direct_vs_p2_gap": gaps, "root_cause": root_decision,
        "private_residual": json.loads((ROOT / ".local/artifacts/tactile_unit/vac_c3dp/locked_test_evaluation.json").read_text())["private_residual"],
        "frozen_shared_metadata": shared_metadata,
        "identities_before": identities_before, "identities_after": identities_after,
        "frozen_identities_unchanged": structural_pass,
        "causal_boundary": {
            "h_t_c_current_causal": True, "u_v_offline_future_derived": True,
            "u_a_offline_planned_transition": True, "u_c_offline_future_target": True,
            "future_contact_input": False, "C5_started": False,
        },
        "scope": config["scope"],
    }
    atomic_json(artifact_root / "locked_test_evaluation.json", locked)
    print(json.dumps({
        "status": "C3R0_COMPLETE" if structural_pass else "C3R0_STRUCTURAL_FAIL",
        "primary": primary, "next_stage": next_stage,
        "VA_gate": deterministic["VA"]["gate"], "VAH_gate": deterministic["VAH"]["gate"],
        "frozen_identities_unchanged": structural_pass,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
