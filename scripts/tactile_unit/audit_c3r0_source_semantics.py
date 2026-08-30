#!/usr/bin/env python3
"""Freeze every C3-R0 diagnostic choice using train and validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3dp_shared_private import load_predictor_checkpoint  # noqa: E402
from gr00t.tactile_unit.c3r0_conditional_sufficiency import (  # noqa: E402
    SOURCE_COMPONENTS,
    SmallContactCeiling,
    TrainStandardizer,
    evaluate_prediction,
    fit_probe,
    knn_target_predictions,
    neighborhood_audit,
    probe_prediction,
    save_ceiling_checkpoint,
    semantic_ratio,
    sha256_file,
    source_features,
)
from scripts.tactile_unit.c3r0_runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    ROOT,
    atomic_json,
    identity_snapshot,
    load_aligned_split,
    load_config,
    write_freeze_files,
)
from scripts.tactile_unit.evaluate_c3dp_cross_prediction import predict_numpy  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def dump_joblib(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)
    return sha256_file(path)


def save_array(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)
    return sha256_file(path)


def legal_arrays(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: split[name] for name in ("u_v", "u_a", "u_c", "z_c", "h_current")}


def source_probe_freeze(config, train, validation, experiment_root, artifact_root):
    results: dict[str, Any] = {}
    records: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for source in (*config["sources"], *config["references"]):
        print(f"[c3r0] fitting fixed direct probes: {source}", flush=True)
        train_x = source_features(source, legal_arrays(train))
        validation_x = source_features(source, legal_arrays(validation))
        records[source] = {}
        results[source] = {}
        predictions[source] = {}
        for metric, classes in (
            ("contact_transition", int(config["probe"]["contact_classes"])),
            ("force_trend_class", int(config["probe"]["force_classes"])),
        ):
            model = fit_probe(train_x, train[metric], alpha=float(config["probe"]["alpha"]))
            path = experiment_root / "probes" / f"{source}_{metric}.joblib"
            digest = dump_joblib(path, model)
            prediction = probe_prediction(model, validation_x)
            majority_class = int(np.bincount(np.asarray(train[metric], dtype=np.int64), minlength=classes).argmax())
            majority = np.full(len(prediction), majority_class, dtype=np.int64)
            results[source][metric] = evaluate_prediction(
                validation[metric], prediction, majority, classes
            )
            predictions[source][metric] = prediction
            records[source][metric] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "alpha": float(config["probe"]["alpha"]),
                "fit_split": "train only",
            }
    for source in config["sources"]:
        for metric in ("contact_transition", "force_trend_class"):
            oracle = results["C"][metric]
            results[source][metric]["semantic_ratio"] = semantic_ratio(
                results[source][metric]["macro_f1"],
                oracle["majority"]["macro_f1"],
                oracle["macro_f1"],
            )
    atomic_json(artifact_root / "direct_source_probes.validation.json", {
        "schema": "tactile3d-unit.vac-c3r0-direct-source-validation.v1",
        "selection_split": "validation only",
        "test_loaded": False,
        "results": results,
    })
    return records, results, predictions


def fit_pca_and_indices(config, train, validation, experiment_root, cache_root, artifact_root, oracle_probes):
    from pynndescent import NNDescent
    from sklearn.decomposition import PCA

    dimension = int(config["knn"]["pca_dimension_candidates"][0])
    atomic_sources = {"V": "u_v", "A": "u_a", "H": "h_current", "C": "u_c"}
    embeddings: dict[str, dict[str, np.ndarray]] = {"train": {}, "validation": {}}
    pca_records: dict[str, Any] = {}
    for offset, (name, key) in enumerate(atomic_sources.items()):
        print(f"[c3r0] fitting train-only PCA: {name}", flush=True)
        standardizer = TrainStandardizer.fit(train[key])
        standardizer_path = experiment_root / "knn" / f"{name}_standardizer.npz"
        standardizer.save(standardizer_path)
        pca = PCA(n_components=dimension, svd_solver="randomized", random_state=int(config["seed"]) + offset)
        train_standard = standardizer.transform(train[key])
        train_embedding = pca.fit_transform(train_standard).astype(np.float32)
        validation_embedding = pca.transform(standardizer.transform(validation[key])).astype(np.float32)
        pca_path = experiment_root / "knn" / f"{name}_pca.joblib"
        pca_digest = dump_joblib(pca_path, pca)
        embeddings["train"][name] = train_embedding
        embeddings["validation"][name] = validation_embedding
        pca_records[name] = {
            "standardizer": standardizer_path.relative_to(ROOT).as_posix(),
            "standardizer_sha256": sha256_file(standardizer_path),
            "pca": pca_path.relative_to(ROOT).as_posix(),
            "pca_sha256": pca_digest,
            "dimension": dimension,
            "fit_split": "train only",
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        }

    projected_components = {
        "V": ("V",), "A": ("A",), "VA": ("V", "A"), "H": ("H",),
        "VH": ("V", "H"), "AH": ("A", "H"), "VAH": ("V", "A", "H"), "C": ("C",),
    }
    global_variance = float(np.var(np.asarray(train["u_c"], dtype=np.float64)))
    index_records: dict[str, Any] = {}
    validation_audit: dict[str, Any] = {}
    selected_k: dict[str, Any] = {}
    max_k = max(map(int, config["knn"]["k"]))
    for offset, source in enumerate((*config["sources"], "C")):
        print(f"[c3r0] building train-reference neighborhood index: {source}", flush=True)
        train_embedding = np.concatenate(
            [embeddings["train"][name] for name in projected_components[source]], axis=1
        )
        validation_embedding = np.concatenate(
            [embeddings["validation"][name] for name in projected_components[source]], axis=1
        )
        index = NNDescent(
            train_embedding,
            n_neighbors=int(config["knn"]["index_neighbors"]),
            metric="euclidean",
            random_state=int(config["seed"]) + 100 + offset,
            n_jobs=-1,
            low_memory=True,
        )
        index.prepare()
        indices, distances = index.query(
            validation_embedding, k=max_k, epsilon=float(config["knn"]["query_epsilon"])
        )
        index_path = experiment_root / "knn" / f"{source}_index.joblib"
        index_digest = dump_joblib(index_path, index)
        indices_path = cache_root / "validation" / f"{source}_indices.npy"
        distances_path = cache_root / "validation" / f"{source}_distances.npy"
        save_array(indices_path, indices.astype(np.int32))
        save_array(distances_path, distances.astype(np.float32))
        validation_audit[source] = {
            str(k): neighborhood_audit(
                indices,
                train["contact_transition"],
                validation["contact_transition"],
                train["u_c"],
                k=int(k),
                global_variance=global_variance,
            )
            for k in config["knn"]["k"]
        }
        index_records[source] = {
            "path": index_path.relative_to(ROOT).as_posix(),
            "sha256": index_digest,
            "reference_split": "train only",
            "rows": len(train_embedding),
            "dimension": train_embedding.shape[1],
            "validation_indices": indices_path.relative_to(ROOT).as_posix(),
            "validation_indices_sha256": sha256_file(indices_path),
            "validation_distances_sha256": sha256_file(distances_path),
        }
        if source == "C":
            continue
        selected_k[source] = {}
        for mode in ("medoid", "mean"):
            candidates = {}
            for k in config["knn"]["mode_k_candidates"]:
                prediction = knn_target_predictions(indices, train["u_c"], int(k))[mode]
                contact_prediction = probe_prediction(oracle_probes["contact_transition"], prediction)
                force_prediction = probe_prediction(oracle_probes["force_trend_class"], prediction)
                contact = evaluate_prediction(
                    validation["contact_transition"], contact_prediction,
                    np.full(len(validation["contact_transition"]), int(np.bincount(np.asarray(train["contact_transition"], dtype=np.int64)).argmax())), 4,
                )
                force = evaluate_prediction(
                    validation["force_trend_class"], force_prediction,
                    np.full(len(validation["force_trend_class"]), int(np.bincount(np.asarray(train["force_trend_class"], dtype=np.int64)).argmax())), 3,
                )
                candidates[str(k)] = {
                    "contact_macro_f1": contact["macro_f1"],
                    "force_macro_f1": force["macro_f1"],
                    "utility": float(contact["macro_f1"] + force["macro_f1"]),
                }
            winner = max(candidates, key=lambda key: (candidates[key]["utility"], -int(key)))
            selected_k[source][mode] = {"k": int(winner), "candidates": candidates}
    atomic_json(artifact_root / "knn_ambiguity.validation.json", {
        "schema": "tactile3d-unit.vac-c3r0-knn-validation.v1",
        "selection_split": "validation only", "test_loaded": False,
        "sources": validation_audit,
    })
    return pca_records, index_records, selected_k, embeddings, projected_components


def exclude_self(indices: np.ndarray) -> np.ndarray:
    result = np.empty((len(indices), indices.shape[1] - 1), dtype=np.int32)
    for row in range(len(indices)):
        filtered = indices[row][indices[row] != row]
        if len(filtered) < result.shape[1]:
            raise RuntimeError("STRUCTURAL_FAIL: self-excluding train kNN has too few neighbors")
        result[row] = filtered[: result.shape[1]]
    return result


def freeze_nonparametric_probes(
    config, train, validation, experiment_root, cache_root, embeddings, projected_components,
    index_records, selected_k,
):
    records: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    max_k = max(map(int, config["knn"]["k"]))
    for source in config["sources"]:
        print(f"[c3r0] freezing nonparametric probes: {source}", flush=True)
        index = joblib.load(ROOT / index_records[source]["path"])
        train_embedding = np.concatenate(
            [embeddings["train"][name] for name in projected_components[source]], axis=1
        )
        raw_indices, _ = index.query(train_embedding, k=max_k + 1, epsilon=float(config["knn"]["query_epsilon"]))
        train_indices = exclude_self(raw_indices)
        train_path = cache_root / "train" / f"{source}_indices.npy"
        save_array(train_path, train_indices)
        validation_indices = np.load(ROOT / index_records[source]["validation_indices"], allow_pickle=False)
        records[source] = {"train_indices": train_path.relative_to(ROOT).as_posix(), "variants": {}}
        validation_results[source] = {}
        variants = {"1nn": 1, "medoid": selected_k[source]["medoid"]["k"], "mean": selected_k[source]["mean"]["k"]}
        for variant, k in variants.items():
            train_prediction = knn_target_predictions(train_indices, train["u_c"], int(k))[variant]
            validation_prediction = knn_target_predictions(validation_indices, train["u_c"], int(k))[variant]
            records[source]["variants"][variant] = {"k": int(k), "probes": {}}
            validation_results[source][variant] = {}
            for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
                model = fit_probe(train_prediction, train[metric], alpha=float(config["probe"]["alpha"]))
                path = experiment_root / "nonparametric_probes" / f"{source}_{variant}_{metric}.joblib"
                digest = dump_joblib(path, model)
                prediction = probe_prediction(model, validation_prediction)
                majority = np.full(len(prediction), int(np.bincount(np.asarray(train[metric], dtype=np.int64), minlength=classes).argmax()))
                validation_results[source][variant][metric] = evaluate_prediction(
                    validation[metric], prediction, majority, classes
                )
                records[source]["variants"][variant]["probes"][metric] = {
                    "path": path.relative_to(ROOT).as_posix(), "sha256": digest,
                    "fit_split": "train only",
                }
    atomic_json(experiment_root / "nonparametric_validation.json", {
        "schema": "tactile3d-unit.vac-c3r0-nonparametric-validation.v1",
        "selection_split": "validation only", "test_loaded": False,
        "results": validation_results,
    })
    return records, validation_results


def predict_ceiling(model, features, device, batch_size):
    result = np.empty((len(features), 8, 32), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            value = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32)).to(device)
            result[start:stop] = model(value).float().cpu().numpy()
    return result


def train_ceiling_trial(config, source, architecture, train_features, validation_features, train_target, validation_target, device, path):
    model = SmallContactCeiling(source, architecture).to(device)
    if model.parameter_count() > int(config["deterministic_ceiling"]["max_parameters"]):
        raise RuntimeError("STRUCTURAL_FAIL: C3-R0 ceiling exceeds parameter bound")
    dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32)),
        torch.from_numpy(np.asarray(train_target, dtype=np.float32)),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=int(config["deterministic_ceiling"]["batch_size"]), shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["deterministic_ceiling"]["learning_rate"]),
        weight_decay=float(config["deterministic_ceiling"]["weight_decay"]),
    )
    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    history = []
    best_state = None
    for epoch in range(int(config["deterministic_ceiling"]["epochs"])):
        model.train()
        losses = []
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = predict_ceiling(model, validation_features, device, 1024)
        validation_loss = float(np.square(validation_prediction.astype(np.float64) - np.asarray(validation_target, dtype=np.float64)).mean())
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "validation_mse": validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(config["deterministic_ceiling"]["patience"]):
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    digest = save_ceiling_checkpoint(path, model, {
        "source": source, "architecture": architecture, "epoch": best_epoch,
        "validation_mse": best_loss, "selection_split": "validation only", "test_loaded": False,
    })
    return model, {"checkpoint": path.relative_to(ROOT).as_posix(), "checkpoint_sha256": digest, "epoch": best_epoch, "validation_mse": best_loss, "history": history, "parameters": model.parameter_count()}


def freeze_deterministic_ceilings(config, train, validation, experiment_root, device):
    trials: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    for source in config["deterministic_ceiling"]["sources"]:
        standardizer = TrainStandardizer.fit(source_features(source, legal_arrays(train)))
        standardizer_path = experiment_root / "ceilings" / f"{source}_standardizer.npz"
        standardizer.save(standardizer_path)
        train_features = standardizer.transform(source_features(source, legal_arrays(train)))
        validation_features = standardizer.transform(source_features(source, legal_arrays(validation)))
        candidates = []
        for architecture in config["deterministic_ceiling"]["architectures"]:
            print(f"[c3r0] training deterministic ceiling: {source}/{architecture}", flush=True)
            trial_id = f"{source}_{architecture}"
            path = experiment_root / "ceilings" / trial_id / "best.pt"
            model, record = train_ceiling_trial(
                config, source, architecture, train_features, validation_features,
                train["u_c"], validation["u_c"], device, path,
            )
            train_prediction = predict_ceiling(model, train_features, device, 1024)
            validation_prediction = predict_ceiling(model, validation_features, device, 1024)
            record["standardizer"] = standardizer_path.relative_to(ROOT).as_posix()
            record["standardizer_sha256"] = sha256_file(standardizer_path)
            record["probes"] = {}
            record["semantics"] = {}
            for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
                probe = fit_probe(train_prediction, train[metric], alpha=float(config["probe"]["alpha"]))
                probe_path = experiment_root / "ceilings" / trial_id / f"{metric}.joblib"
                probe_digest = dump_joblib(probe_path, probe)
                prediction = probe_prediction(probe, validation_prediction)
                majority = np.full(len(prediction), int(np.bincount(np.asarray(train[metric], dtype=np.int64), minlength=classes).argmax()))
                record["semantics"][metric] = evaluate_prediction(validation[metric], prediction, majority, classes)
                record["probes"][metric] = {"path": probe_path.relative_to(ROOT).as_posix(), "sha256": probe_digest}
            record["utility"] = float(record["semantics"]["contact_transition"]["macro_f1"] + record["semantics"]["force_trend_class"]["macro_f1"])
            trials[trial_id] = record
            candidates.append(trial_id)
        winner = max(candidates, key=lambda name: (trials[name]["utility"], -trials[name]["validation_mse"], -trials[name]["parameters"]))
        selected[source] = {"trial": winner, **{key: value for key, value in trials[winner].items() if key != "history"}}
        validation_results[source] = trials[winner]["semantics"]
    atomic_json(experiment_root / "multisource_ceiling_training.json", {
        "schema": "tactile3d-unit.vac-c3r0-ceiling-training.v1",
        "selection_split": "validation only", "test_loaded": False,
        "trials": trials, "selected": selected,
    })
    return trials, selected, validation_results


def freeze_p2_probes(config, train, validation, experiment_root, device):
    predictor, metadata = load_predictor_checkpoint(ROOT / config["runtime"]["c3dp_checkpoint"], device)
    predictor.eval().requires_grad_(False).to(device)
    records = {}
    validation_results = {}
    for source, source_name, key in (("V", "vision", "u_v"), ("A", "action", "u_a")):
        print(f"[c3r0] freezing P2 comparison probes: {source}->C", flush=True)
        train_prediction = predict_numpy(predictor, np.asarray(train[key]), source_name, "contact", device, 512)
        validation_prediction = predict_numpy(predictor, np.asarray(validation[key]), source_name, "contact", device, 512)
        records[source] = {"probes": {}}
        validation_results[source] = {}
        for metric, classes in (("contact_transition", 4), ("force_trend_class", 3)):
            probe = fit_probe(train_prediction, train[metric], alpha=float(config["probe"]["alpha"]))
            path = experiment_root / "p2_probes" / f"{source}_{metric}.joblib"
            digest = dump_joblib(path, probe)
            prediction = probe_prediction(probe, validation_prediction)
            majority = np.full(len(prediction), int(np.bincount(np.asarray(train[metric], dtype=np.int64), minlength=classes).argmax()))
            validation_results[source][metric] = evaluate_prediction(validation[metric], prediction, majority, classes)
            records[source]["probes"][metric] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest}
        records[source]["train_prediction_sha256"] = hashlib_array(train_prediction)
    return records, validation_results, metadata


def hashlib_array(value: np.ndarray) -> str:
    import hashlib
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["deterministic_ceiling"]["batch_size"] = int(args.batch_size)
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    if args.device == "cuda:0":
        raise RuntimeError("use the repository GPU lock helper before enabling C3-R0 CUDA")
    runtime = config["runtime"]
    artifact_root = ROOT / runtime["artifact_root"]
    experiment_root = ROOT / runtime["experiment_root"]
    cache_root = ROOT / runtime["cache_root"]
    for path in (artifact_root, experiment_root, cache_root, ROOT / runtime["log_root"], ROOT / runtime["tmp_root"]):
        path.mkdir(parents=True, exist_ok=True)

    identities_before = identity_snapshot(config)
    if not identities_before["pass"]:
        raise RuntimeError("STRUCTURAL_FAIL: accepted identity mismatch before C3-R0")
    train = load_aligned_split(config, "train")
    validation = load_aligned_split(config, "validation")
    print("[c3r0] immutable train/validation caches verified", flush=True)
    if len(train["u_v"]) != int(config["counts"]["train"]) or len(validation["u_v"]) != int(config["counts"]["validation"]):
        raise RuntimeError("STRUCTURAL_FAIL: C3-R0 split count mismatch")

    probe_records, validation_probes, _ = source_probe_freeze(
        config, train, validation, experiment_root, artifact_root
    )
    oracle_probes = {
        metric: joblib.load(ROOT / probe_records["C"][metric]["path"])
        for metric in ("contact_transition", "force_trend_class")
    }
    pca_records, index_records, selected_k, embeddings, projected_components = fit_pca_and_indices(
        config, train, validation, experiment_root, cache_root, artifact_root, oracle_probes
    )
    nonparametric_records, nonparametric_validation = freeze_nonparametric_probes(
        config, train, validation, experiment_root, cache_root, embeddings,
        projected_components, index_records, selected_k,
    )
    trials, selected_ceilings, ceiling_validation = freeze_deterministic_ceilings(
        config, train, validation, experiment_root, device
    )
    p2_records, p2_validation, p2_metadata = freeze_p2_probes(
        config, train, validation, experiment_root, device
    )
    identities_after = identity_snapshot(config)
    if not identities_after["pass"] or identities_after["actual"] != identities_before["actual"]:
        raise RuntimeError("STRUCTURAL_FAIL: frozen identity changed during C3-R0 selection")

    freeze_values = {
        "audit_protocol.json": {
            "schema": config["schema"], "test_loaded": False,
            "evaluation_type": "POST-C3 DIAGNOSTIC AUDIT", "first_look_untouched": False,
            "splits": config["counts"], "sources": config["sources"], "references": config["references"],
            "feature_standardization": "train fit only", "identities_before": identities_before,
            "identities_after": identities_after, "scope": config["scope"],
        },
        "probe_selection.json": {
            "schema": "tactile3d-unit.vac-c3r0-probe-selection.v1", "test_loaded": False,
            "selection_split": "validation only", "family": config["probe"]["family"],
            "alpha": config["probe"]["alpha"], "records": probe_records,
            "validation": validation_probes, "p2_records": p2_records,
            "p2_validation": p2_validation, "p2_metadata": p2_metadata,
        },
        "knn_protocol.json": {
            "schema": "tactile3d-unit.vac-c3r0-knn-protocol.v1", "test_loaded": False,
            "selection_split": "validation only", "reference_split": "train only",
            "k": config["knn"]["k"], "algorithm": config["knn"]["algorithm"],
            "pca": pca_records, "indices": index_records, "selected_k": selected_k,
            "nonparametric_records": nonparametric_records,
            "nonparametric_validation": nonparametric_validation,
        },
        "deterministic_ceiling_selection.json": {
            "schema": "tactile3d-unit.vac-c3r0-ceiling-selection.v1", "test_loaded": False,
            "selection_split": "validation only", "trials_total": len(trials),
            "maximum_trials": int(config["deterministic_ceiling"]["trials_total"]),
            "selected": selected_ceilings, "validation": ceiling_validation,
        },
    }
    protocol_hashes = write_freeze_files(artifact_root, freeze_values)
    selection = {
        "schema": "tactile3d-unit.vac-c3r0-selection.v1", "test_loaded": False,
        "selection_split": "validation only", "protocol_hashes": protocol_hashes,
        "identities": identities_after, "selected_k": selected_k,
        "selected_ceilings": selected_ceilings, "bounded_trials": len(trials),
    }
    atomic_json(artifact_root / "selection.json", selection)
    print(json.dumps({
        "status": "C3R0_PRETEST_FROZEN", "protocol_hashes": protocol_hashes,
        "selected_ceilings": {name: value["trial"] for name, value in selected_ceilings.items()},
        "test_loaded": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
