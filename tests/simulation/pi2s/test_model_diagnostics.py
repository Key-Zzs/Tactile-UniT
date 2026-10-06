from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/simulation/run_pi2s_model_diagnostics.py"
SOURCE = SCRIPT.read_text()
TREE = ast.parse(SOURCE, filename=str(SCRIPT))
HEAVY_IMPORT_ROOTS = frozenset(
    {
        "flax",
        "jax",
        "jaxlib",
        "openpi",
        "optax",
        "orbax",
        "torch",
    }
)
EXPECTED_FOCUS = (
    ("B0", 43),
    ("B_VA27", 43),
    ("B1", 43),
    ("B_HVA", 42),
    ("B_HVA", 43),
    ("B2", 43),
)
EXPECTED_BUDGET = {
    "checkpoints": 6,
    "phases_per_checkpoint": 2,
    "immutable_shards": 12,
    "action_records": 704,
    "gradient_records": 72,
    "auxiliary_gradient_records": 48,
    "optimizer_steps": 0,
    "checkpoint_writes": 0,
}
EXPECTED_PREFIX_INTEGRATION = {
    "contact_state_shape": [256],
    "adapter_output_shape_per_observation": [8, 2048],
    "token_count": 8,
    "token_width": 2048,
    "placement": "APPEND_AFTER_UNCHANGED_OFFICIAL_PREFIX",
    "token_order": "CONTACT_ADAPTER_RESHAPE_ORDER_0_THROUGH_7",
    "input_mask_for_added_tokens": True,
    "autoregressive_mask_for_added_tokens": False,
    "position_index_rule": "cumsum(input_mask)-1",
    "padding_interaction": (
        "physical append occurs after the padded official prefix; masked padding does not "
        "advance the added tokens' position indices"
    ),
    "state_dimension_changed": False,
    "action_dimension_changed": False,
}
EXPECTED_FIXED_NOISE = {
    "noise_distribution": "jax.random.normal",
    "noise_shape": [30, 32],
    "noise_dtype": "float32",
}
EXPECTED_GRADIENT_NOISE = {
    "nuisance_noise_distribution": "jax.random.normal",
    "nuisance_noise_shape_per_minibatch": [2, 30, 32],
    "nuisance_noise_dtype": "float32",
}


def _heavy_modules() -> set[str]:
    return {name for name in sys.modules if name.partition(".")[0] in HEAVY_IMPORT_ROOTS}


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    module_name = "_pi2s_model_diagnostics_test_subject"
    before = _heavy_modules()
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.__test_new_heavy_modules__ = _heavy_modules() - before
    return module


def _parent_map() -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(TREE) for child in ast.iter_child_nodes(parent)}


def _enclosing_function(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> str | None:
    cursor = node
    while cursor in parents:
        cursor = parents[cursor]
        if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cursor.name
    return None


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _defined(name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in TREE.body
    )


def _flatten_nested_dict(
    value: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    flattened: dict[tuple[str, ...], Any] = {}
    for key, child in value.items():
        path = (*prefix, str(key))
        if isinstance(child, Mapping):
            flattened.update(_flatten_nested_dict(child, path))
        else:
            flattened[path] = child
    return flattened


class _PureState:
    def __init__(self, value: Mapping[str, Any]) -> None:
        self._value = value

    def to_pure_dict(self) -> Mapping[str, Any]:
        return self._value


def _install_fake_flax(monkeypatch: pytest.MonkeyPatch) -> None:
    flax = ModuleType("flax")
    flax.__path__ = []  # type: ignore[attr-defined]
    traverse_util = ModuleType("flax.traverse_util")
    traverse_util.flatten_dict = _flatten_nested_dict  # type: ignore[attr-defined]
    flax.traverse_util = traverse_util  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flax", flax)
    monkeypatch.setitem(sys.modules, "flax.traverse_util", traverse_util)


def _install_fake_numpy_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    jax = ModuleType("jax")
    jax.__path__ = []  # type: ignore[attr-defined]
    jax.device_get = lambda value: value  # type: ignore[attr-defined]
    jnp = ModuleType("jax.numpy")
    for name in (
        "array_equal",
        "asarray",
        "cumsum",
        "square",
        "sum",
        "sqrt",
        "vdot",
    ):
        setattr(jnp, name, getattr(np, name))
    jnp.float32 = np.float32  # type: ignore[attr-defined]
    jnp.linalg = np.linalg  # type: ignore[attr-defined]
    nn = ModuleType("jax.nn")
    nn.gelu = lambda value: value  # type: ignore[attr-defined]
    jax.nn = nn  # type: ignore[attr-defined]
    jax.numpy = jnp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    monkeypatch.setitem(sys.modules, "jax.nn", nn)


def _make_contract(
    runner: ModuleType,
    root: Path,
    *,
    protocol: dict[str, Any] | None = None,
    focus: Sequence[dict[str, Any]] = (),
) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshot"
    train = snapshot / "train_fixed"
    train.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2s-input-snapshot.v1",
        "status": "COMPLETE_VERIFIED",
        "files": [],
    }
    protocol_payload = copy.deepcopy(protocol or {})
    protocol_payload.setdefault("selection_sha256", "selection")
    protocol_payload.setdefault(
        "diagnostic_implementation_sha256",
        {"run_pi2s_model_diagnostics": runner.sha256_file(SCRIPT)},
    )
    protocol_payload.setdefault(
        "prefix_integration_contract", copy.deepcopy(EXPECTED_PREFIX_INTEGRATION)
    )
    fixed_sampling = copy.deepcopy(EXPECTED_FIXED_NOISE)
    fixed_sampling.update(protocol_payload.get("fixed_sampling", {}))
    protocol_payload["fixed_sampling"] = fixed_sampling
    gradient_diagnostic = copy.deepcopy(EXPECTED_GRADIENT_NOISE)
    gradient_diagnostic.update(protocol_payload.get("gradient_diagnostic", {}))
    protocol_payload["gradient_diagnostic"] = gradient_diagnostic
    return runner.Contract(
        protocol=protocol_payload,
        protocol_sha256="protocol",
        runner_sha256="runner",
        experiments=root / "experiments",
        pi2s_root=root,
        snapshot_root=snapshot,
        snapshot_manifest=manifest,
        snapshot_manifest_sha256="manifest",
        train_root=train,
        focus=tuple(focus),
    )


def _stub_binding_authorities(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def source_bindings(contract: Any) -> dict[str, dict[str, str]]:
        return {
            runner._checkpoint_id(row): {
                "cohort": "SYNTHETIC_CONFIG_SOURCE_BINDING",
                "checkpoint_id": runner._checkpoint_id(row),
            }
            for row in contract.focus
        }

    monkeypatch.setattr(runner, "collect_config_source_bindings", source_bindings)
    monkeypatch.setattr(
        runner,
        "checkpoint_authority",
        lambda _experiments, _focus: {"verification": "SYNTHETIC_MANIFEST_REBIND_FOR_UNIT_TEST"},
    )


def _focus_rows() -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "training_seed": seed,
            "tree_sha256": hashlib.sha256(f"{model}:{seed}".encode()).hexdigest(),
            "path": f"$EXPERIMENT_ROOT/checkpoints/{model}_seed{seed}/29999",
        }
        for model, seed in EXPECTED_FOCUS
    ]


def _exact_shards(
    runner: ModuleType, contract: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    non_h_identity = {
        "tokenized_prompt_sha256": hashlib.sha256(b"prompt-tokens").hexdigest(),
        "tokenized_prompt_mask_sha256": hashlib.sha256(b"prompt-mask").hexdigest(),
        "normalized_state_sha256": hashlib.sha256(b"normalized-state").hexdigest(),
        "preprocessed_images_sha256": hashlib.sha256(b"preprocessed-images").hexdigest(),
    }
    parameter_load = runner.strict_parameter_load_evidence()
    state = {"params": {"sha256": hashlib.sha256(b"state").hexdigest()}}
    sentinels = {"params/manifest.ocdbt": {"sha256": hashlib.sha256(b"params").hexdigest()}}
    origins = {
        "openpi_client_image_tools": {"sha256": runner.OPENPI_CLIENT_IMAGE_TOOLS_SHA256},
        "runtime_versions": copy.deepcopy(runner.EXPECTED_RUNTIME_VERSIONS),
        "pip_freeze_all_sha256": runner.EXPECTED_PIP_FREEZE_SHA256,
    }
    environment = {
        "frozen_paligemma_tokenizer": {
            "sha256": runner.PALIGEMMA_TOKENIZER_SHA256,
            "pre_execution_sha256": runner.PALIGEMMA_TOKENIZER_SHA256,
            "post_execution_sha256": runner.PALIGEMMA_TOKENIZER_SHA256,
            "post_execution_status": "PASS",
            "unexpected_cache_files": [],
        }
    }
    for focus in contract.focus:
        conditions = (
            runner.CONDITION_ORDER if focus["model"] in runner.CONTACT_MODELS else ("correct",)
        )
        action_records = []
        for condition in conditions:
            for fixed_position, row in enumerate(range(32)):
                action_records.append(
                    {
                        "fixed_position": fixed_position,
                        "global_row": row,
                        "noise_seed": 431700 + fixed_position,
                        "noise_sha256": hashlib.sha256(
                            f"action-noise:{fixed_position}".encode()
                        ).hexdigest(),
                        "condition": condition,
                        "delta": {
                            "component_rms": {
                                "tcp_xyz": 0.0,
                                "rotation_vector": 0.0,
                                "hand": 0.0,
                            },
                            "chunk_l2": {
                                "early": 0.0,
                                "late": 0.0,
                                "all": 0.0,
                            },
                            "delta_jerk_proxy": 0.0,
                        },
                        "action_jerk_proxy": 0.0,
                        "normalization_boundary_exceedance": {
                            "tcp_xyz": 0.0,
                            "rotation_vector": 0.0,
                            "hand": 0.0,
                        },
                    }
                )
        prefix_diagnostics = [
            {
                "condition": condition,
                "official_prefix_unchanged": True,
                "added_mask_contract": True,
                "adapter_projection_matches_appended_tokens": True,
                "adapter_manual_pipeline_matches": True,
                "position_index_contract": True,
                "records": [
                    {
                        "global_row": row,
                        "condition": condition,
                        "image_tokens_width_2048": {"active_tokens": 768},
                        "language_tokens_width_2048": {"active_tokens": 32},
                        "contact_tokens_width_2048": (
                            {"active_tokens": 8}
                            if focus["model"] in runner.CONTACT_MODELS
                            else None
                        ),
                        "contact_state_pre_layernorm_l2": (
                            1.0 if focus["model"] in runner.CONTACT_MODELS else None
                        ),
                        "contact_state_post_layernorm_l2": (
                            1.0 if focus["model"] in runner.CONTACT_MODELS else None
                        ),
                        "contact_sequence_indices": (
                            list(range(1000, 1008))
                            if focus["model"] in runner.CONTACT_MODELS
                            else []
                        ),
                        "contact_position_indices": (
                            list(range(800, 808)) if focus["model"] in runner.CONTACT_MODELS else []
                        ),
                    }
                    for row in range(32)
                ],
            }
            for condition in conditions
        ]
        actions.append(
            {
                "schema": runner.ACTION_SCHEMA,
                "status": "PASS",
                "binding": runner._artifact_binding(contract, focus, "actions"),
                "gates": {"complete": "PASS"},
                "focus": dict(focus),
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "parameter_load": copy.deepcopy(parameter_load),
                "actual_import_origins": copy.deepcopy(origins),
                "execution_environment": copy.deepcopy(environment),
                "state_before": copy.deepcopy(state),
                "state_after": copy.deepcopy(state),
                "checkpoint_sentinels_before": copy.deepcopy(sentinels),
                "checkpoint_sentinels_after": copy.deepcopy(sentinels),
                "input_batches": [
                    {
                        "condition": condition,
                        "transformed_observation_sha256": hashlib.sha256(
                            f"{runner._checkpoint_id(focus)}:{condition}".encode()
                        ).hexdigest(),
                        "explicit_noise_sha256": hashlib.sha256(
                            b"shared-action-noise-batch"
                        ).hexdigest(),
                        **non_h_identity,
                    }
                    for condition in conditions
                ],
                "prefix_diagnostics": prefix_diagnostics,
                "records": action_records,
            }
        )
        auxiliary = focus["model"] in runner.AUXILIARY_MODELS
        gradient_records = [
            {
                "minibatch_index": minibatch,
                "global_rows": [2 * minibatch, 2 * minibatch + 1],
                "fixed_positions": [2 * minibatch, 2 * minibatch + 1],
                "flow_time": flow_time,
                "noise_seed": 431900 + minibatch,
                "noise_sha256": hashlib.sha256(f"gradient-noise:{minibatch}".encode()).hexdigest(),
                "transformed_batch_sha256": hashlib.sha256(
                    f"{runner._checkpoint_id(focus)}:gradient-input:{minibatch}".encode()
                ).hexdigest(),
                "auxiliary_applicable": auxiliary,
                "losses": {
                    "main": 1.0,
                    "auxiliary": 2.0 if auxiliary else None,
                    "combined": 3.0 if auxiliary else 1.0,
                },
                "groups": {
                    group: {
                        "l2_norm_g_main": 1.0,
                        "l2_norm_g_aux": 2.0 if auxiliary else None,
                        "cosine_g_main_g_aux": 0.0 if auxiliary else None,
                    }
                    for group in runner.PARAMETER_GROUPS
                },
            }
            for minibatch in range(4)
            for flow_time in runner.FLOW_TIMES
        ]
        gradients.append(
            {
                "schema": runner.GRADIENT_SCHEMA,
                "status": "PASS",
                "binding": runner._artifact_binding(contract, focus, "gradients"),
                "gates": {"complete": "PASS"},
                "focus": dict(focus),
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "parameter_load": copy.deepcopy(parameter_load),
                "actual_import_origins": copy.deepcopy(origins),
                "execution_environment": copy.deepcopy(environment),
                "state_before": copy.deepcopy(state),
                "state_after": copy.deepcopy(state),
                "checkpoint_sentinels_before": copy.deepcopy(sentinels),
                "checkpoint_sentinels_after": copy.deepcopy(sentinels),
                "preprocess_mode": "DETERMINISTIC_EVAL_PREPROCESS_NO_AUGMENTATION",
                "fixed_non_h_input_identity": copy.deepcopy(non_h_identity),
                "records": gradient_records,
            }
        )
    return actions, gradients


def _aggregate_contract(runner: ModuleType, root: Path) -> Any:
    protocol = {
        "selection_sha256": "selection",
        "diagnostic_implementation_sha256": {
            "run_pi2s_model_diagnostics": runner.sha256_file(SCRIPT)
        },
        "prefix_integration_contract": copy.deepcopy(EXPECTED_PREFIX_INTEGRATION),
        "fixed_sampling": {
            **EXPECTED_FIXED_NOISE,
            "noise_seeds": list(range(431700, 431732)),
        },
        "samples": {
            "fixed_observation_rows": list(range(32)),
            "gradient_batches": [[0, 1], [2, 3], [4, 5], [6, 7]],
        },
        "metrics": {
            "interpretation": "synthetic unit-test diagnostics",
            "prefix_contract": ["official_prefix_unchanged", "added_token_count"],
            "fixed_observation_actions": {"aggregation": "synthetic exact fixture"},
            "gradients": {
                "aggregation": "synthetic exact fixture",
                "preprocess_mode": "DETERMINISTIC_EVAL_PREPROCESS_NO_AUGMENTATION",
                "training_equivalence_scope": "synthetic exact fixture",
            },
        },
        "h_conditions": {"conditions": list(runner.CONDITION_ORDER)},
        "gradient_diagnostic": {
            **EXPECTED_GRADIENT_NOISE,
            "nuisance_noise_seeds_by_minibatch": [431900, 431901, 431902, 431903],
            "auxiliary_loss_by_model": {
                model: (
                    {"target": "synthetic", "valid_mask": "synthetic"}
                    if model in runner.AUXILIARY_MODELS
                    else "N/A_NO_AUXILIARY_OBJECTIVE"
                )
                for model, _seed in EXPECTED_FOCUS
            },
        },
    }
    return _make_contract(runner, root, protocol=protocol, focus=_focus_rows())


def _write_exact_shards(
    runner: ModuleType,
    output: Path,
    contract: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions, gradients = _exact_shards(runner, contract)
    for focus, action, gradient in zip(contract.focus, actions, gradients):
        checkpoint_id = runner._checkpoint_id(focus)
        runner.write_atomic_json(runner.shard_path(output, checkpoint_id, "actions"), action)
        runner.write_atomic_json(runner.shard_path(output, checkpoint_id, "gradients"), gradient)
    return actions, gradients


def test_runner_import_is_numpy_only_and_does_not_load_accelerator_stacks(
    runner: ModuleType,
) -> None:
    assert runner.__test_new_heavy_modules__ == set()
    parents = _parent_map()
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            roots = {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = {node.module.partition(".")[0]}
        else:
            continue
        if roots & HEAVY_IMPORT_ROOTS:
            assert (
                _enclosing_function(node, parents) is not None
            ), f"heavy import must remain execute-local at line {node.lineno}: {sorted(roots)}"


def test_contract_and_check_report_are_exact_without_jax(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments = tmp_path / "experiments"
    pi2s_root = experiments / runner.PI2S_RELATIVE
    snapshot = pi2s_root / "snapshot"
    train = snapshot / "train_fixed"
    train.mkdir(parents=True)
    generator = tmp_path / "generator.py"
    generator.write_text("# frozen generator\n")
    contact = pi2s_root / "contact.npz"
    va27 = pi2s_root / "va27.npz"
    contact.write_bytes(b"contact")
    va27.write_bytes(b"va27")
    tokenizer = pi2s_root / "paligemma_tokenizer.model"
    tokenizer_manifest = pi2s_root / "paligemma_tokenizer.manifest.json"
    tokenizer.write_bytes(b"synthetic tokenizer")
    tokenizer_manifest.write_bytes(b"synthetic tokenizer manifest\n")
    tokenizer_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    tokenizer_manifest_sha = hashlib.sha256(tokenizer_manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "PALIGEMMA_TOKENIZER_SHA256", tokenizer_sha)
    monkeypatch.setattr(runner, "PALIGEMMA_TOKENIZER_BYTES", tokenizer.stat().st_size)
    monkeypatch.setattr(
        runner,
        "PALIGEMMA_TOKENIZER_MANIFEST_SHA256",
        tokenizer_manifest_sha,
    )

    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2s-input-snapshot.v1",
        "status": "COMPLETE_VERIFIED",
        "files": [],
    }
    manifest_path = snapshot / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    focus = _focus_rows()
    for index, row in enumerate(focus):
        checkpoint = experiments / "checkpoints" / str(index) / "29999"
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "train_state").mkdir()
        (checkpoint / "assets/local_repo").mkdir(parents=True)
        (checkpoint / "params/manifest.ocdbt").write_bytes(b"")
        (checkpoint / "train_state/manifest.ocdbt").write_bytes(b"")
        (checkpoint / "assets/local_repo/norm_stats.json").write_text("{}\n")
        row["path"] = f"$EXPERIMENT_ROOT/checkpoints/{index}/29999"

    rows = np.arange(100, 132, dtype=np.int64)
    expected_aux = {
        "B0": "N/A_NO_AUXILIARY_OBJECTIVE",
        "B1": "N/A_NO_AUXILIARY_OBJECTIVE",
        "B_VA27": {
            "target": "va_target_t_plus_27.npy (source field va_shared_target)",
            "valid_mask": "va_aux_valid.npy (source field va_aux_valid)",
        },
        "B_HVA": {
            "target": "va_target_t_plus_27.npy (source field va_shared_target)",
            "valid_mask": "va_aux_valid.npy (source field va_aux_valid)",
        },
        "B2": {
            "target": "contact_target_t_plus_27.npy (source field contact_shared_target)",
            "valid_mask": "contact_aux_valid.npy (source field physical_aux_valid)",
        },
    }
    protocol = {
        "status": "PREREGISTERED_NOT_EXECUTED",
        "training_budget": {"optimizer_updates": 0, "policy_runs": 0, "teacher_runs": 0},
        "diagnostic_script_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "diagnostic_implementation_sha256": {
            "run_pi2s_model_diagnostics": runner.sha256_file(SCRIPT)
        },
        "selection_sha256": "selection",
        "runtime_dependencies": {
            "openpi_client": {
                "accepted_package_initializer": (
                    "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/packages/"
                    "openpi-client/src/openpi_client/__init__.py"
                ),
                "accepted_package_initializer_sha256": runner.OPENPI_CLIENT_INIT_SHA256,
                "accepted_image_tools": (
                    "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/packages/"
                    "openpi-client/src/openpi_client/image_tools.py"
                ),
                "accepted_image_tools_sha256": runner.OPENPI_CLIENT_IMAGE_TOOLS_SHA256,
                "accepted_python_files": runner.OPENPI_CLIENT_PYTHON_TREE_FILES,
                "accepted_python_tree_sha256": runner.OPENPI_CLIENT_PYTHON_TREE_SHA256,
                "required_distribution_version": "0.1.0",
                "runtime_origin_policy": (
                    "REPOSITORY_SCOPED_AND_BYTE_IDENTICAL_TO_ACCEPTED_SOURCE"
                ),
            },
            "paligemma_tokenizer": {
                "upstream_uri": "gs://big_vision/paligemma_tokenizer.model",
                "persistent_snapshot": "$PI2S_ROOT/paligemma_tokenizer.model",
                "persistent_manifest": "$PI2S_ROOT/paligemma_tokenizer.manifest.json",
                "persistent_manifest_sha256": tokenizer_manifest_sha,
                "sha256": tokenizer_sha,
                "bytes": tokenizer.stat().st_size,
                "required_sentencepiece_version": "0.2.2",
                "execution_cache_policy": (
                    "COPY_VERIFIED_SNAPSHOT_TO_EPHEMERAL_OPENPI_DATA_HOME_NO_NETWORK"
                ),
            },
            "execution_environment": {
                "pip_freeze_all_sha256": runner.EXPECTED_PIP_FREEZE_SHA256,
                "identity_source": "historical starting_integrity environment freeze",
            },
        },
        "focus_checkpoints": focus,
        "source_manifests": {
            "persistent_input_snapshot": {
                "path": "$PI2S_ROOT/snapshot",
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "files": 0,
                "bytes": 0,
            }
        },
        "samples": {
            "fixed_observation_rows": rows.tolist(),
            "gradient_batches": [rows[offset : offset + 2].tolist() for offset in range(0, 8, 2)],
        },
        "gradient_diagnostic": {
            "auxiliary_loss_by_model": expected_aux,
            "flow_times": list(runner.FLOW_TIMES),
            "nuisance_noise_seeds_by_minibatch": [431900, 431901, 431902, 431903],
            **EXPECTED_GRADIENT_NOISE,
            "optimizer_step": False,
            "state_mutation_allowed": False,
        },
        "prefix_integration_contract": copy.deepcopy(EXPECTED_PREFIX_INTEGRATION),
        "fixed_sampling": {
            "noise_seeds": list(range(431700, 431732)),
            **EXPECTED_FIXED_NOISE,
        },
        "data_contract": {
            "contact_sidecar_persistent_snapshot": "$PI2S_ROOT/contact.npz",
            "contact_sidecar_sha256": hashlib.sha256(contact.read_bytes()).hexdigest(),
            "va27_sidecar_persistent_snapshot": "$PI2S_ROOT/va27.npz",
            "va27_sidecar_sha256": hashlib.sha256(va27.read_bytes()).hexdigest(),
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))

    requested_arrays: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}

    def fake_array_contract(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
        requested_arrays[path.name] = (shape, dtype)
        if path.name == "rows.npy":
            return rows
        if path.name == "action_is_pad.npy":
            return np.zeros((32, 30), dtype=np.bool_)
        if path.name in {"contact_aux_valid.npy", "va_aux_valid.npy"}:
            return np.ones(32, dtype=np.bool_)
        return np.empty((0,), dtype=dtype)

    monkeypatch.setattr(runner, "EXPERIMENT_ROOT_LINK", experiments)
    monkeypatch.setattr(runner, "PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(runner, "AUDIT_SCRIPT", generator)
    monkeypatch.setattr(runner, "_array_contract", fake_array_contract)
    _stub_binding_authorities(runner, monkeypatch)
    before = _heavy_modules()
    contract = runner.load_contract(verify_snapshot_contents=True)
    assert _heavy_modules() == before
    assert tuple((row["model"], row["training_seed"]) for row in contract.focus) == EXPECTED_FOCUS
    assert contract.protocol["diagnostic_implementation_sha256"] == {
        "run_pi2s_model_diagnostics": runner.sha256_file(SCRIPT)
    }
    assert contract.protocol["prefix_integration_contract"] == EXPECTED_PREFIX_INTEGRATION
    assert {
        key: contract.protocol["fixed_sampling"][key] for key in EXPECTED_FIXED_NOISE
    } == EXPECTED_FIXED_NOISE
    assert {
        key: contract.protocol["gradient_diagnostic"][key] for key in EXPECTED_GRADIENT_NOISE
    } == EXPECTED_GRADIENT_NOISE
    assert requested_arrays["front_rgb_chw_uint8.npy"] == (
        (32, 3, 640, 640),
        np.dtype(np.uint8),
    )

    report = runner.check_report(contract)
    assert report["status"] == "PASS"
    assert report["execution_performed"] is False
    assert report["jax_imported"] is ("jax" in sys.modules)
    assert report["contracts"] == {
        "fixed_rows": 32,
        "gradient_minibatches": 4,
        "gradient_batch_size": 2,
        "flow_times": [0.1, 0.5, 0.9],
        "action_noise_seeds": list(range(431700, 431732)),
        "gradient_noise_seeds": [431900, 431901, 431902, 431903],
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "global_shard_budget": EXPECTED_BUDGET,
    }


def test_numpy_metrics_match_the_frozen_formulas_exactly(runner: ModuleType) -> None:
    stats = runner.summary_stats([0.0, 1.0, 2.0, 3.0, 4.0])
    assert stats == {
        "count": 5,
        "mean": 2.0,
        "median": 2.0,
        "q10": 0.4,
        "q90": 3.6,
        "min": 0.0,
        "max": 4.0,
    }
    assert runner.summary_stats([]) == {
        "count": 0,
        "mean": None,
        "median": None,
        "q10": None,
        "q90": None,
        "min": None,
        "max": None,
    }
    with pytest.raises(runner.ContractError, match="non-finite"):
        runner.summary_stats([0.0, np.inf])

    time = np.arange(30, dtype=np.float64)[:, None]
    quadratic = np.repeat(time**2, 22, axis=1)
    assert runner.jerk_proxy(quadratic) == pytest.approx(2.0 * np.sqrt(22.0))

    action = np.zeros((30, 22), dtype=np.float32)
    action[0, 0] = 2.0
    action[1, 3] = -2.0
    action[2, 6] = 2.0
    boundary = runner.boundary_exceedance(
        action, np.full(32, -1.0, dtype=np.float32), np.full(32, 1.0, dtype=np.float32)
    )
    assert boundary == {
        "tcp_xyz": 1.0 / (30 * 3),
        "rotation_vector": 1.0 / (30 * 3),
        "hand": 1.0 / (30 * 16),
    }

    reference = np.zeros((30, 22), dtype=np.float32)
    intervention = np.empty_like(reference)
    intervention[:, 0:3] = 1.0
    intervention[:, 3:6] = 2.0
    intervention[:, 6:22] = -3.0
    metrics = runner.action_delta_metrics(reference, intervention)
    assert metrics["component_rms"] == {
        "tcp_xyz": 1.0,
        "rotation_vector": 2.0,
        "hand": 3.0,
    }
    for name in ("early", "late", "all", "max"):
        assert metrics["chunk_l2"][name] == pytest.approx(np.sqrt(159.0))
    assert metrics["delta_jerk_proxy"] == 0.0
    assert metrics["delta_sha256"] == runner.array_sha256(intervention)


def test_prefix_diagnostics_uses_official_prefix_and_exact_adapter_shape(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_numpy_jax(monkeypatch)
    openpi = ModuleType("openpi")
    openpi.__path__ = []  # type: ignore[attr-defined]
    models = ModuleType("openpi.models")
    models.__path__ = []  # type: ignore[attr-defined]
    model_module = ModuleType("openpi.models.model")
    model_module.preprocess_observation = (  # type: ignore[attr-defined]
        lambda _rng, observation, *, train: observation
    )
    pi0_module = ModuleType("openpi.models.pi0")
    official_calls: list[Any] = []

    class FakePi0:
        @staticmethod
        def embed_prefix(candidate: Any, processed: Any) -> tuple[Any, Any, Any]:
            official_calls.append((candidate, processed))
            return candidate.official_prefix

    pi0_module.Pi0 = FakePi0  # type: ignore[attr-defined]
    models.model = model_module  # type: ignore[attr-defined]
    openpi.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.models", models)
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_module)
    monkeypatch.setitem(sys.modules, "openpi.models.pi0", pi0_module)

    batch_size = 2
    official_tokens = np.arange(batch_size * 3 * 2048, dtype=np.float32).reshape(
        batch_size, 3, 2048
    )
    official_mask = np.ones((batch_size, 3), dtype=np.bool_)
    official_ar = np.zeros(3, dtype=np.bool_)
    projected = np.arange(batch_size * 8 * 2048, dtype=np.float32).reshape(batch_size, 8, 2048)

    class Observation:
        contact_state = np.ones((batch_size, 256), dtype=np.float32)
        tokenized_prompt = np.ones((batch_size, 1), dtype=np.int32)
        tokenized_prompt_mask = np.ones((batch_size, 1), dtype=np.bool_)

    class ContactAdapter:
        norm = staticmethod(lambda value: value)
        fc1 = staticmethod(lambda value: value)
        fc2 = staticmethod(lambda value: value)
        shared_projection = staticmethod(lambda _value: projected)

        def __call__(self, contact_state: np.ndarray) -> np.ndarray:
            assert contact_state.shape == (batch_size, 256)
            return projected

    class Model:
        official_prefix = (official_tokens, official_mask, official_ar)
        contact_adapter = ContactAdapter()

        def embed_prefix(self, _processed: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                np.concatenate([official_tokens, projected], axis=1),
                np.concatenate([official_mask, np.ones((batch_size, 8), dtype=np.bool_)], axis=1),
                np.concatenate([official_ar, np.zeros(8, dtype=np.bool_)]),
            )

    model = Model()
    observation = Observation()
    report = runner.prefix_diagnostics(
        model,
        observation,
        "B1",
        [101, 202],
        "correct",
        copy.deepcopy(EXPECTED_PREFIX_INTEGRATION),
    )
    assert official_calls == [(model, observation)]
    assert report["official_prefix_shape"] == [2, 3, 2048]
    assert report["loaded_prefix_shape"] == [2, 11, 2048]
    assert report["added_token_count"] == 8
    assert report["token_width"] == 2048
    assert report["appended_shape_per_observation"] == [8, 2048]
    assert report["official_prefix_unchanged"] is True
    assert report["added_mask_contract"] is True
    assert report["adapter_projection_matches_appended_tokens"] is True
    assert report["adapter_manual_pipeline_matches"] is True
    assert report["position_index_contract"] is True
    assert [row["global_row"] for row in report["records"]] == [101, 202]
    assert all(row["added_input_mask"] == [True] * 8 for row in report["records"])
    assert all(
        row["contact_state_pre_layernorm_l2"] == pytest.approx(16.0)
        and row["contact_state_post_layernorm_l2"] == pytest.approx(16.0)
        and row["image_tokens_width_2048"]["active_tokens"] == 2
        and row["language_tokens_width_2048"]["active_tokens"] == 1
        and row["contact_tokens_width_2048"]["active_tokens"] == 8
        and len(row["contact_position_indices"]) == 8
        for row in report["records"]
    )

    wrong_width = copy.deepcopy(EXPECTED_PREFIX_INTEGRATION)
    wrong_width["token_width"] = 1024
    with pytest.raises(runner.ContractError, match="token-width contract drift"):
        runner.prefix_diagnostics(model, observation, "B1", [101, 202], "correct", wrong_width)


def test_parameter_groups_exclude_both_objective_specific_heads_from_common(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_flax(monkeypatch)
    state = _PureState(
        {
            "PaliGemma": {
                "llm": {"lora_a": np.ones((2,), dtype=np.float32)},
                "embedder": {"kernel": np.ones((2,), dtype=np.float32)},
            },
            "action_in_proj": {"kernel": np.ones((2,), dtype=np.float32)},
            "action_out_proj": {"kernel": np.ones((2,), dtype=np.float32)},
            "contact_adapter": {"kernel": np.ones((2,), dtype=np.float32)},
            "physical_auxiliary": {"kernel": np.ones((2,), dtype=np.float32)},
        }
    )
    paths, groups = runner._gradient_leaf_paths(state)
    objective_specific = {
        "action_out_proj/kernel",
        "physical_auxiliary/kernel",
    }
    assert set(groups) == set(runner.PARAMETER_GROUPS)
    assert set(groups["all_common_trainable"]) == set(paths) - objective_specific
    assert set(groups["all_common_trainable"]).isdisjoint(objective_specific)
    assert groups["lora"] == ["PaliGemma/llm/lora_a"]
    assert groups["contact_adapter"] == ["contact_adapter/kernel"]
    assert groups["physical_auxiliary"] == ["physical_auxiliary/kernel"]


@pytest.mark.parametrize("unused_representation", ["missing", "none"])
def test_gradient_groups_ignore_unused_missing_or_none_leaves_without_casting(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    unused_representation: str,
) -> None:
    _install_fake_flax(monkeypatch)
    _install_fake_numpy_jax(monkeypatch)
    parameter_state = _PureState(
        {
            "PaliGemma": {"lora_a": np.ones((2,), dtype=np.float32)},
            "contact_adapter": {"kernel": np.ones((2,), dtype=np.float32)},
            "action_out_proj": {"kernel": np.ones((2,), dtype=np.float32)},
            "physical_auxiliary": {"kernel": np.ones((2,), dtype=np.float32)},
        }
    )
    _paths, groups = runner._gradient_leaf_paths(parameter_state)
    main_value: dict[str, Any] = {
        "PaliGemma": {"lora_a": np.asarray([3.0, 4.0], dtype=np.float32)},
        "contact_adapter": {"kernel": np.asarray([1.0, 2.0], dtype=np.float32)},
        "action_out_proj": {"kernel": np.asarray([1000.0, 1000.0], dtype=np.float32)},
    }
    if unused_representation == "none":
        main_value["physical_auxiliary"] = {"kernel": None}
    auxiliary_value = {
        "PaliGemma": {"lora_a": np.asarray([6.0, 8.0], dtype=np.float32)},
        "contact_adapter": {"kernel": np.asarray([2.0, 4.0], dtype=np.float32)},
        "physical_auxiliary": {"kernel": np.asarray([5.0, 12.0], dtype=np.float32)},
    }

    metrics = runner.gradient_group_metrics(
        _PureState(main_value), _PureState(auxiliary_value), groups
    )
    assert metrics["lora"]["status"] == "PASS"
    assert metrics["lora"]["leaf_count"] == 1
    assert metrics["lora"]["cosine_intersection_leaf_count"] == 1
    assert metrics["lora"]["l2_norm_g_main"] == pytest.approx(5.0)
    assert metrics["lora"]["l2_norm_g_aux"] == pytest.approx(10.0)
    assert metrics["lora"]["cosine_g_main_g_aux"] == pytest.approx(1.0)
    assert metrics["contact_adapter"]["status"] == "PASS"
    assert metrics["all_common_trainable"]["status"] == "PASS"
    common = metrics["all_common_trainable"]
    assert common["leaf_count"] == 2
    assert common["cosine_intersection_leaf_count"] == 2
    assert common["l2_norm_g_main"] == pytest.approx(np.sqrt(30.0))
    assert common["l2_norm_g_aux"] == pytest.approx(np.sqrt(120.0))
    assert common["l2_norm_g_main_on_cosine_intersection"] == pytest.approx(np.sqrt(30.0))
    assert common["l2_norm_g_aux_on_cosine_intersection"] == pytest.approx(np.sqrt(120.0))
    assert common["cosine_g_main_g_aux"] == pytest.approx(1.0)
    physical = metrics["physical_auxiliary"]
    assert physical["status"].startswith("N/A_")
    assert physical["cosine_g_main_g_aux"] is None


def test_action_aggregate_retains_exact_record_count(runner: ModuleType) -> None:
    records = []
    for value in (1.0, 3.0):
        records.append(
            {
                "delta": {
                    "component_rms": {
                        "tcp_xyz": value,
                        "rotation_vector": value,
                        "hand": value,
                    },
                    "chunk_l2": {"early": value, "late": value, "all": value},
                    "delta_jerk_proxy": value,
                },
                "action_jerk_proxy": value,
                "normalization_boundary_exceedance": {
                    "tcp_xyz": value,
                    "rotation_vector": value,
                    "hand": value,
                },
            }
        )
    aggregate = runner.aggregate_action_records(records)
    assert set(aggregate) == {
        "component_rms/tcp_xyz",
        "component_rms/rotation_vector",
        "component_rms/hand",
        "chunk_l2/early",
        "chunk_l2/late",
        "chunk_l2/all",
        "delta_jerk_proxy",
        "action_jerk_proxy",
        "boundary/tcp_xyz",
        "boundary/rotation_vector",
        "boundary/hand",
    }
    assert all(stats["count"] == 2 and stats["mean"] == 2.0 for stats in aggregate.values())


def test_h_resolver_uses_global_rows_for_lag_and_other_episode(
    runner: ModuleType, tmp_path: Path
) -> None:
    contract = _make_contract(runner, tmp_path)
    rows = np.arange(32, 64, dtype=np.int64)
    np.save(contract.train_root / "rows.npy", rows, allow_pickle=False)
    full_h = np.arange(96 * 256, dtype=np.float32).reshape(96, 256)
    episodes = np.zeros(96, dtype=np.int64)
    episodes[64:] = 1
    ticks = np.arange(96, dtype=np.int64)
    sidecar = tmp_path / "contact.npz"
    np.savez(
        sidecar,
        contact_state=full_h,
        episode_index=episodes,
        control_tick_end=ticks,
    )
    mean_h = np.mean(full_h.astype(np.float64), axis=0).astype(np.float32)
    contract.protocol.update(
        {
            "data_contract": {"contact_sidecar_persistent_snapshot": "$PI2S_ROOT/contact.npz"},
            "h_conditions": {
                "train_mean_h_float32_sha256": hashlib.sha256(
                    mean_h.tobytes(order="C")
                ).hexdigest(),
                "same_episode_lag5_row_map": {
                    str(row): {"row": int(row - 5), "bootstrap_affected": False} for row in rows
                },
                "other_episode_row_map": {str(row): int(row + 32) for row in rows},
            },
        }
    )

    resolved = runner.load_intervention_contact_states(contract)
    assert tuple(resolved) == runner.CONDITION_ORDER
    assert all(
        value.shape == (32, 256) and value.dtype == np.float32 for value in resolved.values()
    )
    assert np.array_equal(resolved["correct"], full_h[rows])
    assert np.array_equal(resolved["same_episode_lag5"], full_h[rows - 5])
    assert np.array_equal(resolved["other_episode"], full_h[rows + 32])
    assert np.array_equal(resolved["train_mean"], np.repeat(mean_h[None, :], 32, axis=0))
    assert not resolved["zero"].any()
    assert not np.array_equal(resolved["same_episode_lag5"][0], full_h[0])


def test_atomic_json_is_immutable_and_resume_binding_is_exact(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _aggregate_contract(runner, tmp_path)
    _stub_binding_authorities(runner, monkeypatch)
    focus = contract.focus[0]
    payload = _exact_shards(runner, contract)[0][0]
    shard = tmp_path / "B0_seed43.actions.json"
    runner.write_atomic_json(shard, payload)
    original = shard.read_bytes()
    assert runner.validate_resumable_artifact(shard, contract, focus, "actions") == payload
    with pytest.raises(FileExistsError, match="refusing to replace"):
        runner.write_atomic_json(shard, {"status": "different"})
    assert shard.read_bytes() == original
    replacement = tmp_path / "replacement-forbidden.json"
    with pytest.raises(ValueError, match="never permit replacement"):
        runner.write_atomic_json(replacement, payload, replace=True)
    assert not os.path.lexists(replacement)

    for key in payload["binding"]:
        changed = copy.deepcopy(payload)
        changed["binding"][key] = f"wrong-{changed['binding'][key]}"
        candidate = tmp_path / f"bad-binding-{key}.json"
        candidate.write_text(json.dumps(changed))
        with pytest.raises(runner.ContractError, match="resume binding drift"):
            runner.validate_resumable_artifact(candidate, contract, focus, "actions")

    truncated = copy.deepcopy(payload)
    truncated["records"].pop()
    truncated_path = tmp_path / "truncated-pass.json"
    truncated_path.write_text(json.dumps(truncated))
    with pytest.raises(runner.ContractError, match="record budget drift"):
        runner.validate_resumable_artifact(truncated_path, contract, focus, "actions")

    mutated = copy.deepcopy(payload)
    mutated["state_after"] = {"params": {"sha256": "changed"}}
    mutated_path = tmp_path / "mutated-state-pass.json"
    mutated_path.write_text(json.dumps(mutated))
    with pytest.raises(runner.ContractError, match="immutability"):
        runner.validate_resumable_artifact(mutated_path, contract, focus, "actions")


def test_atomic_json_rejects_dangling_final_symlink(runner: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "actions.json"
    missing_target = tmp_path / "must-not-be-created.json"
    output.symlink_to(missing_target)
    assert os.path.lexists(output) and output.is_symlink()
    with pytest.raises(FileExistsError):
        runner.write_atomic_json(output, {"publisher": "diagnostic"})
    assert os.path.lexists(output) and output.is_symlink()
    assert os.readlink(output) == str(missing_target)
    assert not missing_target.exists()


def test_write_paths_reject_symlink_ancestors_and_pi2s_escape(
    runner: ModuleType, tmp_path: Path
) -> None:
    contract = _make_contract(runner, tmp_path / "pi2s")
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts = contract.pi2s_root / "artifacts"
    artifacts.symlink_to(outside, target_is_directory=True)
    requested = artifacts / "model_diagnostics_v1"
    with pytest.raises(runner.ContractError, match="contains a symlink"):
        runner._resolve_output_dir(requested, contract, require_pi2s=True)
    with pytest.raises(runner.ContractError, match="ancestor|symlink"):
        runner.write_atomic_json(
            requested / "shards/example.json",
            {"status": "must-not-escape"},
            allowed_root=contract.pi2s_root,
        )
    assert not list(outside.rglob("*.json"))


def test_frozen_tokenizer_uses_isolated_ephemeral_cache_and_restores_environment(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(runner, tmp_path / "pi2s")
    source = contract.pi2s_root / "tokenizer.model"
    source.write_bytes(b"frozen tokenizer bytes")
    observed_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "PALIGEMMA_TOKENIZER_SHA256", observed_sha)
    monkeypatch.setattr(runner, "PALIGEMMA_TOKENIZER_BYTES", source.stat().st_size)
    contract.protocol["runtime_dependencies"] = {
        "paligemma_tokenizer": {
            "persistent_snapshot": "$PI2S_ROOT/tokenizer.model",
            "persistent_manifest": "$PI2S_ROOT/tokenizer.manifest.json",
            "execution_cache_policy": (
                "COPY_VERIFIED_SNAPSHOT_TO_EPHEMERAL_OPENPI_DATA_HOME_NO_NETWORK"
            ),
        }
    }
    monkeypatch.setenv("OPENPI_DATA_HOME", "/preexisting/cache")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "previous")
    previous_dont_write = sys.dont_write_bytecode
    with runner.frozen_openpi_runtime_asset(contract) as (
        asset,
        cache_root,
        evidence,
    ):
        outer = cache_root.parent
        assert os.environ["OPENPI_DATA_HOME"] == str(cache_root.resolve())
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
        assert sys.dont_write_bytecode is True
        assert (outer.stat().st_mode & 0o777) == 0o700
        assert asset == cache_root / "big_vision/paligemma_tokenizer.model"
        assert (asset.stat().st_mode & 0o777) == 0o444
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == observed_sha
        assert evidence["pre_execution_sha256"] == observed_sha
    assert os.environ["OPENPI_DATA_HOME"] == "/preexisting/cache"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "previous"
    assert sys.dont_write_bytecode is previous_dont_write
    assert not outer.exists()
    assert evidence["post_execution_status"] == "PASS"


def test_checkpoint_restore_is_strict_and_disables_extra_parameter_intersection() -> None:
    load_functions = [
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_runtime_model"
    ]
    assert len(load_functions) == 1
    calls = [
        node
        for node in ast.walk(load_functions[0])
        if isinstance(node, ast.Call) and (_qualified_name(node.func) or "").endswith(".load")
    ]
    assert len(calls) == 1
    keyword = next(
        keyword.value.value
        for keyword in calls[0].keywords
        if keyword.arg == "remove_extra_params" and isinstance(keyword.value, ast.Constant)
    )
    assert keyword is False


def test_atomic_json_concurrent_publishers_never_clobber(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "gradient.json"
    barrier = threading.Barrier(2)
    local = threading.local()
    real_link = runner.os.link
    real_getpid = runner.os.getpid

    def distinct_writer_id() -> int:
        return getattr(local, "writer_id", real_getpid())

    def simultaneous_link(source: Path, destination: Path, **kwargs: Any) -> None:
        barrier.wait(timeout=5)
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(runner.os, "getpid", distinct_writer_id)
    monkeypatch.setattr(runner.os, "link", simultaneous_link)
    successes: list[dict[str, int]] = []
    failures: list[BaseException] = []
    result_lock = threading.Lock()

    def publish(writer_id: int) -> None:
        local.writer_id = writer_id
        payload = {"writer": writer_id}
        try:
            runner.write_atomic_json(output, payload)
        except BaseException as exc:  # captured and asserted in the parent thread
            with result_lock:
                failures.append(exc)
        else:
            with result_lock:
                successes.append(payload)

    threads = [threading.Thread(target=publish, args=(90_001 + index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "concurrent publication deadlocked"

    assert len(successes) == 1
    assert len(failures) == 1 and isinstance(failures[0], FileExistsError)
    assert json.loads(output.read_text()) == successes[0]
    assert not list(tmp_path.glob(".gradient.json.tmp.*"))


def test_resume_binding_names_every_checkpoint_and_phase_uniquely(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    focus = _focus_rows()
    contract = _make_contract(runner, tmp_path, focus=focus)
    _stub_binding_authorities(runner, monkeypatch)
    bindings = [
        runner._artifact_binding(contract, row, stage)
        for row in focus
        for stage in ("actions", "gradients")
    ]
    assert len(bindings) == 12
    assert len({runner.canonical_sha(binding) for binding in bindings}) == 12
    assert {(row["model"], row["training_seed"]) for row in bindings} == set(EXPECTED_FOCUS)
    assert {row["stage"] for row in bindings} == {"actions", "gradients"}
    with pytest.raises(runner.ContractError, match="phase|stage"):
        runner._artifact_binding(contract, focus[0], "not-a-phase")


def test_ast_forbids_optimizer_and_checkpoint_writes() -> None:
    banned_call_suffixes = {
        "apply_gradients",
        "apply_updates",
        "save_checkpoint",
        "save_params",
        "save_state",
    }
    offenders: list[tuple[int, str]] = []
    parents = _parent_map()
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_name(node.func) or ""
        suffix = name.rpartition(".")[2]
        lowered = name.lower()
        if suffix in banned_call_suffixes:
            offenders.append((node.lineno, name))
        if suffix in {"save", "write"} and any(
            token in lowered for token in ("checkpoint", "checkpointer", "optimizer", "orbax")
        ):
            offenders.append((node.lineno, name))
        if suffix == "update" and any(token in lowered for token in ("optimizer", "optim", ".tx")):
            offenders.append((node.lineno, name))
        if suffix in {"write_bytes", "write_text", "savetxt", "save", "savez"}:
            enclosing = _enclosing_function(node, parents)
            if enclosing != "write_atomic_json":
                offenders.append((node.lineno, name))
        if suffix == "open" and node.args and isinstance(node.args[0], ast.Constant):
            mode = node.args[0].value
            if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                enclosing = _enclosing_function(node, parents)
                if enclosing != "write_atomic_json":
                    offenders.append((node.lineno, f"{name}({mode!r})"))
        if name == "write_atomic_json":
            replace = next((kw.value for kw in node.keywords if kw.arg == "replace"), None)
            if isinstance(replace, ast.Constant) and replace.value is True:
                offenders.append((node.lineno, "write_atomic_json(replace=True)"))
    assert offenders == []


@pytest.mark.skipif(not _defined("main"), reason="runner CLI is not implemented yet")
def test_ast_requires_explicit_execution_gate() -> None:
    constants = {
        node.value
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "--check" in constants
    assert "--execute" in constants
    required_mode_groups = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and (_qualified_name(node.func) or "").endswith("add_mutually_exclusive_group")
        and any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    assert required_mode_groups, "--check/--execute must be an explicit required choice"
    main_guards = []
    for node in TREE.body:
        if not isinstance(node, ast.If):
            continue
        rendered = ast.dump(node.test, include_attributes=False)
        if "__name__" in rendered and "__main__" in rendered:
            main_guards.append(node)
    assert len(main_guards) == 1
    guard_calls = [
        _qualified_name(node.func)
        for node in ast.walk(main_guards[0])
        if isinstance(node, ast.Call)
    ]
    assert "main" in guard_calls


@pytest.mark.skipif(not _defined("main"), reason="runner CLI is not implemented yet")
def test_cli_check_creates_no_output_and_cannot_import_jax_or_openpi(tmp_path: Path) -> None:
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import importlib.abc\n"
        f"BLOCKED = {sorted(HEAVY_IMPORT_ROOTS)!r}\n"
        "class BlockHeavy(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.partition('.')[0] in BLOCKED:\n"
        "            raise RuntimeError(f'forbidden check-mode import: {fullname}')\n"
        "        return None\n"
        "import sys\n"
        "sys.meta_path.insert(0, BlockHeavy())\n"
    )
    output = tmp_path / "must-not-exist"
    constants = {
        node.value
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    output_flags = [
        flag for flag in ("--output-dir", "--output-root", "--output") if flag in constants
    ]
    assert len(output_flags) == 1, "CLI must expose one explicit diagnostic output option"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(ROOT), environment.get("PYTHONPATH", "")]
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["NVIDIA_VISIBLE_DEVICES"] = "void"
    ungated = subprocess.run(
        [sys.executable, str(SCRIPT), output_flags[0], str(output)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert ungated.returncode != 0
    assert not os.path.lexists(output)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", output_flags[0], str(output)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["execution_performed"] is False
    assert report["jax_imported"] is False
    assert not os.path.lexists(output)


@pytest.mark.skipif(
    not _defined("enforce_global_budgets"),
    reason="global-budget enforcement hook is not implemented yet",
)
def test_global_budget_enforcement_is_exact_and_fails_closed(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _aggregate_contract(runner, tmp_path)
    _stub_binding_authorities(runner, monkeypatch)
    actions, gradients = _exact_shards(runner, contract)
    frozen = runner.canonical_sha([actions, gradients])
    assert runner.enforce_global_budgets(actions, gradients, contract) == EXPECTED_BUDGET
    assert runner.canonical_sha([actions, gradients]) == frozen

    mutations: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    missing_action = copy.deepcopy(actions)
    missing_action.pop()
    mutations.append((missing_action, copy.deepcopy(gradients)))
    duplicate_gradient = copy.deepcopy(gradients)
    duplicate_gradient.append(copy.deepcopy(duplicate_gradient[0]))
    mutations.append((copy.deepcopy(actions), duplicate_gradient))
    extra_record = copy.deepcopy(actions)
    extra_record[0]["records"].append(copy.deepcopy(extra_record[0]["records"][0]))
    mutations.append((extra_record, copy.deepcopy(gradients)))
    missing_aux = copy.deepcopy(gradients)
    for shard in missing_aux:
        if shard["binding"]["model"] in runner.AUXILIARY_MODELS:
            shard["records"][0]["auxiliary_applicable"] = False
            break
    mutations.append((copy.deepcopy(actions), missing_aux))
    for bad_actions, bad_gradients in mutations:
        with pytest.raises(runner.ContractError):
            runner.enforce_global_budgets(bad_actions, bad_gradients, contract)

    identity_mutations = []
    wrong_action_row = copy.deepcopy(actions)
    wrong_action_row[0]["records"][0]["global_row"] = 999
    identity_mutations.append((wrong_action_row, copy.deepcopy(gradients)))
    inconsistent_h_noise = copy.deepcopy(actions)
    contact_shard = next(shard for shard in inconsistent_h_noise if shard["focus"]["model"] == "B1")
    contact_shard["records"][32]["noise_sha256"] = "f" * 64
    identity_mutations.append((inconsistent_h_noise, copy.deepcopy(gradients)))
    cross_checkpoint_gradient_noise = copy.deepcopy(gradients)
    cross_checkpoint_gradient_noise[-1]["records"][0]["noise_sha256"] = "e" * 64
    cross_checkpoint_gradient_noise[-1]["records"][1]["noise_sha256"] = "e" * 64
    cross_checkpoint_gradient_noise[-1]["records"][2]["noise_sha256"] = "e" * 64
    identity_mutations.append((copy.deepcopy(actions), cross_checkpoint_gradient_noise))
    for bad_actions, bad_gradients in identity_mutations:
        with pytest.raises(runner.ContractError):
            runner.enforce_global_budgets(bad_actions, bad_gradients, contract)


@pytest.mark.skipif(
    not (_defined("aggregate_shards") and _defined("enforce_global_budgets")),
    reason="shard aggregation hooks are not implemented yet",
)
def test_aggregate_requires_all_twelve_immutable_shards(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _aggregate_contract(runner, tmp_path)
    _stub_binding_authorities(runner, monkeypatch)
    output = tmp_path / "output"
    _write_exact_shards(runner, output, contract)

    calls: list[tuple[Sequence[dict[str, Any]], Sequence[dict[str, Any]], Any]] = []
    real_enforce: Callable[..., Any] = runner.enforce_global_budgets

    def recording_enforce(
        action_shards: Sequence[dict[str, Any]],
        gradient_shards: Sequence[dict[str, Any]],
        observed_contract: Any | None = None,
    ) -> dict[str, int]:
        calls.append((action_shards, gradient_shards, observed_contract))
        return real_enforce(action_shards, gradient_shards, observed_contract)

    monkeypatch.setattr(runner, "enforce_global_budgets", recording_enforce)
    summary = runner.aggregate_shards(output, contract)
    assert len(calls) == 1
    assert real_enforce(*calls[0]) == EXPECTED_BUDGET
    assert summary["status"] == "PASS"
    assert summary["scientific_budget"] == EXPECTED_BUDGET

    unexpected = output / "shards/unregistered.actions.json"
    runner.write_atomic_json(unexpected, {"status": "must-not-be-ignored"})
    with pytest.raises(runner.ContractError, match="unexpected shard files"):
        runner.aggregate_shards(output, contract)
    unexpected.unlink()

    missing = runner.shard_path(output, "B0_43", "actions")
    missing.rename(tmp_path / missing.name)
    with pytest.raises(runner.ContractError):
        runner.aggregate_shards(output, contract)


def test_canonical_aggregate_payloads_are_complete_and_exact(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _aggregate_contract(runner, tmp_path)
    _stub_binding_authorities(runner, monkeypatch)
    output = tmp_path / "canonical"
    _write_exact_shards(runner, output, contract)

    payloads = runner.canonical_aggregate_payloads(output, contract)
    expected_schemas = {
        "prefix_contract_audit.json": runner.PREFIX_AUDIT_SCHEMA,
        "fixed_observation_interventions.json": runner.INTERVENTION_AUDIT_SCHEMA,
        "read_only_gradient_diagnostics.json": runner.READONLY_GRADIENT_SCHEMA,
    }
    assert list(payloads) == list(expected_schemas)
    expected_inputs = {
        (
            runner._checkpoint_id(focus),
            stage,
            "$PI2S_ROOT/"
            + runner.shard_path(output, runner._checkpoint_id(focus), stage)
            .resolve(strict=True)
            .relative_to(contract.pi2s_root.resolve(strict=True))
            .as_posix(),
        )
        for focus in contract.focus
        for stage in ("actions", "gradients")
    }
    expected_completeness = {
        "required_shards": 12,
        "validated_shards": 12,
        "all_focus_checkpoints": 6,
        "all_phases": ["actions", "gradients"],
        "status": "COMPLETE",
    }
    for name, payload in payloads.items():
        assert payload["schema"] == expected_schemas[name]
        assert payload["status"] == "PASS"
        assert payload["scientific_budget"] == EXPECTED_BUDGET
        assert payload["completeness"] == expected_completeness
        assert len(payload["inputs"]) == 12
        assert {
            (row["checkpoint_id"], row["phase"], row["file"]) for row in payload["inputs"]
        } == expected_inputs
        assert all(len(row["sha256"]) == 64 for row in payload["inputs"])
        assert all(value == "PASS" for value in payload["gates"].values())
    assert payloads["prefix_contract_audit.json"]["condition_batch_count"] == 22
    assert payloads["prefix_contract_audit.json"]["row_record_count"] == 704
    assert payloads["fixed_observation_interventions.json"]["record_count"] == 704
    gradients = payloads["read_only_gradient_diagnostics.json"]
    assert gradients["record_count"] == 72
    assert gradients["auxiliary_record_count"] == 48
    assert not any((output / name).exists() for name in expected_schemas)

    missing = runner.shard_path(output, "B2_43", "gradients")
    missing.rename(tmp_path / missing.name)
    with pytest.raises(runner.ContractError, match="required immutable shard missing"):
        runner.canonical_aggregate_payloads(output, contract)
    assert not any((output / name).exists() for name in expected_schemas)


def test_publish_aggregate_artifacts_no_clobber_and_resume_fills_missing(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _aggregate_contract(runner, tmp_path)
    _stub_binding_authorities(runner, monkeypatch)
    output = tmp_path / "publish"
    _write_exact_shards(runner, output, contract)
    canonical = runner.canonical_aggregate_payloads(output, contract)
    names = list(canonical)
    canonical_root = contract.pi2s_root / "artifacts"

    for name in names[:2]:
        runner.write_atomic_json(canonical_root / name, canonical[name])
    preserved = {name: (canonical_root / name).read_bytes() for name in names[:2]}
    with pytest.raises(FileExistsError, match="refusing to replace aggregate"):
        runner.publish_aggregate_artifacts(output, contract, resume=False)
    assert {(path.name) for path in canonical_root.glob("*.json")} == set(names[:2])
    assert not list(output.glob("*.json"))
    assert all((canonical_root / name).read_bytes() == value for name, value in preserved.items())

    result = runner.publish_aggregate_artifacts(output, contract, resume=True)
    assert result["status"] == "PASS"
    assert result["scientific_budget"] == EXPECTED_BUDGET
    assert set(result["artifacts"]) == {*names, "summary.json"}
    assert {name: result["artifacts"][name]["status"] for name in names} == {
        names[0]: "RESUMED_VERIFIED",
        names[1]: "RESUMED_VERIFIED",
        names[2]: "PUBLISHED",
    }
    assert result["artifacts"]["summary.json"]["status"] == "PUBLISHED"
    assert all((canonical_root / name).read_bytes() == value for name, value in preserved.items())
    assert all(
        result["artifacts"][name]["path"] == f"$PI2S_ROOT/artifacts/{name}" for name in names
    )

    expected_schemas = {
        "prefix_contract_audit.json": runner.PREFIX_AUDIT_SCHEMA,
        "fixed_observation_interventions.json": runner.INTERVENTION_AUDIT_SCHEMA,
        "read_only_gradient_diagnostics.json": runner.READONLY_GRADIENT_SCHEMA,
        "summary.json": runner.SUMMARY_SCHEMA,
    }
    assert {path.name for path in canonical_root.glob("*.json")} == set(names)
    assert {path.name for path in output.glob("*.json")} == {"summary.json"}
    for name, schema in expected_schemas.items():
        path = output / name if name == "summary.json" else canonical_root / name
        payload = json.loads(path.read_text())
        assert payload["schema"] == schema
        assert payload["status"] == "PASS"
        assert payload["scientific_budget"] == EXPECTED_BUDGET
    summary = json.loads((output / "summary.json").read_text())
    assert summary["canonical_artifacts"] == {
        name: {
            "sha256": hashlib.sha256((canonical_root / name).read_bytes()).hexdigest(),
            "schema": expected_schemas[name],
        }
        for name in names
    }

    before_second_resume = {
        name: (output / name if name == "summary.json" else canonical_root / name).read_bytes()
        for name in expected_schemas
    }
    resumed = runner.publish_aggregate_artifacts(output, contract, resume=True)
    assert all(value["status"] == "RESUMED_VERIFIED" for value in resumed["artifacts"].values())
    assert {
        name: (output / name if name == "summary.json" else canonical_root / name).read_bytes()
        for name in expected_schemas
    } == before_second_resume

    corrupted_path = canonical_root / names[0]
    corrupted = json.loads(corrupted_path.read_text())
    corrupted["status"] = "FAIL"
    corrupted_path.chmod(0o644)
    corrupted_path.write_text(json.dumps(corrupted, sort_keys=True))
    corrupted_bytes = corrupted_path.read_bytes()
    with pytest.raises(runner.ContractError, match="binding/content drift"):
        runner.publish_aggregate_artifacts(output, contract, resume=True)
    assert corrupted_path.read_bytes() == corrupted_bytes
