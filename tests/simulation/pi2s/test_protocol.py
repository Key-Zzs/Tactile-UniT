from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "configs/simulation/pi2s/diagnostic_protocol.json"
SCRIPT_PATH = ROOT / "scripts/simulation/audit_pi2s_contracts.py"
DIAGNOSTIC_IMPLEMENTATIONS = {
    "analyze_pi2s_historical_evidence": ROOT
    / "scripts/simulation/analyze_pi2s_historical_evidence.py",
    "audit_pi2s_initialization_rng": ROOT / "scripts/simulation/audit_pi2s_initialization_rng.py",
    "hash_pi2s_intermediate_checkpoints": ROOT
    / "scripts/simulation/hash_pi2s_intermediate_checkpoints.py",
    "run_pi2s_h_parity": ROOT / "scripts/simulation/run_pi2s_h_parity.py",
    "run_pi2s_model_diagnostics": ROOT / "scripts/simulation/run_pi2s_model_diagnostics.py",
}


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_protocol_is_post_hoc_bounded_and_read_only() -> None:
    protocol = _protocol()
    assert protocol["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert protocol["scientific_role"] == "POST_HOC_DEVELOPMENT_DIAGNOSIS"
    assert protocol["formal_track_a_outcomes_read_only"] == 3000
    assert protocol["training_budget"] == {
        "optimizer_updates": 0,
        "policy_runs": 0,
        "teacher_runs": 0,
    }
    assert protocol["development_rollouts"]["base"]["canonical_tuples"] == 72
    assert protocol["development_rollouts"]["optional_h_intervention"]["canonical_tuples"] == 48
    assert protocol["development_rollouts"]["maximum_canonical_tuples"] == 120
    assert protocol["development_rollouts"]["formal_success_rate_replacement_allowed"] is False
    assert protocol["completion_policy"]["automatic_budget_expansion"] is False
    assert protocol["completion_policy"]["automatic_retraining"] is False


def test_checkpoint_and_sample_sets_are_exactly_frozen() -> None:
    protocol = _protocol()
    finals = protocol["all_final_checkpoints"]
    assert len(finals) == 15
    assert len({(row["model"], row["training_seed"]) for row in finals}) == 15
    assert all(row["directory_step"] == 29999 for row in finals)
    assert all(row["expected_restored_train_step"] == 30000 for row in finals)
    assert all(row["params_present"] is True for row in finals)
    assert all(row["train_state_present"] is True for row in finals)
    assert all(row["path"].startswith("$EXPERIMENT_ROOT/") for row in finals)
    assert all(len(row["tree_sha256"]) == 64 for row in finals)
    focus = protocol["focus_checkpoints"]
    assert [(row["model"], row["training_seed"]) for row in focus] == [
        ("B0", 43),
        ("B_VA27", 43),
        ("B1", 43),
        ("B_HVA", 42),
        ("B_HVA", 43),
        ("B2", 43),
    ]
    finals_by_key = {(row["model"], row["training_seed"]): row for row in finals}
    assert all(row == finals_by_key[(row["model"], row["training_seed"])] for row in focus)
    assert [
        (row["model"], row["training_seed"], row["directory_step"])
        for row in protocol["intermediate_checkpoints"]
    ] == [
        ("B_HVA", 42, 10000),
        ("B_HVA", 42, 20000),
        ("B_HVA", 43, 10000),
        ("B_HVA", 43, 20000),
    ]
    assert all(
        row["identity_status"] == "FULL_CONTENT_HASH_VERIFIED"
        for row in protocol["intermediate_checkpoints"]
    )
    assert all(len(row["tree_sha256"]) == 64 for row in protocol["intermediate_checkpoints"])
    assert all(
        row["closed_loop_authorized"] is False for row in protocol["intermediate_checkpoints"]
    )
    samples = protocol["samples"]
    assert samples["train_windows_count"] == 512
    assert len(set(samples["train_windows"])) == 512
    assert [row["row"] for row in samples["train_window_records"]] == samples["train_windows"]
    assert len({row["source_group_id"] for row in samples["train_window_records"]}) == 100
    assert samples["train_source_groups_covered"] == 100
    assert len(samples["h_parity_rows"]) == 64
    assert len(samples["fixed_observation_rows"]) == 32
    assert samples["fixed_observation_contact_strata"] == {"active": 16, "inactive": 16}
    assert set(samples["fixed_observation_contact_labels"]) == {
        str(row) for row in samples["fixed_observation_rows"]
    }
    assert list(samples["fixed_observation_contact_labels"].values()).count("active") == 16
    assert list(samples["fixed_observation_contact_labels"].values()).count("inactive") == 16
    assert len(samples["gradient_batches"]) == 4
    assert all(len(batch) == 2 for batch in samples["gradient_batches"])
    assert samples["gradient_minibatches_per_checkpoint"] == 4
    gradient_rows = [row for batch in samples["gradient_batches"] for row in batch]
    assert len(gradient_rows) == len(set(gradient_rows)) == 8
    assert set(gradient_rows) <= set(samples["fixed_observation_rows"])
    assert len(samples["policy_dev_windows"]) == 256
    assert samples["policy_dev_windows_count"] == 256
    assert samples["policy_dev_source_groups_covered"] == 5
    assert samples["policy_dev_attempts_covered"] == 25
    assert len({row["source_group_id"] for row in samples["policy_dev_windows"]}) == 5
    assert len({row["attempt_id"] for row in samples["policy_dev_windows"]}) == 25
    assert samples["policy_dev_status"] == "AVAILABLE_FOR_H_ACTION_DOMAIN_DISTRIBUTION_ONLY"
    assert samples["policy_dev_parent_cohort"] == (
        "SUCCESS_ONLY_NATIVE_SUCCESS_EXPERT_TRAJECTORIES"
    )
    assert samples["policy_dev_row_selection_scope"] == (
        "OUTCOME_BLIND_ONLY_WITHIN_THE_PREEXISTING_SUCCESS_CONDITIONED_PARENT_COHORT"
    )
    assert samples["policy_dev_population_or_failure_comparison_allowed"] is False
    assert (
        samples["policy_dev_fixed_forward_compatibility"]
        == "NOT_COMPATIBLE_WITH_OFFICIAL_PI05_OBSERVATION_CONTRACT"
    )
    assert protocol["data_contract"]["source_group_mapping_status"] == "AVAILABLE_AND_VERIFIED"
    assert (
        protocol["data_contract"]["policy_dev_original_pi05_train_overlap"]["status"]
        == "OVERLAPS_ORIGINAL_POLICY_TRAIN"
    )


def test_h_interventions_and_noise_are_precommitted() -> None:
    protocol = _protocol()
    conditions = protocol["h_conditions"]
    assert conditions["conditions"] == [
        "correct",
        "train_mean",
        "same_episode_lag5",
        "other_episode",
        "zero",
    ]
    fixed = {str(row) for row in protocol["samples"]["fixed_observation_rows"]}
    assert set(conditions["same_episode_lag5_row_map"]) == fixed
    assert set(conditions["other_episode_row_map"]) == fixed
    for row in conditions["same_episode_lag5_row_map"].values():
        if not row["bootstrap_affected"]:
            assert row["current_control_tick"] - row["actual_control_tick"] == 5
            assert row["requested_control_tick"] == row["actual_control_tick"]
    assert conditions["zero_label"] == "OUT_OF_DISTRIBUTION_NUMERICAL_DIAGNOSTIC"
    sampling = protocol["fixed_sampling"]
    assert sampling["noise_shape"] == [30, 32]
    assert sampling["policy_output_shape"] == [30, 22]
    assert len(sampling["noise_seeds"]) == 32
    assert sampling["shared_noise_across_h_conditions"] is True
    gradient = protocol["gradient_diagnostic"]
    assert gradient["flow_times"] == [0.1, 0.5, 0.9]
    assert gradient["nuisance_noise_distribution"] == "jax.random.normal"
    assert gradient["nuisance_noise_shape_per_minibatch"] == [2, 30, 32]
    assert gradient["nuisance_noise_dtype"] == "float32"
    assert gradient["optimizer_step"] is False
    assert gradient["state_mutation_allowed"] is False
    assert gradient["required_before_after_hash_match"] is True
    assert "config.trainable_filter" in gradient["parameter_membership"]
    assert "physical_auxiliary and action_out_proj" in gradient["main_auxiliary_cosine_scope"]
    assert "finite non-None" in gradient["main_auxiliary_cosine_scope"]


def test_encoder_snapshot_and_metric_formulas_are_frozen() -> None:
    protocol = _protocol()
    prefix = protocol["prefix_integration_contract"]
    assert prefix["adapter_output_shape_per_observation"] == [8, 2048]
    assert prefix["placement"] == "APPEND_AFTER_UNCHANGED_OFFICIAL_PREFIX"
    assert prefix["input_mask_for_added_tokens"] is True
    assert prefix["autoregressive_mask_for_added_tokens"] is False
    assert prefix["position_index_rule"] == "cumsum(input_mask)-1"
    assert "padding does not advance" in prefix["padding_interaction"]
    assert prefix["state_dimension_changed"] is False
    assert prefix["action_dimension_changed"] is False
    encoder = protocol["encoder_contract"]
    assert encoder["input_shape"] == [26, 30]
    assert encoder["output_shape"] == [256]
    assert len(encoder["accepted_et_checkpoint_sha256"]) == 64
    assert set(encoder["paths"]) == {
        "cached_training_sidecar",
        "direct_et",
        "live_sidecar",
        "shared_teacher_loader",
    }
    snapshot = protocol["source_manifests"]["persistent_input_snapshot"]
    assert snapshot["status"] == "COMPLETE_VERIFIED"
    assert snapshot["files"] == 38
    assert snapshot["bytes"] > 80_000_000
    tolerance = protocol["numeric_tolerances"]
    assert tolerance["source"] == (
        "PRE_RESULT_ENGINEERING_TOLERANCE_NO_CLOSED_LOOP_OUTCOME_CONSULTED"
    )
    assert tolerance["post_hoc_relaxation_allowed"] is False
    h_metrics = protocol["metrics"]["h_parity"]
    assert h_metrics["absolute_error_quantile_levels"] == [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    action_metrics = protocol["metrics"]["fixed_observation_actions"]
    assert action_metrics["chunk_l2"]["early"].endswith("positions 0:10")
    assert action_metrics["chunk_l2"]["late"].endswith("positions 20:30")
    assert (
        action_metrics["runtime_clipped_fraction"]
        == "N/A_NO_EXPLICIT_RUNTIME_ACTION_CLIP_IN_POLICY_INFER"
    )
    assert action_metrics["post_hoc_threshold"] is None
    assert protocol["metrics"]["prefix_contract"] == [
        "official_prefix_unchanged",
        "added_token_count",
        "input_mask",
        "autoregressive_mask",
        "position_indices_append_after_official_prefix",
        "contact_state_pre_layernorm_l2",
        "contact_state_post_layernorm_l2",
        "image_token_l2_at_width_2048",
        "language_token_l2_at_width_2048",
        "contact_token_l2_at_width_2048",
    ]
    assert (
        protocol["metrics"]["gradients"]["preprocess_mode"]
        == "DETERMINISTIC_EVAL_PREPROCESS_NO_AUGMENTATION"
    )
    gradient = protocol["gradient_diagnostic"]
    assert gradient["auxiliary_loss_by_model"]["B0"] == "N/A_NO_AUXILIARY_OBJECTIVE"
    assert gradient["auxiliary_loss_by_model"]["B1"] == "N/A_NO_AUXILIARY_OBJECTIVE"
    assert "va_shared_target" in gradient["auxiliary_loss_by_model"]["B_VA27"]["target"]
    assert "va_shared_target" in gradient["auxiliary_loss_by_model"]["B_HVA"]["target"]
    assert "contact_shared_target" in gradient["auxiliary_loss_by_model"]["B2"]["target"]


def test_runtime_dependencies_are_content_addressed_and_recoverable() -> None:
    protocol = _protocol()
    runtime = protocol["runtime_dependencies"]
    client = runtime["openpi_client"]
    assert client["required_distribution_version"] == "0.1.0"
    assert client["accepted_python_files"] == 13
    assert client["accepted_python_tree_sha256"] == (
        "92fa31ab5335cd7613d656fcea70569e7797a73f6584f896ab7ff45a3ebf5cb4"
    )
    assert client["accepted_image_tools_sha256"] == (
        "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
    )
    tokenizer = runtime["paligemma_tokenizer"]
    assert tokenizer["bytes"] == 4_264_023
    assert tokenizer["sha256"] == (
        "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
    )
    assert tokenizer["persistent_snapshot"].startswith("$PI2S_ROOT/")
    assert tokenizer["persistent_manifest"].startswith("$PI2S_ROOT/")
    assert len(tokenizer["persistent_manifest_sha256"]) == 64
    assert tokenizer["required_sentencepiece_version"] == "0.2.2"
    assert runtime["execution_environment"]["pip_freeze_all_sha256"] == (
        "39664447f71898e7ae8c0d7e41a79d74b13fe009546a8efcd69504ca64c91e89"
    )


def test_missing_observability_is_not_fabricated() -> None:
    protocol = _protocol()
    stages = protocol["historical_stage_observability"]
    assert stages["object_contact_ever"] == "AVAILABLE_EPISODE_AGGREGATE"
    assert stages["max_native_pinch_count"] == "AVAILABLE_EPISODE_AGGREGATE"
    assert stages["approach"] == "NOT_AVAILABLE"
    assert stages["stable_grasp"] == "NOT_AVAILABLE"
    assert stages["lift"] == "NOT_AVAILABLE"
    assert stages["exact_first_failure_step"] == "NOT_AVAILABLE"
    assert protocol["missing_inputs"]["attention_weights"].startswith("NOT_AVAILABLE")


def test_protocol_uses_symbolic_paths_and_matches_generator_bytes() -> None:
    protocol = _protocol()
    text = PROTOCOL_PATH.read_text()
    assert "/home/" not in text
    assert "/mnt/" not in text
    assert "$EXPERIMENT_ROOT/" in text
    assert "$REPO_ROOT/" in text
    assert "$PI2S_ROOT/" in text
    assert (
        protocol["diagnostic_script_sha256"] == hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    )
    assert protocol["diagnostic_implementation_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in DIAGNOSTIC_IMPLEMENTATIONS.items()
    }
    selection = {
        "train_records": protocol["samples"]["train_window_records"],
        "policy_dev_records": protocol["samples"]["policy_dev_windows"],
        "fixed": protocol["samples"]["fixed_observation_rows"],
        "gradient": protocol["samples"]["gradient_batches"],
        "resets": protocol["development_rollouts"]["base"]["reset_specs"],
        "noise": protocol["fixed_sampling"]["noise_seeds"],
        "gradient_noise": protocol["gradient_diagnostic"]["nuisance_noise_seeds_by_minibatch"],
        "train_mean_h": protocol["h_conditions"]["train_mean_h_float32_sha256"],
        "lag5": protocol["h_conditions"]["same_episode_lag5_row_map"],
        "other_episode": protocol["h_conditions"]["other_episode_row_map"],
    }
    assert protocol["selection_sha256"] == _canonical_sha(selection)
    assert protocol["samples"]["train_window_records_sha256"] == _canonical_sha(
        protocol["samples"]["train_window_records"]
    )
    assert protocol["samples"]["policy_dev_window_records_sha256"] == _canonical_sha(
        protocol["samples"]["policy_dev_windows"]
    )
    assert len(protocol["samples"]["train_window_records_sha256"]) == 64
    assert len(protocol["samples"]["policy_dev_window_records_sha256"]) == 64
