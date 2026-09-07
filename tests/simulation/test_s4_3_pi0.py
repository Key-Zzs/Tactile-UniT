from pathlib import Path

import importlib.util


ROOT = Path(__file__).resolve().parents[2]
DEXJOCO = ROOT / "third_party/dexjoco"


def read_official(relative: str) -> str:
    return (DEXJOCO / relative).read_text(encoding="utf-8")


def test_official_pi05_entrypoints_are_present() -> None:
    required = (
        "openpi/install.bash",
        "openpi/scripts/train.py",
        "openpi/scripts/serve_policy.py",
        "openpi/scripts/compute_norm_stats.py",
        "dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py",
        "dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py",
        "configs/rand_obj/pinch_tongs.yaml",
    )
    assert all((DEXJOCO / path).is_file() for path in required)


def test_official_single_arm_state_action_contract() -> None:
    wrapper = read_official("dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py")
    evaluator = read_official("dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py")
    assert 'state = env_obs["state"][:23]' in wrapper
    assert "xyz = action[:3]" in wrapper
    assert "rotvec = action[3:6]" in wrapper
    assert "hand = action[6:22]" in wrapper
    assert "R.from_rotvec(rotvec).as_quat(scalar_first=True)" in wrapper
    assert "action_horizon = 30" in evaluator
    assert "replan_ratio: float = 0.8" in evaluator


def test_official_pinch_tongs_pi05_lora_training_contract() -> None:
    config = read_official("openpi/src/openpi/training/dexjoco_configs.py")
    yaml = read_official("openpi/config.yaml")
    task = read_official("configs/rand_obj/pinch_tongs.yaml")
    assert 'name="pinch_tongs"' in config
    assert "pi05=True" in config
    assert "action_horizon=30" in config
    assert 'paligemma_variant="gemma_2b_lora"' in config
    assert 'action_expert_variant="gemma_300m_lora"' in config
    assert "batch_size: 32" in yaml
    assert "single_arm_steps: 30000" in yaml
    assert "base: front" in task
    assert "wrist: wrist" in task
    assert "robot_type: single_arm" in task


def test_pi0_audit_uses_symbolic_tracked_paths() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_3_pi0.py").read_text(encoding="utf-8")
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source
    assert 'Tactile-UniT_inputs": False' in source
    assert 'previous_ACT_protocol_reused": False' in source


def test_official_checkpoint_manifest_contract() -> None:
    module_path = ROOT / "scripts/simulation/validate_s4_3_pi0_checkpoint.py"
    spec = importlib.util.spec_from_file_location("validate_s4_3_pi0_checkpoint", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OFFICIAL_REPOSITORY == "DexJoCo/DexJoCo-Pi05"
    assert module.OFFICIAL_SUBTREE == "pi05_dexjoco_ckpt/pinch_tongs/**"
    assert module.EXPECTED_FILE_COUNT == 57
    assert module.EXPECTED_BYTES == 9_561_967_083
    source = module_path.read_text(encoding="utf-8")
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_official_checkpoint_probe_is_strictly_pi0() -> None:
    source = (ROOT / "scripts/simulation/probe_s4_3_pi0_official.py").read_text(encoding="utf-8")
    assert 'env_name=cfg["env_name"]' in source
    assert "seed=0" in source
    assert "rand_full=False" in source
    assert "randomize_dynamics=False" in source
    assert "actions.shape == (30, 22)" in source
    assert "environment_action.shape == (23,)" in source
    assert "Tactile-UniT" not in source
    assert "raw_tactile" not in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_official_eval_summary_contract() -> None:
    source = (ROOT / "scripts/simulation/summarize_s4_3_pi0_eval.py").read_text(encoding="utf-8")
    assert '"episodes_20": episode_count == 20' in source
    assert '"success_rate_mean": 0.24' in source
    assert '"paper_episodes_per_task": 50' in source
    assert '"rand_full": False' in source
    assert '"randomize_dynamics": False' in source
    assert '"replan_ratio": 0.8' in source
    assert 'choices=("official", "reproduced")' in source
    assert "if episode_group:" in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_official_training_asset_manifest_contract() -> None:
    module_path = ROOT / "scripts/simulation/validate_s4_3_pi0_training_assets.py"
    spec = importlib.util.spec_from_file_location("validate_s4_3_pi0_training_assets", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DATASET_REPOSITORY == "DexJoCo/DexJoCo-Datasets-LeRobot"
    assert module.DATASET_REVISION == "5a57c54e55dc5858dd9fb949c5f67c0c9716e6b3"
    assert module.EXPECTED_DATASET_FILE_COUNT == 10
    assert module.EXPECTED_DATASET_BYTES == 865_831_099
    assert module.BASE_MODEL_REPOSITORY == "DexJoCo/DexJoCo-Pi05"
    assert module.BASE_MODEL_REVISION == "8d253e04e1b82c452c5804273939ff7f002f217f"
    assert module.EXPECTED_BASE_MODEL_FILE_COUNT == 29
    assert module.EXPECTED_BASE_MODEL_BYTES == 12_441_749_581
    source = module_path.read_text(encoding="utf-8")
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_tracked_official_pi05_protocol_is_locked() -> None:
    import json

    protocol_path = ROOT / "configs/simulation/s4_3_pi0_official_pi05_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["scope"] == {
        "task": "pinch_tongs",
        "regime": "rand_obj",
        "robot": "single_arm",
        "canonical_runs": 1,
        "tactile_inputs": False,
        "previous_act_protocol_reused": False,
    }
    assert protocol["training"]["num_train_steps"] == 30_000
    assert protocol["training"]["batch_size"] == 32
    assert protocol["training"]["seed"] == 42
    assert protocol["training"]["save_interval"] == 10_000
    assert protocol["training"]["fsdp_devices"] == 1
    assert protocol["training"]["runtime_parallelism"] == {
        "visible_device_count": 2,
        "mode": "data_parallel",
        "data_parallel_replicas": 2,
        "fsdp_axis_size": 1,
        "global_batch_size": 32,
        "per_device_batch_size": 16,
    }
    assert protocol["model"]["base_model_44d_used"] is False
    assert protocol["evaluation"]["episodes"] == 20
    assert protocol["evaluation"]["seed"] == 0
    assert "/" + "home/" not in protocol_path.read_text(encoding="utf-8")


def test_training_config_freeze_has_no_private_paths() -> None:
    source = (ROOT / "scripts/simulation/freeze_s4_3_pi0_training_config.py").read_text(encoding="utf-8")
    assert "NORM_RECOMPUTE_TOLERANCE = 1e-3" in source
    assert '"official_algorithmic_defaults_overridden": False' in source
    assert '"canonical_norm_stats_selected": True' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_environment_audit_keeps_frozen_environments_locked() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_3_pi0_environment.py").read_text(encoding="utf-8")
    assert '"unit": "3a119880cd4d661259d9476b0d3224302e086ae899ca77250497b5f42fbf6f9b"' in source
    assert '"tactile-unit-dexjoco": "7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78"' in source
    assert '"frozen_environments_mutated": not frozen_unchanged' in source
    assert 'openpi_python.parents[1].name != "openpi"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_pi0_visuals_use_only_available_results() -> None:
    source = (ROOT / "scripts/simulation/visualize_s4_3_pi0.py").read_text(encoding="utf-8")
    assert "official_pi05_pipeline_dataflow.png" in source
    assert "official_state_action_contract.png" in source
    assert "official_checkpoint_rollout_summary.png" in source
    assert "reproduced_training_curves.png" in source
    assert "official_vs_reproduced_rollout_comparison.png" in source
    assert 'evaluation["successes"]' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_training_launch_audit_requires_first_finite_step() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_3_pi0_training_launch.py").read_text(encoding="utf-8")
    assert '"first_step_logged": metrics is not None' in source
    assert '"finite_first_step": finite' in source
    assert '"expected_steps": 30000' in source
    assert '"expected_final_checkpoint": symbolic(args.checkpoint_dir / "29999")' in source
    assert 'physical_gpus = [int(value) for value in args.gpu.split(",")]' in source
    assert '"physical_gpus": physical_gpus' in source
    assert 'environment.get("CUDA_VISIBLE_DEVICES") == gpu_mask' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_base_download_is_selective_resumable_and_strict() -> None:
    source = (ROOT / "scripts/simulation/download_s4_3_pi0_base.sh").read_text(encoding="utf-8")
    assert "hf download DexJoCo/DexJoCo-Pi05" in source
    assert "--include 'pi05_base/**'" in source
    assert "expected_files=29" in source
    assert "expected_bytes=12441749581" in source
    assert "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS" in source
    assert "S4_3_DOWNLOAD_USE_CONFIGURED_PROXY" in source
    assert "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy" in source
    assert "while true" in source
    assert "pi05_base_action_dim_44" not in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_base_model_smoke_is_a_real_cpu_cold_load() -> None:
    source = (ROOT / "scripts/simulation/probe_s4_3_pi0_base.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("JAX_PLATFORMS", "cpu")' in source
    assert "openpi_model.restore_params(PARAMS, restore_type=np.ndarray)" in source
    assert '"all_finite": nonfinite == 0' in source
    assert '"pi05_language_model"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_training_completion_audit_requires_exact_final_state() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_3_pi0_training_completion.py").read_text(encoding="utf-8")
    assert "EXPECTED_CHECKPOINT_STEPS = [10_000, 20_000, 29_999]" in source
    assert '"train_state_step_30000": train_step == 30_000' in source
    assert '"metric_records_300": len(metrics) == 300' in source
    assert '"final_save_finalized"' in source
    assert '"algorithm_defaults_unchanged"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_reproduced_checkpoint_smoke_requires_official_lora_structure() -> None:
    source = (ROOT / "scripts/simulation/probe_s4_3_pi0_reproduced.py").read_text(encoding="utf-8")
    assert "openpi_model.restore_params(PARAMS, restore_type=np.ndarray)" in source
    assert '"lora_arrays_20": len(lora) == 20' in source
    assert '"gemma_2b_lora_present"' in source
    assert '"gemma_300m_lora_present"' in source
    assert '"action_width_32"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_official_vs_reproduced_comparison_is_paired_and_strict() -> None:
    source = (ROOT / "scripts/simulation/compare_s4_3_pi0.py").read_text(encoding="utf-8")
    assert '"official_only_success"' in source
    assert '"reproduced_only_success"' in source
    assert '"exact_mcnemar_two_sided_pvalue"' in source
    assert '"local_success_nontrivial"' in source
    assert '"wilson_intervals_overlap"' in source
    assert '"S4_3_PI0_FULL_OFFICIAL_BASELINE_REPRODUCTION"' in source
    assert '"S4_3_PI0_OFFICIAL_TRAINING_REPRODUCTION_GAP"' in source
    assert '"READY_WITH_WARNINGS"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_eval_client_uses_an_isolated_official_source_overlay() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_3_pi0_eval_environment.py").read_text(encoding="utf-8")
    assert '"official_openpi_client_source"' in source
    assert '"dexjoco_revision_frozen"' in source
    assert '"no package was installed into either frozen environment"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_final_integrity_rehashes_s4_2_and_frozen_source() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_3_pi0_final_integrity.py").read_text(encoding="utf-8")
    assert 'current["matches_pi0_before"]' in source
    assert '"s4_2_byte_identical"' in source
    assert '"dexjoco_revision_frozen"' in source
    assert '"dexjoco_clean"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source


def test_eval_runtime_audit_distinguishes_runtime_from_shutdown_interrupt() -> None:
    source = (ROOT / "scripts/simulation/audit_s4_3_pi0_eval_runtime.py").read_text(encoding="utf-8")
    assert '"official_client_episode_sequence_1_to_20"' in source
    assert '"client_reported_2_of_20"' in source
    assert '"server_no_runtime_traceback"' in source
    assert '"shutdown_interrupt_only"' in source
    assert '"gpu_lock_released"' in source
    assert "/" + "home/" not in source
    assert "/" + "mnt/" not in source
