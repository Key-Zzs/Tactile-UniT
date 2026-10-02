from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]


def load_evaluate():
    path = ROOT / "scripts/simulation/pi2b_policy/evaluate.py"
    spec = importlib.util.spec_from_file_location("pi2b_policy_evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_job_matrix_is_exact_and_pre_registered():
    module = load_evaluate()
    assert len(module.JOBS) == 60
    assert len(set(module.JOBS)) == 60
    assert {row[0] for row in module.JOBS} == set(module.MODEL_ORDER)
    assert {row[1] for row in module.JOBS} == {42, 43, 44}
    assert {row[2] for row in module.JOBS} == {16, 17, 18, 19}


def test_runtime_modes_do_not_send_contact_to_no_h_controls():
    module = load_evaluate()
    assert module.RUNTIME_MODES["B0"] == "NONE"
    assert module.RUNTIME_MODES["B_VA27"] == "NONE"
    assert all(
        module.RUNTIME_MODES[model] == "CONTACT_STATE_TOKENS"
        for model in ("B1", "B_HVA", "B2")
    )


def test_determinism_environment_matches_accepted_pi2n_contract():
    module = load_evaluate()
    assert module.XLA_FLAGS == (
        "--xla_gpu_deterministic_ops=true",
        "--xla_gpu_exclude_nondeterministic_ops=true",
        "--xla_gpu_autotune_level=0",
    )


def test_parallel_attempt_paths_are_disjoint_and_dexjoco_is_bound():
    module = load_evaluate()
    first = module.attempt_paths("B0", 42, 16, 1)
    second = module.attempt_paths("B1", 42, 16, 1)
    assert first.ARTIFACTS != second.ARTIFACTS
    assert first.LOGS != second.LOGS
    assert first.CACHE != second.CACHE
    assert first.TMP != second.TMP
    assert (module.DEXJOCO / "configs/rand_obj/pinch_tongs.yaml").is_file()
    assert module.EVAL_PYTHON.is_file()


def test_publish_file_preserves_source_and_refuses_overwrite():
    module = load_evaluate()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.json"
        destination = root / "nas" / "artifact.json"
        source.write_bytes(b'{"status":"PASS"}\n')
        record = module.publish_file(source, destination)
        assert source.read_bytes() == destination.read_bytes()
        assert record["source_preserved"] is True
        assert record["destination_fsynced_before_atomic_rename"] is True
        try:
            module.publish_file(source, destination)
        except FileExistsError:
            pass
        else:
            raise AssertionError("publish_file overwrote an existing destination")
