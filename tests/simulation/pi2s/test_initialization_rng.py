from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Barrier

import pytest

from scripts.simulation import audit_pi2s_initialization_rng as audit


@pytest.fixture(scope="module")
def artifact() -> dict:
    return audit.build_artifact()


def test_audit_covers_exact_fifteen_rows_and_keeps_source_cohorts_distinct(
    artifact: dict,
) -> None:
    assert audit.OUTPUT.name == "initialization_rng_audit_v2.json"
    assert artifact["schema"] == "tactile3d-unit.s4-3-pi2s-initialization-rng-audit.v2"
    rows = artifact["per_checkpoint"]
    assert len(rows) == 15
    assert {(row["model"], row["training_seed"]) for row in rows} == {
        (model, seed) for model in audit.MODELS for seed in audit.SEEDS
    }
    findings = artifact["cross_run_findings"]
    assert findings["seed43_vs_seed44_source_identity"] == (
        "SAME_IMMUTABLE_TRACK_A_SOURCE_SNAPSHOT_VERIFIED"
    )
    assert findings["seed42_vs_seed43_44_source_identity"] == "NOT_SOURCE_IDENTICAL"
    assert findings["same_numeric_seed_across_modes"] == (
        "DOES_NOT_PROVE_BYTE_IDENTICAL_INITIALIZATION"
    )


def test_rng_static_source_is_not_promoted_to_runtime_subkey_evidence(artifact: dict) -> None:
    track_contract = artifact["track_a_source_and_rng_contract"]["rng_control_flow"]
    assert track_contract["not_runtime_evidence"] is True
    statements = {row["statement"] for row in track_contract["statements"]}
    assert "rng = jax.random.key(config.seed)" in statements
    assert "train_rng, init_rng = jax.random.split(rng)" in statements
    assert "train_rng = jax.random.fold_in(rng, state.step)" in statements

    for row in artifact["per_checkpoint"]:
        assert row["rng"]["realized_root_key_data"] == "UNVERIFIABLE_NOT_PERSISTED"
        assert row["rng"]["realized_model_rng_data"] == "UNVERIFIABLE_NOT_PERSISTED"
        assert row["rng"]["flow_noise_subkey_trace"] == "UNVERIFIABLE_NOT_PERSISTED"
        assert "do not prove byte-identical initialization" in row["claim_boundary"]


def test_seed42_source_claims_match_available_identity_strength(artifact: dict) -> None:
    rows = {row["model"]: row for row in artifact["per_checkpoint"] if row["training_seed"] == 42}
    assert rows["B0"]["rng"]["static_control_flow_reference"] == (
        "SEED42_B0_STATIC_RNG_CONTROL_FLOW"
    )
    for model in ("B_VA27", "B1", "B_HVA", "B2"):
        assert rows[model]["rng"]["static_control_flow_reference"].startswith("UNVERIFIABLE_")
        assert rows[model]["source_identity"]["trainer_runtime_identity"].startswith(
            "UNVERIFIABLE_"
        )
    assert rows["B1"]["initialization"]["loader"]["evidence_kind"].startswith("INFERENCE_")
    assert rows["B2"]["initialization"]["loader"]["evidence_kind"].startswith("INFERENCE_")


def test_recorded_b2_calibration_keys_are_not_relabelled_as_training_trace(
    artifact: dict,
) -> None:
    calibration = artifact["seed42_source_contracts"]["B2"]["pretraining_calibration"]
    assert calibration["scope"] == "B2 lambda calibration only; not a training PRNG trace"
    assert len(calibration["recorded_rng_keys"]) == 4
    assert [row["iterator_batch_index"] for row in calibration["recorded_rng_keys"]] == [
        0,
        1,
        2,
        3,
    ]
    assert all(len(row["loss_rng_key_data"]) == 2 for row in calibration["recorded_rng_keys"])
    b2 = next(
        row
        for row in artifact["per_checkpoint"]
        if row["model"] == "B2" and row["training_seed"] == 42
    )
    assert b2["rng"]["per_step_loss_subkey_trace"] == "UNVERIFIABLE_NOT_PERSISTED"


def test_input_manifest_sha_accounting_is_explicit_and_exhaustive(artifact: dict) -> None:
    rows = artifact["input_manifest"]
    summary = artifact["input_manifest_summary"]
    assert summary["observed_sha256_recorded"] == len(rows)
    assert summary["provided_expected_sha256_match"] + summary[
        "expected_sha256_not_provided"
    ] == len(rows)
    for row in rows:
        if row["expected_sha256"] is None:
            assert row["expected_sha256_status"] == "NOT_PROVIDED"
        else:
            assert row["expected_sha256_status"] == "MATCH"
            assert row["sha256"] == row["expected_sha256"]


def test_training_sidecars_are_bound_to_persistent_source_snapshots(artifact: dict) -> None:
    sidecars = artifact["sidecar_identities"]
    expected = {
        "contact": (
            audit.CONTACT_SIDECAR,
            "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e",
        ),
        "va27": (
            audit.VA27_SIDECAR,
            "7d51a23672273ec3ea46f947080c0c3ea66db55332522b199caab00ad4fc2126",
        ),
    }
    for name, (path, expected_sha) in expected.items():
        assert Path(sidecars[name]["path"]) == path.resolve()
        assert path.parent == audit.PI2S / "source_snapshots/data"
        assert ".local/datasets" not in sidecars[name]["path"]
        assert sidecars[name]["sha256"] == expected_sha


def test_semantic_build_is_deterministic(artifact: dict) -> None:
    rebuilt = audit.build_artifact()
    assert rebuilt["semantic_sha256"] == artifact["semantic_sha256"]
    first = dict(artifact)
    second = dict(rebuilt)
    first.pop("created_at_utc")
    second.pop("created_at_utc")
    assert first == second


def test_source_statement_gate_fails_closed() -> None:
    with pytest.raises(audit.AuditError, match="missing frozen source statement"):
        audit.source_statements(b"seed = 42\n", ["shuffle=True"], "fixture")


def test_input_registry_rejects_expected_sha_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"frozen\n")
    inputs = audit.InputRegistry()
    with pytest.raises(audit.AuditError, match="input SHA mismatch"):
        inputs.bytes(source, "fixture", "0" * 64)


def test_no_clobber_publish_and_check_are_semantic_and_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    inputs = audit.InputRegistry()
    inputs.json(source, "fixture")
    artifact = audit.output_artifact({"status": "PASS"}, inputs)
    output = tmp_path / "audit.json"
    result = audit.execute_artifact(output, artifact, check=False)
    before = output.read_bytes()
    before_stat = output.stat()
    assert result == {
        "status": "PUBLISH_NO_CLOBBER_AND_READBACK_PASS",
        "operation": "ATOMIC_SAME_DIRECTORY_HARD_LINK_NO_CLOBBER_PUBLISH",
        "write_attempted": True,
        "overwrite_allowed": False,
        "output_sha256": hashlib.sha256(before).hexdigest(),
    }

    checked = audit.execute_artifact(output, artifact, check=True)
    assert checked == {
        "status": "CHECK_PASS",
        "operation": "READ_ONLY_RECOMPUTE_AND_COMPARE",
        "write_attempted": False,
        "overwrite_allowed": False,
        "output_sha256": hashlib.sha256(before).hexdigest(),
    }
    assert output.read_bytes() == before
    assert output.stat().st_mtime_ns == before_stat.st_mtime_ns

    changed_artifact = dict(artifact)
    changed_artifact["status"] = "FAIL"
    with pytest.raises(audit.AuditError, match="refusing to overwrite existing output"):
        audit.execute_artifact(output, changed_artifact, check=False)
    assert output.read_bytes() == before

    output.chmod(0o644)
    output.write_text(json.dumps(changed_artifact))
    with pytest.raises(audit.AuditError, match="bad semantic SHA"):
        audit.check_artifact(output, artifact)


def test_no_clobber_publish_has_one_winner_under_race(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    barrier = Barrier(2)
    payloads = ({"writer": "alpha"}, {"writer": "beta"})

    def contender(payload: dict[str, str]) -> tuple[str, str]:
        barrier.wait()
        try:
            return "published", audit.publish_json_no_clobber(output, payload)
        except audit.AuditError as error:
            return "refused", str(error)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(contender, payloads))

    assert sorted(status for status, _ in results) == ["published", "refused"]
    assert any(
        "refusing to overwrite existing output" in value
        for status, value in results
        if status == "refused"
    )
    assert json.loads(output.read_bytes()) in payloads
    assert not list(tmp_path.glob(f".{output.name}.tmp.*"))


def test_no_clobber_publish_refuses_broken_symlink_destination(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    missing_target = tmp_path / "missing.json"
    output.symlink_to(missing_target)

    with pytest.raises(audit.AuditError, match="refusing to overwrite existing output"):
        audit.publish_json_no_clobber(output, {"status": "PASS"})

    assert output.is_symlink()
    assert output.readlink() == missing_target
    assert not missing_target.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp.*"))
