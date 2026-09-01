from gr00t.simulation.s4_3_pd import REQUIRED_SOURCE_FIELDS, TASKS
from scripts.simulation.generate_s4_3_policy_expert_data import formal_attempt_manifest
from scripts.simulation.run_s4_3_pd_pilot import pilot_seed_manifest


def fake_official_root(tmp_path):
    for task in TASKS:
        for index in range(29):
            data = tmp_path / task / f"{task}-official-{index:02d}" / "replay.zarr/data"
            data.mkdir(parents=True)
            for field in REQUIRED_SOURCE_FIELDS:
                (data / field).mkdir()
    return tmp_path


def test_formal_manifest_has_exact_frozen_cardinality_and_split(tmp_path):
    value = formal_attempt_manifest(fake_official_root(tmp_path))
    assert value["groups_per_task"] == 25
    assert value["attempts_per_group"] == 5
    assert value["attempts_per_task"] == 125
    assert value["total_attempts"] == 375
    assert len({row["attempt_id"] for row in value["attempts"]}) == 375
    assert len({row["reset_seed"] for row in value["attempts"]}) == 375
    for task in TASKS:
        rows = [row for row in value["attempts"] if row["task"] == task]
        assert len(rows) == 125
        assert sum(row["split_role"] == "POLICY_TRAIN" for row in rows) == 100
        assert sum(row["split_role"] == "POLICY_DEV" for row in rows) == 25
        assert len({row["source_group_id"] for row in rows}) == 25


def test_formal_groups_and_seed_tuples_are_disjoint_from_pilot(tmp_path):
    root = fake_official_root(tmp_path)
    pilot = pilot_seed_manifest(root)
    formal = formal_attempt_manifest(root)
    pilot_groups = {
        (row["task"], row["source_group_id"]) for row in pilot["attempts"]
    }
    formal_groups = {
        (row["task"], row["source_group_id"]) for row in formal["attempts"]
    }
    assert pilot_groups.isdisjoint(formal_groups)
    pilot_seeds = {
        (row["reset_seed"], row["perturbation_seed"], row["controller_seed"])
        for row in pilot["attempts"]
    }
    formal_seeds = {
        (row["reset_seed"], row["perturbation_seed"], row["controller_seed"])
        for row in formal["attempts"]
    }
    assert pilot_seeds.isdisjoint(formal_seeds)
    assert {row["seed_namespace"] for row in formal["attempts"]} == {"PD_FORMAL"}
