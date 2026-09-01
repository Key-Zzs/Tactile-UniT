from gr00t.simulation.s4_3_pd import REQUIRED_SOURCE_FIELDS, TASKS
from scripts.simulation.run_s4_3_pd_pilot import pilot_seed_manifest


def fake_official_root(tmp_path):
    for task in TASKS:
        for index in range(29):
            data = tmp_path / task / f"{task}-official-{index:02d}" / "replay.zarr/data"
            data.mkdir(parents=True)
            for field in REQUIRED_SOURCE_FIELDS:
                (data / field).mkdir()
    return tmp_path


def test_pilot_manifest_is_exact_and_unique(tmp_path):
    value = pilot_seed_manifest(fake_official_root(tmp_path))
    assert value["attempts_per_task"] == 20
    assert value["total_attempts"] == 60
    assert value["formal_membership"] is False
    attempts = value["attempts"]
    assert len({row["attempt_id"] for row in attempts}) == 60
    assert len({row["reset_seed"] for row in attempts}) == 60
    assert {row["seed_namespace"] for row in attempts} == {"PD_PILOT"}


def test_pilot_uses_four_disjoint_groups_per_task(tmp_path):
    value = pilot_seed_manifest(fake_official_root(tmp_path))
    for task in ("pinch_tongs", "hammer_nail", "click_mouse"):
        rows = [row for row in value["attempts"] if row["task"] == task]
        assert len(rows) == 20
        assert len({row["source_group_id"] for row in rows}) == 4
        groups = {item["source_group_id"] for item in rows}
        assert all(
            sum(item["source_group_id"] == group for item in rows) == 5
            for group in groups
        )
