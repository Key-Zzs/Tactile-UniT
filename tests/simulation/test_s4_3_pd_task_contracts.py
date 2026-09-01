from scripts.simulation.audit_s4_3_pd_task_contracts import TASKS, contracts


def test_native_contracts_cover_all_tasks_and_use_native_success():
    value = contracts()
    assert set(value) == set(TASKS)
    for task in TASKS:
        item = value[task]
        assert "success" in item["success_predicate"] or task in {
            "pinch_tongs",
            "hammer_nail",
            "click_mouse",
        }
        assert item["reward"].startswith("1.0 exactly on native success")
        assert item["action"]["policy"].startswith("22D")
        assert item["action"]["environment"].startswith("23D")


def test_success_contracts_preserve_task_semantics():
    for item in contracts().values():
        assert item["negative_state"]
        assert item["known_positive_state"]
        assert item["termination_predicate"].startswith("native success")
