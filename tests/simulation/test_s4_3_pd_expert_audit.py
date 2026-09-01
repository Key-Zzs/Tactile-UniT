from scripts.simulation.audit_s4_3_pd_experts import RAW_DATASET_REVISION, TASKS, audit


def test_official_expert_audit_selects_actual_demonstrations():
    value = audit()
    assert value["status"] == "PASS"
    assert value["official_raw_dataset"]["revision"] == RAW_DATASET_REVISION
    assert set(value["tasks"]) == set(TASKS)
    for task in TASKS:
        record = value["tasks"][task]
        assert record["selected_source_class"] == "E0_OFFICIAL_DEMONSTRATION"
        selected = [item for item in record["available_sources"] if item["selected"]]
        assert len(selected) == 1
        assert selected[0]["action_form"].endswith("= 22D")


def test_contact_probe_and_non_executable_sources_are_rejected():
    value = audit()
    assert "NOT_AN_EXPERT" in value["rejections"]["old_s4_2_contact_probe"]
    assert "not executable" in value["rejections"]["openpi_norm_stats"]
