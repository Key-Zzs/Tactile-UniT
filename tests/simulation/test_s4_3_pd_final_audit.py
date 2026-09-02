from scripts.simulation.audit_s4_3_pd_final import final_classification


def passing_gates():
    return {
        "PD1": "PASS",
        "PD2": "PASS",
        "PD3": "PASS",
        "PD4": "PASS",
        "PD5": "PASS",
        "PD6": "PASS",
        "PD7": "PASS",
        "PD8": "PASS",
        "S4.2 immutability": "PASS",
        "Environment": "PASS",
        "Regression": "PASS",
    }


def test_all_hard_gates_map_to_policy_data_ready():
    assert final_classification(passing_gates()) == "S4_3_PD_COMPLETE_POLICY_DATA_READY"


def test_immutability_failure_has_specific_terminal_decision():
    gates = passing_gates()
    gates["S4.2 immutability"] = "FAIL"
    assert final_classification(gates) == "S4_3_PD_S4_2_MUTATION_FAIL"


def test_regression_failure_is_structural():
    gates = passing_gates()
    gates["Regression"] = "FAIL"
    assert final_classification(gates) == "STRUCTURAL_FAIL"
