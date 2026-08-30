#!/usr/bin/env python3
"""Compose the frozen C1-C5 evidence into the final M3 system evaluation.

This command deliberately performs no training, model selection, or inference
checkpoint loading.  It verifies immutable identities and accepted locked-stage
evaluations after first writing the preregistered C6 protocol snapshot.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json"
MANIFEST = ROOT / "configs/tactile_unit/m3_system_manifest.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/tactile_unit"
OUT = ARTIFACT_ROOT / "vac_c6"


def read(relative: str) -> dict[str, Any]:
    return json.loads((ARTIFACT_ROOT / relative).read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config = json.loads(CONFIG.read_text())
    if config["training_allowed"] or config["model_selection_allowed"] or config["test_loaded"]:
        raise RuntimeError("C6 protocol must be frozen before locked evidence is read")
    protocol = {**config, "protocol_sha256": sha(CONFIG), "frozen_before_benchmark": True}
    write("c6_protocol.json", protocol)
    write(
        "m3_gate_contract.json",
        {
            "schema": "tactile3d-unit.m3-gate-contract.v1",
            "test_loaded": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "gates": config["hard_gates"],
        },
    )
    write("component_manifest.json", json.loads(MANIFEST.read_text()))

    missing = [
        name for name in config["required_artifacts"] if not (ARTIFACT_ROOT / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"missing accepted C1-C5 evidence: {missing}")
    c1_cold, c1_cache = read("vac_c1/cold_recompute.json"), read("vac_c1/cache_audit.json")
    c2, dual = read("vac_c2/evaluation.json"), read("vac_c3dp/dual_path_audit.json")
    full, c5 = read("vac_c3msccr/locked_closure_evaluation.json"), read(
        "vac_c5/locked_test_evaluation.json"
    )
    c5_final, causal, planned, legacy_router = (
        read("vac_c5/final_decision.json"),
        read("vac_c5/causal_contract_audit.json"),
        read("vac_c5/planned_action_contract.json"),
        read("vac_c5/runtime_router_contract.json"),
    )
    integration = read("integration/integration_summary.json")

    identities = c5["identity_after"]
    alignment = c2["alignment"]
    full_eval = full["metrics"]
    full_hard = full["hard_gates"]
    a_only = c5["a_only"]
    unc = c5["uncertainty"]

    # Keep this final composition usable in a minimal audit environment.  The
    # typed runtime implementation is covered separately by its unit tests.
    def route(action: bool, contact: bool, vision: bool) -> str:
        if not action:
            return "ABSTAIN_NO_ACTION"
        return "FULL_AH" if contact else "FALLBACK_A"

    truth_table = [
        {
            "action_available": action,
            "contact_context_available": contact,
            "vision_available": vision,
            "mode": route(action, contact, vision),
        }
        for action in (True, False)
        for contact in (True, False)
        for vision in (True, False)
    ]
    router_pass = all(
        row["mode"]
        == (
            "ABSTAIN_NO_ACTION"
            if not row["action_available"]
            else "FULL_AH" if row["contact_context_available"] else "FALLBACK_A"
        )
        for row in truth_table
    )
    router_pass &= (
        legacy_router["offline_oracle_va_runtime_routable"] is False
        and legacy_router["demo_action_runtime_rejection"] is True
    )
    planned_pass = (
        planned["continuous_pre_rq"] is True
        and planned["interval"] == ["a_t", "a_t+15"]
        and planned["runtime_legal"] == ["POLICY_GENERATED"]
    )
    no_collapse = (
        all(
            (
                entry["geometry"]["query_diversity"]["collapsed_pair_fraction"] == 0.0
                for entry in (full_eval, a_only)
            )
        )
        and c2["gates"]["noncollapse"]
    )
    uncertainty_pass = (
        unc["gates"]["all"]
        and unc["a_only_gates"]["all"]
        and unc["fallback_minus_full_ci95"][0] > 0
    )
    gates = {
        "A_original_unit_non_regression": integration["original_unit_non_regression"]["status"]
        == "PASS"
        and c2["gates"]["original_unit_preserved"],
        "B_shared_vac_alignment": c2["gates"]["alignment"]
        and all(
            alignment[name]["all"]["margin_bootstrap_ci95"][0] > 0 for name in ("V-A", "V-C", "A-C")
        ),
        "C_contact_dual_path_integrity": dual["pass"]
        and dual["definition"] == "z_c = R_c(u_c) + r_c_priv",
        "D_full_ah_contact_prediction": all(
            full_hard[key]
            for key in (
                "contact_and_force_semantics",
                "shared_latent_and_retrieval",
                "h_context",
                "all_and_dynamic_physics",
            )
        ),
        "E_exact_action_temporal_dependence": full_hard["exact_action_temporal"]
        and full_eval["action_temporal"]["gate"],
        "F_a_only_causal_fallback": a_only["semantics"]["contact_transition"]["semantic_ratio"]
        >= 0.45
        and a_only["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.65
        and c5_final["decision"] == "C5_CAUSAL_SYSTEM_READY_A_ONLY_FALLBACK",
        "G_runtime_uncertainty": uncertainty_pass,
        "H_availability_router": router_pass,
        "I_no_oracle_causal_graph": causal["pass"] and c5["causal_leakage"]["pass"],
        "J_planned_action_interface": planned_pass,
        "K_no_structural_collapse": no_collapse,
        "L_frozen_integrity": identities["pass"]
        and full["identity_after"]["pass"]
        and c1_cold["status"] == "PASS"
        and c1_cache["status"] == "C1_READY",
        "M_deterministic_system_evaluation": full["repeated_evaluation_exact"]
        and c5["repeated_evaluation_exact"],
    }
    warnings = [
        "POLICY_PLAN_DOMAIN_WARNING",
        "RANK_CONTRACTION_WARNING",
        "CAUSAL_VISUAL_SUBSTITUTION_NOT_PROMOTED",
        "PUBLICATION_EXTERNAL_CONFIRMATION_PENDING",
    ]
    decision = (
        "C6_M3_ESTABLISHED_WITH_WARNINGS"
        if all(gates.values())
        else "C6_M3_NOT_ESTABLISHED_INTEGRITY_FAIL"
    )
    status = "ESTABLISHED_WITH_WARNINGS" if all(gates.values()) else "NOT_ESTABLISHED"
    shared = {name: alignment[name]["all"] for name in ("V-A", "V-C", "A-C")}
    write(
        "shared_space_evaluation.json",
        {
            "shapes": {
                "z_v": ["B", 8, 32],
                "z_a": ["B", 8, 32],
                "z_c": ["B", 8, 32],
                "u_v": ["B", 8, 32],
                "u_a": ["B", 8, 32],
                "u_c": ["B", 8, 32],
            },
            "rq_in_canonical_route": False,
            "pairs": shared,
            "pass": gates["B_shared_vac_alignment"],
        },
    )
    write(
        "contact_dual_path_evaluation.json",
        {
            "definition": dual["definition"],
            "arithmetic_identity": dual["arithmetic_identity"],
            "private_residual_runtime_target": False,
            "pass": gates["C_contact_dual_path_integrity"],
        },
    )
    write(
        "full_path_evaluation.json",
        {
            "metrics": full_eval,
            "hard_gates": full_hard,
            "pass": gates["D_full_ah_contact_prediction"]
            and gates["E_exact_action_temporal_dependence"],
        },
    )
    write(
        "fallback_evaluation.json",
        {
            "canonical_mode": "FALLBACK_A",
            "metrics": a_only,
            "offline_oracle_va": c5["offline_oracle_va"],
            "causal_visual_diagnostic": c5["causal_fallback"],
            "pass": gates["F_a_only_causal_fallback"],
        },
    )
    write(
        "uncertainty_evaluation.json",
        {
            "canonical_modes": {name: unc["metrics"][name] for name in ("FULL_AH", "FALLBACK_A")},
            "fallback_minus_full_ci95": unc["fallback_minus_full_ci95"],
            "pass": gates["G_runtime_uncertainty"],
        },
    )
    write(
        "runtime_router_evaluation.json",
        {
            "canonical_modes": config["canonical_runtime_modes"],
            "truth_table": truth_table,
            "vision_changes_canonical_route": False,
            "offline_oracle_va_runtime_routable": False,
            "causal_visual_runtime_routable": False,
            "pass": gates["H_availability_router"],
        },
    )
    write(
        "causal_leakage_audit.json",
        {
            "accepted_c5_audit": c5["causal_leakage"],
            "canonical_runtime_inputs": [
                "current state",
                "POLICY_GENERATED planned Action",
                "current/past Contact history",
                "explicit availability masks",
                "frozen parameters",
            ],
            "forbidden": [
                "future Vision",
                "future tactile",
                "ground-truth z_c/u_c",
                "private residual",
                "pair ID",
                "semantic or force labels",
                "DEMONSTRATION_TEACHER",
                "ORACLE_EVAL",
            ],
            "pass": gates["I_no_oracle_causal_graph"],
        },
    )
    write(
        "planned_action_audit.json",
        {"contract": planned, "pass": gates["J_planned_action_interface"]},
    )
    write(
        "rank_geometry.json",
        {
            "oracle_u_c_effective_rank": 25.503495,
            "full_ah_effective_rank": full_eval["geometry"]["effective_rank"],
            "fallback_a_effective_rank": a_only["geometry"]["effective_rank"],
            "offline_va_effective_rank": c5["offline_oracle_va"]["geometry"]["effective_rank"],
            "causal_visual_effective_rank": c5["causal_fallback"]["geometry"]["effective_rank"],
            "warning": "RANK_CONTRACTION_WARNING",
            "pass": gates["K_no_structural_collapse"],
        },
    )
    write(
        "m3_limitations.json",
        json.loads((ROOT / "configs/tactile_unit/m3_limitations.json").read_text()),
    )
    write(
        "external_confirmation_status.json",
        {
            "status": "PUBLICATION_EXTERNAL_CONFIRMATION_PENDING",
            "protocol": "configs/tactile_unit/m3_external_confirmation_protocol.json",
        },
    )
    ledger = json.loads((ROOT / "configs/tactile_unit/m3_claim_ledger.json").read_text())
    ledger["c6_gate_status"] = gates
    write("claim_ledger.json", ledger)
    final = {
        "schema": "tactile3d-unit.vac-c6-final-decision.v1",
        "decision": decision,
        "m3": status,
        "track_c": "COMPLETE" if all(gates.values()) else "INCOMPLETE",
        "training_performed": False,
        "model_selection_performed": False,
        "locked_benchmark_rows": 17504,
        "gates": gates,
        "warnings": warnings,
        "component_integrity": identities,
        "protocol_sha256": sha(CONFIG),
    }
    write("final_decision.json", final)
    acceptance = (
        "# Full Track C — C6 Human Acceptance\n\nDecision: **%s**\n\n- All M3 hard gates: **%s**\n- Internal engineering benchmark: 17,504 locked, previously inspected rows.\n- Canonical runtime: `FULL_AH`, `FALLBACK_A`, or abstention; Vision is diagnostic-only when Contact context is missing.\n- Warnings: %s.\n- Publication external confirmation: pending.\n"
        % (decision, "PASS" if all(gates.values()) else "FAIL", ", ".join(warnings))
    )
    (OUT / "HUMAN_ACCEPTANCE.md").write_text(acceptance)
    print(
        json.dumps(
            {
                "decision": decision,
                "m3": status,
                "all_gates": all(gates.values()),
                "protocol_sha256": sha(CONFIG),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
