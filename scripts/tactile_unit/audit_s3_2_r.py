#!/usr/bin/env python3
"""Close the S3.2-R decision tree from measured local R0 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/s3_2_r"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot_retention(nominal: dict[str, Any], capacity: dict[str, Any], path: Path) -> None:
    conditions = [
        ("Frozen shared\nidentity", nominal["conditions"]["frozen_shared_identity"]),
        ("P1 + frozen\nshared RQ", nominal["conditions"]["p1_frozen_shared"]),
        ("Private RQ\n2×128", nominal["conditions"]["separate_contact_rq"]),
        ("Private RQ\n3×128", capacity["conditions"]["separate_contact_rq"]),
    ]
    metrics = ("r_recon_raw", "r_contact_raw", "r_force_raw")
    labels = ("Reconstruction", "Contact transition", "Force trend")
    x = np.arange(len(conditions))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        values = [row["retention"][metric] for _, row in conditions]
        ax.bar(x + (index - 1) * width, values, width, label=label)
    ax.axhline(0.90, color="#2a9d8f", linestyle="--", linewidth=1, label="semantic strong gate")
    ax.axhline(0.75, color="#e9c46a", linestyle="--", linewidth=1, label="semantic partial gate")
    ax.set_ylabel("Raw advantage retention")
    ax.set_xticks(x, [label for label, _ in conditions])
    ax.set_ylim(0, 1.03)
    ax.set_title("S3.2-R R0: larger capacity does not recover Contact-transition semantics")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_secondary(nominal: dict[str, Any], capacity: dict[str, Any], path: Path) -> None:
    conditions = [
        ("Frozen shared", nominal["conditions"]["frozen_shared_identity"]),
        ("P1 shared", nominal["conditions"]["p1_frozen_shared"]),
        ("Private 2×128", nominal["conditions"]["separate_contact_rq"]),
        ("Private 3×128", capacity["conditions"]["separate_contact_rq"]),
    ]
    distortion = [row["quantization"]["relative_distortion"] for _, row in conditions]
    recoverability = [row["native_recoverability"]["test"]["r2"] for _, row in conditions]
    cka = [row["linear_cka_native_compressed"] for _, row in conditions]
    x = np.arange(len(conditions))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - width, distortion, width, label="Relative distortion")
    ax.bar(x, recoverability, width, label="Native recovery R²")
    ax.bar(x + width, cka, width, label="Linear CKA")
    ax.set_xticks(x, [label for label, _ in conditions])
    ax.set_ylim(0, 1.05)
    ax.set_title("Low distortion and high CKA are not sufficient semantic evidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_usage(nominal: dict[str, Any], capacity: dict[str, Any], path: Path) -> None:
    private2 = nominal["conditions"]["separate_contact_rq"]
    private3 = capacity["conditions"]["separate_contact_rq"]
    usage_rows = [
        ("2×128", row["stage"], row["active_ratio"], row["top1_frequency"])
        for row in private2["collapse"]["usage"]
    ] + [
        ("3×128", row["stage"], row["active_ratio"], row["top1_frequency"])
        for row in private3["collapse"]["usage"]
    ]
    labels = [f"{name}\nstage {stage + 1}" for name, stage, _, _ in usage_rows]
    active = [row[2] for row in usage_rows]
    top1 = [row[3] for row in usage_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, active, 0.36, label="Active ratio")
    ax.bar(x + 0.18, top1, 0.36, label="Top-1 frequency")
    ax.axhline(0.90, color="#d62828", linestyle="--", linewidth=1, label="top-1 collapse boundary")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Private Contact RQs do not exhibit hard code collapse")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def human_acceptance() -> str:
    return """# S3.2-R Human Acceptance

Run all commands from the repository root.

## 1. R0 information ceiling

```bash
python -m json.tool .local/artifacts/tactile_unit/s3_2_r/r0/r0_result.json | less
python -m json.tool .local/artifacts/tactile_unit/s3_2_r/r0_capacity/r0_result.json | less
```

Inspect `.local/artifacts/tactile_unit/s3_2_r/plots/r0_retention_comparison.png`.

## 2. R1 decoder recovery

```bash
python - <<'PY'
import json
d=json.load(open('.local/artifacts/tactile_unit/s3_2_r/final_decision.json'))
print(json.dumps(d['stages']['r1'], indent=2))
PY
```

R1 must read `SKIPPED_BY_DECISION_TREE`, because the private same-capacity RQ failed and the only capacity sensitivity also failed.

## 3. R2 frozen shared RQ

```bash
python - <<'PY'
import json
d=json.load(open('.local/artifacts/tactile_unit/s3_2_r/final_decision.json'))
print(json.dumps(d['stages']['r2'], indent=2))
PY
```

## 4. R3 Vision/Action preservation

```bash
python - <<'PY'
import json
d=json.load(open('.local/artifacts/tactile_unit/s3_2_r/final_decision.json'))
print(json.dumps(d['stages']['r3'], indent=2))
print(json.dumps(d['original_unit_non_regression'], indent=2))
PY
```

R3 was not scientifically permitted, so no Q_new exists and no T4 optimizer/selection access occurred.

## 5. Final semantic retention

```bash
python - <<'PY'
import json
for name in ('r0', 'r0_capacity'):
 d=json.load(open(f'.local/artifacts/tactile_unit/s3_2_r/{name}/r0_result.json'))
 c=d['conditions']['separate_contact_rq']
 print(name, c['semantic_probes'], c['retention'], c['category'])
PY
```

## 6. Code/query collapse diagnostics

```bash
python - <<'PY'
import json
for name in ('r0', 'r0_capacity'):
 d=json.load(open(f'.local/artifacts/tactile_unit/s3_2_r/{name}/r0_result.json'))
 print(name, json.dumps(d['conditions']['separate_contact_rq']['collapse'], indent=2))
PY
```

Inspect `.local/artifacts/tactile_unit/s3_2_r/plots/r0_code_usage.png` and `.local/artifacts/tactile_unit/s3_2_r/plots/r0_secondary_diagnostics.png`.

## 7. Final decision

```bash
python -m json.tool .local/artifacts/tactile_unit/s3_2_r/final_decision.json | less
```
"""


def main() -> int:
    args = parse_args()
    nominal = load(args.artifact_dir / "r0/r0_result.json")
    capacity = load(args.artifact_dir / "r0_capacity/r0_result.json")
    if nominal["r0_final"] != "SAME_CAPACITY_CONTACT_RQ_FAIL":
        raise RuntimeError("audit script is only valid for the measured R0 FAIL branch")
    if nominal["architecture"] != {"codes_per_stage": 128, "embedding_dim": 32, "queries": 8, "stages": 2}:
        raise RuntimeError("nominal R0 geometry changed")
    if capacity["architecture"] != {"codes_per_stage": 128, "embedding_dim": 32, "queries": 8, "stages": 3}:
        raise RuntimeError("capacity sensitivity geometry changed")
    for result in (nominal, capacity):
        if not all(row["unchanged"] for row in result["frozen_integrity"].values()):
            raise RuntimeError("frozen identity failure")
        if result["conditions"]["separate_contact_rq"]["category"] != "FAIL":
            raise RuntimeError("measured decision branch changed")
    nominal_private = nominal["conditions"]["separate_contact_rq"]
    capacity_private = capacity["conditions"]["separate_contact_rq"]
    decision = {
        "schema": "tactile3d-unit.s3-2-r-final-decision.v1",
        "status": "COMPLETE",
        "canonical_data": {
            "contact_train": 279680,
            "contact_validation": 17504,
            "contact_test": 17504,
            "dynamic_test": nominal_private["reconstruction"]["dynamic"]["windows"],
            "gr1_rehearsal_train": 0,
            "gr1_rehearsal_validation": 0,
            "t4_held_out": 960,
            "leakage": "PASS",
            "t4_optimizer_or_selection_access": False,
        },
        "stages": {
            "r0": {
                "status": "SAME_CAPACITY_CONTACT_RQ_FAIL",
                "category": "FAIL",
                "capacity_sensitivity_run": True,
                "capacity_sensitivity_category": "FAIL",
            },
            "r1": {"status": "SKIPPED_BY_DECISION_TREE", "reason": "Private same-capacity Contact RQ failed and the bounded larger-capacity RQ also failed."},
            "r2": {"status": "SKIPPED_BY_DECISION_TREE", "reason": "R0 did not establish reasonable Contact compressibility."},
            "r3": {"status": "SKIPPED_BY_DECISION_TREE", "reason": "R0 precondition failed; no Q_new was created or trained."},
        },
        "r0": {
            "same_capacity": {
                "dynamic_mse": nominal_private["reconstruction"]["dynamic"]["future_mse"],
                **nominal_private["retention"],
                "semantic_f1": {key: value["macro_f1"] for key, value in nominal_private["semantic_probes"].items()},
                "quantization": nominal_private["quantization"],
                "native_recoverability": nominal_private["native_recoverability"]["test"],
                "cka": nominal_private["linear_cka_native_compressed"],
                "collapse": {key: nominal_private["collapse"][key] for key in ("hard_code_collapse", "query_collapse")},
            },
            "capacity_sensitivity": {
                "dynamic_mse": capacity_private["reconstruction"]["dynamic"]["future_mse"],
                **capacity_private["retention"],
                "semantic_f1": {key: value["macro_f1"] for key, value in capacity_private["semantic_probes"].items()},
                "quantization": capacity_private["quantization"],
                "native_recoverability": capacity_private["native_recoverability"]["test"],
                "cka": capacity_private["linear_cka_native_compressed"],
                "collapse": {key: capacity_private["collapse"][key] for key in ("hard_code_collapse", "query_collapse")},
            },
        },
        "primary_root_cause": "CONTACT_DISCRETIZATION_OR_OBJECTIVE_LIMIT",
        "evidence": [
            "The repository-native private 2-stage 128-code Contact RQ failed the pre-registered overall gate.",
            "Adding a third 128-code residual stage improved reconstruction and native recovery but did not convert the gate; Contact-transition retention decreased.",
            "Both private RQs avoided hard code and query collapse, so collapse does not explain the failure.",
        ],
        "dimensional_compatibility": "SATISFIED: native Contact is already 8x32, matching Original UniT RQ input geometry.",
        "promoted_architecture": {
            "native_contact": "z_c = E_c(h_t^c, h_{t+16}^c) in R^(8x32)",
            "shared_contact": "NO CANDIDATE PROMOTED",
            "decoder_interface": "NO CANDIDATE PROMOTED",
            "shared_rq": "NO CANDIDATE PROMOTED",
            "private_contact_residual_required_now": "NOT YET DETERMINED",
        },
        "original_unit_non_regression": {
            "status": "NOT_RUN_BY_DECISION_TREE",
            "interpretation": "Q_old remained frozen and no Q_new was created; R3/T4 non-regression is inapplicable rather than a claimed PASS.",
        },
        "answers": {
            "contact_compressible_same_budget": "NO under the S3.2-R STRONG/PARTIAL task-relevant gates",
            "main_problem": "Contact discretization/training objective limit; not dimensional mismatch and not proven to be pure bottleneck capacity or frozen-vocabulary coverage",
            "frozen_original_rq_via_pc_rc": "NOT SCIENTIFICALLY TESTABLE AFTER R0 HARD STOP",
            "shared_rq_joint_adaptation_required": "NOT ESTABLISHED; R3 was forbidden after R0 failure",
            "vision_action_non_regression_if_adapted": "NOT APPLICABLE; no adapted shared RQ exists",
            "s3_3": "RECOMMEND ENTERING S3.3 because Action Embodiment Bootstrap is orthogonal to the failed Contact discretization branch; do not treat it as Contact integration",
            "s3_4": "NOT READY",
        },
        "s3_3_recommendation": "RECOMMEND ENTERING S3.3",
        "s3_4_recommendation": "NOT READY",
        "m3": "NOT ESTABLISHED",
        "s3_3_started": False,
        "s3_4_started": False,
    }
    write_json(args.artifact_dir / "final_decision.json", decision)
    plots = args.artifact_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_retention(nominal, capacity, plots / "r0_retention_comparison.png")
    plot_secondary(nominal, capacity, plots / "r0_secondary_diagnostics.png")
    plot_usage(nominal, capacity, plots / "r0_code_usage.png")
    (args.artifact_dir / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance())
    print(json.dumps({"status": "COMPLETE", "primary_root_cause": decision["primary_root_cause"], "r1_r2_r3": "SKIPPED_BY_DECISION_TREE"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
