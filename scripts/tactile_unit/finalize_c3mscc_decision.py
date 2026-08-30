#!/usr/bin/env python3
"""Finalize C3-MS-CC interpretation from the already-written locked result."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3mscc_contact_context import sha256_file  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import state_dict_digest  # noqa: E402
from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from scripts.tactile_unit.c3mscc_runtime import (  # noqa: E402
    atomic_json, load_config, load_frozen_shared_space, validate_selection_lock,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402


def main() -> None:
    config = load_config()
    root = ROOT / config["runtime"]["artifact_root"]
    selection = validate_selection_lock(config)
    evaluation = json.loads((root / "locked_test_evaluation.json").read_text())
    if evaluation.get("test_loaded") is not True or evaluation.get("selection_sha256") is None:
        raise RuntimeError("STRUCTURAL_FAIL: locked evaluation is absent")
    sources = evaluation["sources"]
    increments = evaluation["vision_incremental"]
    contact_ci = increments["contact_transition"]["bootstrap_ci95"]
    force_ci = increments["force_trend_class"]["bootstrap_ci95"]
    ah = sources["AH"]
    if contact_ci[0] > 0 and force_ci[0] > 0:
        classification = "VISION_MATERIALLY_IMPROVES_CONTACT_PREDICTION"
    elif contact_ci[0] > 0 or force_ci[0] > 0:
        classification = "VISION_SMALL_BUT_POSITIVE_GAIN"
    elif (
        ah["semantics"]["contact_transition"]["semantic_ratio"] >= 0.75
        and ah["semantics"]["force_trend_class"]["semantic_ratio"] >= 0.75
        and ah["shared_target"]["gate"] and ah["physics"]["gate"]
        and ah["h_context"]["gate"]
    ):
        classification = "A_PLUS_H_SUFFICIENT_VISION_OPTIONAL"
    else:
        classification = "VISION_NO_MEASURABLE_GAIN"
    evaluation["vision_classification"] = classification
    actual_gpu = evaluation["gpu"].get("actual_physical")
    runtime_gpu = {
        **evaluation["gpu"],
        "preferred_physical": int(config["gpu"]["preferred_physical"]),
        "fallback": actual_gpu != int(config["gpu"]["preferred_physical"]),
        "gpu1_authorization": config["gpu"].get("gpu1_authorization"),
    }
    evaluation["gpu"] = runtime_gpu
    shared, _, shared_digest = load_frozen_shared_space(config, torch.device("cpu"))
    s2 = load_s2_model(ROOT / config["runtime"]["s2_checkpoint"], torch.device("cpu"))
    c3dp_root = ROOT / config["runtime"]["c3dp_cache_root"]
    private_digests = {}
    for split in ("train", "validation", "test"):
        manifest = json.loads((c3dp_root / split / "manifest.json").read_text())
        private_digests[split] = manifest["arrays"]["r_c_priv"]["sha256"]
        if sha256_file(c3dp_root / manifest["arrays"]["r_c_priv"]["path"]) != private_digests[split]:
            raise RuntimeError("STRUCTURAL_FAIL: Contact private residual cache changed")
    component_digests = {
        "P_v": state_dict_digest(shared.projectors["vision"]),
        "P_a": state_dict_digest(shared.projectors["action"]),
        "P_c": state_dict_digest(shared.projectors["contact"]),
        "R_v": state_dict_digest(shared.recovery["vision"]),
        "R_a": state_dict_digest(shared.recovery["action"]),
        "R_c": state_dict_digest(shared.recovery["contact"]),
        "D_c": parameter_digest(s2.decoder),
        "shared_space": shared_digest,
    }
    evaluation["frozen_components"] = {
        "digests": component_digests,
        "private_residual_sha256": private_digests,
        "c3r0_protocol_sha256": {
            name: sha256_file(ROOT / config["runtime"]["artifact_root"].replace("vac_c3mscc", "vac_c3r0") / name)
            for name in ("selection.json", "root_cause_decision.json", "locked_test_evaluation.json")
        },
        "d_c_matches_accepted": component_digests["D_c"] == config["accepted"]["s2_decoder_parameter_sha256"],
        "all_requires_grad_false": all(not value.requires_grad for value in shared.parameters())
        and all(not value.requires_grad for value in s2.parameters()),
        "before_after_pass": evaluation["identity_before"]["pass"]
        and evaluation["identity_after"]["pass"]
        and evaluation["shared_state_before"] == evaluation["shared_state_after"]
        and component_digests["D_c"] == config["accepted"]["s2_decoder_parameter_sha256"],
    }
    atomic_json(root / "locked_test_evaluation.json", evaluation)
    source_ablation = json.loads((root / "source_ablation.json").read_text())
    source_ablation["vision_classification"] = classification
    atomic_json(root / "source_ablation.json", source_ablation)
    training = json.loads((root / "training_summary.json").read_text())
    training["selection"] = selection
    training["gpu"] = runtime_gpu
    atomic_json(root / "training_summary.json", training)
    manifest = json.loads((root / "trial_manifest.json").read_text())
    manifest["gpu"] = runtime_gpu
    atomic_json(root / "trial_manifest.json", manifest)
    print(classification)


if __name__ == "__main__":
    main()
