#!/usr/bin/env python3
"""Cold-load and structurally validate the locally reproduced final params."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.traverse_util as traverse_util  # noqa: E402
import numpy as np  # noqa: E402

from openpi.models import model as openpi_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999"
PARAMS = CHECKPOINT / "params"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi0/reproduced_checkpoint_smoke.json"


def symbolic(path: Path) -> str:
    return "$REPO_ROOT/" + path.resolve().relative_to(ROOT).as_posix()


def write_json(payload: Any) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(ARTIFACT)


def main() -> None:
    started = perf_counter()
    params = openpi_model.restore_params(PARAMS, restore_type=np.ndarray)
    elapsed = perf_counter() - started
    flat = traverse_util.flatten_dict(params, sep="/")
    arrays = {key: np.asarray(value) for key, value in flat.items()}
    dtype_counts = Counter(str(array.dtype) for array in arrays.values())
    lora = {key: value for key, value in arrays.items() if "lora_" in key}
    gemma_2b_lora = {key: value for key, value in lora.items() if "_1" not in key}
    gemma_300m_lora = {key: value for key, value in lora.items() if "_1" in key}
    action_bias = arrays.get("action_out_proj/bias")
    nonfinite = sum(int(array.size - np.count_nonzero(np.isfinite(array))) for array in arrays.values())
    gates = {
        "params_directory": PARAMS.is_dir(),
        "cold_load": bool(arrays),
        "numpy_restore": all(isinstance(value, np.ndarray) for value in flat.values()),
        "array_leaves_71": len(arrays) == 71,
        "all_nonempty": all(array.size > 0 for array in arrays.values()),
        "all_finite": nonfinite == 0,
        "lora_arrays_20": len(lora) == 20,
        "gemma_2b_lora_present": bool(gemma_2b_lora)
        and any(16 in array.shape for array in gemma_2b_lora.values()),
        "gemma_300m_lora_present": bool(gemma_300m_lora)
        and any(32 in array.shape for array in gemma_300m_lora.values()),
        "action_width_32": action_bias is not None and action_bias.shape == (32,),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-reproduced-checkpoint-smoke.v1",
        "stage": "PI0-9",
        "checkpoint": symbolic(CHECKPOINT),
        "params": symbolic(PARAMS),
        "restore_type": "numpy.ndarray",
        "jax_platform": os.environ["JAX_PLATFORMS"],
        "elapsed_seconds": elapsed,
        "array_leaves": len(arrays),
        "array_elements": sum(array.size for array in arrays.values()),
        "array_bytes": sum(array.nbytes for array in arrays.values()),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "lora_array_count": len(lora),
        "gemma_2b_lora_array_count": len(gemma_2b_lora),
        "gemma_300m_lora_array_count": len(gemma_300m_lora),
        "nonfinite_elements": nonfinite,
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "array_leaves": len(arrays),
                "array_bytes": payload["array_bytes"],
                "lora_arrays": len(lora),
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_REPRODUCED_CHECKPOINT_SMOKE_FAIL")


if __name__ == "__main__":
    main()
