#!/usr/bin/env python3
"""Cold-load the official pi0.5 base parameters and record structural gates."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

# This validation is intentionally host-side; the training GPU gate is separate.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.traverse_util as traverse_util  # noqa: E402
import numpy as np  # noqa: E402

from openpi.models import model as openpi_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / ".local/external/s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params"
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi0/official_base_model_smoke.json"


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
    arrays = [np.asarray(value) for value in flat.values()]
    dtype_counts = Counter(str(array.dtype) for array in arrays)
    nonfinite = sum(int(array.size - np.count_nonzero(np.isfinite(array))) for array in arrays)
    gates = {
        "params_directory": PARAMS.is_dir(),
        "cold_load": bool(arrays),
        "numpy_restore": all(isinstance(value, np.ndarray) for value in flat.values()),
        "all_nonempty": all(array.size > 0 for array in arrays),
        "all_finite": nonfinite == 0,
        "pi05_language_model": any("llm" in key.lower() for key in flat),
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-official-base-model-smoke.v1",
        "stage": "PI0-4",
        "params_path": symbolic(PARAMS),
        "restore_type": "numpy.ndarray",
        "jax_platform": os.environ["JAX_PLATFORMS"],
        "elapsed_seconds": elapsed,
        "array_leaves": len(arrays),
        "array_elements": sum(array.size for array in arrays),
        "array_bytes": sum(array.nbytes for array in arrays),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "nonfinite_elements": nonfinite,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(payload)
    print(
        json.dumps(
            {key: payload[key] for key in ("status", "array_leaves", "array_bytes", "elapsed_seconds")},
            sort_keys=True,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_BASE_MODEL_SMOKE_FAIL")


if __name__ == "__main__":
    main()
