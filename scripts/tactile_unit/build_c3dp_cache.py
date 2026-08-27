#!/usr/bin/env python3
"""Build ignored C3-DP shared/private derived arrays without touching C1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.tactile_unit.c3dp_runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    build_split,
    load_config,
    validate_selection_lock,
)
from scripts.tactile_unit.vac_runtime_common import (  # noqa: E402
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if "test" in args.splits:
        validate_selection_lock(config)
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("2", "3"))
    try:
        set_seed(int(config["seed"]))
        manifests = {
            split: build_split(config, split, device, args.batch_size) for split in args.splits
        }
        print(json.dumps({"gpu": gpu, "splits": manifests}, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
