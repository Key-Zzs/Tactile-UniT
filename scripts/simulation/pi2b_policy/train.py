#!/usr/bin/env python3
"""Run one frozen PI2B policy training job or bounded engineering fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.pi2b_policy.contract import MODEL_ORDER, Workspace
from gr00t.simulation.pi2b_policy.training import (
    build_config,
    config_summary,
    load_official_train_module,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    parser.add_argument("--seed", type=int, choices=(43, 44), required=True)
    parser.add_argument("--fsdp-devices", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--engineering-fixture", action="store_true")
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Workspace.load(ROOT)
    if args.fixture_output is not None and not args.engineering_fixture:
        raise SystemExit("--fixture-output is only valid for an engineering fixture")
    output = args.fixture_output.resolve() if args.fixture_output else None
    if output is not None and workspace.write_root not in output.parents:
        raise SystemExit("fixture output must remain inside the policy write root")
    config = build_config(
        workspace,
        args.model_id,
        args.seed,
        fsdp_devices=args.fsdp_devices,
        steps=args.steps,
        checkpoint_base_dir=output,
        engineering_fixture=args.engineering_fixture,
    )
    summary = config_summary(config, args.model_id, args.engineering_fixture)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.print_config:
        return
    load_official_train_module(workspace).main(config)


if __name__ == "__main__":
    main()
