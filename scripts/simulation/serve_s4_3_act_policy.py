#!/usr/bin/env python3
"""Serve one cold-loaded causal S4.3 ACT checkpoint over a local Unix socket."""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_act import FrozenS42PolicyStack  # noqa: E402
from gr00t.simulation.s4_3_transport import (  # noqa: E402
    cleanup_server_endpoint,
    load_endpoint_manifest,
    prepare_server_endpoint,
    register_server_endpoint,
)
from gr00t.simulation.s4_3_training import set_deterministic, sha256_file  # noqa: E402
from gr00t.tactile_unit.paired_contract import preprocess_trex_rgb  # noqa: E402
from scripts.simulation.evaluate_s4_3_policy_offline import cold_load  # noqa: E402
from scripts.tactile_unit.build_c5_causal_visual_cache import frozen_frame_features  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_frozen_vision  # noqa: E402

VISION_IDENTITY = ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--unit-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint, _ = load_endpoint_manifest(args.endpoint_manifest)
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("policy server checkpoint SHA256 mismatch")
    set_deterministic(args.seed)
    device = torch.device(args.device)
    row: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_sha256": args.checkpoint_sha256,
        "task": args.task,
        "variant": args.variant,
        "training_seed": args.seed,
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    row["best_dev_step"] = checkpoint["training_step"]
    model = cold_load(row, device)
    identity = json.loads(VISION_IDENTITY.read_text(encoding="utf-8"))["checkpoint_file_sha256"]
    vision, vision_load = load_frozen_vision(
        args.unit_checkpoint,
        {"frozen_identity": {"original_unit_tokenizer_files_sha256": identity}},
        device,
    )
    if vision.training or vision_load["trainable_parameters"] != 0:
        raise RuntimeError("policy server Vision boundary is not frozen")
    stack = FrozenS42PolicyStack().to(device) if args.variant in {"P2", "P3"} else None
    prepare_server_endpoint(endpoint)
    first_inference = True
    listener = None
    try:
        listener = Listener(endpoint.path, family="AF_UNIX", authkey=b"s4_3_local_v1")
        register_server_endpoint(endpoint)
        print(
            json.dumps(
                {
                    "endpoint": endpoint.socket_path.name,
                    "endpoint_bytes": endpoint.encoded_length,
                    "status": "READY",
                }
            ),
            flush=True,
        )
        connection = listener.accept()
        try:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                command = request.get("command")
                if command == "ping":
                    connection.send({"status": "READY"})
                    continue
                if command == "shutdown":
                    connection.send({"status": "STOPPING"})
                    break
                if command != "infer":
                    raise RuntimeError("unknown policy server command")
                encoded = np.frombuffer(request["rgb_jpeg"], dtype=np.uint8)
                bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError("policy server received unreadable RGB")
                processed = preprocess_trex_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[None]
                vision_np = frozen_frame_features(vision, processed, device)
                vision_t = torch.from_numpy(vision_np).to(device)
                proprio = torch.from_numpy(
                    np.asarray(request["proprio"], dtype=np.float32)[None]
                ).to(device)
                tactile = torch.from_numpy(
                    np.asarray(request["tactile_history"], dtype=np.float32)[None]
                ).to(device)
                kwargs: dict[str, torch.Tensor] = {}
                contact = None
                if args.variant == "P1":
                    kwargs["tactile_history"] = tactile
                elif args.variant in {"P2", "P3"}:
                    assert stack is not None
                    contact = stack.encode_contact_state(tactile)
                    kwargs["contact_state"] = contact
                with torch.inference_mode():
                    output = model(vision_t, proprio, **kwargs)
                    action = output["physical_action"]
                    predicted_contact = None
                    if args.variant == "P3":
                        assert stack is not None and contact is not None
                        predicted_contact = (
                            stack.predict_shared_contact(proprio, action, contact)
                            .float()
                            .cpu()
                            .numpy()[0]
                        )
                    if first_inference:
                        repeated = model(vision_t, proprio, **kwargs)["physical_action"]
                        if not torch.equal(action, repeated):
                            raise RuntimeError(
                                "policy server deterministic inference repeat failed"
                            )
                        first_inference = False
                action_np = action.float().cpu().numpy()[0]
                if action_np.shape != (27, 22) or not np.isfinite(action_np).all():
                    raise RuntimeError("policy server produced invalid Action chunk")
                connection.send(
                    {
                        "action_chunk": action_np,
                        "p3_predicted_contact": predicted_contact,
                        "vision_shape": list(vision_np.shape[1:]),
                        "uncertainty": {
                            "status": "UNAVAILABLE_CAUSAL_INPUT_MISMATCH",
                            "invoked": False,
                            "intervention": False,
                        },
                    }
                )
        finally:
            connection.close()
    finally:
        if listener is not None:
            listener.close()
        cleanup_server_endpoint(endpoint, allow_unregistered_own_socket=True)


if __name__ == "__main__":
    main()
