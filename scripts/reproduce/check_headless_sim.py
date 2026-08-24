#!/usr/bin/env python3
"""Layered headless EGL checks: MuJoCo physics, robosuite, then RoboCasa GR1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".local" / "artifacts" / "reproduction" / "s0_env"


def require_headless_egl() -> None:
    if os.environ.get("DISPLAY"):
        raise RuntimeError(f"DISPLAY must be unset for this check, got {os.environ['DISPLAY']!r}")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("MUJOCO_GL=egl is required")
    if os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("PYOPENGL_PLATFORM=egl is required")
    print("DISPLAY=<unset>")
    print("MUJOCO_GL=egl")
    print("PYOPENGL_PLATFORM=egl")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID', '<unset>')}")


def save_rgb(rgb: np.ndarray, name: str) -> Path:
    from PIL import Image

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    Image.fromarray(rgb).save(path)
    print(f"saved: {path}")
    return path


def simple_model():
    import mujoco

    xml = """
    <mujoco model='s0_egl_test'>
      <option gravity='0 0 -9.81'/>
      <worldbody>
        <geom type='plane' size='2 2 .1' rgba='.2 .3 .4 1'/>
        <body pos='0 0 1'>
          <freejoint/><geom type='box' size='.15 .15 .15' rgba='.9 .2 .2 1'/>
        </body>
        <light pos='0 0 3' dir='0 0 -1'/>
        <camera name='s0_camera' pos='2 -2 1.6' xyaxes='.707 .707 0 -.408 .408 .816'/>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def check_mujoco_physics() -> None:
    import mujoco

    model = simple_model()
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        raise RuntimeError("MuJoCo physics produced non-finite qpos")
    print(f"MuJoCo: {mujoco.__version__}; qpos={data.qpos.tolist()}")
    print("MuJoCo physics PASS")


def check_mujoco_egl() -> None:
    import mujoco

    require_headless_egl()
    model = simple_model()
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        renderer.update_scene(data, camera="s0_camera")
        rgb = renderer.render()
    finally:
        renderer.close()
    if rgb.shape != (240, 320, 3) or not np.isfinite(rgb).all() or not np.any(rgb):
        raise RuntimeError(f"Invalid EGL render: shape={rgb.shape}, min={rgb.min()}, max={rgb.max()}")
    save_rgb(rgb, "mujoco_egl_test.png")
    print(f"MuJoCo EGL RGB: shape={rgb.shape}; dtype={rgb.dtype}; min={rgb.min()}; max={rgb.max()}")
    print("MuJoCo EGL PASS")


def check_robosuite() -> None:
    require_headless_egl()
    import robosuite as suite

    env = suite.make(
        env_name="Lift", robots="Panda", has_renderer=False,
        has_offscreen_renderer=True, use_camera_obs=True, camera_names="agentview",
        camera_heights=240, camera_widths=320, control_freq=20,
    )
    try:
        observation = env.reset()
        rgb = observation["agentview_image"]
        for _ in range(3):
            observation, reward, done, info = env.step(np.zeros(env.action_dim))
        if rgb.shape != (240, 320, 3) or not np.isfinite(rgb).all() or not np.any(rgb):
            raise RuntimeError(f"Invalid robosuite camera image: {rgb.shape}")
        save_rgb(rgb, "robosuite_headless.png")
        print(f"robosuite: {suite.__version__}; camera=agentview; RGB={rgb.shape}; max={rgb.max()}")
        print(f"robosuite zero-action steps: reward={reward}; done={done}")
    finally:
        env.close()
    print("robosuite headless PASS")


def check_robocasa() -> None:
    require_headless_egl()
    import robocasa  # Registers actual RoboCasa environments.
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    robot = "GR1ArmsAndWaistFourierHands"
    controller = load_composite_controller_config(controller=None, robot=robot)
    controller["type"] = "BASIC"
    controller["composite_controller_specific_configs"] = {}
    controller["control_delta"] = False
    env = robosuite.make(
        env_name="PnPCupToDrawerClose", robots=robot, controller_configs=controller,
        camera_names=["egoview"], camera_widths=320, camera_heights=240,
        has_renderer=False, has_offscreen_renderer=True, ignore_done=True,
        use_object_obs=True, use_camera_obs=True, camera_depths=False,
        translucent_robot=False,
    )
    try:
        observation = env.reset()
        rgb = observation["egoview_image"]
        for _ in range(2):
            observation, reward, done, info = env.step(np.zeros(env.action_dim))
        if rgb.shape != (240, 320, 3) or not np.isfinite(rgb).all() or not np.any(rgb):
            raise RuntimeError(f"Invalid RoboCasa camera image: {rgb.shape}")
        print(f"RoboCasa: {robocasa.__version__}; env=PnPCupToDrawerClose; robot={robot}")
        print(f"RoboCasa RGB: {rgb.shape}; max={rgb.max()}; zero-action reward={reward}; done={done}")
    finally:
        env.close()
    print("RoboCasa GR1 PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("mujoco-physics", "mujoco-egl", "robosuite", "robocasa"))
    args = parser.parse_args()
    {
        "mujoco-physics": check_mujoco_physics,
        "mujoco-egl": check_mujoco_egl,
        "robosuite": check_robosuite,
        "robocasa": check_robocasa,
    }[args.stage]()


if __name__ == "__main__":
    main()
