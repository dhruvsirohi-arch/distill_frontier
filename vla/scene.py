"""Scene for the VLA task: SO-101 + coloured bowls + coloured pads sliding on rails + wrist camera."""
from __future__ import annotations

import mujoco
import numpy as np

from env.scene import _lookat_quat, add_bowl, base_spec

# Wrist camera in the "gripper" body frame (body -z = finger direction, +x = jaw closing direction).
WRIST_CAM_POS = [0.0, -0.075, 0.005]
WRIST_CAM_TARGET = [0.0, 0.0, -0.11]


def build_spec(cfg: dict) -> mujoco.MjSpec:
    v = cfg["vla"]
    spec = base_spec(cfg)
    spec.modelname = "so101_vla"
    wb = spec.worldbody

    for b in v["bowls"]:
        add_bowl(wb, v["bowl"], name=f"bowl_{b['name']}", rgba=b["rgba"], lug=False)

    for p in v["pads"]:
        cx, cy = p["center"]
        sx, sy = p["size"]
        h = p["height"]
        ax = np.array([p["axis"][0], p["axis"][1], 0.0], float)
        ax /= np.linalg.norm(ax)
        name = f"pad_{p['name']}"
        # rail (visual only)
        L = p["travel"] + max(sx, sy) / 2
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), float(np.arctan2(ax[1], ax[0])))
        wb.add_geom(name=f"{name}_rail", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, 0.001], quat=q,
                    size=[L, 0.012, 0.001], rgba=[0.12, 0.12, 0.13, 1], contype=0, conaffinity=0)
        body = wb.add_body(name=name, pos=[cx, cy, 0.0])
        body.add_joint(name=f"{name}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=ax,
                       range=[-p["travel"] - 0.01, p["travel"] + 0.01], damping=5.0)
        # hovers 5 mm above the floor so it never rubs on it; heavy relative to the 50 g bowl
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, (h + 0.005) / 2],
                      size=[sx / 2, sy / 2, (h - 0.005) / 2], mass=1.0, rgba=p["rgba"],
                      friction=[1.0, 0.005, 0.0001])
        act = spec.add_actuator(name=f"{name}_drive", target=f"{name}_slide",
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=5000.0, kv=140.0)

    # closer scene camera than the video "front" camera: workspace fills the 224 px frame
    sc = v.get("scene_camera", {})
    sp, st = sc.get("pos", [0.47, -0.30, 0.40]), sc.get("target", [0.13, 0.04, 0.04])
    wb.add_camera(name="scene", pos=sp, quat=_lookat_quat(sp, st), fovy=sc.get("fovy", 45))
    cam_pos, cam_tgt = np.array(WRIST_CAM_POS), np.array(WRIST_CAM_TARGET)
    spec.body("gripper").add_camera(name="wrist", pos=cam_pos, quat=_lookat_quat(cam_pos, cam_tgt), fovy=75)
    return spec


def build_model(cfg: dict) -> mujoco.MjModel:
    return build_spec(cfg).compile()
