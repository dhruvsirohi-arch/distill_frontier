"""Build the MuJoCo scene: Menagerie SO-101 + floor + low table with target region + rimmed bowl.

The arm model is loaded unmodified from mujoco_menagerie/robotstudio_so101/so101.xml; we only
*add* elements (a TCP site on the gripper body, world geometry, cameras) via MjSpec.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Pan axis location in the base frame (body "shoulder" pos in so101.xml).
PAN_AXIS_XY = np.array([0.0388353, 0.0])
# TCP: point between the jaws, expressed in the "gripper" body frame. Body -z is the finger
# direction, body +x is the jaw closing direction (fixed jaw on the -x side, inner face ~-0.0098
# at this height). We place the TCP ~9 mm from the fixed-jaw face so a 5 mm wall centred on the
# TCP has clearance on both sides while the gripper is open.
TCP_IN_GRIPPER = [-0.001, 0.0, -0.090]


def _lookat_quat(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    z = -f
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1).reshape(-1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def build_spec(cfg: dict) -> mujoco.MjSpec:
    ecfg = cfg["env"]
    spec = mujoco.MjSpec.from_file(str(REPO / cfg["paths"]["robot_xml"]))
    spec.modelname = "so101_bowl"
    spec.option.timestep = ecfg["timestep"]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # --- assets
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=300, height=300,
                     rgb1=[0.26, 0.30, 0.34], rgb2=[0.20, 0.24, 0.28])
    mat = spec.add_material(name="grid", texrepeat=[8, 8], texuniform=True, reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=1536,
                     rgb1=[0.45, 0.55, 0.65], rgb2=[0.1, 0.1, 0.12])

    wb = spec.worldbody
    # Mount the arm on a pedestal: the SO-101 wrist-flex limit only allows a vertical (top-down)
    # gripper when the TCP is <~0.09 m above the arm base, so the base must sit above the floor.
    bh = ecfg["base_height"]
    spec.body("base").pos = [0.0, 0.0, bh]
    wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, bh / 2],
                size=[0.055, 0.05, bh / 2], rgba=[0.25, 0.25, 0.28, 1], contype=0, conaffinity=0)
    wb.add_light(name="top", pos=[0.15, 0.0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], material="grid")

    # spawn region marker (visual only)
    sx, sy = ecfg["spawn"]["x"], ecfg["spawn"]["y"]
    wb.add_geom(name="spawn_marker", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[(sx[0] + sx[1]) / 2, (sy[0] + sy[1]) / 2, 0.0005],
                size=[(sx[1] - sx[0]) / 2 + ecfg["bowl"]["radius"], (sy[1] - sy[0]) / 2 + ecfg["bowl"]["radius"], 0.0005],
                rgba=[0.3, 0.5, 0.9, 0.25], contype=0, conaffinity=0, group=1)

    # table (low platform) + target region
    t = ecfg["table"]
    th = t["height"]
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[t["center"][0], t["center"][1], th / 2],
                size=[t["size"][0] / 2, t["size"][1] / 2, th / 2], rgba=[0.55, 0.40, 0.28, 1])
    tg = ecfg["target"]
    wb.add_geom(name="target_region", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[tg["center"][0], tg["center"][1], th + 0.0006],
                size=[tg["size"][0] / 2, tg["size"][1] / 2, 0.0006],
                rgba=[0.2, 0.85, 0.3, 0.55], contype=0, conaffinity=0, group=1)

    # bowl: base disc + ring of wall boxes + a yaw-marking lug
    b = ecfg["bowl"]
    R, wall, H, bt, N = b["radius"], b["wall"], b["height"], b["base_thickness"], b["segments"]
    bowl = wb.add_body(name="bowl", pos=[0.2, 0.0, 0.0])
    bowl.add_freejoint(name="bowl_free")
    per = b["mass"] / (N + 2)
    bowl_rgba = [0.92, 0.92, 0.95, 1]
    bowl.add_geom(name="bowl_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, bt / 2],
                  size=[R - wall * 0.5, bt / 2, 0], mass=per, rgba=bowl_rgba, friction=[1.0, 0.005, 0.0001])
    rmid = R - wall / 2
    seg_half = math.pi * rmid / N * 1.08
    for i in range(N):
        a = 2 * math.pi * i / N
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), a)
        bowl.add_geom(name=f"bowl_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[rmid * math.cos(a), rmid * math.sin(a), (H + 0.001) / 2],
                      quat=q, size=[wall / 2, seg_half, (H - 0.001) / 2], mass=per, rgba=bowl_rgba,
                      friction=[1.0, 0.005, 0.0001])
    L = b["lug_length"]
    bowl.add_geom(name="bowl_lug", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[R + L / 2, 0, H - 0.003], size=[L / 2, 0.008, 0.003], mass=per,
                  rgba=[0.85, 0.25, 0.2, 1])
    bowl.add_site(name="bowl_center", pos=[0, 0, 0], size=[0.003, 0, 0], group=4)

    # TCP site on the gripper body
    spec.body("gripper").add_site(name="tcp", pos=TCP_IN_GRIPPER, size=[0.004, 0, 0],
                                  rgba=[1, 0, 1, 1], group=4)

    # cameras: front view and top-down view
    front_pos, front_tgt = [0.60, -0.42, 0.50], [0.12, 0.02, 0.10]
    wb.add_camera(name="front", pos=front_pos, quat=_lookat_quat(front_pos, front_tgt), fovy=48)
    wb.add_camera(name="top", pos=[0.16, 0.02, 0.72], quat=[1, 0, 0, 0], fovy=45)
    return spec


def build_model(cfg: dict) -> mujoco.MjModel:
    return build_spec(cfg).compile()
