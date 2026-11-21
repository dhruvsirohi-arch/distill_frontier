"""MuJoCo scene: Unitree G1-D (wheeled base, humanoid torso, two arms, Dex1-1 grippers) inside a
Reaction Ring drop-stick machine.

The machine follows the manufacturer's description (madeforarcade.com Reaction Ring manual): a
2.05 m tall stand whose curved overhead bar holds 8, 10 or 12 sticks on electromagnets and drops
them one at a time. The hanging sticks end about 1.55 m above the floor, just above the reach of
the G1-D's level grippers (pinch height 1.0-1.45 m at the bar radius), as they would for a
human player's raised hands.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "robot_model" / "g1d"
URDF = MODEL_DIR / "g1_d_with_dex1_1.urdf"

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
SIDES = ("left", "right")

BASE_Z = 0.1108          # wheel centre (z=-0.026 in AGV frame) + wheel radius 0.0848
LIFT = 0.08              # each of the two lift stages; shoulders end up at about 1.25 m


@dataclass(frozen=True)
class Machine:
    n_sticks: int = 8              # manual menu F0: 8 (default), 10 or 12
    radius: float = 0.50           # arc radius around the player's standing spot
    magnet_z: float = 1.95         # underside of the bar (the machine is 2.05 m tall)
    stick_length: float = 0.40     # foam baton, end to end
    stick_radius: float = 0.016
    stick_mass: float = 0.12
    centre_x: float = 0.0          # arc centre in world frame (robot starts here)

    def slots(self) -> np.ndarray:
        """Magnet positions (n, 3): a half circle from the player's left (+y) through the front (+x)
        to the right (-y)."""
        a = np.linspace(pi / 2, -pi / 2, self.n_sticks)
        return np.stack([self.centre_x + self.radius * np.cos(a), self.radius * np.sin(a),
                         np.full_like(a, self.magnet_z)], axis=1)


def _robot_spec() -> mujoco.MjSpec:
    text = URDF.read_text().replace(
        '<compiler meshdir="meshes" discardvisual="false"/>',
        f'<compiler meshdir="{MODEL_DIR}" discardvisual="false" fusestatic="false" '
        f'balanceinertia="true"/>')
    return mujoco.MjSpec.from_string(text)


def _look_xyaxes(direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    z = -d
    x = np.cross(d, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.concatenate([x, y])


def build(machine: Machine = Machine()) -> mujoco.MjSpec:
    spec = _robot_spec()
    spec.modelname = "reaction_ring_g1d"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    spec.compiler.autolimits = True
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.quality.shadowsize = 2048
    spec.visual.map.znear = 0.01
    spec.stat.center = [0.2, 0, 1.0]
    spec.stat.extent = 2.0

    # ---- materials ------------------------------------------------------------------------------
    def mat(name, rgba, **kw):
        m = spec.add_material(name=name, rgba=rgba)
        for k, v in kw.items():
            setattr(m, k, v)
        return m
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                     rgb1=[.42, .45, .47], rgb2=[.38, .41, .43])
    floor_mat = spec.add_material(name="floor", texrepeat=[8, 8], reflectance=0.05)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=64, height=512,
                     rgb1=[.70, .74, .78], rgb2=[.90, .92, .93])
    mat("shell", [.80, .82, .83, 1], specular=0.3)
    mat("dark", [.13, .14, .16, 1], specular=0.3)
    mat("steel", [.08, .08, .09, 1], specular=0.5, shininess=0.6)
    mat("glow", [.45, .85, .95, 1], emission=0.6)
    mat("stick", [.92, .16, .42, 1], specular=0.2)
    mat("magnet", [.25, .25, .28, 1])
    mat("pad", [.05, .05, .05, 1])

    world = spec.worldbody
    world.add_light(pos=[0.5, 0.8, 3.2], dir=[-0.1, -0.2, -1], diffuse=[.7, .7, .7],
                    castshadow=True)
    world.add_light(pos=[-1.5, -1.0, 2.5], dir=[0.5, 0.3, -1], diffuse=[.35, .35, .35],
                    castshadow=False)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.05],
                   material="floor", contype=1, conaffinity=4, friction=[0.9, 0.01, 0.001])

    # ---- robot ----------------------------------------------------------------------------------
    for g in spec.geoms:                       # URDF meshes become visual only
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        g.contype = 0
        g.conaffinity = 0
        g.group = 2
        name = g.parent.name.lower()
        g.material = "dark" if any(k in name for k in ("wheel", "caster", "finger", "wrist",
                                                       "camera", "lidar", "head")) else "shell"
    removable = {"LZ_mt_Joint", "LZ_it_Joint", "Yaw_Joint"} | {
        j.name for j in spec.joints if "Wheel" in j.name or "Caster" in j.name}
    for j in list(spec.joints):
        if j.name in removable:
            if j.name.startswith("LZ_"):        # freeze the lift at its working height
                j.parent.pos = j.parent.pos + np.array([0, 0, LIFT])
            spec.delete(j)
    base = spec.body("AGV_link")
    base.pos = [machine.centre_x, 0, BASE_Z]
    base.add_joint(name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    base.add_joint(name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    base.add_joint(name="base_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    # chassis bumper so sticks bounce off the base instead of passing through it
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.26, 0.24, 0.09], pos=[0, 0, 0.03],
                  contype=2, conaffinity=4, group=3, rgba=[0, 0, 0, 0], mass=0)
    base.add_site(name="base_origin", pos=[0, 0, 0], size=[0.01] * 3, group=5)

    for j in spec.joints:
        if j.name in ("base_x", "base_y", "base_yaw"):
            j.damping = [0, 0, 0]
            continue
        j.armature = 0.01 if "finger" not in j.name else 0.002
        j.damping = [0.5 if "finger" not in j.name else 2.0, 0, 0]
    for b in spec.bodies:
        if b.name.startswith(("left_", "right_", "torso", "head")):
            b.gravcomp = 1.0                   # servo firmware gravity feed-forward

    for side in SIDES:
        # arm links: convex-hull collisions against sticks only
        for part in ("elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"):
            body = spec.body(f"{side}_{part}_link")
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=[g for g in body.geoms][0].meshname,
                          contype=2, conaffinity=4, group=3, mass=0, rgba=[0, 0, 0, 0])
        hand = spec.body(f"{side}_dex1_base_link")
        hand.add_geom(name=f"{side}_palm", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0.032, 0, 0], size=[0.036, 0.029, 0.03], contype=2, conaffinity=4,
                      group=3, mass=0, rgba=[0, 0, 0, 0])
        hand.add_site(name=f"{side}_pinch", pos=[0.125, 0, 0], size=[0.006] * 3, group=4,
                      rgba=[1, .8, 0, 1])
        for k, sgn in ((1, -1), (2, 1)):
            finger = spec.body(f"{side}_dex1_finger_link_{k}")
            finger.add_geom(name=f"{side}_pad{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[0.0625, sgn * 0.030, sgn * 0.0105],
                            size=[0.0175, 0.006, 0.0145], material="pad",
                            contype=2, conaffinity=4, group=1, mass=0, condim=4,
                            friction=[1.5, 0.02, 0.002], solref=[0.006, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                            priority=1)

    # head cameras: a wide stereo-free pair covering the half circle in front of the robot.
    head = spec.body("head_link")
    fovy = float(np.degrees(2 * np.arctan(np.tan(np.radians(53)) * 0.75)))   # 106 deg wide
    for name, yaw in (("head_left", 0.95), ("head_right", -0.95)):
        d = [cos(yaw) * cos(0.45), sin(yaw) * cos(0.45), sin(0.45)]
        cam = head.add_camera(name=name, pos=[0.07, 0, 0.45], fovy=fovy,
                              mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = _look_xyaxes(d)

    # ---- actuators ------------------------------------------------------------------------------
    limits = {}
    for j in spec.joints:
        limits[j.name] = j.range.copy()
    def pos_act(joint, kp, kv, force, crange=None):
        a = spec.add_actuator(name=joint, target=joint,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp, kv=kv)
        a.forcelimited = True
        a.forcerange = [-force, force]
        a.ctrllimited = True
        a.ctrlrange = crange if crange is not None else limits[joint]
    def vel_act(joint, kv, force, crange):
        a = spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_velocity(kv=kv)
        a.forcelimited = True
        a.forcerange = [-force, force]
        a.ctrllimited = True
        a.ctrlrange = crange

    vel_act("base_x", 4000, 300, [-1.0, 1.0])
    vel_act("base_y", 4000, 300, [-1.0, 1.0])
    vel_act("base_yaw", 400, 60, [-2.5, 2.5])
    # The G1-D URDF lists 30 N·m for the waist yaw; the G1 humanoid's waist-yaw motor is rated
    # 88 N·m, and at 30 N·m the torso visibly sways when an arm swings. Use the motor rating.
    pos_act("torso_Joint", 4000, 150, 88, [-1.6, 1.6])
    effort = {"shoulder_pitch": 25, "shoulder_roll": 25, "shoulder_yaw": 25, "elbow": 25,
              "wrist_roll": 25, "wrist_pitch": 5, "wrist_yaw": 5}
    gains = {"shoulder_pitch": (400, 20), "shoulder_roll": (400, 20), "shoulder_yaw": (300, 14),
             "elbow": (300, 14), "wrist_roll": (120, 5), "wrist_pitch": (60, 2.5),
             "wrist_yaw": (60, 2.5)}
    for side in SIDES:
        for axis in ARM_JOINTS:
            kp, kv = gains[axis]
            pos_act(f"{side}_{axis}_joint", kp, kv, effort[axis])
        for k in (1, 2):
            # 20 N rated grip force; the URDF's 0.2 m/s slide speed comes from the damping
            pos_act(f"{side}_dex1_finger_joint_{k}", 3000, 60, 20, [0, 0.041])

    # ---- the machine ----------------------------------------------------------------------------
    m = machine
    cx, R, top = m.centre_x, m.radius, m.magnet_z
    frame = world.add_body(name="machine", pos=[cx, 0, 0])
    # U-shaped floor base (open toward -x, where the player steps in)
    nseg = 24
    for i in range(nseg):
        a0 = pi / 2 - pi * i / nseg
        a1 = pi / 2 - pi * (i + 1) / nseg
        am = 0.5 * (a0 + a1)
        rr = R + 0.28
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[rr * cos(am), rr * sin(am), 0.025],
                       size=[0.09, rr * (a0 - a1) / 2 + 0.005, 0.025],
                       euler=[0, 0, am], material="steel", contype=0, conaffinity=0, group=1)
    # post behind the apex of the arc
    post_x = R + 0.28
    frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[post_x, 0, top / 2 + 0.06],
                   size=[0.07, 0.10, top / 2 + 0.06], material="steel", contype=0,
                   conaffinity=0, group=1)
    frame.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[post_x - 0.072, 0, 1.30],
                   size=[0.035, 0.004], euler=[0, pi / 2, 0], material="dark", contype=0,
                   conaffinity=0, group=1)
    # the overhead arc ("ring"), drawn as short segments, with a lit strip on top
    nseg = 36
    for i in range(nseg):
        a0 = pi / 2 - pi * i / nseg
        a1 = pi / 2 - pi * (i + 1) / nseg
        am = 0.5 * (a0 + a1)
        half = R * (a0 - a1) / 2 + 0.004
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[R * cos(am), R * sin(am), top + 0.03],
                       size=[0.05, half, 0.03], euler=[0, 0, am], material="steel",
                       contype=0, conaffinity=0, group=1)
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[R * cos(am), R * sin(am), top + 0.062],
                       size=[0.035, half, 0.003], euler=[0, 0, am], material="glow",
                       contype=0, conaffinity=0, group=1)
    frame.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[(R + post_x) / 2 + 0.02, 0, top + 0.03],
                   size=[(post_x - R) / 2 + 0.05, 0.06, 0.03], material="steel",
                   contype=0, conaffinity=0, group=1)

    half_len = m.stick_length / 2 - m.stick_radius
    for i, (x, y, z) in enumerate(m.slots()):
        frame.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x - cx, y, z - 0.008],
                       size=[0.022, 0.008], material="magnet", contype=0, conaffinity=0,
                       group=1)
        stick = world.add_body(name=f"stick{i}", pos=[x, y, z - m.stick_length / 2])
        stick.add_freejoint(name=f"stick{i}")
        stick.add_geom(name=f"stick{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       size=[m.stick_radius, half_len, 0], mass=m.stick_mass,
                       material="stick", contype=4, conaffinity=7, condim=4,
                       friction=[1.0, 0.02, 0.002])
        stick.add_site(name=f"stick{i}_top", pos=[0, 0, m.stick_length / 2], size=[0.004] * 3,
                       group=5)
        # electromagnet: a weld between the stick top and the bar, switched off to drop it
        eq = spec.add_equality(name=f"magnet{i}", type=mujoco.mjtEq.mjEQ_WELD,
                               objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"stick{i}",
                               name2="world")
        eq.data[0:3] = [x, y, z]       # anchor at the magnet face (world frame = body2)
        eq.data[3:10] = 0              # relative pose taken from the hanging configuration
        eq.data[10] = 1.0
        eq.solref = [0.004, 1]

    # spectator cameras
    for name, pos, look in (("overview", [-1.55, -1.65, 2.55], [0.25, 0, 1.25]),
                            ("front", [2.5, -1.0, 2.0], [0.2, 0, 1.35])):
        cam = world.add_camera(name=name, pos=pos, fovy=50,
                               mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = _look_xyaxes(np.subtract(look, pos))
    cam = world.add_camera(name="side", pos=[0.25, -2.2, 1.35], fovy=45,
                           mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = _look_xyaxes([0, 1, -0.05])
    return spec


def compile_model(machine: Machine = Machine()):
    spec = build(machine)
    model = spec.compile()
    return spec, model
