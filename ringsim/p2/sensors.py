"""Phase 2 sensor models, published in ROS-style messages (REP-103 axes, REP-105 frames).

  /head/rgb, /head/depth   head RGB-D pair (two panels, 320x240 each, ~106 deg wide), 30 Hz
  /lidar/points            Livox MID-360 on the head: 360 x (-7..+52) deg, non-repeating scan
                           pattern, 20 000 points per scan, 10 Hz, 2 cm range noise
  /joint_states            sensor_msgs/JointState: position, velocity, effort; 30 Hz here
  /imu                     sensor_msgs/Imu in the torso: orientation, angular velocity, linear
                           acceleration (gravity included, as a real accelerometer reports it)
  /odom                    nav_msgs/Odometry from the wheel encoders
  /tf                      forward kinematics: gripper pinch points in base_link

Everything an agent sees is in `base_link` (x forward, y left, z up from the floor), the same
frame its arm commands use. Simulator-only quantities (stick poses, the release schedule, the
judge) are never published.
"""
from __future__ import annotations

import io

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..scene import SIDES

GOLDEN = (np.sqrt(5) - 1) / 2


class Mid360:
    RATE_HZ = 10
    POINTS = 20_000
    V_FOV = (-7.0, 52.0)
    RANGE = (0.1, 40.0)
    SIGMA = 0.02

    def __init__(self, game, seed=0):
        self.g = game
        m = game.m
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "head_lidar")
        self.rng = np.random.default_rng(seed + 7919)
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "AGV_link")
        robot = set()
        for b in range(m.nbody):
            k = b
            while k > 0:
                if k == root:
                    robot.add(b)
                    break
                k = m.body_parentid[k]
        self.robot_geom = np.isin(m.geom_bodyid, list(robot))
        self.groups = np.array([1, 1, 1, 0, 0, 0], np.uint8)     # visible geometry only
        self.scan_no = 0
        self.gid = np.zeros(self.POINTS, np.int32)
        self.dist = np.zeros(self.POINTS)

    def _pattern(self):
        """Non-repetitive coverage like the MID-360's rosette: a rotating low-discrepancy set."""
        i = np.arange(self.POINTS) + self.scan_no * self.POINTS
        az = 2 * np.pi * np.mod(i * GOLDEN, 1.0)
        lo, hi = np.radians(self.V_FOV)
        u = np.mod(i * 0.7548776662466927, 1.0)            # plastic-number sequence
        el = np.arcsin(np.sin(lo) + u * (np.sin(hi) - np.sin(lo)))
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)

    def sweep(self, part, parts):
        """Cast this frame's share of the scan (the MID-360 sweeps continuously, so a scan is
        built over consecutive frames). Returns the finished scan after the last part."""
        g, m, d = self.g, self.g.m, self.g.d
        if part == 0:
            self._dirs = self._pattern()
            self._acc = []
            self._open = True
        elif not getattr(self, "_open", False):
            return None                              # joined mid-sweep: wait for the next one
        n = self.POINTS
        lo, hi = part * n // parts, (part + 1) * n // parts
        R = d.site_xmat[self.site].reshape(3, 3)
        dirs = self._dirs[lo:hi] @ R.T
        origin = d.site_xpos[self.site].copy()
        k = hi - lo
        gid, dist = self.gid[:k], self.dist[:k]
        mujoco.mj_multiRay(m, d, origin, dirs.ravel(), self.groups, 1, -1, gid, dist,
                           None, k, self.RANGE[1])
        hit = (gid >= 0) & (dist >= self.RANGE[0])
        hit &= ~self.robot_geom[np.clip(gid, 0, None)]              # self-filter
        r = dist[hit] + self.rng.normal(0, self.SIGMA, hit.sum())
        world = origin + dirs[hit] * r[:, None]
        self._acc.append((_to_base_many(g, world), self._dirs[lo:hi][hit] * r[:, None], r))
        if part < parts - 1:
            return None
        self.scan_no += 1
        self._open = False
        xyz, sxyz, rr = (np.concatenate(a, 0) for a in zip(*self._acc))
        return dict(stamp=round(g.t, 4), frame_id="base_link", n=int(len(rr)),
                    xyz=xyz.astype(np.float32), sensor_xyz=sxyz.astype(np.float32),
                    range=rr.astype(np.float32))

    def scan(self):
        return self.sweep(0, 1)


def _to_base_many(g, world):
    xy, yaw = g.ctrl.base_pose(g.d)
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = world[:, 0] - xy[0], world[:, 1] - xy[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, world[:, 2]], 1)


# ---- point cloud products for the agent -----------------------------------------------------------
CROP = dict(r=0.62, z=(0.9, 1.93))        # arm reach (cylinder about the torso), below the bar
VOXEL = 0.03


def workspace_cloud(scan, voxel=VOXEL):
    """Crop to the arms' workspace (a cylinder of the arm reach about base_link, between the
    lowest grasp height and just below the bar) and voxel-filter, the usual PCL/Open3D
    preprocessing. Returns points in centimetres, one per occupied voxel (the voxel's mean)."""
    p = scan["xyz"]
    keep = (np.hypot(p[:, 0], p[:, 1]) <= CROP["r"]) & (p[:, 2] >= CROP["z"][0]) & \
        (p[:, 2] <= CROP["z"][1])
    p = p[keep]
    if not len(p):
        return np.zeros((0, 3), int)
    key = np.floor(p / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    sums = np.zeros((inv.max() + 1, 3))
    np.add.at(sums, inv, p)
    cnt = np.bincount(inv)[:, None]
    pts = np.round(sums / cnt * 100).astype(int)
    order = np.lexsort((pts[:, 2], np.arctan2(pts[:, 1], pts[:, 0])))
    return pts[order]


def cloud_text(pts):
    return ";".join(f"{x},{y},{z}" for x, y, z in pts)


def _cmap(v):
    """0..1 -> RGB (dark blue -> cyan -> yellow -> red), no matplotlib dependency."""
    v = np.clip(v, 0, 1)[..., None]
    stops = np.array([[20, 30, 120], [0, 190, 230], [250, 230, 40], [230, 40, 30]], float)
    x = v * 3
    i = np.clip(np.floor(x).astype(int), 0, 2)
    f = x - i
    return (stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f).astype(np.uint8)


def lidar_views(scan):
    """Two 320x200 views stacked (320x400): an unrolled panorama of the scan (azimuth +120 deg on
    the left to -120 deg on the right, elevation -7..+52 deg, colour = range 0.2-2.5 m) and a
    top-down view of base_link x -0.3..1.0 m, y +0.9..-0.9 m (colour = height 0.6-2.2 m)."""
    W, H = 320, 200
    s = scan["sensor_xyz"]
    pano = np.full((H, W, 3), 12, np.uint8)
    if len(s):
        az = np.degrees(np.arctan2(s[:, 1], s[:, 0]))
        el = np.degrees(np.arctan2(s[:, 2], np.hypot(s[:, 0], s[:, 1])))
        u = ((120 - az) / 240 * W).astype(int)
        v = ((52 - el) / 59 * H).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        order = np.argsort(-scan["range"][ok])                    # near points drawn last
        uu, vv, rr = u[ok][order], v[ok][order], scan["range"][ok][order]
        col = _cmap((rr - 0.2) / 2.3)
        for du in (0, 1):
            for dv in (0, 1):
                pano[np.clip(vv + dv, 0, H - 1), np.clip(uu + du, 0, W - 1)] = col
    top = np.full((H, W, 3), 12, np.uint8)
    p = scan["xyz"]
    sel = (p[:, 2] > 0.6) & (p[:, 2] < 2.2)
    p = p[sel]
    if len(p):
        u = ((0.9 - p[:, 1]) / 1.8 * W).astype(int)
        v = ((1.0 - p[:, 0]) / 1.3 * H).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        order = np.argsort(p[ok, 2])
        col = _cmap((p[ok, 2][order] - 0.6) / 1.6)
        top[v[ok][order], u[ok][order]] = col
    img = Image.fromarray(np.concatenate([pano, top], 0))
    dr = ImageDraw.Draw(img)
    dr.text((4, 2), "lidar panorama  L(+120)  ...  front  ...  R(-120)  colour=range", fill=(230,) * 3)
    dr.line([(W // 2, 0), (W // 2, H - 1)], fill=(80, 80, 80))
    dr.text((4, H + 2), "top-down base_link: up=+x(front) left=+y  colour=height", fill=(230,) * 3)
    ru, rv = int(0.9 / 1.8 * W), int(1.0 / 1.3 * H)               # robot origin
    dr.ellipse([ru - 3, H + rv - 3, ru + 3, H + rv + 3], outline=(255, 255, 255))
    return np.asarray(img)


# ---- proprioception -------------------------------------------------------------------------------
class Proprio:
    GYRO_SIGMA, ACCEL_SIGMA = 0.003, 0.03

    def __init__(self, game, seed=0):
        self.g = game
        m = game.m
        c = game.ctrl
        self.names, self.qadr, self.dadr, self.act = [], [], [], []
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        add = lambda name, j, a: (self.names.append(name), self.qadr.append(m.jnt_qposadr[j]),
                                  self.dadr.append(m.jnt_dofadr[j]), self.act.append(a))
        tj = m.actuator_trnid[c.torso_act, 0]
        add("waist_yaw", tj, c.torso_act)
        for side in SIDES:
            for a in c.arm[side]["act"]:
                j = m.actuator_trnid[a, 0]
                add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), j, a)
            for k, a in enumerate(c.arm[side]["fingers"]):
                j = m.actuator_trnid[a, 0]
                add(f"{side}_gripper_finger{k + 1}", j, a)
        self.sen = {n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
                    for n in ("imu_quat", "imu_gyro", "imu_accel")}
        self.rng = np.random.default_rng(seed + 104729)

    def joint_state(self):
        d = self.g.d
        return dict(stamp=round(self.g.t, 4),
                    position=np.round(d.qpos[self.qadr], 4).tolist(),
                    velocity=np.round(d.qvel[self.dadr], 3).tolist(),
                    effort=np.round(d.actuator_force[self.act], 2).tolist())

    def imu(self):
        d, s = self.g.d, self.sen
        q = d.sensordata[s["imu_quat"]:s["imu_quat"] + 4]
        w = d.sensordata[s["imu_gyro"]:s["imu_gyro"] + 3] + self.rng.normal(0, self.GYRO_SIGMA, 3)
        a = d.sensordata[s["imu_accel"]:s["imu_accel"] + 3] + self.rng.normal(0, self.ACCEL_SIGMA, 3)
        return dict(stamp=round(self.g.t, 4), frame_id="imu_link",
                    orientation_wxyz=np.round(q, 4).tolist(),
                    angular_velocity=np.round(w, 3).tolist(),
                    linear_acceleration=np.round(a, 2).tolist())

    def odom(self):
        p = self.g.proprio()["odometry"]
        return dict(stamp=round(self.g.t, 4), frame_id="odom", child_frame_id="base_link", **p)

    def tf(self):
        g = self.g
        p = g.proprio()
        gap = {s: round(1000 * sum(g.d.qpos[q] for q in g.finger_q[s]), 1) for s in SIDES}
        return dict(stamp=round(g.t, 4), frame_id="base_link",
                    pinch={s: p["pinch"][s] for s in SIDES},
                    grip=p["grip"], jaw_gap_mm=gap)


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
