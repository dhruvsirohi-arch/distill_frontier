"""Phase 2 agent interface: the prompt, the reply schema and the per-call sensor payload.

Every call is stateless (a fresh thread); the only memory is the notes memo. Each reply starts
with an explicit reasoning record (observation -> inference -> decision) written before the
action, so the action can be read against what the model said it saw and concluded.
"""
from __future__ import annotations

import base64
import json

import numpy as np
from PIL import Image, ImageDraw

from ..exam import ACTION_SCHEMA, GAME, PRIV, ROBOT
from .sensors import cloud_text, lidar_views, png_bytes, workspace_cloud


def _sensors_text(joint_names):
    return f"""\
SENSORS (a standard robot sensor suite, published continuously; every message carries its
capture time "stamp" on the game clock; positions are in base_link: x forward, y left, z up from
the floor, metres unless noted):
 image 1  /head/rgb: head colour camera, the newest frame (30 frames per second). Two panels side
          by side: left half looks forward-left, right half forward-right; each 320x240, about
          106 degrees wide.
 image 2  /head/depth: the same frame's depth, grayscale from black (0 m) to white (2.5 m+).
 image 3  clip: the head colour camera 0.1 s (top) and 0.2 s (bottom) before the newest frame,
          at half size, for seeing motion.
 image 4  /lidar/points views of the newest lidar scan (Livox MID-360 on top of the head, 10 scans
          per second, 20 000 points per scan): top, an unrolled panorama (azimuth +120 deg on the
          left through the front to -120 deg on the right, elevation -7 to +52 deg, colour = range,
          blue near to red far); bottom, a top-down view of base_link (up = forward, left = +y,
          colour = height, white ring = robot origin).
 text     lidar_points_cm: the same scan cropped to the arms' workspace (within 0.62 m of the
          torso axis, 0.90-1.93 m above the floor) and voxel-filtered at 3 cm, as "x,y,z" in
          centimetres separated by ";".
 text     joint_states (sensor_msgs/JointState), joints in this order: {", ".join(joint_names)}.
          position rad (gripper fingers: metres of travel, 0.041 = fully open), velocity, effort
          (N m; fingers N).
 text     imu (sensor_msgs/Imu, in the torso): orientation quaternion w,x,y,z; angular velocity
          rad/s; linear acceleration m/s^2 (includes gravity).
 text     odom (nav_msgs/Odometry from the wheel encoders): x, y, yaw, v, w.
 text     tf: each gripper's pinch point in base_link, grip (1 open, 0 closed), jaw gap in mm.
 text     command_in_force (what the controller is executing) and your_notes."""


REPLY = """\
REPLY: JSON only, fields in this order:
  reasoning.observation  what the sensors show right now that matters (<= 300 characters)
  reasoning.inference    what it means: which stick, where, when (<= 300 characters)
  reasoning.decision     what you will command and why (<= 300 characters)
  action                 the full command (base, torso_yaw, left, right)
  notes                  <= 400 characters shown back to you on the next call. Each call sees
                         only the newest sensors and these notes: keep what you need here."""

TIMING = {
    "standard": """\
TIMING (Standard): the simulation is paused while you think. After each reply it applies your
command and advances exactly one camera frame (1/30 s), then calls you with the next frame. You
see every frame.""",
    "realtime": """\
TIMING (Realtime): the simulation runs at real speed while you think and the sensors keep
publishing. Your command is applied at the first frame after your reply arrives; you are then
called again with the newest sensors. Thinking time is game time: compare the stamps.""",
    "pipelined": """\
TIMING (Realtime, pipelined): the simulation runs at real speed while you think and the sensors
keep publishing. A new call like this one starts every 0.25 s with the newest sensors while
earlier calls are still thinking. Each reply's command is applied at the first frame after it
arrives, unless a reply based on newer sensors has already been applied. Your notes reach the
calls that start after your reply arrives. Thinking time is game time: compare the stamps.""",
}


def instructions(mode, joint_names):
    head = ("You control a real robot through its sensors and motion controller in a physics "
            "simulation. Reply with JSON only.")
    return "\n\n".join([head, ROBOT, GAME, _sensors_text(joint_names), TIMING[mode], REPLY])


REASONING_SCHEMA = {"type": "object", "additionalProperties": False,
                    "required": ["observation", "inference", "decision"],
                    "properties": {k: {"type": "string"}
                                   for k in ("observation", "inference", "decision")}}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["reasoning", "action", "notes"],
          "properties": {"reasoning": REASONING_SCHEMA, "action": ACTION_SCHEMA,
                         "notes": {"type": "string"}}}


def _image(arr):
    return {"type": "image",
            "url": "data:image/png;base64," + base64.b64encode(png_bytes(arr)).decode()}


def depth_u8(depth, max_m=2.5):
    return np.uint8(np.clip(depth / max_m, 0, 1) * 255)


def clip_image(snap):
    parts = []
    for t, rgb in snap["clip"]:
        im = Image.fromarray(rgb).resize((rgb.shape[1] // 2, rgb.shape[0] // 2))
        ImageDraw.Draw(im).text((3, 2), f"t={t:.3f}", fill=(255, 255, 255))
        parts.append(np.asarray(im))
    return np.concatenate(parts, 0)


def observation_text(snap, notes):
    pts = workspace_cloud(snap["scan"])
    head = {"frame": snap["frame"], "t": snap["t"], "joint_states": snap["joint_states"],
            "imu": snap["imu"], "odom": snap["odom"], "tf": snap["tf"],
            "lidar": {"stamp": snap["scan"]["stamp"], "points_in_scan": snap["scan"]["n"],
                      "workspace_voxels": int(len(pts))},
            "command_in_force": snap["held"], "your_notes": notes}
    return ("Observation " + json.dumps(head, default=float) +
            f"\nlidar_points_cm (stamp {snap['scan']['stamp']}): " + cloud_text(pts))


def build_inputs(snap, notes):
    """Returns (inputs for the model, the exact text part) so the text can be traced verbatim."""
    text = observation_text(snap, notes)
    return [{"type": "text", "text": text}, _image(snap["rgb"]), _image(depth_u8(snap["depth"])),
            _image(clip_image(snap)), _image(lidar_views(snap["scan"]))], text


# ---- privileged ---------------------------------------------------------------------------------
PRIV2 = PRIV.replace("each\naction takes effect at the first 40 ms control tick",
                     "each\naction takes effect at the first control tick (1/30 s)") \
            .replace("control tick 40 ms", "control tick 1/30 s (one camera frame)")
PRIV_SCHEMA2 = {"type": "object", "additionalProperties": False,
                "required": ["reasoning", "op", "plan", "notes"],
                "properties": {"reasoning": {"type": "string"},
                               "op": {"type": "string", "enum": ["simulate", "submit"]},
                               "plan": {"type": "array", "items": {
                                   "type": "object", "additionalProperties": False,
                                   "required": ["t", "action"],
                                   "properties": {"t": {"type": "number"},
                                                  "action": ACTION_SCHEMA}}},
                               "notes": {"type": "string"}}}


def priv_instructions():
    return "\n\n".join([
        "You control a robot in a physics simulation. Reply with JSON only. Start each reply "
        "with 'reasoning': what you learned from the state or the last simulation and why your "
        "plan should work (<= 800 characters).", ROBOT, GAME, PRIV2])
