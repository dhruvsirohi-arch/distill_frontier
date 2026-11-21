# Unitree G1-D with Dex1-1

URDF and meshes from Unitree Robotics, revision `157ff950351e5f535e03c5d85a16b7d594b3dbcc`
([upstream](https://github.com/unitreerobotics/unitree_ros/tree/157ff950351e5f535e03c5d85a16b7d594b3dbcc/robots/g1_d_description),
[BSD 3-Clause](LICENSE)). Only meshes used by the Dex1-1 variant are included; large meshes were
decimated to 5,500 triangles per part.

How `ringsim/scene.py` adapts it (MuJoCo's URDF importer, static links kept):

- Link frames, masses, inertias, joint axes and ranges come from the URDF unchanged.
- Base: planar x/y/yaw joints driven as a differential drive (no sideways slip), |v| <= 0.6 m/s,
  |w| <= 2 rad/s. Wheels and casters are visual only.
- Lift fixed at 0.08 m per stage (shoulders at 1.25 m); waist pitch fixed upright.
- Waist yaw actuated at 88 N·m (the G1 waist-yaw motor rating; the URDF lists 30 N·m, at which
  the torso sways several degrees when an arm swings).
- Arms: position servos at the URDF effort limits with gravity feed-forward; joint setpoints are
  rate-limited (12 rad/s proximal, 16 rad/s wrist) by the motion controller.
- Dex1-1: two prismatic fingers, 41 mm travel each (80 mm max gap), 20 N, box contact pads on the
  inner fingertip faces.
- Two wide head cameras (106 degrees each, yawed ±0.95 rad, pitched up 0.45 rad) are a simulation
  addition so the head view covers the whole half circle of sticks.
- Collisions: fingertip pads, palms, forearm/wrist links and a chassis bumper collide with sticks;
  robot self-collision is off.
