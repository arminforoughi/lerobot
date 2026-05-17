#!/usr/bin/env bash
# Gaze + spherical-orbit servoing for SO-101 + OAK-D eye-in-hand.
#
# Architecture: PREPOSITIONING moves the EE on a sphere around the back-projected
# object to (azimuth, elevation, radius). APPROACHING then shrinks the radius
# along the **vector to the back-projected object** (not the camera +Z) while a
# soft visibility regulator throttles depth speed by pixel error so the cube
# never leaves the FoV.
#
# State machine: SEARCH → TRACKING (center with gaze) → APPROACHING → HOLD.
# Optional orbit: type ``preposition`` or ``top`` on stdin (or set
# ``--preposition-from-search=true`` to jump SEARCH→PREPOSITIONING like before).
#
# Visibility-first: PREPOSITIONING skips large IK when bbox is off-center and
# uses emergency gaze; object Z is floored for IK so bad depth does not pull
# the arm through the table.
#
# Control handles you'll touch most:
#   --approach-el-deg   : elevation 0..90 (90 = top-down)
#   --approach-az-deg   : azimuth around the object's vertical axis
#   --final-standoff-m  : terminal camera→object distance (depth)
#   --preposition-initial-radius-m : entry radius (must be > final-standoff)
#
# Camera mount: ONE source of truth — `--gripper-camera-tf="x,y,z,rx,ry,rz"`,
# translation in the EE frame (meters) plus a rotation vector (axis*angle, rad).
# Fine-tune pitch WITHOUT re-deriving the rotvec via
#   --gripper-camera-pitch-trim-deg=N
# (positive N rotates the camera about its own image-right axis, tilting the
# optical axis further DOWN in the world — frame-agnostic).
#
# At startup the engine logs `T_ee_cam` and the cam +Z direction in BASE frame
# plus the pitch-below-horizon angle. If those don't match your physical mount,
# adjust the tf string or the pitch trim until they do.
#
# If bbox depth reads short vs a tape measure, raise --bbox-depth-scale.
#
# Live tuning (same terminal): add --live-control-stdin=true, then type lines:
#   preposition      — start orbit IK (after you are roughly centered)
#   top              — overhead preset + preposition
#   el 88            — approach elevation (degrees)
#   depth 0.05      — target standoff (meters)
#   radius 0.22     — orbit radius before final approach
#   pan 0.15        — damp shoulder_pan gaze (0..1);  pan auto  — auto-weak when centered
#   up [deg]        — raise arm (shoulder_lift trim +3° default, optional step)
#   down [deg]      — lower arm (same)
#   lift 5 | lift reset — set trim to ±deg or clear
#   status | help
# Single keys (no Enter): add --live-control-keypress=true with stdin; then
# every key drives orbit IK (joints 2/3/4) around the OBJECT and look-at:
#   [ ] or ↓ ↑  — elevation around object (orbit)
#   ,           — back away (larger orbit radius)
#   .           — come in (smaller orbit radius)
#   - / =       — back / in with 2× step (--live-radius-step-m-default)
#   (standoff / final approach distance only applies in APPROACHING, not PREPOSITION)
#   p           — re-enter PREPOSITION immediately
#   ?           — print key help
# Or append to a file: --live-control-file=/tmp/gaze_cmd.txt
#   echo "top" >> /tmp/gaze_cmd.txt
set -euo pipefail

export PYTHONUNBUFFERED=1
export LEROBOT_LOG_LEVEL="${LEROBOT_LOG_LEVEL:-INFO}"

lerobot-gaze-engine \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5A4B0479741 \
  --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \
  --urdf=./SO101/so101_new_calib.urdf \
  --query="red cube" \
  --model-path=./yolov8s-worldv2.pt \
  --gripper-camera-tf="0.04,0,0.09,-0.2690,0.2824,-1.6014" \
  --gripper-camera-pitch-trim-deg=0 \
  --target-physical-size-m=0.03 \
  --bbox-depth-scale=1.0 \
  --bbox-depth-offset-m=0.02 \
  --approach-az-deg=0 \
  --approach-el-deg=82 \
  --final-standoff-m=0.06 \
  --preposition-enabled=true \
  --preposition-from-search=false \
  --preposition-emergency-gaze-pixel-threshold-px=55 \
  --ik-object-floor-z-m=0.015 \
  --preposition-initial-radius-m=0.20 \
  --live-control-stdin=true \
  --live-control-keypress=true \
  --gaze-pan-scale-when-aligned=0.22 \
  --gaze-pan-scale-aligned-enabled=true \
  --preposition-position-tolerance-m=0.04 \
  --preposition-apply-gaze=false \
  --preposition-require-centered-bbox=false \
  --preposition-max-joint-step-deg=5 \
  --preposition-max-lin-vel-m-s=0.08 \
  --preposition-max-ang-vel-deg-s=150 \
  --live-el-step-deg-default=5 \
  --live-el-min-deg=-30 \
  --live-el-max-deg=135 \
  --live-radius-step-m-default=0.03 \
  --live-closeness-step-m-default=0.03 \
  --live-key-max-steps-per-tick=2 \
  --live-radius-slew-m-s=0.18 \
  --live-standoff-slew-m-s=0.06 \
  --live-standoff-step-m-default=0.03 \
  --live-approach-boost-lin-vel-m-s=0.08 \
  --live-approach-boost-fov-scale-min=0.85 \
  --live-preposition-boost-duration-s=0.55 \
  --live-preposition-boost-lin-vel-m-s=0.14 \
  --live-preposition-boost-joint-step-deg=9 \
  --preposition-ik-orientation-weight=1.5 \
  --approach-use-radial-to-object=true \
  --approach-fov-soft-threshold-px=35 \
  --approach-regress-pixel-threshold-px=80 \
  --approach-done-tolerance-m=0.012 \
  --approach-kp=0.6 \
  --approach-max-lin-vel-m-s=0.05 \
  --approach-max-joint-step-deg=2.5 \
  --gaze-kp-pan=0.45 \
  --gaze-kp-tilt=0.35 \
  --gaze-max-step-pan-deg=3.0 \
  --gaze-max-step-tilt-deg=3.0 \
  --gaze-deadband-px=6 \
  --lock-required-frames=4 \
  --track-lost-frames=12 \
  --approach-pixel-threshold-px=35 \
  --approach-consecutive-centered-frames=3 \
  --depth-ema-alpha=0.35 \
  --search-wrist-flex-start-deg=45 \
  --search-wrist-flex-end-deg=4 \
  --search-look-up-period-s=14 \
  --ik-position-weight=2.0 \
  --ik-orientation-weight=1.0 \
  --loop-hz=25 \
  --viz-clamp-object-to-ground=true \
  --viz-ground-plane-z-m=0.0 \
  --display-data=true \
  --display-sim3d=true
