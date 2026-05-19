#!/usr/bin/env bash
# SEARCH → PAN_ALIGN → APPROACHING. [ ] , . move arm (PREPOSITION); - = depth.
set -euo pipefail

export PYTHONUNBUFFERED=1
export LEROBOT_LOG_LEVEL="${LEROBOT_LOG_LEVEL:-INFO}"

lerobot-gaze-engine \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A4B0479741 \
  --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \
  --urdf=./SO101/so101_new_calib.urdf \
  --query="red cube" \
  --model-path=./yolov8s-worldv2.pt \
  --gripper-camera-tf="0.04,0,0.09,-0.2690,0.2824,-1.6014" \
  --gripper-camera-pitch-trim-deg=0 \
  --target-physical-size-m=0.03 \
  --bbox-depth-scale=1.0 \
  --bbox-depth-offset-m=0.02 \
  --final-standoff-m=0.06 \
  --search-startup-probe=true \
  --search-startup-lock-frames=2 \
  --search-min-detection-confidence=0.18 \
  --approach-el-deg=55 \
  --approach-steep-always-optical=true \
  --gaze-kp-tilt=0.48 \
  --approach-gaze-max-tilt-deg-coarse=4.0 \
  --approach-pause-vertical-err-px=28 \
  --preposition-enabled=true \
  --preposition-apply-gaze=true \
  --live-control-stdin=true \
  --live-control-keypress=true \
  --live-keys-auto-preposition=true \
  --live-keys-snap-targets=true \
  --live-key-max-steps-per-tick=8 \
  --live-el-slew-deg-s=72 \
  --live-preposition-boost-duration-s=1.2 \
  --live-preposition-boost-lin-vel-m-s=0.16 \
  --display-data=true \
  --display-sim3d=true
