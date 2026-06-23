#!/usr/bin/env bash
# Minimal live tracker: detect -> center -> depth standoff (no grasp / no phased planner)
set -euo pipefail

lerobot-minimal-tracker \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5A4B0479741 \
  --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \
  --urdf=./SO101/so101_new_calib.urdf \
  --query="red cube" \
  --model-path=./yolov8s-worldv2.pt \
  --camera-mount=gripper \
  --camera-frame-convention=opencv \
  --gripper-camera-tf="0.04,0,0.02,0,-0.35,0" \
  --conf-threshold=0.25 \
  --axis-mode=xyz \
  --z-only-when-centered=true \
  --z-backoff-enable=false \
  --z-backoff-deadband-m=0.01 \
  --depth-policy=bbox_preferred \
  --max-depth-jump-m=0.12 \
  --kp-xy=0.52 \
  --kp-z=0.55 \
  --max-step-m=0.02 \
  --center-deadband-px=18 \
  --target-depth-m=0.22 \
  --min-depth-m=0.11 \
  --pan-center-enable=true \
  --pan-deadband-px=18 \
  --pan-max-step-deg=6 \
  --search-scan-enabled=true \
  --search-scan-start-after-lost-frames=6 \
  --search-pan-amplitude-deg=20 \
  --search-pan-period-iters=22 \
  --stop-area-frac=0.18 \
  --display-data=true \
  --display-sim3d=true \
  --loop-sleep-s=0.03 \
  --lost-patience=25
