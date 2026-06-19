#!/usr/bin/env bash
# Estimate-then-plan look-at grasp engine.
# The object lives as a filtered 3D point in the base frame (ray ∩ known
# surface height), so arm motion and dropped detections never blind the
# controller. Each tick: one full-pose IK that keeps the camera aimed at the
# object (look-at) while the gripper tip closes on a standoff along the
# approach ray. SEARCH -> APPROACH -> GRASP.
set -euo pipefail

export PYTHONUNBUFFERED=1
export LEROBOT_LOG_LEVEL="${LEROBOT_LOG_LEVEL:-INFO}"

lerobot-lookat-engine \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A4B0479741 \
  --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \
  --urdf=./SO101/so101_new_calib.urdf \
  --query="red cube" \
  --model-path=./yolov8s-worldv2.pt \
  --gripper-camera-tf="0.04,0,0.09,-0.2690,0.2824,-1.6014" \
  --gripper-camera-pitch-trim-deg=0 \
  --gripper-tip-offset-m=0.10 \
  --target-physical-size-m=0.03 \
  --min-detection-confidence=0.20 \
  --bbox-grasp-v-frac=0.5 \
  `# Pin the object-centre height below base for rock-stable plane tracking,` \
  `# e.g. --object-center-z-m=-0.17 . Leave unset to auto-track from depth.` \
  --approach-az-deg=0 \
  --approach-el-deg=55 \
  --final-standoff-m=0.05 \
  --grasp-trigger-tip-m=0.10 \
  --tip-floor-below-object-m=0.03 \
  --tip-floor-abs-z-m=-0.30 \
  --loop-hz=25 \
  --max-lin-vel-m-s=0.05 \
  --max-joint-step-deg=3.0 \
  --ik-orientation-weight=0.15 \
  --grasp-center-pixel-err-px=45 \
  --acquire-frames=3 \
  --lost-grace-ticks=40 \
  --grasp-enable=true \
  --grasp-final-approach-m=0.04 \
  --grasp-final-gap-m=0.012 \
  --grasp-max-inch-m=0.16 \
  --grasp-final-approach-along-optical=false \
  --grasp-post-contact-squeeze-pct=8.0 \
  --grasp-lift-confirm=true \
  --grasp-hold-after=true \
  --display-data=true \
  --display-sim3d=true
