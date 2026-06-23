#!/usr/bin/env bash
# Hybrid YOLO (coarse) + VLM (fine) + current-sensing grasp.
#
# Requires: GEMINI_API_KEY in env (or pass --vlm-api-key=...).
# - YOLO `plan_top` drives the arm to a hover above the target (safe, deterministic).
# - Gemini then suggests small dx/dy/dyaw refinements until it says "close now".
# - Gripper closes in small percent-steps while we watch the gripper motor's Present_Current;
#   contact is declared on a current spike or a position stall.
# - Arm lifts a few cm and re-samples current to confirm the object is still held.

set -euo pipefail

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY (or pass --vlm-api-key=...)}"

lerobot-yolo-vlm-grasp \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5A4B0479741 \
  --robot.cameras='{"front": {"type": "oakd", "fps": 30, "width": 640, "height": 480, "use_depth": true}}' \
  --urdf=./SO101/so101_new_calib.urdf \
  --query="red cube" \
  --model-path=./yolov8s-worldv2.pt \
  --camera-mount=fixed \
  --camera-frame-convention=opsencv \
  --gripper-camera-tf="0.04,0,0.02,0,-0.35,0" \
  --target-from-gripper-tf=true \
  --search-scan-enabled=true \
  --depth-from-bbox-enabled=true \
  --target-physical-size-m=0.03 \
  --depth-source-policy=bbox_preferred \
  --top-approach-height-m=0.05 \
  --top-max-reach-m=0.35 \
  --table-z-m=-0.02 \
  --table-clearance-m=0.005 \
  --plan-top-tilt-fraction=0.0 \
  --plan-top-tilt-tolerance-deg=90 \
  --plan-top-final-hover-m=0.05 \
  --plan-top-gripper-tip-offset-m=0.02 \
  --vlm-backend=gemini \
  --vlm-model=gemini-3-flash \
  --vlm-max-refine-iters=6 \
  --vlm-nudge-step-m=0.012 \
  --vlm-nudge-total-budget-m=0.06 \
  --vlm-wrist-roll-step-deg=15 \
  --vlm-wrist-roll-total-deg=45 \
  --vlm-auto-close-on-exhaustion=true \
  --hybrid-final-tip-gap-m=0.012 \
  --grasp-open-pct=100 \
  --grasp-close-pct=0 \
  --grasp-close-step-pct=2 \
  --grasp-close-step-wait-s=0.08 \
  --grasp-contact-delta-current-counts=40 \
  --grasp-contact-stall-steps=3 \
  --grasp-hold-seconds=0.8 \
  --grasp-lift-height-m=0.06 \
  --grasp-lift-confirm-delta-counts=20 \
  --hybrid-log-jsonl="hybrid_grasp_$(date +%Y%m%d_%H%M%S).jsonl" \
  --show-window=true --display-data=true --display-sim3d=true
