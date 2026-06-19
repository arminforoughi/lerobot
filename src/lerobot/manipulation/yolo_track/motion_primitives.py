# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Low-level motion helpers: smooth joint streaming + gripper-aware Cartesian nudges.

These wrap :mod:`lerobot.utils.motion_executor` with the extra knobs the YOLO-track controller
needs (look-at orientation, rotation clamps, gripper width streaming).
"""

from __future__ import annotations

import time as _time

import numpy as np

from lerobot.manipulation.yolo_track.math_utils import (
    R_ee_step_clamp,
    T_ee_aim_cam_at,
    look_at_R_base_cam,
)
from lerobot.utils.motion_executor import (
    MotionExecutionConfig,
    execute_cartesian_nudge_base,
    execute_pose_waypoint_base,
)


def send_joint_target_smoothly(
    robot,
    motor_names: list[str],
    current_deg: np.ndarray,
    target_deg: np.ndarray,
    *,
    step_deg: float,
    sleep_s: float,
    gripper_open: bool,
    gripper_width_pct: float,
) -> None:
    """Stream small joint updates from ``current_deg`` → ``target_deg`` to avoid jerk.

    Pure joint control (no IK). Used for the search scan where we only want a smooth shoulder
    pan + lift sweep.
    """
    cur = np.asarray(current_deg, dtype=np.float64).copy()
    tgt = np.asarray(target_deg, dtype=np.float64).copy()
    max_move = float(np.max(np.abs(tgt - cur))) if cur.size else 0.0
    n_steps = max(1, int(np.ceil(max_move / max(float(step_deg), 1e-6))))
    for i in range(1, n_steps + 1):
        alpha = float(i) / float(n_steps)
        q = (1.0 - alpha) * cur + alpha * tgt
        action: dict[str, float] = {}
        for j, m in enumerate(motor_names):
            action[f"{m}.pos"] = float(q[j])
        action["gripper.pos"] = 100.0 if gripper_open else float(gripper_width_pct)
        robot.send_action(action)
        _time.sleep(max(0.0, float(sleep_s)))


def execute_gripper_nudge(
    *,
    robot,
    kinematics,
    motor_names: list[str],
    motion_default: MotionExecutionConfig,
    motion_look: MotionExecutionConfig,
    use_look: bool,
    T_base_ee: np.ndarray,
    T_ee_cam: np.ndarray,
    delta_base: np.ndarray,
    aim_point_base: np.ndarray | None,
    eye_cam_base: np.ndarray,
    max_look_rot_step_deg: float = 0.0,
    label: str = "gripper_visual_servo",
) -> None:
    """Translate by ``delta_base`` and optionally reorient the camera to look at ``aim_point_base``.

    When ``use_look`` is true and we have a target point, we issue a single pose waypoint that
    combines the translation and a clamped look-at rotation (``max_look_rot_step_deg`` caps the
    per-call angular change). Otherwise falls back to a pure Cartesian nudge.
    """
    d = np.asarray(delta_base, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(d)) < 1e-9 and not use_look:
        return
    if use_look and aim_point_base is not None:
        R_bc = look_at_R_base_cam(eye_cam_base, aim_point_base)
        if R_bc is not None:
            R_ee_cam = np.asarray(T_ee_cam, dtype=np.float64)[:3, :3]
            R_des_ee = R_bc @ R_ee_cam.T
            if max_look_rot_step_deg > 1e-6:
                R_curr_ee = np.asarray(T_base_ee, dtype=np.float64)[:3, :3]
                R_ee = R_ee_step_clamp(R_curr_ee, R_des_ee, max_look_rot_step_deg)
                R_bc = R_ee @ R_ee_cam
            T_tgt = T_ee_aim_cam_at(T_base_ee, T_ee_cam, d, R_base_cam_desired=R_bc)
            execute_pose_waypoint_base(
                robot,
                kinematics,
                motor_names,
                T_tgt,
                motion_look,
                label=label,
            )
            return
    execute_cartesian_nudge_base(robot, kinematics, motor_names, d, motion_default)
