# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Main control loop for the YOLO-track approach pipeline.

The loop iterates once per camera frame:

1. Get an RGB/(depth) frame and read the current joints.
2. Run YOLO-World to pick the best box for the text query.
3. Update Rerun overlays (bbox, depth, p_target_base, EE trail, camera frustum).
4. If not engaged: run the mount-agnostic search-scan (pan + lift + wrist) and
   the optional gripper-raise, until a stable detection is seen.
5. Once engaged, dispatch per ``approach_style``:
   - ``plan_top``: gather 3D target → phased lift/align/tilt/descend/center.
   - ``umbrella``: phased lift → planar align → base-Z descend with visual servo.
   - ``servo``: continuous image-centering + depth/bbox-area closing.

This module is intentionally a long, linear function so the phase-by-phase logic stays easy to
read top-to-bottom and to debug from the logs. Stateless helpers live in the sibling modules
(:mod:`.math_utils`, :mod:`.depth`, :mod:`.motion_primitives`, :mod:`.rerun_viz`).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from lerobot.configs import parser
from lerobot.manipulation.yolo_track.config import YoloTrackApproachConfig
from lerobot.manipulation.yolo_track.depth import depth_from_bbox_size, median_depth_m
from lerobot.manipulation.yolo_track.math_utils import (
    camera_opencv_to_robot_rotation,
    ema_p_base,
    look_at_R_base_cam,
    parse_horizon_dir_base,
    parse_tf_string,
    point_cam_to_base,
    rot_step_toward,
    rotation_base_cam,
    visual_servo_delta_base,
    wrist_down_R_base_ee,
)
from lerobot.manipulation.yolo_track.motion_primitives import (
    execute_gripper_nudge,
    send_joint_target_smoothly,
)
from lerobot.manipulation.yolo_track.rerun_viz import log_rerun_iter
from lerobot.robots import make_robot_from_config
from lerobot.utils.motion_executor import (
    MotionExecutionConfig,
    execute_cartesian_nudge_base,
    execute_pose_waypoint_base,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

SO100_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _camera_on_arm(cfg: YoloTrackApproachConfig, mount: str) -> bool:
    """True when the pipeline uses the EE camera chain for geometry (pan centers the bbox).

    Users sometimes set ``camera_mount=fixed`` while still passing ``--target-from-gripper-tf``
    and a non-trivial ``gripper_camera_tf``; in that case shoulder_pan still moves the stereo
    on the wrist, so yaw centering is meaningful.
    """
    return (mount or "").lower() == "gripper" or bool(cfg.target_from_gripper_tf)


def _send_shoulder_pan_toward_image_center(
    robot,
    cfg: YoloTrackApproachConfig,
    joints_deg: np.ndarray,
    *,
    cx_img: float,
    fx: float,
    cx0: float,
    deadband_px: float,
    kp: float,
    max_step_deg: float,
    wrist_bias_deg: float,
    iter_idx: int,
    log_prefix: str,
) -> bool:
    """Bounded shoulder_pan step from horizontal pixel error (theta ≈ atan((cx-cx0)/fx))."""
    ex_px = float(cx_img - cx0)
    if abs(ex_px) <= deadband_px:
        return False
    try:
        i_pan = SO100_MOTOR_NAMES.index("shoulder_pan")
    except ValueError:
        return False
    ang_err_rad = float(math.atan(ex_px / max(fx, 1e-6)))
    d_deg = float(np.clip(np.degrees(kp * ang_err_rad), -max_step_deg, max_step_deg))
    target_joints = joints_deg.copy()
    target_joints[i_pan] = float(target_joints[i_pan] + d_deg)
    if abs(wrist_bias_deg) > 1e-6:
        try:
            i_wrist = SO100_MOTOR_NAMES.index("wrist_flex")
            target_joints[i_wrist] = float(target_joints[i_wrist] + wrist_bias_deg)
        except ValueError:
            pass
    obs_gr = robot.get_observation()
    raw_g = obs_gr.get("gripper.pos")
    g_open = True
    g_pct = 100.0
    if raw_g is not None:
        v = float(raw_g)
        g_open = v >= 90.0
        g_pct = float(np.clip(v, 0.0, 100.0))
    logger.info(
        "%s iter=%d ex=%.0fpx Δpan=%+.2f° (kp=%.2f max=%.1f°)",
        log_prefix,
        iter_idx,
        ex_px,
        d_deg,
        kp,
        max_step_deg,
    )
    send_joint_target_smoothly(
        robot,
        SO100_MOTOR_NAMES,
        joints_deg,
        target_joints,
        step_deg=float(cfg.search_joint_step_deg),
        sleep_s=float(cfg.search_joint_sleep_s),
        gripper_open=g_open,
        gripper_width_pct=g_pct,
    )
    return True


def _plan_top_trans_nudge(
    *,
    robot,
    kinematics,
    cfg: YoloTrackApproachConfig,
    mount: str,
    motion: MotionExecutionConfig,
    motion_look: MotionExecutionConfig,
    T_base_ee: np.ndarray,
    T_ee_cam: np.ndarray,
    p_target_base: np.ndarray | None,
    delta_base: np.ndarray,
    use_look_flag: bool,
    label: str,
) -> None:
    """Cartesian delta in base; optional look-at to ``p_target_base`` (often makes IK erratic; default off)."""
    d = np.asarray(delta_base, dtype=np.float64).reshape(3)
    look_ok = (
        bool(use_look_flag)
        and _camera_on_arm(cfg, mount)
        and p_target_base is not None
        and bool(cfg.gripper_point_at_target)
    )
    if look_ok:
        T_bc = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(T_ee_cam, dtype=np.float64)
        execute_gripper_nudge(
            robot=robot,
            kinematics=kinematics,
            motor_names=SO100_MOTOR_NAMES,
            motion_default=motion,
            motion_look=motion_look,
            use_look=True,
            T_base_ee=np.asarray(T_base_ee, dtype=np.float64),
            T_ee_cam=np.asarray(T_ee_cam, dtype=np.float64),
            delta_base=d,
            aim_point_base=np.asarray(p_target_base, dtype=np.float64).reshape(3),
            eye_cam_base=np.asarray(T_bc[:3, 3], dtype=np.float64),
            max_look_rot_step_deg=float(cfg.gripper_max_look_rot_step_deg),
            label=label,
        )
    else:
        execute_cartesian_nudge_base(robot, kinematics, SO100_MOTOR_NAMES, d, motion)


def _plan_top_apply_wrist_flex_approach_bias(
    robot: Any,
    cfg: YoloTrackApproachConfig,
    joints_deg: np.ndarray,
    applied_so_far: float,
) -> float:
    """Cumulative negative wrist_flex nudge so the gripper cam looks up during Cartesian plan_top.

    Returns updated ``applied_so_far`` (algebraic sum of increments, toward ``total_deg`` cap).
    """
    if not bool(cfg.plan_top_approach_wrist_flex_bias_enable):
        return float(applied_so_far)
    cap = float(cfg.plan_top_approach_wrist_flex_total_deg)
    step = float(cfg.plan_top_approach_wrist_flex_step_deg)
    if cap >= -1e-6 or abs(step) < 1e-6 or step > 1e-9:
        return float(applied_so_far)
    applied = float(applied_so_far)
    remaining = cap - applied
    if remaining >= -1e-6:
        return applied
    this = float(max(step, remaining))
    if abs(this) < 1e-6:
        return applied
    try:
        i_w = SO100_MOTOR_NAMES.index("wrist_flex")
    except ValueError:
        return applied
    obs_gr = robot.get_observation()
    cur = np.array([float(obs_gr[f"{m}.pos"]) for m in SO100_MOTOR_NAMES], dtype=np.float64)
    tgt = cur.copy()
    tgt[i_w] = float(tgt[i_w] + this)
    raw_g = obs_gr.get("gripper.pos")
    g_open = True
    g_pct = 100.0
    if raw_g is not None:
        v = float(raw_g)
        g_open = v >= 90.0
        g_pct = float(np.clip(v, 0.0, 100.0))
    send_joint_target_smoothly(
        robot,
        SO100_MOTOR_NAMES,
        cur,
        tgt,
        step_deg=float(cfg.plan_top_approach_wrist_flex_joint_step_deg),
        sleep_s=float(cfg.plan_top_approach_wrist_flex_joint_sleep_s),
        gripper_open=g_open,
        gripper_width_pct=g_pct,
    )
    obs2 = robot.get_observation()
    joints_deg[:] = np.array(
        [float(obs2[f"{m}.pos"]) for m in SO100_MOTOR_NAMES], dtype=np.float64
    )
    applied += this
    logger.debug(
        "[yolo-track] plan_top: approach wrist_flex Δ=%.2f° (cumulative=%.2f°, cap=%.2f°)",
        this,
        applied,
        cap,
    )
    return applied


def _plan_top_vertical_recovery_use_shoulder(cfg: YoloTrackApproachConfig) -> bool:
    return (cfg.plan_top_vertical_recovery_mode or "shoulder_lift").lower().strip() == "shoulder_lift"


def _plan_top_apply_shoulder_lift_recovery(
    robot: Any,
    cfg: YoloTrackApproachConfig,
    joints_deg: np.ndarray,
    applied_so_far: float,
) -> float:
    """Negative shoulder_lift delta raises the arm on SO-101; returns updated cumulative applied."""
    if not _plan_top_vertical_recovery_use_shoulder(cfg):
        return float(applied_so_far)
    cap = float(cfg.plan_top_vertical_recovery_shoulder_lift_total_deg)
    step = float(cfg.plan_top_vertical_recovery_shoulder_lift_step_deg)
    if cap >= -1e-6 or abs(step) < 1e-6 or step > 1e-9:
        return float(applied_so_far)
    applied = float(applied_so_far)
    remaining = cap - applied
    if remaining >= -1e-6:
        return applied
    this = float(max(step, remaining))
    if abs(this) < 1e-6:
        return applied
    try:
        i_sl = SO100_MOTOR_NAMES.index("shoulder_lift")
    except ValueError:
        return applied
    obs_gr = robot.get_observation()
    cur = np.array([float(obs_gr[f"{m}.pos"]) for m in SO100_MOTOR_NAMES], dtype=np.float64)
    tgt = cur.copy()
    tgt[i_sl] = float(tgt[i_sl] + this)
    raw_g = obs_gr.get("gripper.pos")
    g_open = True
    g_pct = 100.0
    if raw_g is not None:
        v = float(raw_g)
        g_open = v >= 90.0
        g_pct = float(np.clip(v, 0.0, 100.0))
    send_joint_target_smoothly(
        robot,
        SO100_MOTOR_NAMES,
        cur,
        tgt,
        step_deg=float(cfg.plan_top_vertical_recovery_shoulder_joint_step_deg),
        sleep_s=float(cfg.plan_top_vertical_recovery_shoulder_joint_sleep_s),
        gripper_open=g_open,
        gripper_width_pct=g_pct,
    )
    obs2 = robot.get_observation()
    joints_deg[:] = np.array(
        [float(obs2[f"{m}.pos"]) for m in SO100_MOTOR_NAMES], dtype=np.float64
    )
    applied += this
    logger.info(
        "[yolo-track] plan_top: vertical recover shoulder_lift Δ=%.2f° (cumulative=%.1f° cap=%.1f°)",
        this,
        applied,
        cap,
    )
    return applied


@dataclass
class PlanTopHandoffState:
    """State snapshot passed to ``on_plan_top_descend_done`` when plan_top finishes descend.

    The callback owns the remaining behaviour (fine VLM refinement, grasping, lifting, etc.).
    When it returns, the runner breaks out of the control loop and runs its normal teardown
    (``robot.disconnect()``). The callback may mutate the robot freely.

    Attributes are the exact objects the runner is using — pass them to motion primitives
    directly to stay coordinated with the runner's kinematics / motion config.
    """

    cfg: YoloTrackApproachConfig
    robot: Any  # lerobot.robots.Robot
    kinematics: Any  # lerobot.model.kinematics.RobotKinematics
    motion: Any  # MotionExecutionConfig (translation-friendly preset)
    motion_look: Any  # MotionExecutionConfig (orientation-weighted preset)
    T_ee_cam: np.ndarray  # 4x4 camera pose in ee frame
    T_base_ee: np.ndarray  # 4x4 current ee pose in base frame at handoff
    p_target_base: np.ndarray  # 3-vec, object position in base frame (smoothed)
    det: Any  # YoloWorldDetector Detection (label, xyxy, confidence) or None
    rgb: np.ndarray  # last RGB frame the runner observed (HxWx3 uint8)
    depth: np.ndarray | None  # last depth frame (uint16 mm) or None
    intrinsics: dict[str, float]
    fx: float
    fy: float
    cx0: float
    cy0: float
    cam_to_robot: np.ndarray  # 4x4 (fixed-mount only)
    mount: str  # "fixed" | "gripper"
    # Optional PBVS / close-range handoff (defaults keep YOLO-only handoffs working).
    depth_ema_m: float | None = None
    du_px: float | None = None
    dv_px: float | None = None
    depth_goal_m: float | None = None
    yolo_detector: Any | None = None


@parser.wrap()
def yolo_track_approach(cfg: YoloTrackApproachConfig) -> None:
    """CLI entrypoint (``lerobot-yolo-track-approach``).

    Behaviour is identical to the pre-refactor script. See :func:`run_yolo_track_approach`
    for the programmatic variant that accepts a ``on_plan_top_descend_done`` callback.
    """
    run_yolo_track_approach(cfg)


def run_yolo_track_approach(
    cfg: YoloTrackApproachConfig,
    *,
    on_plan_top_descend_done: Callable[[PlanTopHandoffState], None] | None = None,
) -> None:
    """Run the YOLO-track approach pipeline.

    Args:
        cfg: parsed configuration.
        on_plan_top_descend_done: optional callback invoked once after ``plan_top`` finishes
            the ``descend`` phase and the gripper is hovering above the target. When provided,
            it overrides the built-in ``center`` phase: the callback takes over control of the
            (still-connected) robot, and when it returns the runner breaks and disconnects.
            Ignored for approach styles other than ``plan_top``.
    """
    init_logging()
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.perception.yolo_world import YoloWorldDetector

    mount = (cfg.camera_mount or "fixed").lower().strip()
    if mount not in ("fixed", "gripper"):
        raise ValueError("camera_mount must be 'fixed' or 'gripper'")

    style = (cfg.approach_style or "servo").lower().strip()
    if style not in ("servo", "umbrella", "plan_top"):
        raise ValueError("approach_style must be 'servo', 'umbrella', or 'plan_top'")

    T_ee_cam = parse_tf_string(cfg.gripper_camera_tf)
    cam_to_robot = parse_tf_string(cfg.camera_to_robot_tf)
    if mount == "fixed" and (cfg.camera_frame_convention or "").lower() == "opencv":
        t = cam_to_robot[:3, 3].copy()
        R = camera_opencv_to_robot_rotation(flip_lateral=bool(cfg.camera_flip_lateral))
        cam_to_robot = np.eye(4, dtype=np.float64)
        cam_to_robot[:3, :3] = R
        cam_to_robot[:3, 3] = t

    kinematics = RobotKinematics(
        urdf_path=cfg.urdf,
        target_frame_name=cfg.ee_frame,
        joint_names=SO100_MOTOR_NAMES,
    )

    detector = YoloWorldDetector(
        cfg.model_path,
        device=(cfg.device or None),
        conf=float(cfg.conf_threshold),
        sahi_enable=bool(getattr(cfg, "sahi_enable", False)),
        sahi_slice_wh=(int(getattr(cfg, "sahi_slice_w", 512)), int(getattr(cfg, "sahi_slice_h", 512))),
        sahi_overlap=float(getattr(cfg, "sahi_overlap", 0.20)),
        sahi_iou_threshold=float(getattr(cfg, "sahi_iou_threshold", 0.55)),
        sahi_include_full_image=bool(getattr(cfg, "sahi_include_full_image", True)),
    )
    detector.set_query(cfg.query)

    if mount == "fixed" and cfg.gripper_camera_tf.replace(" ", "") not in ("0,0,0,0,0,0",):
        logger.warning(
            "[yolo-track] camera_mount=fixed but gripper_camera_tf=%r looks non-trivial. "
            "If the stereo camera is physically mounted on the gripper, pass "
            "--camera-mount=gripper so the controller can point the wrist at the target "
            "(otherwise the object drifts out of frame as soon as the arm lifts). "
            "If you keep fixed+--target-from-gripper-tf, shoulder_pan centering still runs, "
            "but Rerun/aim paths that key off camera_mount may be inconsistent.",
            cfg.gripper_camera_tf,
        )
    if mount == "gripper" and cfg.gripper_camera_tf.replace(" ", "") == "0,0,0,0,0,0":
        logger.warning(
            "[yolo-track] camera_mount=gripper but gripper_camera_tf is identity. "
            "Pass --gripper-camera-tf='x,y,z,rx,ry,rz' (camera pose in ee frame). "
            "E.g. camera 5cm forward, 2cm up, pitched down 20°: '0.05,0,0.02,-0.35,0,0'."
        )

    rerun_enabled = bool(cfg.display_data) or bool(cfg.display_sim3d)
    if rerun_enabled:
        try:
            from lerobot.utils.visualization_utils import init_rerun, send_agentic_rerun_blueprint

            init_rerun(
                session_name="yolo_track_approach",
                ip=cfg.display_ip,
                port=cfg.display_port,
            )
            send_agentic_rerun_blueprint(
                show_camera_stream=bool(cfg.display_data),
                show_sim3d=bool(cfg.display_sim3d),
                camera_key=cfg.camera_key,
            )
        except Exception as e:
            logger.warning("Rerun init failed (continuing without viz): %s", e)
            rerun_enabled = False
    ee_trail: list[np.ndarray] = []

    motion = MotionExecutionConfig(
        cartesian_step_m=float(cfg.motion_cartesian_step_m),
        min_steps_per_segment=int(cfg.motion_min_steps_per_segment),
        inter_step_sleep_s=float(cfg.motion_inter_step_sleep_s),
        settle_timeout_s=float(cfg.motion_settle_timeout_s),
        settle_threshold_deg=float(cfg.motion_settle_threshold_deg),
        settle_last_step=bool(cfg.motion_settle_last_step),
        cartesian_translate_first_on_reach=False,
        lock_wrist_roll_on_coarse_reach=False,
    )
    motion_look = MotionExecutionConfig(
        cartesian_step_m=float(cfg.motion_cartesian_step_m),
        min_steps_per_segment=int(cfg.motion_min_steps_per_segment),
        inter_step_sleep_s=float(cfg.motion_inter_step_sleep_s),
        settle_timeout_s=float(cfg.motion_settle_timeout_s),
        settle_threshold_deg=float(cfg.motion_settle_threshold_deg),
        settle_last_step=bool(cfg.motion_settle_last_step),
        cartesian_translate_first_on_reach=False,
        lock_wrist_roll_on_coarse_reach=False,
        ik_position_weight=1.0,
        ik_orientation_weight=float(cfg.gripper_ik_orientation_weight),
    )
    try:
        horiz_dir = parse_horizon_dir_base(cfg.gripper_search_horizon_dir)
    except ValueError as e:
        raise ValueError(f"Invalid gripper_search_horizon_dir: {e}") from e

    robot = None
    depth_key = f"{cfg.camera_key}_depth"
    intrinsics: dict[str, float] = {}

    if cfg.dry_run:
        if not (cfg.dry_run_image or "").strip():
            raise ValueError("dry_run requires --dry-run-image")
        rgb_np = cv2.imread(cfg.dry_run_image.strip())
        if rgb_np is None:
            raise FileNotFoundError(cfg.dry_run_image)
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        h, w = rgb_np.shape[:2]
        intrinsics = {"fx": 525.0, "fy": 525.0, "cx": w / 2.0, "cy": h / 2.0, "depth_scale": 0.001}
        logger.info("[yolo-track] dry-run: no motion, image=%s query=%r", cfg.dry_run_image, cfg.query)
    else:
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        dc = robot.cameras.get(cfg.camera_key)
        if dc is None:
            raise ValueError(f"camera {cfg.camera_key!r} missing; have {list(robot.cameras.keys())}")
        if hasattr(dc, "get_depth_intrinsics"):
            try:
                intrinsics = dict(dc.get_depth_intrinsics())
            except Exception as e:
                logger.warning("get_depth_intrinsics failed: %s", e)
        if not intrinsics:
            obs = robot.get_observation()
            im = obs.get(cfg.camera_key)
            if im is not None:
                hh, ww = np.asarray(im).shape[:2]
                intrinsics = {"fx": 525.0, "fy": 525.0, "cx": ww / 2.0, "cy": hh / 2.0, "depth_scale": 0.001}

    fx = float(intrinsics.get("fx", 525.0))
    fy = float(intrinsics.get("fy", 525.0))
    cx0 = float(intrinsics.get("cx", 320.0))
    cy0 = float(intrinsics.get("cy", 240.0))
    depth_scale = float(intrinsics.get("depth_scale", 0.001))

    phased = style == "umbrella" and bool(cfg.umbrella_phased)
    if cfg.dry_run:
        phased = False
    if phased:
        uma_phase = "align" if float(cfg.umbrella_clearance_z_m) <= 1e-6 else "lift"
    else:
        uma_phase = "descend"
    z_start: float | None = None
    lift_iter_total = 0
    engaged = False
    seen_streak = 0
    search_z0: float | None = None
    search_raise_iters = 0

    lost = 0
    smooth_cx: float | None = None
    smooth_cy: float | None = None
    smooth_depth_m: float | None = None
    prev_smoothed_delta = np.zeros(3, dtype=np.float64)
    p_target_base: np.ndarray | None = None
    hold_age = 0
    good_depth_samples = 0
    last_det = None
    last_det_t = 0.0
    baseline_joints_deg: np.ndarray | None = None
    scan_step_idx = 0
    top_pose_committed = False
    top_descend_iters = 0
    # Phased plan_top state machine.
    plan_top_phase: str = "gather"  # gather → lift → align → tilt → descend
    plan_top_target_xyz: np.ndarray | None = None  # committed hover XY (target + up offset)
    plan_top_lift_z_goal: float | None = None  # absolute base-z to reach in lift phase
    # Per-phase iter counters and rolling progress history (for stuck detection). Each phase
    # pushes a scalar metric (z_ee for lift, xy-error for align, orientation-remain for tilt)
    # into its window; if the spread over the window is below the threshold we declare stuck.
    plan_top_lift_iters = 0
    plan_top_align_iters = 0
    plan_top_tilt_iters = 0
    plan_top_descend_recover_iters = 0
    plan_top_descend_recover_warned = False
    plan_top_lift_hist: list[float] = []
    plan_top_align_hist: list[float] = []
    plan_top_tilt_hist: list[float] = []
    # Target rotation for the tilt phase, computed once on first entry as a partial rotation
    # (``plan_top_tilt_fraction``) from the pose at tilt-entry toward strict wrist-down. We hold
    # it fixed across iterations so the per-step rotation error decreases monotonically.
    plan_top_tilt_R_tgt: np.ndarray | None = None
    plan_top_center_no_det_lift_accum = 0.0
    plan_top_center_shoulder_lift_applied = 0.0
    plan_top_descend_shoulder_lift_applied = 0.0
    plan_top_wrist_flex_bias_applied = 0.0
    plan_top_frame_smooth_cx: float | None = None
    plan_top_frame_smooth_cy: float | None = None
    try:
        for it in range(int(cfg.max_iters)):
            if cfg.dry_run:
                rgb = rgb_np
                depth = None
                joints_deg = np.zeros(5, dtype=np.float64)
            else:
                assert robot is not None
                obs = robot.get_observation()
                rgb = obs.get(cfg.camera_key)
                depth = obs.get(depth_key)
                if rgb is None:
                    logger.warning("no RGB")
                    time.sleep(float(cfg.loop_sleep_s))
                    continue
                rgb = np.asarray(rgb)
                joints_deg = np.array([float(obs[f"{m}.pos"]) for m in SO100_MOTOR_NAMES], dtype=np.float64)

            det = detector.best_detection(rgb)
            det_held = False
            now_t = time.time()
            hold_s = float(max(0.0, getattr(cfg, "detection_hold_s", 0.0) or 0.0))
            if det is None and last_det is not None and hold_s > 1e-6:
                if (now_t - float(last_det_t)) <= hold_s:
                    det = last_det
                    det_held = True
            if det is not None and not det_held:
                last_det = det
                last_det_t = now_t

            if det is None:
                seen_streak = 0
            else:
                if not det_held:
                    hold_age = 0
                seen_streak = seen_streak + 1 if float(det.confidence) >= float(cfg.engage_min_conf) else 0

            if rerun_enabled and not cfg.dry_run:
                try:
                    T_base_ee_now = kinematics.forward_kinematics(joints_deg)
                    T_base_cam_now = T_base_ee_now @ T_ee_cam if mount == "gripper" else cam_to_robot
                    ee_pos_now = np.asarray(T_base_ee_now, dtype=np.float64)[:3, 3].copy()
                    if not ee_trail or float(np.linalg.norm(ee_pos_now - ee_trail[-1])) > 1e-3:
                        ee_trail.append(ee_pos_now)
                        if len(ee_trail) > 600:
                            ee_trail.pop(0)
                    bbox_xyxy_log = tuple(det.xyxy) if det is not None else None
                    center_raw_log = None
                    center_s_log = None
                    if det is not None:
                        x1_l, y1_l, x2_l, y2_l = det.xyxy
                        center_raw_log = (0.5 * (x1_l + x2_l), 0.5 * (y1_l + y2_l))
                        center_s_log = (
                            (float(smooth_cx), float(smooth_cy))
                            if smooth_cx is not None and smooth_cy is not None
                            else center_raw_log
                        )
                    d_log = None
                    if depth is not None and det is not None:
                        d_log = median_depth_m(
                            np.asarray(depth), det.xyxy, depth_scale=depth_scale
                        )
                        if d_log is not None and float(d_log) > float(cfg.max_plausible_depth_m):
                            d_log = None
                    log_rerun_iter(
                        frame=it,
                        camera_key=cfg.camera_key,
                        rgb=np.asarray(rgb),
                        depth=depth,
                        bbox_xyxy=bbox_xyxy_log,
                        bbox_center_raw=center_raw_log,
                        bbox_center_smoothed=center_s_log,
                        cx0=cx0,
                        cy0=cy0,
                        conf=float(det.confidence) if det is not None else None,
                        phase=(uma_phase if engaged else "search"),
                        z_ee=float(ee_pos_now[2]),
                        depth_m=d_log,
                        kinematics=kinematics,
                        joints_deg=joints_deg,
                        T_base_ee=T_base_ee_now,
                        T_base_cam=T_base_cam_now,
                        p_target_base=p_target_base,
                        ee_trail=ee_trail,
                        object_half_size_m=float(cfg.display_object_half_size_m),
                        show_sim3d=bool(cfg.display_sim3d),
                        show_camera=bool(cfg.display_data),
                        object_semantic_label=str(cfg.query),
                    )
                except Exception as _rr_e:
                    logger.debug("rerun per-iter log failed: %s", _rr_e)

            if not engaged:
                T_search = kinematics.forward_kinematics(joints_deg)
                z_search = float(T_search[2, 3])
                if search_z0 is None:
                    search_z0 = z_search
                if baseline_joints_deg is None:
                    baseline_joints_deg = joints_deg.copy()

                # Mount-agnostic search scan: raise shoulder + oscillate shoulder_pan around the
                # baseline (folded) pose so the (gripper-mounted) camera sweeps the scene.
                if (
                    bool(cfg.search_scan_enabled)
                    and det is None
                    and not cfg.dry_run
                    and baseline_joints_deg is not None
                ):
                    assert robot is not None
                    amp = float(cfg.search_pan_amplitude_deg)
                    period = max(2, int(cfg.search_pan_period_iters))
                    pan_delta = amp * float(np.sin(2.0 * np.pi * (scan_step_idx % period) / period))
                    lift_delta = float(cfg.search_lift_up_delta_deg) * (
                        1.0 - float(np.exp(-scan_step_idx / max(period / 2.0, 1.0)))
                    )
                    lift_delta = float(
                        np.clip(
                            lift_delta,
                            -abs(float(cfg.search_lift_up_max_deg)),
                            abs(float(cfg.search_lift_up_max_deg)),
                        )
                    )
                    if float(cfg.search_lift_up_max_deg) < 0:
                        lift_delta = -abs(lift_delta)
                    target_joints = baseline_joints_deg.copy()
                    try:
                        i_pan = SO100_MOTOR_NAMES.index("shoulder_pan")
                        target_joints[i_pan] = baseline_joints_deg[i_pan] + pan_delta
                    except ValueError:
                        pass
                    try:
                        i_lift = SO100_MOTOR_NAMES.index("shoulder_lift")
                        target_joints[i_lift] = baseline_joints_deg[i_lift] + lift_delta
                    except ValueError:
                        pass
                    # Pitch the wrist UP during search so the camera (which is mounted pitched
                    # downward in ``gripper_camera_tf``) actually looks out into the workspace
                    # instead of at the table. Ramped in over ~half the pan period so we don't
                    # yank the wrist on the first iteration.
                    try:
                        i_wrist = SO100_MOTOR_NAMES.index("wrist_flex")
                        wrist_ramp = 1.0 - float(np.exp(-scan_step_idx / max(period / 2.0, 1.0)))
                        wrist_delta = float(cfg.search_wrist_up_delta_deg) * wrist_ramp
                        target_joints[i_wrist] = baseline_joints_deg[i_wrist] + wrist_delta
                    except ValueError:
                        wrist_delta = 0.0
                    obs_gr = robot.get_observation()
                    raw_g = obs_gr.get("gripper.pos")
                    g_open = True
                    g_pct = 100.0
                    if raw_g is not None:
                        v = float(raw_g)
                        g_open = v >= 90.0
                        g_pct = float(np.clip(v, 0.0, 100.0))
                    logger.info(
                        "[yolo-track] iter=%d search-scan pan=%+.1f° lift=%+.1f° wrist=%+.1f° (baseline+Δ)",
                        it,
                        pan_delta,
                        lift_delta,
                        wrist_delta,
                    )
                    send_joint_target_smoothly(
                        robot,
                        SO100_MOTOR_NAMES,
                        joints_deg,
                        target_joints,
                        step_deg=float(cfg.search_joint_step_deg),
                        sleep_s=float(cfg.search_joint_sleep_s),
                        gripper_open=g_open,
                        gripper_width_pct=g_pct,
                    )
                    scan_step_idx += 1
                    time.sleep(float(cfg.loop_sleep_s))
                    continue

                if (
                    mount == "gripper"
                    and bool(cfg.gripper_search_raise)
                    and det is None
                    and not cfg.dry_run
                ):
                    hit_h = z_search >= float(search_z0) + float(cfg.search_raise_max_m) - 1e-6
                    hit_n = search_raise_iters >= int(cfg.search_raise_max_iters)
                    if not hit_h and not hit_n:
                        search_raise_iters += 1
                        d_raise = np.zeros(3, dtype=np.float64)
                        d_raise[2] = float(cfg.search_raise_z_sign) * float(cfg.search_raise_step_m)
                        mmax_sr = float(cfg.max_step_m)
                        n_sr = float(np.linalg.norm(d_raise))
                        if n_sr > mmax_sr and n_sr > 1e-9:
                            d_raise *= mmax_sr / n_sr
                        assert robot is not None
                        T0_sr = np.asarray(T_search, dtype=np.float64)
                        t_ee1 = T0_sr[:3, 3] + d_raise
                        eye_cam = (T0_sr @ T_ee_cam)[:3, 3]
                        logger.info(
                            "[yolo-track] iter=%d search-raise z=%.4f (z0+%.3f target) Δ=%s [%d/%d]",
                            it,
                            z_search,
                            float(cfg.search_raise_max_m),
                            np.round(d_raise, 4).tolist(),
                            search_raise_iters,
                            int(cfg.search_raise_max_iters),
                        )
                        if bool(cfg.gripper_search_level_camera):
                            aim_h = eye_cam + float(cfg.gripper_search_look_distance_m) * horiz_dir
                            R_bc = look_at_R_base_cam(eye_cam, aim_h)
                            if R_bc is not None:
                                R_ee = R_bc @ T_ee_cam[:3, :3].T
                                T_tgt = np.eye(4, dtype=np.float64)
                                T_tgt[:3, :3] = R_ee
                                T_tgt[:3, 3] = t_ee1
                                execute_pose_waypoint_base(
                                    robot,
                                    kinematics,
                                    SO100_MOTOR_NAMES,
                                    T_tgt,
                                    motion_look,
                                    label="gripper_search_raise",
                                )
                            else:
                                execute_cartesian_nudge_base(
                                    robot, kinematics, SO100_MOTOR_NAMES, d_raise, motion
                                )
                        else:
                            execute_cartesian_nudge_base(
                                robot, kinematics, SO100_MOTOR_NAMES, d_raise, motion
                            )
                        time.sleep(float(cfg.loop_sleep_s))
                        continue

                if det is None:
                    logger.info("[yolo-track] iter=%d search: no detection", it)
                    time.sleep(float(cfg.loop_sleep_s))
                    continue
                if seen_streak < int(cfg.engage_min_consecutive_detections):
                    # If we can already see the object but we're not "engaged" yet, use a bounded
                    # shoulder_pan correction to center it in the image. This avoids the search
                    # sinusoid turning too far past an object that appears in a corner.
                    if (
                        bool(getattr(cfg, "preengage_pan_center_enable", True))
                        and _camera_on_arm(cfg, mount)
                        and not cfg.dry_run
                        and det is not None
                    ):
                        assert robot is not None
                        x1c, y1c, x2c, y2c = det.xyxy
                        cxc = 0.5 * (x1c + x2c)
                        _send_shoulder_pan_toward_image_center(
                            robot,
                            cfg,
                            joints_deg,
                            cx_img=float(cxc),
                            fx=fx,
                            cx0=cx0,
                            deadband_px=float(cfg.preengage_pan_deadband_px),
                            kp=float(cfg.preengage_pan_kp),
                            max_step_deg=float(cfg.preengage_pan_max_step_deg),
                            wrist_bias_deg=float(cfg.preengage_wrist_up_deg),
                            iter_idx=it,
                            log_prefix="[yolo-track] preengage-center",
                        )
                    # Accumulate p_target_base during pre-engage whenever we have depth, so plan_top
                    # has a usable 3D target at the moment of engagement. Uses gripper_camera_tf
                    # through T_base_ee when mount=gripper or target_from_gripper_tf=True.
                    if (
                        det is not None
                        and depth is not None
                        and not cfg.dry_run
                        and (mount == "gripper" or bool(cfg.target_from_gripper_tf))
                    ):
                        d_pe = median_depth_m(
                            np.asarray(depth), det.xyxy, depth_scale=depth_scale
                        )
                        if d_pe is not None and float(d_pe) > float(cfg.max_plausible_depth_m):
                            d_pe = None
                        if d_pe is not None and float(d_pe) > 1e-4:
                            T_be_a = kinematics.forward_kinematics(joints_deg)
                            T_bc_a = T_be_a @ T_ee_cam
                            x1a, y1a, x2a, y2a = det.xyxy
                            cxa = 0.5 * (x1a + x2a)
                            cya = 0.5 * (y1a + y2a)
                            meas = point_cam_to_base(
                                T_bc_a, u=cxa, v_pix=cya, depth_m=float(d_pe),
                                fx=fx, fy=fy, cx0=cx0, cy0=cy0,
                            )
                            p_target_base = ema_p_base(
                                p_target_base, meas, float(cfg.gripper_track_3d_ema_alpha),
                            )
                            good_depth_samples += 1
                    if (
                        mount == "gripper"
                        and bool(cfg.gripper_preengage_aim)
                        and bool(cfg.gripper_point_at_target)
                        and not cfg.dry_run
                    ):
                        assert robot is not None
                        T_be_pe = kinematics.forward_kinematics(joints_deg)
                        T_bc_pe = T_be_pe @ T_ee_cam
                        eye_pe = T_bc_pe[:3, 3]
                        depth_pe = None
                        if depth is not None:
                            depth_pe = median_depth_m(
                                np.asarray(depth),
                                det.xyxy,
                                depth_scale=depth_scale,
                            )
                        if depth_pe is not None and float(depth_pe) > float(cfg.max_plausible_depth_m):
                            depth_pe = None
                        d_aim = float(cfg.gripper_aim_default_depth_m)
                        if depth_pe is not None and float(depth_pe) > 1e-4:
                            d_aim = float(depth_pe)
                        x1p, y1p, x2p, y2p = det.xyxy
                        cxp = 0.5 * (x1p + x2p)
                        cyp = 0.5 * (y1p + y2p)
                        aim_pe = point_cam_to_base(
                            T_bc_pe,
                            u=cxp,
                            v_pix=cyp,
                            depth_m=d_aim,
                            fx=fx,
                            fy=fy,
                            cx0=cx0,
                            cy0=cy0,
                        )
                        depth_ok_pe = depth_pe is not None and float(depth_pe) > 1e-4
                        if bool(cfg.gripper_track_3d_enable) and (
                            (not bool(cfg.gripper_3d_ema_require_depth)) or depth_ok_pe
                        ):
                            p_target_base = ema_p_base(
                                p_target_base,
                                aim_pe,
                                float(cfg.gripper_track_3d_ema_alpha),
                            )
                        aim_cmd_pe = (
                            p_target_base
                            if bool(cfg.gripper_aim_use_filtered_3d) and p_target_base is not None
                            else aim_pe
                        )
                        pe_cap = float(cfg.gripper_preengage_max_look_rot_step_deg or 0.0)
                        if pe_cap > 1e-6:
                            preengage_look_deg = min(
                                float(cfg.gripper_max_look_rot_step_deg), pe_cap
                            )
                        else:
                            preengage_look_deg = float(cfg.gripper_max_look_rot_step_deg)
                        execute_gripper_nudge(
                            robot=robot,
                            kinematics=kinematics,
                            motor_names=SO100_MOTOR_NAMES,
                            motion_default=motion,
                            motion_look=motion_look,
                            use_look=True,
                            T_base_ee=T_be_pe,
                            T_ee_cam=T_ee_cam,
                            delta_base=np.zeros(3, dtype=np.float64),
                            aim_point_base=aim_cmd_pe,
                            eye_cam_base=eye_pe,
                            max_look_rot_step_deg=preengage_look_deg,
                            label="gripper_preengage_aim",
                        )
                    logger.info(
                        "[yolo-track] iter=%d search: detection conf=%.2f (%d/%d)",
                        it,
                        float(det.confidence),
                        seen_streak,
                        int(cfg.engage_min_consecutive_detections),
                    )
                    time.sleep(float(cfg.loop_sleep_s))
                    continue
                engaged = True
                smooth_cx = smooth_cy = smooth_depth_m = None
                prev_smoothed_delta.fill(0.0)
                plan_top_frame_smooth_cx = None
                plan_top_frame_smooth_cy = None
                z_start = None
                lift_iter_total = 0
                logger.info("[yolo-track] engaged: starting approach (style=%s)", style)

            # plan_top after commit: we already have a frozen 3D target in the base frame. Detection
            # loss is *expected* (camera tilts away as the wrist rotates down), so don't gate on it
            # or bail out — fall through to the plan_top state machine against the frozen target.
            plan_top_blind_ok = (
                style == "plan_top" and engaged and plan_top_phase != "gather"
            )

            if det is None and not plan_top_blind_ok:
                if (
                    engaged
                    and mount == "gripper"
                    and bool(cfg.gripper_track_3d_enable)
                    and p_target_base is not None
                    and hold_age < int(cfg.gripper_hold_target_frames)
                    and not cfg.dry_run
                ):
                    hold_age += 1
                    lost = 0
                    assert robot is not None
                    T_be_h = kinematics.forward_kinematics(joints_deg)
                    T_bc_h = T_be_h @ T_ee_cam
                    execute_gripper_nudge(
                        robot=robot,
                        kinematics=kinematics,
                        motor_names=SO100_MOTOR_NAMES,
                        motion_default=motion,
                        motion_look=motion_look,
                        use_look=bool(cfg.gripper_point_at_target),
                        T_base_ee=T_be_h,
                        T_ee_cam=T_ee_cam,
                        delta_base=np.zeros(3, dtype=np.float64),
                        aim_point_base=p_target_base,
                        eye_cam_base=T_bc_h[:3, 3],
                        max_look_rot_step_deg=float(cfg.gripper_max_look_rot_step_deg),
                        label="gripper_hold_3d",
                    )
                    logger.info(
                        "[yolo-track] iter=%d hold 3d aim age=%d/%d p_base=%s",
                        it,
                        hold_age,
                        int(cfg.gripper_hold_target_frames),
                        np.round(p_target_base, 3).tolist(),
                    )
                    time.sleep(float(cfg.loop_sleep_s))
                    continue

                lost += 1
                logger.info("[yolo-track] iter=%d no detection (%d/%d)", it, lost, cfg.lost_patience)
                if lost >= cfg.lost_patience:
                    logger.warning("[yolo-track] stopping: lost target too long")
                    break
                time.sleep(float(cfg.loop_sleep_s))
                continue

            if det is None and plan_top_blind_ok:
                lost = 0
                hold_age = 0
                # Fall through: plan_top state machine below runs against frozen target.

            reacquire = lost > 0
            lost = 0
            if reacquire:
                smooth_cx = smooth_cy = smooth_depth_m = None
                prev_smoothed_delta.fill(0.0)
                plan_top_frame_smooth_cx = None
                plan_top_frame_smooth_cy = None
                if phased and uma_phase == "descend":
                    uma_phase = "align"
                    logger.info("[yolo-track] umbrella → align (re-acquire)")

            skip_visual = False

            T_base_ee = kinematics.forward_kinematics(joints_deg)
            z_ee = float(T_base_ee[2, 3])
            if z_start is None:
                z_start = z_ee

            # Translation-free look-at each tick (optional): keep camera on ``p_target_base`` during
            # plan_top without coupling orientation into Cartesian nudges (see config notes).
            period_m = max(1, int(cfg.gripper_maintain_aim_period_iters))
            plan_top_only = bool(cfg.gripper_maintain_aim_plan_top_only)
            skip_maintain_plan_top_smooth = (
                style == "plan_top"
                and bool(cfg.gripper_maintain_aim_skip_plan_top_lift_align_tilt)
                and plan_top_phase in ("gather", "lift", "align", "tilt", "descend", "center")
            )
            if (
                bool(cfg.gripper_maintain_aim_at_target)
                and engaged
                and not cfg.dry_run
                and p_target_base is not None
                and bool(cfg.gripper_point_at_target)
                and _camera_on_arm(cfg, mount)
                and (not plan_top_only or style == "plan_top")
                and (it % period_m == 0)
                and not skip_maintain_plan_top_smooth
            ):
                assert robot is not None
                T_bc_m = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(
                    T_ee_cam, dtype=np.float64
                )
                execute_gripper_nudge(
                    robot=robot,
                    kinematics=kinematics,
                    motor_names=SO100_MOTOR_NAMES,
                    motion_default=motion,
                    motion_look=motion_look,
                    use_look=True,
                    T_base_ee=np.asarray(T_base_ee, dtype=np.float64),
                    T_ee_cam=np.asarray(T_ee_cam, dtype=np.float64),
                    delta_base=np.zeros(3, dtype=np.float64),
                    aim_point_base=np.asarray(p_target_base, dtype=np.float64).reshape(3),
                    eye_cam_base=np.asarray(T_bc_m[:3, 3], dtype=np.float64),
                    max_look_rot_step_deg=float(cfg.gripper_max_look_rot_step_deg),
                    label="gripper_maintain_aim",
                )
                obs_m = robot.get_observation()
                joints_deg = np.array(
                    [float(obs_m[f"{m}.pos"]) for m in SO100_MOTOR_NAMES], dtype=np.float64
                )
                T_base_ee = kinematics.forward_kinematics(joints_deg)
                z_ee = float(T_base_ee[2, 3])

            # ----- plan_top: accumulate 3D target, commit a single top-pose waypoint, then descend
            if style == "plan_top":
                reject_reason: str | None = None
                d_pt: float | None = None
                d_src: str = ""
                # Keep updating p_target_base while we have a detection + depth (EMA-filtered).
                use_tf_path = mount == "gripper" or bool(cfg.target_from_gripper_tf)

                # Stereo depth (may be None if inside OAK-D near-field blind zone).
                d_stereo: float | None = None
                stereo_reason: str | None = None
                if cfg.dry_run:
                    stereo_reason = "dry_run"
                elif det is None:
                    stereo_reason = "no_detection"
                elif not use_tf_path:
                    stereo_reason = "target_from_gripper_tf=false"
                elif depth is None:
                    stereo_reason = "no depth frame"
                else:
                    d_raw = median_depth_m(
                        np.asarray(depth), det.xyxy, depth_scale=depth_scale
                    )
                    if d_raw is None:
                        stereo_reason = "too few valid stereo px (near-field blind zone?)"
                    elif float(d_raw) > float(cfg.max_plausible_depth_m):
                        stereo_reason = (
                            f"stereo {float(d_raw):.3f}m > max_plausible_depth_m="
                            f"{float(cfg.max_plausible_depth_m):.3f}m"
                        )
                    elif float(d_raw) <= 1e-4:
                        stereo_reason = f"stereo {float(d_raw):.4f}m too small"
                    else:
                        d_stereo = float(d_raw)

                # Bounding-box size depth (known-size object, works in near-field).
                d_bbox: float | None = None
                bbox_reason: str | None = None
                if not cfg.depth_from_bbox_enabled:
                    bbox_reason = "depth_from_bbox_enabled=false"
                elif cfg.dry_run:
                    bbox_reason = "dry_run"
                elif det is None:
                    bbox_reason = "no_detection"
                elif not use_tf_path:
                    bbox_reason = "target_from_gripper_tf=false"
                else:
                    d_bbox_raw = depth_from_bbox_size(
                        tuple(det.xyxy),
                        fx=fx, fy=fy,
                        target_physical_size_m=float(cfg.target_physical_size_m),
                    )
                    if d_bbox_raw is None:
                        bbox_reason = "bbox too small for size-depth"
                    elif d_bbox_raw <= 1e-4:
                        bbox_reason = f"bbox-depth {d_bbox_raw:.4f}m too small"
                    else:
                        d_bbox = float(d_bbox_raw)

                # Combine per policy.
                policy = str(cfg.depth_source_policy).lower()
                if policy == "bbox_only":
                    if d_bbox is not None:
                        d_pt, d_src = d_bbox, "bbox"
                    else:
                        reject_reason = f"bbox_only: {bbox_reason}"
                elif policy == "bbox_preferred":
                    if d_bbox is not None:
                        d_pt, d_src = d_bbox, "bbox"
                    elif d_stereo is not None:
                        d_pt, d_src = d_stereo, "stereo"
                    else:
                        reject_reason = f"bbox={bbox_reason}; stereo={stereo_reason}"
                else:  # "stereo_preferred" (default behavior)
                    if d_stereo is not None:
                        d_pt, d_src = d_stereo, "stereo"
                    elif d_bbox is not None:
                        d_pt, d_src = d_bbox, "bbox"
                    else:
                        reject_reason = f"stereo={stereo_reason}; bbox={bbox_reason}"

                # Only refine the 3D target during the gather phase. Once we've committed a top
                # pose and started the phased motion (lift/align/tilt/descend), the target must
                # stay FROZEN — otherwise every new detection reprojected through a pitched-down
                # camera would drift the marker below the ground as the arm lifts, and we'd be
                # chasing a moving point instead of executing a deterministic plan.
                if d_pt is not None and plan_top_phase == "gather":
                    T_bc_pt = T_base_ee @ T_ee_cam
                    x1t, y1t, x2t, y2t = det.xyxy
                    meas = point_cam_to_base(
                        T_bc_pt, u=0.5 * (x1t + x2t), v_pix=0.5 * (y1t + y2t),
                        depth_m=float(d_pt), fx=fx, fy=fy, cx0=cx0, cy0=cy0,
                    )
                    # Bbox-depth + a pitched-down camera can project the ray well below the
                    # actual table surface. Clamp to the user-provided table height so the
                    # commit/descend math doesn't drive the fingertip into the table.
                    if cfg.table_z_m is not None and float(meas[2]) < float(cfg.table_z_m):
                        logger.info(
                            "[yolo-track] plan_top: clamped cube_z %.3f → %.3f (table_z_m floor)",
                            float(meas[2]), float(cfg.table_z_m),
                        )
                        meas = np.array([meas[0], meas[1], float(cfg.table_z_m)], dtype=np.float64)
                    p_target_base = ema_p_base(
                        p_target_base, meas, float(cfg.gripper_track_3d_ema_alpha),
                    )
                    good_depth_samples += 1

                # Keep the gripper camera from letting the bbox slide out of view while the
                # frozen-target plan_top phases run. This is intentionally pan-only and smoothed:
                # it preserves the committed 3D target while damping YOLO jitter.
                if (
                    bool(cfg.plan_top_keep_in_frame_enable)
                    and plan_top_phase in ("lift", "align", "tilt", "descend")
                    and det is not None
                    and float(det.confidence) >= float(cfg.plan_top_keep_in_frame_min_conf)
                    and _camera_on_arm(cfg, mount)
                    and not cfg.dry_run
                    and (it % max(1, int(cfg.plan_top_keep_in_frame_period_iters))) == 0
                ):
                    x1k, y1k, x2k, y2k = det.xyxy
                    cx_raw_k = 0.5 * (x1k + x2k)
                    cy_raw_k = 0.5 * (y1k + y2k)
                    ac_k = float(np.clip(cfg.smooth_center_alpha, 0.0, 1.0))
                    if plan_top_frame_smooth_cx is None or ac_k >= 1.0 - 1e-9:
                        plan_top_frame_smooth_cx = float(cx_raw_k)
                        plan_top_frame_smooth_cy = float(cy_raw_k)
                    else:
                        plan_top_frame_smooth_cx = (
                            ac_k * float(cx_raw_k)
                            + (1.0 - ac_k) * float(plan_top_frame_smooth_cx)
                        )
                        plan_top_frame_smooth_cy = (
                            ac_k * float(cy_raw_k)
                            + (1.0 - ac_k) * float(plan_top_frame_smooth_cy)
                        )
                    assert robot is not None
                    if _send_shoulder_pan_toward_image_center(
                        robot,
                        cfg,
                        joints_deg,
                        cx_img=float(plan_top_frame_smooth_cx),
                        fx=fx,
                        cx0=cx0,
                        deadband_px=float(cfg.plan_top_keep_in_frame_deadband_px),
                        kp=float(cfg.plan_top_keep_in_frame_kp),
                        max_step_deg=float(cfg.plan_top_keep_in_frame_max_step_deg),
                        wrist_bias_deg=0.0,
                        iter_idx=it,
                        log_prefix=f"[yolo-track] plan_top: {plan_top_phase}-frame-pan",
                    ):
                        time.sleep(float(cfg.loop_sleep_s))
                        continue

                if not top_pose_committed:
                    # ----- Phase transitions: gather → lift → align → tilt → descend -----
                    if plan_top_phase == "gather":
                        enough = good_depth_samples >= int(cfg.top_min_good_depth_samples)
                        have_target = p_target_base is not None
                        if enough and have_target and not cfg.dry_run:
                            p_xyz = np.asarray(p_target_base).reshape(3)
                            reach = float(np.linalg.norm(p_xyz))
                            if reach > float(cfg.top_max_reach_m):
                                logger.warning(
                                    "[yolo-track] plan_top: target xyz=%s reach=%.3fm > top_max_reach_m=%.3fm — "
                                    "object is OUTSIDE the arm's reach envelope. Either move the cube %.2fm "
                                    "closer to the base, or (if depth is correct) raise --top-max-reach-m. "
                                    "Resetting samples.",
                                    np.round(p_xyz, 3).tolist(),
                                    reach, float(cfg.top_max_reach_m),
                                    reach - float(cfg.top_max_reach_m),
                                )
                                good_depth_samples = 0
                                p_target_base = None
                                time.sleep(float(cfg.loop_sleep_s))
                                continue
                            # Commit the hover point and start phased motion.
                            plan_top_target_xyz = p_xyz + np.array(
                                [0.0, 0.0, float(cfg.top_approach_height_m)], dtype=np.float64
                            )
                            plan_top_lift_z_goal = float(z_ee) + float(cfg.plan_top_lift_height_m)
                            plan_top_phase = "lift"
                            plan_top_wrist_flex_bias_applied = 0.0
                            logger.info(
                                "[yolo-track] plan_top: target=%s reach=%.3fm → phased approach "
                                "(lift z:%.3f→%.3f, hover=%s, tilt=wrist-down)",
                                np.round(p_xyz, 3).tolist(), reach,
                                float(z_ee), plan_top_lift_z_goal,
                                np.round(plan_top_target_xyz, 3).tolist(),
                            )
                            # Fall through to execute the first lift step this iteration.
                        else:
                            if (
                                bool(cfg.plan_top_gather_pan_center_enable)
                                and det is not None
                                and _camera_on_arm(cfg, mount)
                                and not cfg.dry_run
                            ):
                                assert robot is not None
                                x1g, y1g, x2g, y2g = det.xyxy
                                cxg = 0.5 * (x1g + x2g)
                                if _send_shoulder_pan_toward_image_center(
                                    robot,
                                    cfg,
                                    joints_deg,
                                    cx_img=float(cxg),
                                    fx=fx,
                                    cx0=cx0,
                                    deadband_px=float(cfg.preengage_pan_deadband_px),
                                    kp=float(cfg.preengage_pan_kp),
                                    max_step_deg=float(cfg.preengage_pan_max_step_deg),
                                    wrist_bias_deg=0.0,
                                    iter_idx=it,
                                    log_prefix="[yolo-track] plan_top: gather-pan",
                                ):
                                    time.sleep(float(cfg.loop_sleep_s))
                                    continue
                            if d_pt is not None:
                                logger.info(
                                    "[yolo-track] plan_top: gathering depth samples %d/%d (conf=%.2f, d=%.3fm src=%s)",
                                    good_depth_samples,
                                    int(cfg.top_min_good_depth_samples),
                                    float(det.confidence) if det is not None else 0.0,
                                    d_pt,
                                    d_src,
                                )
                            else:
                                logger.warning(
                                    "[yolo-track] plan_top: depth REJECTED (%s) samples=%d/%d conf=%.2f — "
                                    "bbox=%s. For near-field (<0.35m) targets enable --depth-from-bbox-enabled=true "
                                    "and set --target-physical-size-m to the object's largest side.",
                                    reject_reason or "unknown",
                                    good_depth_samples,
                                    int(cfg.top_min_good_depth_samples),
                                    float(det.confidence) if det is not None else 0.0,
                                    tuple(round(v, 1) for v in det.xyxy) if det is not None else None,
                                )
                            time.sleep(float(cfg.loop_sleep_s))
                            continue

                    # Phase 1: LIFT — translate EE straight up, orientation unchanged.
                    if plan_top_phase == "lift":
                        assert plan_top_lift_z_goal is not None
                        dz_remain = float(plan_top_lift_z_goal) - float(z_ee)
                        # Track base-frame z progress so we can abort when the IK oscillates
                        # around the goal (commanded ±step_m nudges → actual ΔZ within tol).
                        plan_top_lift_hist.append(float(z_ee))
                        if len(plan_top_lift_hist) > int(cfg.plan_top_stuck_window):
                            plan_top_lift_hist = plan_top_lift_hist[-int(cfg.plan_top_stuck_window):]
                        stuck = (
                            len(plan_top_lift_hist) >= int(cfg.plan_top_stuck_window)
                            and (max(plan_top_lift_hist) - min(plan_top_lift_hist))
                                < float(cfg.plan_top_stuck_progress_m)
                        )
                        iter_cap = plan_top_lift_iters >= int(cfg.plan_top_lift_max_iters)
                        tol = float(cfg.plan_top_lift_tolerance_m)
                        if dz_remain <= tol or stuck or iter_cap:
                            reason = (
                                "reached" if dz_remain <= tol
                                else ("stuck" if stuck else "iter-cap")
                            )
                            plan_top_phase = "align"
                            plan_top_lift_hist = []
                            logger.info(
                                "[yolo-track] plan_top: lift done z_ee=%.3f (remain=%.3f, %s) → align phase",
                                z_ee, dz_remain, reason,
                            )
                        else:
                            step = float(min(dz_remain, float(cfg.plan_top_lift_step_m)))
                            if not cfg.dry_run:
                                assert robot is not None
                                _plan_top_trans_nudge(
                                    robot=robot,
                                    kinematics=kinematics,
                                    cfg=cfg,
                                    mount=mount,
                                    motion=motion,
                                    motion_look=motion_look,
                                    T_base_ee=T_base_ee,
                                    T_ee_cam=T_ee_cam,
                                    p_target_base=p_target_base,
                                    delta_base=np.array([0.0, 0.0, step], dtype=np.float64),
                                    use_look_flag=bool(cfg.plan_top_lift_look_at_target),
                                    label="plan_top_lift",
                                )
                                plan_top_lift_iters += 1
                                plan_top_wrist_flex_bias_applied = _plan_top_apply_wrist_flex_approach_bias(
                                    robot, cfg, joints_deg, plan_top_wrist_flex_bias_applied
                                )
                            logger.info(
                                "[yolo-track] plan_top: lift step Δz=%.3f (remain=%.3f, iter=%d/%d)",
                                step, dz_remain - step,
                                plan_top_lift_iters, int(cfg.plan_top_lift_max_iters),
                            )
                            time.sleep(float(cfg.loop_sleep_s))
                            continue

                    # Phase 2: ALIGN — translate EE toward hover point in XY while actively
                    # holding Z at plan_top_target_xyz[2]. Without the Z term, the IK collapses
                    # the wrist downward as the arm reaches forward (near-singular shoulder/elbow
                    # config on SO-101), which then forces a long "recover lift" in descend that
                    # visually looks like the arm retreating. Commanding dz keeps the nudge
                    # diagonal and preserves the hover altitude.
                    if plan_top_phase == "align":
                        assert plan_top_target_xyz is not None
                        ee_xyz = np.asarray(T_base_ee[:3, 3], dtype=np.float64).reshape(3)
                        dz_hold = float(plan_top_target_xyz[2]) - float(ee_xyz[2])
                        dxy = np.array(
                            [plan_top_target_xyz[0] - ee_xyz[0],
                             plan_top_target_xyz[1] - ee_xyz[1],
                             dz_hold], dtype=np.float64,
                        )
                        d_xy_norm = float(np.linalg.norm(dxy[:2]))
                        plan_top_align_hist.append(d_xy_norm)
                        if len(plan_top_align_hist) > int(cfg.plan_top_stuck_window):
                            plan_top_align_hist = plan_top_align_hist[-int(cfg.plan_top_stuck_window):]
                        stuck = (
                            len(plan_top_align_hist) >= int(cfg.plan_top_stuck_window)
                            and (max(plan_top_align_hist) - min(plan_top_align_hist))
                                < float(cfg.plan_top_stuck_progress_m)
                        )
                        iter_cap = plan_top_align_iters >= int(cfg.plan_top_align_max_iters)
                        if d_xy_norm <= float(cfg.plan_top_align_tolerance_m) or stuck or iter_cap:
                            reason = (
                                "reached" if d_xy_norm <= float(cfg.plan_top_align_tolerance_m)
                                else ("stuck" if stuck else "iter-cap")
                            )
                            plan_top_phase = "tilt"
                            plan_top_align_hist = []
                            logger.info(
                                "[yolo-track] plan_top: align done ee_xy=%s (err=%.3fm, %s) → tilt phase",
                                np.round(ee_xyz[:2], 3).tolist(), d_xy_norm, reason,
                            )
                        else:
                            max_step = float(cfg.plan_top_xy_step_m)
                            if d_xy_norm <= float(cfg.plan_top_align_near_xy_m):
                                max_step = min(max_step, float(cfg.plan_top_align_near_xy_step_m))
                            if d_xy_norm > max_step:
                                dxy[:2] = dxy[:2] * (max_step / d_xy_norm)
                            # Clamp Z component to the lift step so we don't jump vertically in
                            # a single iteration while IK is already stressed by the XY move.
                            max_dz = float(cfg.plan_top_lift_step_m)
                            if abs(dxy[2]) > max_dz:
                                dxy[2] = math.copysign(max_dz, dxy[2])
                            if not cfg.dry_run:
                                assert robot is not None
                                _plan_top_trans_nudge(
                                    robot=robot,
                                    kinematics=kinematics,
                                    cfg=cfg,
                                    mount=mount,
                                    motion=motion,
                                    motion_look=motion_look,
                                    T_base_ee=T_base_ee,
                                    T_ee_cam=T_ee_cam,
                                    p_target_base=p_target_base,
                                    delta_base=dxy,
                                    use_look_flag=bool(cfg.plan_top_align_look_at_target),
                                    label="plan_top_align",
                                )
                                plan_top_align_iters += 1
                                plan_top_wrist_flex_bias_applied = _plan_top_apply_wrist_flex_approach_bias(
                                    robot, cfg, joints_deg, plan_top_wrist_flex_bias_applied
                                )
                            logger.info(
                                "[yolo-track] plan_top: align step Δxyz=%s (remain_xy=%.3fm dz=%.3f, iter=%d/%d)",
                                np.round(dxy, 3).tolist(), d_xy_norm, float(dxy[2]),
                                plan_top_align_iters, int(cfg.plan_top_align_max_iters),
                            )
                            time.sleep(float(cfg.loop_sleep_s))
                            continue

                    # Phase 3: TILT — rotate wrist toward straight-down in small increments,
                    # holding current XY/Z.
                    if plan_top_phase == "tilt":
                        R_cur = np.asarray(T_base_ee[:3, :3], dtype=np.float64)
                        if plan_top_tilt_R_tgt is None:
                            # Compute the partial target ONCE on entry: slerp a fraction of the
                            # way from R_cur toward strict wrist-down. Full wrist-down often
                            # makes IK collapse the arm; the user can tune the fraction.
                            R_full_down = wrist_down_R_base_ee(R_cur)
                            frac = float(np.clip(cfg.plan_top_tilt_fraction, 0.0, 1.0))
                            if frac >= 0.999:
                                plan_top_tilt_R_tgt = R_full_down
                                full_deg = 0.0  # unused
                            else:
                                # Axis-angle from R_cur → R_full_down, scale the angle.
                                R_rel = R_full_down @ R_cur.T
                                cos_a = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
                                full_angle = float(np.arccos(cos_a))
                                full_deg = float(np.degrees(full_angle))
                                sin_a = float(np.sin(full_angle))
                                if full_angle < 1e-6 or abs(sin_a) < 1e-8:
                                    plan_top_tilt_R_tgt = R_cur.copy()
                                else:
                                    axis = np.array(
                                        [
                                            R_rel[2, 1] - R_rel[1, 2],
                                            R_rel[0, 2] - R_rel[2, 0],
                                            R_rel[1, 0] - R_rel[0, 1],
                                        ],
                                        dtype=np.float64,
                                    ) / (2.0 * sin_a)
                                    axis = axis / float(np.linalg.norm(axis))
                                    partial = float(full_angle * frac)
                                    c, s = float(np.cos(partial)), float(np.sin(partial))
                                    K = np.array(
                                        [
                                            [0, -axis[2], axis[1]],
                                            [axis[2], 0, -axis[0]],
                                            [-axis[1], axis[0], 0],
                                        ],
                                        dtype=np.float64,
                                    )
                                    R_partial = np.eye(3) + s * K + (1.0 - c) * (K @ K)
                                    plan_top_tilt_R_tgt = R_partial @ R_cur
                            logger.info(
                                "[yolo-track] plan_top: tilt target = %.0f%% toward wrist-down "
                                "(full=%.1f° → partial=%.1f°)",
                                100.0 * frac, full_deg, full_deg * frac,
                            )
                        R_tgt = plan_top_tilt_R_tgt
                        R_next, remain_deg = rot_step_toward(
                            R_cur, R_tgt, float(cfg.plan_top_tilt_step_deg),
                        )
                        plan_top_tilt_hist.append(float(remain_deg))
                        if len(plan_top_tilt_hist) > int(cfg.plan_top_stuck_window):
                            plan_top_tilt_hist = plan_top_tilt_hist[-int(cfg.plan_top_stuck_window):]
                        stuck = (
                            len(plan_top_tilt_hist) >= int(cfg.plan_top_stuck_window)
                            and (max(plan_top_tilt_hist) - min(plan_top_tilt_hist))
                                < float(np.degrees(cfg.plan_top_stuck_progress_rad))
                        )
                        iter_cap = plan_top_tilt_iters >= int(cfg.plan_top_tilt_max_iters)
                        if remain_deg <= float(cfg.plan_top_tilt_tolerance_deg) or stuck or iter_cap:
                            reason = (
                                "reached" if remain_deg <= float(cfg.plan_top_tilt_tolerance_deg)
                                else ("stuck" if stuck else "iter-cap")
                            )
                            plan_top_phase = "descend"
                            top_pose_committed = True
                            top_descend_iters = 0
                            plan_top_descend_recover_iters = 0
                            plan_top_descend_recover_warned = False
                            plan_top_descend_shoulder_lift_applied = 0.0
                            plan_top_tilt_hist = []
                            plan_top_tilt_R_tgt = None
                            logger.info(
                                "[yolo-track] plan_top: tilt done (remain=%.1f deg, %s) → descend phase",
                                remain_deg, reason,
                            )
                        else:
                            T_tilt = np.eye(4, dtype=np.float64)
                            T_tilt[:3, :3] = R_next
                            T_tilt[:3, 3] = np.asarray(T_base_ee[:3, 3], dtype=np.float64)
                            if not cfg.dry_run:
                                assert robot is not None
                                execute_pose_waypoint_base(
                                    robot, kinematics, SO100_MOTOR_NAMES,
                                    T_tilt, motion, label="plan_top_tilt",
                                )
                                plan_top_tilt_iters += 1
                            logger.info(
                                "[yolo-track] plan_top: tilt step (remain=%.1f deg, iter=%d/%d)",
                                remain_deg,
                                plan_top_tilt_iters, int(cfg.plan_top_tilt_max_iters),
                            )
                            time.sleep(float(cfg.loop_sleep_s))
                            continue

                # Phase 4: DESCEND — lower EE straight down until the fingertip hovers
                # ``plan_top_final_hover_m`` above the object's **top** (target Z + physical half
                # size). Does NOT plunge to the ground.
                if plan_top_phase == "descend":
                    assert plan_top_target_xyz is not None and p_target_base is not None
                    cube_z = float(np.asarray(p_target_base).reshape(3)[2])
                    obj_half_z = max(0.0, float(cfg.target_physical_size_m) * 0.5)
                    tip_goal_z = cube_z + obj_half_z + float(cfg.plan_top_final_hover_m)
                    # EE origin (wrist joint) must sit tip_offset ABOVE the desired tip hover
                    # height, otherwise the fingers will plunge through the cube into the table.
                    stop_z = tip_goal_z + float(cfg.plan_top_gripper_tip_offset_m)
                    # Absolute fingertip floor: tip = z_ee - tip_offset must stay above the
                    # configured table height (+ clearance) and the global safety floor. Lift
                    # stop_z accordingly so we never command the tip into the table even if the
                    # cube depth estimate is wrong.
                    tip_floor_z = float(cfg.top_descend_min_z_m)
                    if cfg.table_z_m is not None:
                        tip_floor_z = max(
                            tip_floor_z,
                            float(cfg.table_z_m) + float(cfg.table_clearance_m),
                        )
                    floor_stop_z = tip_floor_z + float(cfg.plan_top_gripper_tip_offset_m)
                    if stop_z < floor_stop_z:
                        logger.info(
                            "[yolo-track] plan_top: descend stop_z %.3f raised to floor %.3f "
                            "(tip_floor_z=%.3f, tip_offset=%.3f) — cube_z estimate %.3f below table.",
                            stop_z, floor_stop_z, tip_floor_z,
                            float(cfg.plan_top_gripper_tip_offset_m), cube_z,
                        )
                        stop_z = floor_stop_z
                    # If partial tilt + IK dropped the EE below the planned hover, recover with
                    # small upward Z nudges (same step as lift) before descending. Without this,
                    # dz_remain stays negative, descend is skipped, and the tip can sit inside the
                    # object while the center phase only nudges XY.
                    recover_tol = 0.005
                    if float(z_ee) < stop_z - recover_tol:
                        rec_cap = int(cfg.plan_top_lift_max_iters)
                        gap_up = float(stop_z) - float(z_ee)
                        step_up = float(min(gap_up, float(cfg.plan_top_lift_step_m)))
                        if plan_top_descend_recover_iters >= rec_cap:
                            if not plan_top_descend_recover_warned:
                                tip_w = float(z_ee) - float(
                                    cfg.plan_top_gripper_tip_offset_m
                                )
                                logger.warning(
                                    "[yolo-track] plan_top: descend recover hit iter cap "
                                    "(mode=%s) — z_ee=%.3f still below stop=%.3f (tip=%.3f). "
                                    "Lower --plan-top-tilt-fraction or raise lift before tilt.",
                                    cfg.plan_top_vertical_recovery_mode,
                                    z_ee, stop_z, tip_w,
                                )
                                plan_top_descend_recover_warned = True
                        elif _plan_top_vertical_recovery_use_shoulder(cfg):
                            if not cfg.dry_run:
                                assert robot is not None
                                plan_top_descend_shoulder_lift_applied = _plan_top_apply_shoulder_lift_recovery(
                                    robot,
                                    cfg,
                                    joints_deg,
                                    plan_top_descend_shoulder_lift_applied,
                                )
                                if bool(cfg.plan_top_approach_wrist_flex_in_descend_recover):
                                    plan_top_wrist_flex_bias_applied = _plan_top_apply_wrist_flex_approach_bias(
                                        robot, cfg, joints_deg, plan_top_wrist_flex_bias_applied
                                    )
                            plan_top_descend_recover_iters += 1
                            logger.info(
                                "[yolo-track] plan_top: descend recover shoulder "
                                "(z_ee=%.3f → stop=%.3f, shoulder_applied=%.1f°, iter=%d/%d)",
                                z_ee,
                                stop_z,
                                plan_top_descend_shoulder_lift_applied,
                                plan_top_descend_recover_iters,
                                rec_cap,
                            )
                            time.sleep(float(cfg.loop_sleep_s))
                            continue
                        elif step_up > 1e-4:
                            if not cfg.dry_run:
                                assert robot is not None
                                _plan_top_trans_nudge(
                                    robot=robot,
                                    kinematics=kinematics,
                                    cfg=cfg,
                                    mount=mount,
                                    motion=motion,
                                    motion_look=motion_look,
                                    T_base_ee=T_base_ee,
                                    T_ee_cam=T_ee_cam,
                                    p_target_base=p_target_base,
                                    delta_base=np.array([0.0, 0.0, step_up], dtype=np.float64),
                                    use_look_flag=bool(cfg.plan_top_descend_recover_look_at_target),
                                    label="plan_top_descend_recover",
                                )
                                if bool(cfg.plan_top_approach_wrist_flex_in_descend_recover):
                                    plan_top_wrist_flex_bias_applied = _plan_top_apply_wrist_flex_approach_bias(
                                        robot, cfg, joints_deg, plan_top_wrist_flex_bias_applied
                                    )
                            plan_top_descend_recover_iters += 1
                            logger.info(
                                "[yolo-track] plan_top: descend recover lift Δz=%.4f "
                                "(z_ee=%.3f → stop=%.3f, iter=%d/%d)",
                                step_up, z_ee, stop_z,
                                plan_top_descend_recover_iters, rec_cap,
                            )
                            time.sleep(float(cfg.loop_sleep_s))
                            continue

                    # Descend is ONLY downward toward stop_z.
                    dz_remain = float(z_ee) - stop_z
                    hit_cap = top_descend_iters >= int(cfg.top_descend_max_iters)
                    if dz_remain <= 0.003 or hit_cap:
                        tip_z = float(z_ee) - float(cfg.plan_top_gripper_tip_offset_m)
                        if dz_remain < -0.01:
                            logger.warning(
                                "[yolo-track] plan_top: skipping descend — z_ee=%.3f "
                                "below hover target %.3f (dz=%.3f). Tip at %.3f. Tune "
                                "--plan-top-tilt-fraction lower so tilt preserves Z better.",
                                z_ee, stop_z, dz_remain, tip_z,
                            )
                        logger.info(
                            "[yolo-track] plan_top: descend done z_ee=%.3f tip_z=%.3f "
                            "(tip_goal=%.3f = cube_z=%.3f + obj_half=%.3f + hover=%.3f, "
                            "tip_offset=%.3fm) descend_iters=%d recover_iters=%d",
                            z_ee, tip_z, tip_goal_z, cube_z, obj_half_z,
                            float(cfg.plan_top_final_hover_m),
                            float(cfg.plan_top_gripper_tip_offset_m),
                            top_descend_iters, plan_top_descend_recover_iters,
                        )
                        if on_plan_top_descend_done is not None:
                            logger.info(
                                "[yolo-track] plan_top: descend done — handing off to callback "
                                "(on_plan_top_descend_done)."
                            )
                            try:
                                on_plan_top_descend_done(
                                    PlanTopHandoffState(
                                        cfg=cfg,
                                        robot=robot,
                                        kinematics=kinematics,
                                        motion=motion,
                                        motion_look=motion_look,
                                        T_ee_cam=T_ee_cam,
                                        T_base_ee=np.asarray(T_base_ee, dtype=np.float64).copy(),
                                        p_target_base=np.asarray(p_target_base, dtype=np.float64).reshape(3).copy(),
                                        det=det,
                                        rgb=np.asarray(rgb).copy() if rgb is not None else None,
                                        depth=np.asarray(depth).copy() if depth is not None else None,
                                        intrinsics=dict(intrinsics),
                                        fx=float(fx),
                                        fy=float(fy),
                                        cx0=float(cx0),
                                        cy0=float(cy0),
                                        cam_to_robot=np.asarray(cam_to_robot, dtype=np.float64).copy(),
                                        mount=str(mount),
                                    )
                                )
                            except Exception as hcb_exc:
                                logger.exception(
                                    "[yolo-track] plan_top handoff callback raised: %s", hcb_exc
                                )
                            break
                        if bool(cfg.plan_top_center_enable):
                            plan_top_phase = "center"
                            top_descend_iters = 0
                            plan_top_center_no_det_lift_accum = 0.0
                            plan_top_center_shoulder_lift_applied = 0.0
                            plan_top_frame_smooth_cx = None
                            plan_top_frame_smooth_cy = None
                        else:
                            break
                    else:
                        step = float(min(dz_remain, float(cfg.plan_top_descend_step_m)))
                        if not cfg.dry_run:
                            assert robot is not None
                            _plan_top_trans_nudge(
                                robot=robot,
                                kinematics=kinematics,
                                cfg=cfg,
                                mount=mount,
                                motion=motion,
                                motion_look=motion_look,
                                T_base_ee=T_base_ee,
                                T_ee_cam=T_ee_cam,
                                p_target_base=p_target_base,
                                delta_base=np.array([0.0, 0.0, -step], dtype=np.float64),
                                use_look_flag=bool(cfg.plan_top_descend_look_at_target),
                                label="plan_top_descend",
                            )
                            top_descend_iters += 1
                            plan_top_wrist_flex_bias_applied = _plan_top_apply_wrist_flex_approach_bias(
                                robot, cfg, joints_deg, plan_top_wrist_flex_bias_applied
                            )
                        logger.info(
                            "[yolo-track] plan_top: descend step Δz=-%.3f (z_ee=%.3f → stop=%.3f)",
                            step, z_ee, stop_z,
                        )
                        time.sleep(float(cfg.loop_sleep_s))
                        continue

                # Phase 5: CENTER — with the wrist pointing straight down, re-detect the cube in
                # the camera image and issue small base-XY nudges until the bbox center sits in the
                # image center. Stops open-loop drift from the bbox-size depth estimate and from
                # any compliance in the tilt phase. Uses the OpenCV convention (+u right, +v down);
                # camera Z points down (-Z_base), so image +u maps to one horizontal base-XY axis
                # and image +v to the other — we convert through T_base_cam explicitly.
                if plan_top_phase == "center":
                    center_iter = int(top_descend_iters)  # reusing counter
                    if center_iter >= int(cfg.plan_top_center_max_iters):
                        logger.info(
                            "[yolo-track] plan_top: center phase hit iter cap (%d) — stopping.",
                            int(cfg.plan_top_center_max_iters),
                        )
                        break
                    if det is None or float(det.confidence) < float(cfg.plan_top_center_min_conf):
                        rec_every = max(1, int(getattr(cfg, "plan_top_center_reacquire_every_n_iters", 5)))
                        if (
                            bool(cfg.plan_top_center_reacquire_look)
                            and _camera_on_arm(cfg, mount)
                            and p_target_base is not None
                            and not cfg.dry_run
                            and (center_iter % rec_every) == 0
                        ):
                            assert robot is not None
                            T_bc_rq = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(
                                T_ee_cam, dtype=np.float64
                            )
                            execute_gripper_nudge(
                                robot=robot,
                                kinematics=kinematics,
                                motor_names=SO100_MOTOR_NAMES,
                                motion_default=motion,
                                motion_look=motion_look,
                                use_look=bool(cfg.gripper_point_at_target),
                                T_base_ee=np.asarray(T_base_ee, dtype=np.float64),
                                T_ee_cam=np.asarray(T_ee_cam, dtype=np.float64),
                                delta_base=np.zeros(3, dtype=np.float64),
                                aim_point_base=np.asarray(
                                    p_target_base, dtype=np.float64
                                ).reshape(3),
                                eye_cam_base=np.asarray(T_bc_rq[:3, 3], dtype=np.float64),
                                max_look_rot_step_deg=float(cfg.gripper_max_look_rot_step_deg),
                                label="plan_top_center_reacquire_aim",
                            )
                        z_rec_step = float(cfg.plan_top_center_no_det_lift_step_m)
                        z_rec_max = float(cfg.plan_top_center_no_det_lift_max_m)
                        z_rec_n = max(1, int(cfg.plan_top_center_no_det_recover_every_n))
                        if (center_iter % z_rec_n) == 0 and not cfg.dry_run:
                            assert robot is not None
                            if _plan_top_vertical_recovery_use_shoulder(cfg):
                                prev_sl = float(plan_top_center_shoulder_lift_applied)
                                plan_top_center_shoulder_lift_applied = _plan_top_apply_shoulder_lift_recovery(
                                    robot,
                                    cfg,
                                    joints_deg,
                                    plan_top_center_shoulder_lift_applied,
                                )
                                if plan_top_center_shoulder_lift_applied != prev_sl:
                                    logger.info(
                                        "[yolo-track] plan_top: center blind shoulder "
                                        "(cumulative=%.1f° cap=%.1f°)",
                                        plan_top_center_shoulder_lift_applied,
                                        float(cfg.plan_top_vertical_recovery_shoulder_lift_total_deg),
                                    )
                            elif z_rec_step > 1e-6 and z_rec_max > 1e-6:
                                room = z_rec_max - float(plan_top_center_no_det_lift_accum)
                                dz_cmd = float(min(z_rec_step, max(0.0, room)))
                                if dz_cmd > 1e-6:
                                    execute_cartesian_nudge_base(
                                        robot,
                                        kinematics,
                                        SO100_MOTOR_NAMES,
                                        np.array([0.0, 0.0, dz_cmd], dtype=np.float64),
                                        motion,
                                    )
                                    plan_top_center_no_det_lift_accum += dz_cmd
                                    obs_cz = robot.get_observation()
                                    joints_deg[:] = np.array(
                                        [float(obs_cz[f"{m}.pos"]) for m in SO100_MOTOR_NAMES],
                                        dtype=np.float64,
                                    )
                                    logger.info(
                                        "[yolo-track] plan_top: center blind +Z=%.4f (accum=%.3f/%.3fm)",
                                        dz_cmd,
                                        plan_top_center_no_det_lift_accum,
                                        z_rec_max,
                                    )
                        logger.info(
                            "[yolo-track] plan_top: center waiting for detection (iter=%d/%d, conf=%.2f)",
                            center_iter, int(cfg.plan_top_center_max_iters),
                            float(det.confidence) if det is not None else 0.0,
                        )
                        top_descend_iters = center_iter + 1
                        time.sleep(float(cfg.loop_sleep_s))
                        continue
                    plan_top_center_no_det_lift_accum = 0.0
                    plan_top_center_shoulder_lift_applied = 0.0
                    x1c, y1c, x2c, y2c = det.xyxy
                    u_raw = 0.5 * (x1c + x2c)
                    v_raw = 0.5 * (y1c + y2c)
                    ac_c = float(np.clip(cfg.smooth_center_alpha, 0.0, 1.0))
                    if plan_top_frame_smooth_cx is None or ac_c >= 1.0 - 1e-9:
                        plan_top_frame_smooth_cx = float(u_raw)
                        plan_top_frame_smooth_cy = float(v_raw)
                    else:
                        plan_top_frame_smooth_cx = (
                            ac_c * float(u_raw)
                            + (1.0 - ac_c) * float(plan_top_frame_smooth_cx)
                        )
                        plan_top_frame_smooth_cy = (
                            ac_c * float(v_raw)
                            + (1.0 - ac_c) * float(plan_top_frame_smooth_cy)
                        )
                    u = float(plan_top_frame_smooth_cx)
                    v_pix = float(plan_top_frame_smooth_cy)
                    du = float(u - cx0)
                    dv = float(v_pix - cy0)
                    err_px = float(np.hypot(du, dv))
                    if err_px <= float(cfg.plan_top_center_tol_px):
                        logger.info(
                            "[yolo-track] plan_top: centered! err=%.1fpx (tol=%.1f) — hovering over cube. Done.",
                            err_px, float(cfg.plan_top_center_tol_px),
                        )
                        break
                    if (
                        bool(cfg.plan_top_center_pan_enable)
                        and _camera_on_arm(cfg, mount)
                        and not cfg.dry_run
                    ):
                        assert robot is not None
                        if _send_shoulder_pan_toward_image_center(
                            robot,
                            cfg,
                            joints_deg,
                            cx_img=float(u),
                            fx=fx,
                            cx0=cx0,
                            deadband_px=float(cfg.plan_top_center_pan_deadband_px),
                            kp=float(cfg.plan_top_center_pan_kp),
                            max_step_deg=float(cfg.plan_top_center_pan_max_step_deg),
                            wrist_bias_deg=0.0,
                            iter_idx=it,
                            log_prefix="[yolo-track] plan_top: center-pan",
                        ):
                            top_descend_iters = center_iter + 1
                            time.sleep(float(cfg.loop_sleep_s))
                            continue
                    # Convert pixel error to a small base-XY nudge. With wrist pointing down, the
                    # camera looks along -Z_base; at the current hover height h above the cube, a
                    # unit pixel offset corresponds to h/fx meters in the camera X axis (and h/fy
                    # in camera Y). We rotate through T_base_cam = T_base_ee @ T_ee_cam.
                    T_bc = T_base_ee @ T_ee_cam
                    h_above_cube = max(
                        0.02,
                        float(T_bc[2, 3]) - float(np.asarray(p_target_base).reshape(3)[2]),
                    )
                    dx_cam = du * h_above_cube / float(fx)
                    dy_cam = dv * h_above_cube / float(fy)
                    # Point at bbox center, at the cube's Z, expressed in base frame.
                    R_bc = np.asarray(T_bc[:3, :3], dtype=np.float64)
                    t_bc = np.asarray(T_bc[:3, 3], dtype=np.float64)
                    ray_cam = np.array([dx_cam, dy_cam, h_above_cube], dtype=np.float64)
                    p_bbox_base = R_bc @ ray_cam + t_bc
                    # Only nudge in XY; keep Z constant.
                    ee_xy = np.asarray(T_base_ee[:3, 3], dtype=np.float64)[:2]
                    dxy_full = p_bbox_base[:2] - ee_xy
                    norm = float(np.linalg.norm(dxy_full))
                    if norm < 1e-5:
                        logger.info(
                            "[yolo-track] plan_top: center ray colinear — stopping. err=%.1fpx",
                            err_px,
                        )
                        break
                    max_step = float(cfg.plan_top_center_step_m)
                    if norm > max_step:
                        dxy_full = dxy_full * (max_step / norm)
                    d3 = np.array([dxy_full[0], dxy_full[1], 0.0], dtype=np.float64)
                    if not cfg.dry_run:
                        assert robot is not None
                        _plan_top_trans_nudge(
                            robot=robot,
                            kinematics=kinematics,
                            cfg=cfg,
                            mount=mount,
                            motion=motion,
                            motion_look=motion_look,
                            T_base_ee=T_base_ee,
                            T_ee_cam=T_ee_cam,
                            p_target_base=p_target_base,
                            delta_base=d3,
                            use_look_flag=bool(cfg.plan_top_center_xy_look_at_target),
                            label="plan_top_center_xy",
                        )
                    logger.info(
                        "[yolo-track] plan_top: center step du=%.0fpx dv=%.0fpx err=%.1fpx Δxy=%s",
                        du, dv, err_px, np.round(dxy_full, 4).tolist(),
                    )
                    top_descend_iters = center_iter + 1
                    time.sleep(float(cfg.loop_sleep_s))
                    continue

                # No phase matched — stop to avoid undefined behavior.
                logger.warning(
                    "[yolo-track] plan_top: unknown phase %r — stopping.", plan_top_phase,
                )
                break

            clearance_m = float(cfg.umbrella_clearance_z_m)
            if phased and uma_phase == "lift" and clearance_m > 1e-6:
                lift_iter_total += 1
                hit_clear = z_ee >= z_start + clearance_m
                hit_cap = lift_iter_total >= int(cfg.umbrella_lift_max_iters)
                if hit_clear or hit_cap:
                    uma_phase = "align"
                    lift_iter_total = 0
                    prev_smoothed_delta.fill(0.0)
                    logger.info(
                        "[yolo-track] umbrella → align z_ee=%.4f (z0+z_clear=%.4f) reason=%s",
                        z_ee,
                        z_start + clearance_m,
                        "clearance" if hit_clear else "max_lift_iters",
                    )
                    time.sleep(float(cfg.loop_sleep_s))
                    continue
                # Lift fast, but keep the object centered (XY) so we don't lose it.
                raw_delta = np.zeros(3, dtype=np.float64)
                if bool(cfg.lift_track_xy):
                    assert det is not None
                    x1, y1, x2, y2 = det.xyxy
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    ac = float(cfg.smooth_center_alpha)
                    if smooth_cx is None or ac >= 1.0 - 1e-9:
                        smooth_cx, smooth_cy = float(cx), float(cy)
                    else:
                        smooth_cx = ac * float(cx) + (1.0 - ac) * float(smooth_cx)
                        smooth_cy = ac * float(cy) + (1.0 - ac) * float(smooth_cy)
                    cx_ctl, cy_ctl = smooth_cx, smooth_cy

                    if mount == "gripper":
                        T_base_cam = T_base_ee @ T_ee_cam
                    else:
                        T_base_cam = cam_to_robot
                    R_bc = rotation_base_cam(T_base_cam)
                    depth_m_lift: float | None = None
                    if depth is not None:
                        depth_m_lift = median_depth_m(
                            np.asarray(depth), det.xyxy, depth_scale=depth_scale
                        )
                    if depth_m_lift is not None and float(depth_m_lift) > float(
                        cfg.max_plausible_depth_m
                    ):
                        depth_m_lift = None
                    lift_xy_cap = min(float(cfg.max_step_m), float(cfg.lift_track_max_step_m))
                    xy_delta = visual_servo_delta_base(
                        cx=cx_ctl,
                        cy=cy_ctl,
                        fx=fx,
                        fy=fy,
                        cx0=cx0,
                        cy0=cy0,
                        depth_m=depth_m_lift,
                        R_base_cam=R_bc,
                        kp_xy=float(cfg.lift_track_xy_gain),
                        kp_z=0.0,
                        max_step_m=lift_xy_cap,
                        target_depth_m=float(cfg.target_depth_m),
                        min_depth_m=float(cfg.min_depth_m),
                        bbox_area_frac=0.0,
                        area_close_frac=float(cfg.area_close_frac),
                        forward_if_no_depth_m=0.0,
                        invert_lateral=bool(cfg.invert_lateral),
                    )
                    xy_delta[2] = 0.0
                    raw_delta += xy_delta

                raw_delta[2] += float(cfg.umbrella_lift_z_sign) * float(cfg.umbrella_lift_step_m)

                # Clip and send WITHOUT EMA (keep lift responsive).
                mmax = float(cfg.max_step_m)
                nrm = float(np.linalg.norm(raw_delta))
                if nrm > mmax and nrm > 1e-9:
                    raw_delta *= mmax / nrm
                delta_base = raw_delta
                if float(np.linalg.norm(delta_base)) >= 1e-6 and not cfg.dry_run:
                    assert robot is not None
                    logger.info(
                        "[yolo-track] iter=%d phase=lift z_ee=%.4f Δbase=%s conf=%.2f",
                        it,
                        z_ee,
                        np.round(delta_base, 4).tolist(),
                        float(det.confidence),
                    )
                    decouple_lift_look = (
                        bool(cfg.umbrella_lift_decouple_look)
                        and mount == "gripper"
                        and bool(cfg.gripper_point_at_target)
                    )
                    if decouple_lift_look:
                        if p_target_base is not None:
                            T_bc_lift = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(
                                T_ee_cam, dtype=np.float64
                            )
                            execute_gripper_nudge(
                                robot=robot,
                                kinematics=kinematics,
                                motor_names=SO100_MOTOR_NAMES,
                                motion_default=motion,
                                motion_look=motion_look,
                                use_look=True,
                                T_base_ee=np.asarray(T_base_ee, dtype=np.float64),
                                T_ee_cam=np.asarray(T_ee_cam, dtype=np.float64),
                                delta_base=np.zeros(3, dtype=np.float64),
                                aim_point_base=np.asarray(p_target_base, dtype=np.float64).reshape(3),
                                eye_cam_base=np.asarray(T_bc_lift[:3, 3], dtype=np.float64),
                                max_look_rot_step_deg=float(cfg.umbrella_lift_look_rot_step_deg),
                                label="gripper_lift_maintain_aim",
                            )
                            obs_l = robot.get_observation()
                            joints_deg[:] = np.array(
                                [float(obs_l[f"{m}.pos"]) for m in SO100_MOTOR_NAMES],
                                dtype=np.float64,
                            )
                            T_base_ee = kinematics.forward_kinematics(joints_deg)
                        execute_cartesian_nudge_base(
                            robot, kinematics, SO100_MOTOR_NAMES, delta_base, motion
                        )
                    elif mount == "gripper" and bool(cfg.gripper_point_at_target):
                        if bool(cfg.gripper_aim_use_filtered_3d) and p_target_base is not None:
                            aim_lift = np.asarray(p_target_base, dtype=np.float64).copy()
                        else:
                            aim_lift = point_cam_to_base(
                                T_base_cam,
                                u=cx_ctl,
                                v_pix=cy_ctl,
                                depth_m=float(cfg.gripper_aim_default_depth_m),
                                fx=fx,
                                fy=fy,
                                cx0=cx0,
                                cy0=cy0,
                            )
                        execute_gripper_nudge(
                            robot=robot,
                            kinematics=kinematics,
                            motor_names=SO100_MOTOR_NAMES,
                            motion_default=motion,
                            motion_look=motion_look,
                            use_look=True,
                            T_base_ee=T_base_ee,
                            T_ee_cam=T_ee_cam,
                            delta_base=delta_base,
                            aim_point_base=aim_lift,
                            eye_cam_base=T_base_cam[:3, 3],
                            max_look_rot_step_deg=float(cfg.gripper_max_look_rot_step_deg),
                            label="gripper_lift_track",
                        )
                    else:
                        execute_cartesian_nudge_base(
                            robot, kinematics, SO100_MOTOR_NAMES, delta_base, motion
                        )
                time.sleep(float(cfg.loop_sleep_s))
                continue

            if skip_visual:
                lost += 1
                logger.info("[yolo-track] iter=%d no detection (%d/%d)", it, lost, cfg.lost_patience)
                if lost >= cfg.lost_patience:
                    logger.warning("[yolo-track] stopping: lost target too long")
                    break
                time.sleep(float(cfg.loop_sleep_s))
                continue

            assert det is not None
            x1, y1, x2, y2 = det.xyxy
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            h, w = rgb.shape[:2]
            area_frac = max(0.0, (x2 - x1) * (y2 - y1) / float(w * h))

            ac = float(cfg.smooth_center_alpha)
            if smooth_cx is None or ac >= 1.0 - 1e-9:
                smooth_cx, smooth_cy = float(cx), float(cy)
            else:
                smooth_cx = ac * float(cx) + (1.0 - ac) * float(smooth_cx)
                smooth_cy = ac * float(cy) + (1.0 - ac) * float(smooth_cy)
            cx_ctl, cy_ctl = smooth_cx, smooth_cy

            depth_m_raw = None
            if depth is not None:
                depth_m_raw = median_depth_m(
                    np.asarray(depth),
                    det.xyxy,
                    depth_scale=depth_scale,
                )
            if depth_m_raw is not None and float(depth_m_raw) > float(cfg.max_plausible_depth_m):
                depth_m_raw = None

            depth_for_ctl: float | None = None
            ad = float(cfg.smooth_depth_alpha)
            if depth_m_raw is not None:
                if smooth_depth_m is None or ad >= 1.0 - 1e-9:
                    smooth_depth_m = float(depth_m_raw)
                else:
                    smooth_depth_m = ad * float(depth_m_raw) + (1.0 - ad) * float(smooth_depth_m)
                depth_for_ctl = smooth_depth_m
            else:
                smooth_depth_m = None

            if mount == "gripper":
                T_base_cam = T_base_ee @ T_ee_cam
            else:
                T_base_cam = cam_to_robot

            aim_meas_main: np.ndarray | None = None
            if mount == "gripper":
                aim_d_ema = float(cfg.gripper_aim_default_depth_m)
                if depth_m_raw is not None and float(depth_m_raw) > 1e-4:
                    aim_d_ema = float(depth_m_raw)
                elif depth_for_ctl is not None and float(depth_for_ctl) > 1e-4:
                    aim_d_ema = float(depth_for_ctl)
                aim_meas_main = point_cam_to_base(
                    T_base_cam,
                    u=cx_ctl,
                    v_pix=cy_ctl,
                    depth_m=aim_d_ema,
                    fx=fx,
                    fy=fy,
                    cx0=cx0,
                    cy0=cy0,
                )
                if bool(cfg.gripper_track_3d_enable) and not cfg.dry_run:
                    depth_ok_tr = depth_m_raw is not None and float(depth_m_raw) > 1e-4
                    if (not bool(cfg.gripper_3d_ema_require_depth)) or depth_ok_tr:
                        p_target_base = ema_p_base(
                            p_target_base,
                            aim_meas_main,
                            float(cfg.gripper_track_3d_ema_alpha),
                        )

            R_bc = rotation_base_cam(T_base_cam)

            ex = abs(cx_ctl - cx0)
            ey = abs(cy_ctl - cy0)
            centered = ex < float(cfg.center_deadband_px) and ey < float(cfg.center_deadband_px)
            close_enough = False
            if depth_for_ctl is not None:
                close_enough = depth_for_ctl <= float(cfg.target_depth_m)
            else:
                close_enough = area_frac >= float(cfg.area_close_frac)

            if phased and uma_phase == "align" and centered:
                uma_phase = "descend"
                prev_smoothed_delta.fill(0.0)
                logger.info("[yolo-track] umbrella → descend (centered in image)")
                time.sleep(float(cfg.loop_sleep_s))
                continue

            can_finish = (not phased) or uma_phase == "descend"
            if can_finish and centered and close_enough:
                logger.info(
                    "[yolo-track] done iter=%d phase=%s centered=%s depth_m=%s area_frac=%.3f",
                    it,
                    uma_phase,
                    centered,
                    f"{depth_for_ctl:.3f}" if depth_for_ctl is not None else "n/a",
                    area_frac,
                )
                break

            depth_for_servo = depth_for_ctl
            if phased and uma_phase == "align":
                depth_for_servo = None

            raw_delta = visual_servo_delta_base(
                cx=cx_ctl,
                cy=cy_ctl,
                fx=fx,
                fy=fy,
                cx0=cx0,
                cy0=cy0,
                depth_m=depth_for_servo,
                R_base_cam=R_bc,
                kp_xy=float(cfg.kp_xy),
                kp_z=float(cfg.kp_z),
                max_step_m=float(cfg.max_step_m),
                target_depth_m=float(cfg.target_depth_m),
                min_depth_m=float(cfg.min_depth_m),
                bbox_area_frac=area_frac,
                area_close_frac=float(cfg.area_close_frac),
                forward_if_no_depth_m=float(cfg.forward_if_no_depth_m),
                invert_lateral=bool(cfg.invert_lateral),
            )

            if phased and uma_phase == "align":
                raw_delta[2] = 0.0
                nrm_h = float(np.linalg.norm(raw_delta))
                mmax = float(cfg.max_step_m)
                if nrm_h > mmax and nrm_h > 1e-9:
                    raw_delta *= mmax / nrm_h

            if style == "umbrella" and (not phased or uma_phase == "descend") and centered:
                raw_delta[2] += float(cfg.umbrella_base_z_sign) * float(cfg.umbrella_descent_step_m)

            if centered and (not phased or uma_phase == "descend"):
                raw_delta[0] = 0.0
                raw_delta[1] = 0.0

            mmax = float(cfg.max_step_m)
            nrm = float(np.linalg.norm(raw_delta))
            if nrm > mmax and nrm > 1e-9:
                raw_delta *= mmax / nrm

            sd = float(cfg.smooth_delta_alpha)
            if sd >= 1.0 - 1e-9:
                delta_base = raw_delta.copy()
            else:
                prev_smoothed_delta[:] = sd * raw_delta + (1.0 - sd) * prev_smoothed_delta
                delta_base = prev_smoothed_delta.copy()

            if float(np.linalg.norm(delta_base)) < 1e-6:
                logger.info("[yolo-track] iter=%d no delta (centered or at limits)", it)
                time.sleep(float(cfg.loop_sleep_s))
                continue

            logger.info(
                "[yolo-track] iter=%d phase=%s conf=%.2f bbox=%s center_raw=(%.0f,%.0f) center_s=(%.0f,%.0f) depth=%s Δbase=%s",
                it,
                uma_phase,
                det.confidence,
                tuple(round(v, 1) for v in det.xyxy),
                cx,
                cy,
                cx_ctl,
                cy_ctl,
                f"{depth_for_ctl:.3f}" if depth_for_ctl is not None else "n/a",
                np.round(delta_base, 4).tolist(),
            )

            if not cfg.dry_run:
                assert robot is not None
                if mount == "gripper" and bool(cfg.gripper_point_at_target):
                    assert aim_meas_main is not None
                    aim_main = (
                        p_target_base
                        if bool(cfg.gripper_aim_use_filtered_3d) and p_target_base is not None
                        else aim_meas_main
                    )
                    execute_gripper_nudge(
                        robot=robot,
                        kinematics=kinematics,
                        motor_names=SO100_MOTOR_NAMES,
                        motion_default=motion,
                        motion_look=motion_look,
                        use_look=True,
                        T_base_ee=T_base_ee,
                        T_ee_cam=T_ee_cam,
                        delta_base=delta_base,
                        aim_point_base=aim_main,
                        eye_cam_base=T_base_cam[:3, 3],
                        max_look_rot_step_deg=float(cfg.gripper_max_look_rot_step_deg),
                        label="gripper_visual_servo",
                    )
                else:
                    execute_cartesian_nudge_base(
                        robot,
                        kinematics,
                        SO100_MOTOR_NAMES,
                        delta_base,
                        motion,
                    )

            if cfg.show_window:
                vis = cv2.cvtColor(np.asarray(rgb).copy(), cv2.COLOR_RGB2BGR)
                xi1, yi1, xi2, yi2 = (int(round(v)) for v in det.xyxy)
                cv2.rectangle(vis, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)
                cv2.drawMarker(vis, (int(round(cx0)), int(round(cy0))), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.imshow("yolo_track", vis)
                cv2.waitKey(1)

            time.sleep(float(cfg.loop_sleep_s))

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        if cfg.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if robot is not None:
            robot.disconnect()
