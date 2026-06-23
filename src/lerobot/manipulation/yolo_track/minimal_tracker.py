# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal live tracker: detect -> center -> depth approach (servo only).

This module intentionally keeps the control surface small and avoids the larger phased state
machines in ``runner.py``. It is designed for live tracking where the object should remain near the
image center while the robot approaches to a depth standoff.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.manipulation.yolo_track.depth import depth_from_bbox_size, median_depth_m
from lerobot.manipulation.yolo_track.math_utils import (
    camera_opencv_to_robot_rotation,
    parse_tf_string,
    point_cam_to_base,
    rotation_base_cam,
    visual_servo_delta_base,
)
from lerobot.manipulation.yolo_track.motion_primitives import execute_gripper_nudge
from lerobot.manipulation.yolo_track.motion_primitives import send_joint_target_smoothly
from lerobot.manipulation.yolo_track.rerun_viz import log_rerun_iter
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig  # noqa: F401
from lerobot.utils.motion_executor import MotionExecutionConfig, execute_cartesian_nudge_base
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

SO100_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _apply_axis_mode(delta_base: np.ndarray, axis_mode: str) -> np.ndarray:
    d = np.asarray(delta_base, dtype=np.float64).reshape(3).copy()
    mode = (axis_mode or "xyz").lower().strip()
    if mode == "xyz":
        return d
    if mode == "xy":
        d[2] = 0.0
    elif mode == "z":
        d[0] = 0.0
        d[1] = 0.0
    elif mode == "xz":
        d[1] = 0.0
    elif mode == "yz":
        d[0] = 0.0
    elif mode == "x":
        d[1] = 0.0
        d[2] = 0.0
    elif mode == "y":
        d[0] = 0.0
        d[2] = 0.0
    return d


@dataclass
class MinimalTrackerConfig:
    # Concrete default (``RobotConfig(type=...)`` is invalid on the abstract registry class).
    robot: RobotConfig = field(default_factory=lambda: SOFollowerRobotConfig(port="", cameras={}))
    urdf: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    camera_key: str = "front"
    camera_mount: str = "fixed"  # fixed | gripper
    camera_to_robot_tf: str = "0.4,0,0.1,0,0,0"
    gripper_camera_tf: str = "0,0,0,0,0,0"
    camera_frame_convention: str = "opencv"
    camera_flip_lateral: bool = False

    query: str = "cup"
    model_path: str = "yolov8s-worldv2.pt"
    device: str = ""
    conf_threshold: float = 0.25
    # SAHI-style sliced inference for small objects
    sahi_enable: bool = False
    sahi_slice_w: int = 512
    sahi_slice_h: int = 512
    sahi_overlap: float = 0.20
    sahi_iou_threshold: float = 0.55
    sahi_include_full_image: bool = True

    kp_xy: float = 0.52
    kp_z: float = 0.55
    max_step_m: float = 0.02
    center_deadband_px: float = 18.0
    target_depth_m: float = 0.22
    min_depth_m: float = 0.11
    max_plausible_depth_m: float = 1.35
    area_close_frac: float = 0.18
    forward_if_no_depth_m: float = 0.01
    invert_lateral: bool = False
    # Which base-frame axes are allowed in control output.
    # One of: xyz, xy, z, xz, yz, x, y
    axis_mode: str = "xyz"
    # When closer than target_depth_m by this margin, command negative camera-Z (back off).
    z_backoff_enable: bool = True
    z_backoff_deadband_m: float = 0.01
    z_backoff_kp: float = 0.55
    # If True, gate forward/back (camera Z) when the bbox is far from the image center.
    # Uses ``z_center_pixel_tol_px`` (not the tiny final deadband) so the arm can still approach
    # while roughly centering.
    z_only_when_centered: bool = True
    z_center_pixel_tol_px: float = 80.0

    # Depth selection. OAK-D stereo is unreliable below MinZ (~0.35m unless extended disparity
    # is enabled), so default behavior is to prefer bbox-size depth when available.
    depth_policy: str = "bbox_preferred"  # bbox_preferred | stereo_preferred | bbox_only | stereo_only
    max_depth_jump_m: float = 0.12
    use_bbox_depth_fallback: bool = True  # kept for backward compatibility; implied by depth_policy
    target_physical_size_m: float = 0.03

    smooth_center_alpha: float = 0.35
    smooth_depth_alpha: float = 0.35
    smooth_delta_alpha: float = 0.45

    gripper_point_at_target: bool = False
    gripper_aim_default_depth_m: float = 0.25
    gripper_max_look_rot_step_deg: float = 10.0

    motion_cartesian_step_m: float = 0.01
    motion_sleep_s: float = 0.025

    # Fast, stable centering: use shoulder_pan (joint space) when object is far from image center.
    pan_center_enable: bool = True
    # Use raw bbox center for pan (smoothed center lags and causes overshoot / ping-pong).
    pan_use_raw_center: bool = True
    pan_deadband_px: float = 40.0
    pan_kp: float = 0.55
    pan_max_step_deg: float = 3.0

    # Lost-target behavior: smooth joint-space scan around startup baseline.
    search_scan_enabled: bool = True
    search_scan_start_after_lost_frames: int = 6
    search_pan_amplitude_deg: float = 20.0
    search_pan_period_iters: int = 22
    search_lift_up_delta_deg: float = -8.0
    search_lift_up_max_deg: float = -18.0
    search_wrist_up_delta_deg: float = -22.0
    search_joint_step_deg: float = 1.25
    search_joint_sleep_s: float = 0.022

    # Optional rerun visualization.
    display_data: bool = True
    display_sim3d: bool = True
    display_ip: str = ""
    display_port: int = 9876
    display_object_half_size_m: float = 0.015

    # Stop condition override: if > 0, stop when bbox covers this fraction of the image.
    # This is the most reliable “go very close” proxy when stereo depth is invalid in near-field.
    stop_area_frac: float = 0.0

    loop_sleep_s: float = 0.03
    lost_patience: int = 25
    max_iters: int = 1000


@parser.wrap()
def minimal_tracker(cfg: MinimalTrackerConfig) -> None:
    run_minimal_tracker(cfg)


def run_minimal_tracker(cfg: MinimalTrackerConfig) -> None:
    init_logging()
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.perception.yolo_world import YoloWorldDetector

    mount = (cfg.camera_mount or "fixed").lower().strip()
    if mount not in ("fixed", "gripper"):
        raise ValueError("camera_mount must be 'fixed' or 'gripper'")

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

    motion = MotionExecutionConfig(
        use_cartesian_interp=True,
        cartesian_step_m=float(cfg.motion_cartesian_step_m),
        min_steps_per_segment=3,
        inter_step_sleep_s=float(cfg.motion_sleep_s),
        settle_last_step=False,
        ik_position_weight=1.0,
        ik_orientation_weight=0.01,
    )
    motion_look = MotionExecutionConfig(
        use_cartesian_interp=True,
        cartesian_step_m=float(cfg.motion_cartesian_step_m),
        min_steps_per_segment=3,
        inter_step_sleep_s=float(cfg.motion_sleep_s),
        settle_last_step=False,
        ik_position_weight=1.0,
        ik_orientation_weight=0.12,
    )
    rerun_enabled = bool(cfg.display_data) or bool(cfg.display_sim3d)
    if rerun_enabled:
        try:
            from lerobot.utils.visualization_utils import init_rerun, send_agentic_rerun_blueprint

            init_rerun(
                session_name="minimal_tracker",
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

    robot = None
    smooth_cx: float | None = None
    smooth_cy: float | None = None
    smooth_depth_m: float | None = None
    p_target_base: np.ndarray | None = None
    ee_trail: list[np.ndarray] = []
    prev_delta = np.zeros(3, dtype=np.float64)
    lost_frames = 0
    baseline_joints_deg: np.ndarray | None = None
    scan_step_idx = 0
    try:
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        obs0 = robot.get_observation()
        rgb0 = np.asarray(obs0[cfg.camera_key])
        h, w = rgb0.shape[:2]
        intrinsics = {"fx": 525.0, "fy": 525.0, "cx": w / 2.0, "cy": h / 2.0, "depth_scale": 0.001}
        dc = getattr(robot, "cameras", {}).get(cfg.camera_key, None)
        if dc is not None and hasattr(dc, "get_depth_intrinsics"):
            try:
                intrinsics = dict(dc.get_depth_intrinsics())
            except Exception as e:
                logger.warning("get_depth_intrinsics failed: %s", e)
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx0 = float(intrinsics["cx"])
        cy0 = float(intrinsics["cy"])
        depth_scale = float(intrinsics.get("depth_scale", 0.001))

        for it in range(int(cfg.max_iters)):
            obs = robot.get_observation()
            rgb = obs.get(cfg.camera_key)
            depth = obs.get(f"{cfg.camera_key}_depth")
            if rgb is None:
                time.sleep(float(cfg.loop_sleep_s))
                continue
            rgb = np.asarray(rgb)
            joints_deg = np.array([float(obs[f"{m}.pos"]) for m in SO100_MOTOR_NAMES], dtype=np.float64)
            T_base_ee = kinematics.forward_kinematics(joints_deg)
            T_base_cam = T_base_ee @ T_ee_cam if mount == "gripper" else np.asarray(cam_to_robot, dtype=np.float64)
            if baseline_joints_deg is None:
                baseline_joints_deg = joints_deg.copy()
            ee_now = np.asarray(T_base_ee, dtype=np.float64)[:3, 3].copy()
            if not ee_trail or float(np.linalg.norm(ee_now - ee_trail[-1])) > 1e-3:
                ee_trail.append(ee_now)
                if len(ee_trail) > 600:
                    ee_trail.pop(0)

            det = detector.best_detection(rgb)
            if det is None:
                lost_frames += 1
                if (
                    bool(cfg.search_scan_enabled)
                    and lost_frames >= int(cfg.search_scan_start_after_lost_frames)
                    and baseline_joints_deg is not None
                ):
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
                    wrist_ramp = 1.0 - float(np.exp(-scan_step_idx / max(period / 2.0, 1.0)))
                    wrist_delta = float(cfg.search_wrist_up_delta_deg) * wrist_ramp
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
                    try:
                        i_wrist = SO100_MOTOR_NAMES.index("wrist_flex")
                        target_joints[i_wrist] = baseline_joints_deg[i_wrist] + wrist_delta
                    except ValueError:
                        pass
                    raw_g = obs.get("gripper.pos")
                    g_open = True
                    g_pct = 100.0
                    if raw_g is not None:
                        v = float(raw_g)
                        g_open = v >= 90.0
                        g_pct = float(np.clip(v, 0.0, 100.0))
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
                if lost_frames >= int(cfg.lost_patience):
                    logger.info("[minimal-tracker] lost target for %d frames; stopping.", lost_frames)
                    break
                if lost_frames % 5 == 0:
                    logger.info("[minimal-tracker] no detection (%d/%d).", lost_frames, int(cfg.lost_patience))
                if rerun_enabled:
                    try:
                        log_rerun_iter(
                            frame=it,
                            camera_key=cfg.camera_key,
                            rgb=np.asarray(rgb),
                            depth=depth,
                            bbox_xyxy=None,
                            bbox_center_raw=None,
                            bbox_center_smoothed=None,
                            cx0=cx0,
                            cy0=cy0,
                            conf=None,
                            phase="search",
                            z_ee=float(ee_now[2]),
                            depth_m=None,
                            kinematics=kinematics,
                            joints_deg=joints_deg,
                            T_base_ee=T_base_ee,
                            T_base_cam=T_base_cam,
                            p_target_base=p_target_base,
                            ee_trail=ee_trail,
                            object_half_size_m=float(cfg.display_object_half_size_m),
                            show_sim3d=bool(cfg.display_sim3d),
                            show_camera=bool(cfg.display_data),
                            object_semantic_label=str(cfg.query),
                        )
                    except Exception:
                        pass
                time.sleep(float(cfg.loop_sleep_s))
                continue
            lost_frames = 0
            scan_step_idx = 0

            x1, y1, x2, y2 = det.xyxy
            cx_raw = 0.5 * (x1 + x2)
            cy_raw = 0.5 * (y1 + y2)
            if smooth_cx is None:
                smooth_cx, smooth_cy = float(cx_raw), float(cy_raw)
            else:
                a_c = float(np.clip(cfg.smooth_center_alpha, 0.0, 1.0))
                smooth_cx = a_c * float(cx_raw) + (1.0 - a_c) * float(smooth_cx)
                smooth_cy = a_c * float(cy_raw) + (1.0 - a_c) * float(smooth_cy)

            # If the object is far off-center, do a bounded shoulder_pan correction first.
            # This is much more stable than trying to “side-step” in Cartesian space while the
            # camera is on the wrist.
            ex_px = float(smooth_cx) - float(cx0)
            if (
                mount == "gripper"
                and bool(cfg.pan_center_enable)
                and abs(ex_px) > float(cfg.pan_deadband_px)
                and baseline_joints_deg is not None
            ):
                try:
                    i_pan = SO100_MOTOR_NAMES.index("shoulder_pan")
                except ValueError:
                    i_pan = -1
                if i_pan >= 0:
                    ang_err_rad = float(np.arctan(ex_px / max(float(fx), 1e-6)))
                    d_deg = float(
                        np.clip(
                            np.degrees(float(cfg.pan_kp) * ang_err_rad),
                            -float(cfg.pan_max_step_deg),
                            float(cfg.pan_max_step_deg),
                        )
                    )
                    target_joints = joints_deg.copy()
                    target_joints[i_pan] = float(target_joints[i_pan] + d_deg)
                    raw_g = obs.get("gripper.pos")
                    g_open = True
                    g_pct = 100.0
                    if raw_g is not None:
                        v = float(raw_g)
                        g_open = v >= 90.0
                        g_pct = float(np.clip(v, 0.0, 100.0))
                    logger.info(
                        "[minimal-tracker] pan-center ex=%.0fpx Δpan=%+.2f° (deadband=%.0fpx)",
                        ex_px,
                        d_deg,
                        float(cfg.pan_deadband_px),
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
                    time.sleep(float(cfg.loop_sleep_s))
                    continue

            # Depth sources:
            # - stereo ROI median can jump to background at close range / invalid band
            # - bbox-size depth is stable in near-field if physical size is correct
            d_stereo: float | None = None
            if depth is not None:
                d_raw = median_depth_m(np.asarray(depth), det.xyxy, depth_scale=depth_scale)
                if d_raw is not None and 1e-4 < float(d_raw) <= float(cfg.max_plausible_depth_m):
                    d_stereo = float(d_raw)
            d_bbox: float | None = None
            if bool(cfg.use_bbox_depth_fallback):
                d_bbox_raw = depth_from_bbox_size(
                    tuple(det.xyxy),
                    fx=fx,
                    fy=fy,
                    target_physical_size_m=float(cfg.target_physical_size_m),
                )
                if d_bbox_raw is not None and float(d_bbox_raw) > 1e-4:
                    d_bbox = float(d_bbox_raw)

            policy = (cfg.depth_policy or "bbox_preferred").lower().strip()
            depth_m: float | None = None
            if policy == "bbox_only":
                depth_m = d_bbox
            elif policy == "stereo_only":
                depth_m = d_stereo
            elif policy == "stereo_preferred":
                depth_m = d_stereo if d_stereo is not None else d_bbox
            else:  # bbox_preferred (default)
                depth_m = d_bbox if d_bbox is not None else d_stereo

            # Reject sudden depth jumps (stereo often flips to background near-field).
            if (
                smooth_depth_m is not None
                and depth_m is not None
                and abs(float(depth_m) - float(smooth_depth_m)) > float(cfg.max_depth_jump_m)
                and d_bbox is not None
            ):
                depth_m = d_bbox
            if smooth_depth_m is None or depth_m is None:
                smooth_depth_m = depth_m
            else:
                a_d = float(np.clip(cfg.smooth_depth_alpha, 0.0, 1.0))
                smooth_depth_m = a_d * float(depth_m) + (1.0 - a_d) * float(smooth_depth_m)

            if smooth_depth_m is not None and float(smooth_depth_m) > 1e-4:
                p_target_base = point_cam_to_base(
                    T_base_cam,
                    u=float(smooth_cx),
                    v_pix=float(smooth_cy),
                    depth_m=float(smooth_depth_m),
                    fx=fx,
                    fy=fy,
                    cx0=cx0,
                    cy0=cy0,
                )

            area_frac = float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(rgb.shape[0] * rgb.shape[1])))
            R_base_cam_now = rotation_base_cam(T_base_cam)
            delta_base = visual_servo_delta_base(
                cx=float(smooth_cx),
                cy=float(smooth_cy),
                fx=fx,
                fy=fy,
                cx0=cx0,
                cy0=cy0,
                depth_m=smooth_depth_m,
                R_base_cam=R_base_cam_now,
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
            centered_for_motion = (
                abs(float(smooth_cx) - cx0) <= float(cfg.center_deadband_px)
                and abs(float(smooth_cy) - cy0) <= float(cfg.center_deadband_px)
            )
            if bool(cfg.z_only_when_centered) and not centered_for_motion:
                # Keep target in view: fix centering first, then move along depth.
                delta_base[2] = 0.0
            if (
                bool(cfg.z_backoff_enable)
                and smooth_depth_m is not None
                and float(smooth_depth_m) < float(cfg.target_depth_m) - float(cfg.z_backoff_deadband_m)
            ):
                dz_back = float(
                    min(
                        float(cfg.max_step_m),
                        float(cfg.z_backoff_kp) * (float(cfg.target_depth_m) - float(smooth_depth_m)),
                    )
                )
                delta_base = np.asarray(delta_base, dtype=np.float64) + (
                    R_base_cam_now @ np.array([0.0, 0.0, -dz_back], dtype=np.float64)
                )
            if abs(float(smooth_cx) - cx0) <= float(cfg.center_deadband_px) and abs(float(smooth_cy) - cy0) <= float(cfg.center_deadband_px):
                delta_base[:2] = 0.0

            a_delta = float(np.clip(cfg.smooth_delta_alpha, 0.0, 1.0))
            delta_base = a_delta * np.asarray(delta_base, dtype=np.float64) + (1.0 - a_delta) * prev_delta
            delta_base = _apply_axis_mode(delta_base, cfg.axis_mode)
            prev_delta = np.asarray(delta_base, dtype=np.float64).copy()
            norm = float(np.linalg.norm(delta_base))
            if norm > float(cfg.max_step_m) and norm > 1e-9:
                delta_base *= float(cfg.max_step_m) / norm

            centered = (
                abs(float(smooth_cx) - cx0) <= float(cfg.center_deadband_px)
                and abs(float(smooth_cy) - cy0) <= float(cfg.center_deadband_px)
            )
            if float(cfg.stop_area_frac) > 1e-6:
                close_enough = area_frac >= float(cfg.stop_area_frac)
            else:
                close_enough = (
                    smooth_depth_m is not None and float(smooth_depth_m) <= float(cfg.target_depth_m)
                ) or (smooth_depth_m is None and area_frac >= float(cfg.area_close_frac))
            if centered and close_enough:
                logger.info(
                    "[minimal-tracker] reached standoff: centered=%s depth=%s area=%.3f stop_area=%.3f",
                    centered,
                    "none" if smooth_depth_m is None else f"{float(smooth_depth_m):.3f}m",
                    area_frac,
                    float(cfg.stop_area_frac),
                )
                break

            if rerun_enabled:
                try:
                    log_rerun_iter(
                        frame=it,
                        camera_key=cfg.camera_key,
                        rgb=np.asarray(rgb),
                        depth=depth,
                        bbox_xyxy=tuple(det.xyxy),
                        bbox_center_raw=(float(cx_raw), float(cy_raw)),
                        bbox_center_smoothed=(float(smooth_cx), float(smooth_cy)),
                        cx0=cx0,
                        cy0=cy0,
                        conf=float(det.confidence),
                        phase="track",
                        z_ee=float(ee_now[2]),
                        depth_m=float(smooth_depth_m) if smooth_depth_m is not None else None,
                        kinematics=kinematics,
                        joints_deg=joints_deg,
                        T_base_ee=T_base_ee,
                        T_base_cam=T_base_cam,
                        p_target_base=p_target_base,
                        ee_trail=ee_trail,
                        object_half_size_m=float(cfg.display_object_half_size_m),
                        show_sim3d=bool(cfg.display_sim3d),
                        show_camera=bool(cfg.display_data),
                        object_semantic_label=str(cfg.query),
                    )
                except Exception:
                    pass

            if float(np.linalg.norm(delta_base)) < 1e-5:
                time.sleep(float(cfg.loop_sleep_s))
                continue

            if mount == "gripper" and bool(cfg.gripper_point_at_target):
                d_aim = (
                    float(smooth_depth_m)
                    if smooth_depth_m is not None and float(smooth_depth_m) > 1e-4
                    else float(cfg.gripper_aim_default_depth_m)
                )
                p_target_base = point_cam_to_base(
                    T_base_cam,
                    u=float(smooth_cx),
                    v_pix=float(smooth_cy),
                    depth_m=d_aim,
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
                    T_base_ee=np.asarray(T_base_ee, dtype=np.float64),
                    T_ee_cam=np.asarray(T_ee_cam, dtype=np.float64),
                    delta_base=np.asarray(delta_base, dtype=np.float64),
                    aim_point_base=np.asarray(p_target_base, dtype=np.float64),
                    eye_cam_base=np.asarray(T_base_cam[:3, 3], dtype=np.float64),
                    max_look_rot_step_deg=float(cfg.gripper_max_look_rot_step_deg),
                    label="minimal_tracker_servo",
                )
            else:
                execute_cartesian_nudge_base(
                    robot,
                    kinematics,
                    SO100_MOTOR_NAMES,
                    np.asarray(delta_base, dtype=np.float64),
                    motion,
                )

            logger.info(
                "[minimal-tracker] iter=%d conf=%.2f center=(%.0f,%.0f) depth=%s area=%.3f axis=%s dxyz=[%.3f, %.3f, %.3f]",
                it,
                float(det.confidence),
                float(smooth_cx),
                float(smooth_cy),
                "none" if smooth_depth_m is None else f"{float(smooth_depth_m):.3f}m",
                area_frac,
                cfg.axis_mode,
                float(delta_base[0]),
                float(delta_base[1]),
                float(delta_base[2]),
            )
            time.sleep(float(cfg.loop_sleep_s))
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
