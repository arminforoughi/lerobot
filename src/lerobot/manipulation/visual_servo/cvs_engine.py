# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Continuous Visual Servoing engine for SO-101 + OAK-D eye-in-hand.

Inputs: query, approach (azimuth, elevation), standoff distance.
Loop: search -> locate -> single SE(3) rate-limited approach with the camera
optical axis pointed at the object the entire time.

Compared to ``pbvs_engine.py`` this is intentionally minimal: no posture
biasing, no vantage waypoints, no translation gate, no IK-overriding pixel
controller. The kinematics are direct math on (p_obj, n_hat, standoff).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.manipulation.yolo_track.depth import depth_from_bbox_size, median_depth_m
from lerobot.manipulation.yolo_track.math_utils import (
    cam_z_of_base_point,
    parse_tf_string,
    point_cam_to_base,
)
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.motion_executor import interpolate_se3
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@dataclass
class CVSEngineConfig:
    # Robot / camera
    robot: RobotConfig = field(
        default_factory=lambda: SOFollowerRobotConfig(port="", cameras={})
    )
    urdf: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    camera_key: str = "front"
    gripper_camera_tf: str = "0.04,0,0.02,0,-0.35,0"
    # If the object appears behind the arm in Rerun while still visible in the
    # image, the stored calibration may be T_cam_ee instead of T_ee_cam; try True.
    invert_gripper_camera_tf: bool = False

    # Detection
    query: str = "red cube"
    model_path: str = "yolov8s-worldv2.pt"
    target_physical_size_m: float = 0.03

    # Approach geometry — the three primary inputs
    standoff_m: float = 0.12
    approach_az_deg: float = 0.0
    approach_el_deg: float = 90.0

    # Hybrid PBVS/IBVS handoff. While the perceived camera–object distance is
    # large we run the standard PBVS approach (look-at + standoff at the fixed
    # ``(approach_az_deg, approach_el_deg)`` direction). Once we get close, we
    # switch to an IBVS-flavored final phase: depth comes from the bbox apparent
    # size (focal · physical_size / pixel_size) — far more reliable inside the
    # ~35 cm range where OAK-D stereo dies and where the ray–plane assumption is
    # most sensitive to plane-Z mis-calibration — and the approach direction is
    # the current optical axis (i.e. fly straight down the line of sight) rather
    # than the global ``n_hat``. ``ibvs_final_standoff_m`` is the close-range
    # stop distance. ``ibvs_handoff_hysteresis_m`` prevents chattering between
    # the two modes when sitting near the threshold.
    ibvs_enabled: bool = True
    ibvs_handoff_distance_m: float = 0.10
    ibvs_final_standoff_m: float = 0.07
    ibvs_handoff_hysteresis_m: float = 0.015

    # Motion limits / IK weights. Orientation must DOMINATE position so the
    # 5-DoF IK actually points the camera at the object — otherwise the IK
    # satisfies p_target with a stretched-out wrist and the cube drifts off
    # the optical axis.
    max_lin_vel_m_s: float = 0.04
    max_ang_vel_deg_s: float = 80.0
    ik_position_weight: float = 0.5
    ik_orientation_weight: float = 2.0
    loop_hz: float = 25.0

    # Tolerate brief detection drops (YOLO flicker on small/oblique objects)
    # without falling back to rotate-only recovery. While inside the tolerance
    # window we keep servoing toward the last p_target using KF-predicted
    # position; only after this many consecutive misses do we switch to the
    # rotate-toward-last-p_obj branch.
    miss_tolerance_frames: int = 5

    # Soft additive pixel feedback (capped, deadbanded). Never gates IK.
    # Larger gains than pure look-at would suggest because the look-at IK is
    # underdetermined on a 5-DoF arm — we use pixel error as the precision
    # final-cm centering layer on top of the look-at approximation.
    pixel_feedback_enabled: bool = True
    pixel_kp_pan: float = 0.35
    pixel_kp_tilt: float = 0.30
    pixel_max_step_pan_deg: float = 2.5
    pixel_max_step_tilt_deg: float = 2.5
    pixel_deadband_px: float = 10.0
    pan_sign: float = 1.0
    wrist_tilt_sign: float = 1.0

    # Search (configurable scan pose + sinusoidal pan sweep).
    # NOTE on signs (SO-101): +wrist_flex tilts the gripper DOWN, so the
    # camera looks DOWN. To scan forward we want the gripper tilted UP,
    # which means a NEGATIVE absolute wrist_flex target. The parameter is
    # named ``search_wrist_flex_up_deg`` for historical reasons but it is
    # used as the absolute commanded value — set it negative to look up.
    # The shoulder_lift target is also absolute; less-negative values stand
    # the arm up more (negative = pitched forward toward the table).
    search_enabled: bool = True
    search_pan_amplitude_deg: float = 35.0
    search_pan_period_s: float = 6.0
    search_shoulder_lift_target_deg: float = -10.0
    search_wrist_flex_up_deg: float = -25.0
    search_pose_ramp_iters: int = 12

    # Servo-phase orbit: once the KF has locked, slowly sweep the approach
    # direction (elevation/azimuth) and/or standoff radius. ``n_hat`` and
    # standoff are recomputed each tick so the EE orbits the object on a
    # (near-)constant-radius sphere while look-at keeps the camera pointed
    # at it. Sweeps are centered on (approach_az_deg, approach_el_deg,
    # standoff_m). Set any amplitude to 0 to disable that axis.
    orbit_enabled: bool = False
    orbit_el_amp_deg: float = 30.0
    orbit_el_period_s: float = 12.0
    orbit_az_amp_deg: float = 0.0
    orbit_az_period_s: float = 18.0
    orbit_standoff_amp_m: float = 0.0
    orbit_standoff_period_s: float = 10.0
    orbit_warmup_s: float = 2.0

    # KF / depth
    kf_sigma_proc_pos_m: float = 0.005
    kf_sigma_proc_vel_m_s: float = 0.025
    kf_sigma_meas_m: float = 0.012
    kf_jump_reject_m: float = 0.20
    min_valid_depth_m: float = 0.03
    max_valid_depth_m: float = 0.80
    # Added along the camera ray (m) after any range estimate: fused stereo /
    # pinhole depth before back-projection, or ray–plane distance before the
    # point is pushed outward on the same ray. Typical use: OAK stereo reads
    # short vs. tape measure — try +0.12.
    depth_offset_m: float = 0.0
    lost_abort_frames: int = 80
    # Floor clamp for the back-projected object Z. None = don't clamp; let the
    # depth/back-projection speak for itself. Set to 0.0 if the robot base
    # plane and table top coincide and you want to reject below-table noise.
    target_base_z_min_m: float | None = None
    # Ray-plane localization. When set, we IGNORE depth and instead intersect
    # the camera ray through the bbox centroid with the horizontal plane
    # z=object_plane_z_m in base frame. Far more stable than depth at <35 cm
    # (where OAK-D stereo fails) — relies only on a well-calibrated
    # gripper_camera_tf and the ground-truth that the object sits at a known
    # table height. Set to ``None`` to fall back to the depth-fusion path.
    object_plane_z_m: float | None = 0.0
    # Reject 3D measurements with cam_z below this (OpenCV +Z = forward). Use a
    # small value (e.g. 5 mm): a 20 mm floor wrongly rejects valid close work
    # (~18 mm) and triggers lost→rotate-only while the object is still in view.
    meas_min_cam_z_m: float = 0.005

    # Display
    display_data: bool = False
    display_sim3d: bool = False
    log_every_n: int = 0


class ObjectKalmanFilter:
    """3D constant-velocity KF on object position in base frame."""

    def __init__(
        self,
        sigma_proc_pos_m: float,
        sigma_proc_vel_m_s: float,
        sigma_meas_m: float,
        jump_reject_m: float,
    ) -> None:
        self.sp = float(sigma_proc_pos_m)
        self.sv = float(sigma_proc_vel_m_s)
        self.sm = float(sigma_meas_m)
        self.jump = float(jump_reject_m)
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.initialized = False

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        dt = float(max(0.0, dt))
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        Q = np.diag(
            [self.sp**2, self.sp**2, self.sp**2, self.sv**2, self.sv**2, self.sv**2]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray) -> bool:
        z = np.asarray(z, dtype=np.float64).reshape(3)
        if not self.initialized:
            self.x = np.concatenate([z, np.zeros(3, dtype=np.float64)])
            self.P = np.diag([0.01**2] * 3 + [0.05**2] * 3).astype(np.float64)
            self.initialized = True
            return True
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, :3] = np.eye(3)
        R = (self.sm**2) * np.eye(3, dtype=np.float64)
        y = z - H @ self.x
        if float(np.linalg.norm(y)) > self.jump:
            return False
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ H) @ self.P
        return True


def approach_unit_vector(approach_az_deg: float, approach_el_deg: float) -> np.ndarray:
    """Unit vector FROM object TO EE in base frame.

    el=+90 -> (0, 0, +1)         top-down (EE above object, camera looks down)
    el=  0, az=0 -> (-1, 0, 0)   side approach (EE between base and object)
    az=+90, el=0 -> (0, +1, 0)   approach from object's +Y side
    """
    az = math.radians(float(approach_az_deg))
    el = math.radians(float(approach_el_deg))
    horiz = math.cos(el)
    return np.array(
        [-horiz * math.cos(az), horiz * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )


def build_look_at_R(
    R_base_ee_cur: np.ndarray,
    T_ee_cam: np.ndarray,
    optical_axis_target_base: np.ndarray,
) -> np.ndarray:
    """Solve for R_base_ee that aligns the camera optical axis (cam +Z) with
    ``optical_axis_target_base``. Wrist roll is left free — required on a 5-DoF
    arm without an independent roll joint.
    """
    R_ee_cam = np.asarray(T_ee_cam, dtype=np.float64)[:3, :3]
    R_base_cam_cur = np.asarray(R_base_ee_cur, dtype=np.float64) @ R_ee_cam

    z_target = np.asarray(optical_axis_target_base, dtype=np.float64).reshape(3)
    zn = float(np.linalg.norm(z_target))
    z_target = (z_target / zn) if zn > 1e-9 else np.array([0.0, 0.0, -1.0])

    x_cur = R_base_cam_cur[:, 0]
    x_proj = x_cur - float(np.dot(x_cur, z_target)) * z_target
    n = float(np.linalg.norm(x_proj))
    if n < 1e-6:
        for cand in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
            x_proj = cand - float(np.dot(cand, z_target)) * z_target
            n = float(np.linalg.norm(x_proj))
            if n >= 1e-6:
                break
    x_target = x_proj / n
    y_target = np.cross(z_target, x_target)
    R_base_cam_target = np.column_stack([x_target, y_target, z_target])
    return R_base_cam_target @ R_ee_cam.T


def _fuse_depth(
    depth_map: np.ndarray | None,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    fx: float,
    fy: float,
    depth_scale: float,
    target_physical_size_m: float,
    min_valid_depth_m: float,
    max_valid_depth_m: float,
) -> float | None:
    d_stereo: float | None = None
    if depth_map is not None:
        try:
            d_stereo = median_depth_m(
                np.asarray(depth_map),
                bbox_xyxy,
                depth_scale=float(depth_scale),
                min_mm=float(min_valid_depth_m) * 1000.0,
                max_mm=float(max_valid_depth_m) * 1000.0,
            )
        except Exception:
            d_stereo = None
    if d_stereo is not None and min_valid_depth_m <= d_stereo <= max_valid_depth_m:
        return float(d_stereo)
    d_pinhole = depth_from_bbox_size(
        bbox_xyxy,
        fx=float(fx),
        fy=float(fy),
        target_physical_size_m=float(target_physical_size_m),
    )
    if d_pinhole is not None and min_valid_depth_m <= d_pinhole <= max_valid_depth_m:
        return float(d_pinhole)
    return None


def _project_pixel_to_plane(
    T_base_cam: np.ndarray,
    *,
    u: float,
    v_pix: float,
    fx: float,
    fy: float,
    cx0: float,
    cy0: float,
    plane_z_m: float,
) -> np.ndarray | None:
    """Intersect the camera ray through pixel (u, v_pix) with the horizontal
    plane z = plane_z_m in base frame. Returns None if the plane is parallel
    to the ray or behind the camera.

    Independent of any depth measurement — relies only on the camera intrinsics
    and the (extrinsic) gripper_camera_tf via ``T_base_cam``.
    """
    Tbc = np.asarray(T_base_cam, dtype=np.float64)
    R = Tbc[:3, :3]
    t = Tbc[:3, 3]
    d_cam = np.array(
        [
            (float(u) - float(cx0)) / max(float(fx), 1e-6),
            (float(v_pix) - float(cy0)) / max(float(fy), 1e-6),
            1.0,
        ],
        dtype=np.float64,
    )
    d_base = R @ d_cam
    if abs(float(d_base[2])) < 1e-9:
        return None
    s = (float(plane_z_m) - float(t[2])) / float(d_base[2])
    if s <= 1e-6:
        return None
    return t + s * d_base


def _se3_rate_limited_step(
    T_start: np.ndarray,
    T_target: np.ndarray,
    *,
    dt: float,
    max_lin_vel_m_s: float,
    max_ang_vel_deg_s: float,
) -> np.ndarray:
    """LERP+SLERP from T_start toward T_target, capped by per-tick velocity."""
    d_t = float(np.linalg.norm(T_target[:3, 3] - T_start[:3, 3]))
    R_rel = T_target[:3, :3] @ T_start[:3, :3].T
    d_r_rad = float(np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec()))
    alpha_t = 1.0 if d_t < 1e-9 else min(1.0, float(max_lin_vel_m_s) * dt / d_t)
    alpha_r = (
        1.0
        if d_r_rad < 1e-6
        else min(1.0, math.radians(float(max_ang_vel_deg_s)) * dt / d_r_rad)
    )
    alpha = min(alpha_t, alpha_r, 1.0)
    return interpolate_se3(T_start, T_target, alpha)


def _bearing(p_obj_base, T_base_cam, T_base_ee):
    if p_obj_base is None or T_base_cam is None:
        return None, None, None, None
    Tbc = np.asarray(T_base_cam, dtype=np.float64)
    p_cam = Tbc[:3, :3].T @ (np.asarray(p_obj_base, dtype=np.float64) - Tbc[:3, 3])
    x_c, y_c, z_c = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    z_safe = z_c if abs(z_c) > 1e-9 else (1e-9 if z_c >= 0 else -1e-9)
    az = math.degrees(math.atan2(x_c, z_safe))
    el = math.degrees(math.atan2(-y_c, z_safe))
    d_cam = float(np.linalg.norm(p_cam))
    d_ee = None
    if T_base_ee is not None:
        d_ee = float(
            np.linalg.norm(
                np.asarray(p_obj_base) - np.asarray(T_base_ee, dtype=np.float64)[:3, 3]
            )
        )
    return az, el, d_cam, d_ee


def _try_init_rerun(cfg: CVSEngineConfig) -> bool:
    if not (cfg.display_data or cfg.display_sim3d):
        return False
    try:
        from lerobot.utils.visualization_utils import init_rerun, send_agentic_rerun_blueprint

        init_rerun(session_name="cvs_engine")
        send_agentic_rerun_blueprint(
            show_camera_stream=bool(cfg.display_data),
            show_sim3d=bool(cfg.display_sim3d),
            camera_key=cfg.camera_key,
        )
        return True
    except Exception as e:
        logger.warning("[cvs-engine] rerun init failed: %s", e)
        return False


def _rerun_log(
    *,
    cfg: CVSEngineConfig,
    frame: int,
    rgb,
    depth,
    bbox_xyxy,
    uv,
    cx0,
    cy0,
    conf,
    phase,
    depth_m,
    kin,
    joints_deg,
    T_base_ee,
    T_base_cam,
    p_obj_base,
) -> None:
    if rgb is None or not (cfg.display_data or cfg.display_sim3d):
        return
    try:
        from lerobot.manipulation.yolo_track.rerun_viz import log_rerun_iter
    except Exception:
        return
    try:
        az, el, d_cam, d_ee = _bearing(p_obj_base, T_base_cam, T_base_ee)
        log_rerun_iter(
            frame=int(frame),
            camera_key=cfg.camera_key,
            rgb=np.asarray(rgb),
            depth=depth,
            bbox_xyxy=bbox_xyxy,
            bbox_center_raw=uv,
            bbox_center_smoothed=uv,
            cx0=float(cx0),
            cy0=float(cy0),
            conf=conf,
            phase=phase,
            z_ee=float(T_base_ee[2, 3]) if T_base_ee is not None else None,
            depth_m=depth_m,
            kinematics=kin,
            joints_deg=np.asarray(joints_deg, dtype=np.float64),
            T_base_ee=T_base_ee,
            T_base_cam=T_base_cam,
            p_target_base=p_obj_base,
            ee_trail=[],
            object_half_size_m=0.015,
            show_sim3d=bool(cfg.display_sim3d),
            show_camera=bool(cfg.display_data),
            object_semantic_label=str(cfg.query) or None,
            bearing_az_deg=az,
            bearing_el_deg=el,
        )
        try:
            import rerun as rr

            if az is not None:
                rr.log("scalars/bearing_az_deg", rr.Scalars(float(az)))
            if el is not None:
                rr.log("scalars/bearing_el_deg", rr.Scalars(float(el)))
            if d_cam is not None:
                rr.log("scalars/dist_cam_obj_m", rr.Scalars(float(d_cam)))
            if d_ee is not None:
                rr.log("scalars/dist_ee_obj_m", rr.Scalars(float(d_ee)))
        except Exception:
            pass
    except Exception as e:
        logger.debug("[cvs-engine] rerun log failed: %s", e)


def _search_command(
    *,
    seed: np.ndarray,
    elapsed_s: float,
    ramp_alpha: float,
    pan_amp_deg: float,
    pan_period_s: float,
    lift_target_deg: float,
    wrist_up_deg: float,
) -> np.ndarray:
    target = seed.copy()
    if pan_period_s > 1e-3:
        omega = 2.0 * math.pi / float(pan_period_s)
        target[0] = seed[0] + float(pan_amp_deg) * math.sin(omega * float(elapsed_s))
    target[1] = float(lift_target_deg)
    target[3] = float(wrist_up_deg)
    a = float(np.clip(ramp_alpha, 0.0, 1.0))
    return (1.0 - a) * seed + a * target


def _pixel_correction(
    *,
    uv: tuple[float, float],
    cx0: float,
    cy0: float,
    fx: float,
    fy: float,
    cfg: CVSEngineConfig,
) -> tuple[float, float]:
    du = float(uv[0]) - float(cx0)
    dv = float(uv[1]) - float(cy0)
    if abs(du) > float(cfg.pixel_deadband_px):
        d_pan_deg = math.degrees(math.atan2(du, max(1e-6, fx)))
        delta_pan = float(
            np.clip(
                float(cfg.pan_sign) * float(cfg.pixel_kp_pan) * d_pan_deg,
                -float(cfg.pixel_max_step_pan_deg),
                +float(cfg.pixel_max_step_pan_deg),
            )
        )
    else:
        delta_pan = 0.0
    if abs(dv) > float(cfg.pixel_deadband_px):
        d_tilt_deg = math.degrees(math.atan2(dv, max(1e-6, fy)))
        delta_tilt = float(
            np.clip(
                float(cfg.wrist_tilt_sign) * float(cfg.pixel_kp_tilt) * d_tilt_deg,
                -float(cfg.pixel_max_step_tilt_deg),
                +float(cfg.pixel_max_step_tilt_deg),
            )
        )
    else:
        delta_tilt = 0.0
    return delta_pan, delta_tilt


@parser.wrap()
def cvs_engine_main(cfg: CVSEngineConfig) -> None:
    run_cvs_engine(cfg)


def run_cvs_engine(cfg: CVSEngineConfig) -> None:
    init_logging(console_level=os.environ.get("LEROBOT_LOG_LEVEL", "INFO"))
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.perception.yolo_world import YoloWorldDetector

    kin = RobotKinematics(
        urdf_path=cfg.urdf, target_frame_name=cfg.ee_frame, joint_names=ARM_MOTORS
    )
    detector = YoloWorldDetector(cfg.model_path)
    detector.set_query(cfg.query)
    T_ee_cam = parse_tf_string(cfg.gripper_camera_tf)
    if bool(cfg.invert_gripper_camera_tf):
        T_ee_cam = np.linalg.inv(np.asarray(T_ee_cam, dtype=np.float64))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    rerun_enabled = _try_init_rerun(cfg)

    intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0, "depth_scale": 0.001}
    cam = getattr(robot, "cameras", {}).get(cfg.camera_key)
    if cam is not None and hasattr(cam, "get_depth_intrinsics"):
        try:
            intrinsics = dict(cam.get_depth_intrinsics())
        except Exception as e:
            logger.warning("[cvs-engine] get_depth_intrinsics failed (%s); using fallback", e)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx0 = float(intrinsics["cx"])
    cy0 = float(intrinsics["cy"])
    depth_scale = float(intrinsics.get("depth_scale", 0.001))

    n_hat = approach_unit_vector(cfg.approach_az_deg, cfg.approach_el_deg)
    logger.info(
        "[cvs-engine] query=%r az=%.1f° el=%.1f° standoff=%.3fm n_hat=(%+.3f,%+.3f,%+.3f) "
        "loop_hz=%.1f rerun=%s",
        cfg.query,
        float(cfg.approach_az_deg),
        float(cfg.approach_el_deg),
        float(cfg.standoff_m),
        float(n_hat[0]),
        float(n_hat[1]),
        float(n_hat[2]),
        float(cfg.loop_hz),
        rerun_enabled,
    )
    logger.info(
        "[cvs-engine] intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f depth_scale=%.6f",
        fx,
        fy,
        cx0,
        cy0,
        depth_scale,
    )

    meas_reject_warned = False

    try:
        obs_init = robot.get_observation()
        joints_init = np.array(
            [float(obs_init[f"{m}.pos"]) for m in ARM_MOTORS], dtype=np.float64
        )
        T_base_ee_init = np.asarray(
            kin.forward_kinematics(joints_init), dtype=np.float64
        )
        T_base_cam_init = T_base_ee_init @ T_ee_cam
        logger.info(
            "[cvs-engine] calibration check (R_base_cam columns = OpenCV cam axes in base):"
        )
        logger.info(
            "  cam +X (image right)  = %s",
            np.round(T_base_cam_init[:3, 0], 3).tolist(),
        )
        logger.info(
            "  cam +Y (image down)   = %s",
            np.round(T_base_cam_init[:3, 1], 3).tolist(),
        )
        logger.info(
            "  cam +Z (optical axis) = %s",
            np.round(T_base_cam_init[:3, 2], 3).tolist(),
        )
        # Camera position in base frame — useful for sanity-checking
        # ray-plane depth: in object_plane mode, d_meas is essentially
        # (cam_z - object_plane_z) / |optical_axis · z_base|. If this z
        # disagrees with your physical tape-measure, the depth label in
        # Rerun will be off by the same factor.
        logger.info(
            "  cam position in base = %s (object_plane_z_m=%s depth_offset_m=%.3f)",
            np.round(T_base_cam_init[:3, 3], 3).tolist(),
            (
                f"{float(cfg.object_plane_z_m):.3f}m (ray-plane)"
                if cfg.object_plane_z_m is not None
                else "None (using OAK-D depth)"
            ),
            float(cfg.depth_offset_m),
        )
        logger.info(
            "[cvs-engine] gripper_camera_tf is camera(optical) → %s (OpenCV +Z forward). "
            "If Rerun places the object behind the arm while the image shows it in front, "
            "fix the transform or try --invert-gripper-camera-tf=true.",
            cfg.ee_frame,
        )
    except Exception as e:
        logger.warning("[cvs-engine] calibration check skipped: %s", e)

    kf = ObjectKalmanFilter(
        sigma_proc_pos_m=float(cfg.kf_sigma_proc_pos_m),
        sigma_proc_vel_m_s=float(cfg.kf_sigma_proc_vel_m_s),
        sigma_meas_m=float(cfg.kf_sigma_meas_m),
        jump_reject_m=float(cfg.kf_jump_reject_m),
    )

    dt_target = 1.0 / max(1.0, float(cfg.loop_hz))
    log_every_n = (
        max(1, int(cfg.log_every_n))
        if int(cfg.log_every_n) > 0
        else max(1, int(round(float(cfg.loop_hz))))
    )

    last_t = time.time()
    tick = 0
    lost = 0
    consecutive_misses = 0
    last_phase: str | None = None
    last_p_obj_base: np.ndarray | None = None
    search_seed: np.ndarray | None = None
    search_t0: float | None = None
    search_ramp = 0
    servo_t0: float | None = None
    servo_mode: str = "pbvs"  # hysteretic state for PBVS↔IBVS handoff
    i_pan = ARM_MOTORS.index("shoulder_pan")
    i_tilt = ARM_MOTORS.index("wrist_flex")

    try:
        while True:
            loop_t = time.time()
            dt = max(1e-3, loop_t - last_t)
            last_t = loop_t
            tick += 1

            obs = robot.get_observation()
            rgb = obs.get(cfg.camera_key)
            depth = obs.get(f"{cfg.camera_key}_depth")
            if rgb is None:
                _sleep(loop_t, dt_target)
                continue
            joints = np.array(
                [float(obs[f"{m}.pos"]) for m in ARM_MOTORS], dtype=np.float64
            )

            # ---------- Detect ----------
            det = detector.best_detection(np.asarray(rgb))
            bbox_xyxy: tuple[float, float, float, float] | None = None
            uv: tuple[float, float] | None = None
            if det is not None:
                bbox_xyxy = (
                    float(det.xyxy[0]),
                    float(det.xyxy[1]),
                    float(det.xyxy[2]),
                    float(det.xyxy[3]),
                )
                uv = (
                    0.5 * (bbox_xyxy[0] + bbox_xyxy[2]),
                    0.5 * (bbox_xyxy[1] + bbox_xyxy[3]),
                )

            kf.predict(dt)

            measured = False
            d_meas: float | None = None
            p_meas_base: np.ndarray | None = None
            geom_z_reject: bool = False
            T_base_ee_cur = np.asarray(kin.forward_kinematics(joints), dtype=np.float64)
            T_base_cam_cur = T_base_ee_cur @ T_ee_cam

            if det is not None and bbox_xyxy is not None and uv is not None:
                if cfg.object_plane_z_m is not None:
                    # Ray-plane localization: ignore depth, intersect the
                    # camera ray with the known horizontal object plane.
                    p_meas_base = _project_pixel_to_plane(
                        T_base_cam_cur,
                        u=float(uv[0]),
                        v_pix=float(uv[1]),
                        fx=fx,
                        fy=fy,
                        cx0=cx0,
                        cy0=cy0,
                        plane_z_m=float(cfg.object_plane_z_m),
                    )
                    if p_meas_base is not None:
                        cam_pos = np.asarray(T_base_cam_cur[:3, 3], dtype=np.float64)
                        ray = np.asarray(p_meas_base, dtype=np.float64) - cam_pos
                        d0 = float(np.linalg.norm(ray))
                        d_meas = d0
                        off = float(cfg.depth_offset_m)
                        if off != 0.0 and d0 > 1e-9:
                            d_meas = d0 + off
                            p_meas_base = cam_pos + (ray / d0) * d_meas
                else:
                    d_meas = _fuse_depth(
                        depth,
                        bbox_xyxy,
                        fx=fx,
                        fy=fy,
                        depth_scale=depth_scale,
                        target_physical_size_m=float(cfg.target_physical_size_m),
                        min_valid_depth_m=float(cfg.min_valid_depth_m),
                        max_valid_depth_m=float(cfg.max_valid_depth_m),
                    )
                    if d_meas is not None:
                        d_meas = float(d_meas) + float(cfg.depth_offset_m)
                        p_meas_base = point_cam_to_base(
                            T_base_cam_cur,
                            u=float(uv[0]),
                            v_pix=float(uv[1]),
                            depth_m=float(d_meas),
                            fx=fx,
                            fy=fy,
                            cx0=cx0,
                            cy0=cy0,
                        )
                        if cfg.target_base_z_min_m is not None and math.isfinite(
                            float(cfg.target_base_z_min_m)
                        ):
                            p_meas_base = np.asarray(
                                p_meas_base, dtype=np.float64
                            ).copy()
                            p_meas_base[2] = max(
                                float(p_meas_base[2]),
                                float(cfg.target_base_z_min_m),
                            )
                if p_meas_base is not None:
                    z_cam_m = cam_z_of_base_point(T_base_cam_cur, p_meas_base)
                    if z_cam_m < float(cfg.meas_min_cam_z_m):
                        geom_z_reject = True
                        p_meas_base = None
                        d_meas = None
                        if not meas_reject_warned and z_cam_m <= 0.0:
                            meas_reject_warned = True
                            logger.warning(
                                "[cvs-engine] rejected 3D measurement: point on or behind the "
                                "camera optical plane (cam_z=%.4f m). Check gripper_camera_tf "
                                "or --object-plane-z-m; try --invert-gripper-camera-tf=true.",
                                z_cam_m,
                            )
                if p_meas_base is not None and kf.update(p_meas_base):
                    measured = True

            # ---------- SEARCH (KF never initialized yet) ----------
            if not kf.initialized:
                if not bool(cfg.search_enabled):
                    _sleep(loop_t, dt_target)
                    continue
                servo_t0 = None
                servo_mode = "pbvs"
                if search_seed is None:
                    search_seed = joints.copy()
                    search_t0 = loop_t
                    search_ramp = 0
                search_ramp += 1
                ramp_alpha = min(
                    1.0, float(search_ramp) / max(1.0, float(cfg.search_pose_ramp_iters))
                )
                elapsed = float(loop_t - (search_t0 or loop_t))
                q_search = _search_command(
                    seed=search_seed,
                    elapsed_s=elapsed,
                    ramp_alpha=ramp_alpha,
                    pan_amp_deg=float(cfg.search_pan_amplitude_deg),
                    pan_period_s=float(cfg.search_pan_period_s),
                    lift_target_deg=float(cfg.search_shoulder_lift_target_deg),
                    wrist_up_deg=float(cfg.search_wrist_flex_up_deg),
                )
                act = {f"{m}.pos": float(q_search[i]) for i, m in enumerate(ARM_MOTORS)}
                if "gripper.pos" in obs:
                    act["gripper.pos"] = float(obs["gripper.pos"])
                try:
                    robot.send_action(act)
                except Exception as e:
                    logger.warning("[cvs-engine] send_action failed (search): %s", e)
                if last_phase != "search":
                    last_phase = "search"
                    logger.info("[cvs-engine] phase=search ramp=%.2f", ramp_alpha)
                _rerun_log(
                    cfg=cfg,
                    frame=tick,
                    rgb=rgb,
                    depth=depth,
                    bbox_xyxy=bbox_xyxy,
                    uv=uv,
                    cx0=cx0,
                    cy0=cy0,
                    conf=float(getattr(det, "confidence", 0.0)) if det is not None else None,
                    phase="search",
                    depth_m=d_meas,
                    kin=kin,
                    joints_deg=joints,
                    T_base_ee=T_base_ee_cur,
                    T_base_cam=T_base_cam_cur,
                    p_obj_base=None,
                )
                _sleep(loop_t, dt_target)
                continue

            # KF was initialized at some point — clear search state.
            search_seed = None
            search_t0 = None
            search_ramp = 0

            # ---------- MISSED MEASUREMENT HANDLING ----------
            # Brief flickers (consecutive_misses < tolerance) fall through to
            # servo using the KF-predicted position. Sustained drops switch to
            # rotate-only recovery (camera tracks last known cube; EE held).
            # If YOLO still sees the target but only the cam_z gate dropped the
            # update, do not count that as a miss — avoids lost→rotate-away while
            # the bbox is still visible (e.g. user-tight meas_min_cam_z_m).
            det_ok = (
                det is not None and bbox_xyxy is not None and uv is not None
            )
            visual_miss = det_ok is False or (
                not measured and not geom_z_reject
            )
            if visual_miss:
                consecutive_misses += 1
                lost += 1
                if lost >= int(cfg.lost_abort_frames):
                    logger.error(
                        "[cvs-engine] target lost for %d frames; exiting.", lost
                    )
                    break

                if (
                    consecutive_misses >= int(cfg.miss_tolerance_frames)
                    and last_p_obj_base is not None
                ):
                    servo_t0 = None
                    servo_mode = "pbvs"
                    cam_pos = T_base_cam_cur[:3, 3]
                    look_dir = last_p_obj_base - cam_pos
                    ld_norm = float(np.linalg.norm(look_dir))
                    optical_axis = (
                        look_dir / ld_norm
                        if ld_norm > 1e-3
                        else -approach_unit_vector(cfg.approach_az_deg, cfg.approach_el_deg)
                    )
                    R_target = build_look_at_R(
                        T_base_ee_cur[:3, :3], T_ee_cam, optical_axis
                    )
                    T_target = np.eye(4, dtype=np.float64)
                    T_target[:3, :3] = R_target
                    T_target[:3, 3] = T_base_ee_cur[:3, 3]
                    T_step = _se3_rate_limited_step(
                        T_base_ee_cur,
                        T_target,
                        dt=dt_target,
                        max_lin_vel_m_s=float(cfg.max_lin_vel_m_s),
                        max_ang_vel_deg_s=float(cfg.max_ang_vel_deg_s),
                    )
                    try:
                        q_ik = kin.inverse_kinematics(
                            joints,
                            T_step,
                            position_weight=float(cfg.ik_position_weight),
                            orientation_weight=float(cfg.ik_orientation_weight),
                        )
                        act = {
                            f"{m}.pos": float(q_ik[i]) for i, m in enumerate(ARM_MOTORS)
                        }
                        if "gripper.pos" in obs:
                            act["gripper.pos"] = float(obs["gripper.pos"])
                        robot.send_action(act)
                    except Exception as e:
                        logger.warning("[cvs-engine] lost-recovery IK/send failed: %s", e)
                    if last_phase != "lost":
                        last_phase = "lost"
                        logger.info(
                            "[cvs-engine] phase=lost consecutive_misses=%d (rotate-only)",
                            int(consecutive_misses),
                        )
                    _rerun_log(
                        cfg=cfg,
                        frame=tick,
                        rgb=rgb,
                        depth=depth,
                        bbox_xyxy=bbox_xyxy,
                        uv=uv,
                        cx0=cx0,
                        cy0=cy0,
                        conf=float(getattr(det, "confidence", 0.0)) if det is not None else None,
                        phase="lost",
                        depth_m=d_meas,
                        kin=kin,
                        joints_deg=joints,
                        T_base_ee=T_base_ee_cur,
                        T_base_cam=T_base_cam_cur,
                        p_obj_base=kf.position,
                    )
                    _sleep(loop_t, dt_target)
                    continue
                # else: fall through to servo using KF-predicted position
            else:
                consecutive_misses = 0
                lost = 0

            # ---------- SERVO (kinematics is direct: look-at + standoff) ----------
            p_obj_base = kf.position
            if measured:
                last_p_obj_base = p_obj_base.copy()

            if servo_t0 is None:
                servo_t0 = loop_t

            # Live approach vector: optionally orbit (az/el/standoff) around
            # the object while continuing to look at it. Disabled by default.
            az_live = float(cfg.approach_az_deg)
            el_live = float(cfg.approach_el_deg)
            standoff_live = float(cfg.standoff_m)
            if bool(cfg.orbit_enabled):
                t_orbit = (loop_t - servo_t0) - float(cfg.orbit_warmup_s)
                if t_orbit > 0.0:
                    if (
                        float(cfg.orbit_el_amp_deg) != 0.0
                        and float(cfg.orbit_el_period_s) > 1e-3
                    ):
                        el_live += float(cfg.orbit_el_amp_deg) * math.sin(
                            2.0 * math.pi * t_orbit / float(cfg.orbit_el_period_s)
                        )
                    if (
                        float(cfg.orbit_az_amp_deg) != 0.0
                        and float(cfg.orbit_az_period_s) > 1e-3
                    ):
                        az_live += float(cfg.orbit_az_amp_deg) * math.sin(
                            2.0 * math.pi * t_orbit / float(cfg.orbit_az_period_s)
                        )
                    if (
                        float(cfg.orbit_standoff_amp_m) != 0.0
                        and float(cfg.orbit_standoff_period_s) > 1e-3
                    ):
                        standoff_live += float(cfg.orbit_standoff_amp_m) * math.sin(
                            2.0 * math.pi * t_orbit / float(cfg.orbit_standoff_period_s)
                        )
            n_hat_live = approach_unit_vector(az_live, el_live)

            # ---- PBVS ↔ IBVS handoff -------------------------------------
            # bbox-size pinhole depth: ``d_bbox = f · S_real / W_px``. This is
            # the only depth signal we trust inside ~30 cm (OAK-D stereo
            # blind, ray-plane sensitive to plane-Z mis-calibration). When it
            # says we're close, switch to the IBVS final phase: approach
            # along the *current* optical axis at ``ibvs_final_standoff_m``
            # using a bbox-size-back-projected p_obj, while pixel feedback
            # keeps the cube centered. Hysteresis prevents thrashing across
            # the threshold.
            d_bbox: float | None = None
            if bbox_xyxy is not None:
                d_bbox = depth_from_bbox_size(
                    bbox_xyxy,
                    fx=fx,
                    fy=fy,
                    target_physical_size_m=float(cfg.target_physical_size_m),
                )
            d_for_mode = (
                float(d_bbox)
                if d_bbox is not None
                else (float(d_meas) if d_meas is not None else None)
            )
            if bool(cfg.ibvs_enabled) and d_for_mode is not None:
                handoff = float(cfg.ibvs_handoff_distance_m)
                hyst = float(cfg.ibvs_handoff_hysteresis_m)
                upper = handoff + hyst  # leave IBVS when crossing this going out
                lower = handoff         # enter IBVS when crossing this going in
                if servo_mode == "ibvs":
                    if d_for_mode > upper:
                        servo_mode = "pbvs"
                else:
                    if d_for_mode <= lower:
                        servo_mode = "ibvs"
            else:
                servo_mode = "pbvs"

            cam_pos = T_base_cam_cur[:3, 3]

            if servo_mode == "ibvs" and d_bbox is not None and uv is not None:
                # IBVS branch: trust bbox-size depth + line-of-sight approach.
                p_obj_for_look = point_cam_to_base(
                    T_base_cam_cur,
                    u=float(uv[0]),
                    v_pix=float(uv[1]),
                    depth_m=float(d_bbox),
                    fx=fx,
                    fy=fy,
                    cx0=cx0,
                    cy0=cy0,
                )
                look_dir = np.asarray(p_obj_for_look, dtype=np.float64) - cam_pos
                ld_norm = float(np.linalg.norm(look_dir))
                if ld_norm > 1e-3:
                    optical_axis_target = look_dir / ld_norm
                else:
                    optical_axis_target = T_base_cam_cur[:3, 2]
                # Position target: along the line of sight, stop at the
                # configured close-range standoff. Equivalent to
                # ``p_obj − optical_axis · ibvs_final_standoff_m``.
                p_target_base = (
                    np.asarray(p_obj_for_look, dtype=np.float64)
                    - optical_axis_target * float(cfg.ibvs_final_standoff_m)
                )
            else:
                # PBVS branch: KF-fused p_obj + fixed (az, el) approach.
                p_obj_for_look = p_obj_base
                p_target_base = p_obj_base + n_hat_live * standoff_live
                look_dir = p_obj_base - cam_pos
                ld_norm = float(np.linalg.norm(look_dir))
                optical_axis_target = (
                    look_dir / ld_norm if ld_norm > 1e-3 else -n_hat_live
                )

            R_target = build_look_at_R(T_base_ee_cur[:3, :3], T_ee_cam, optical_axis_target)
            T_target = np.eye(4, dtype=np.float64)
            T_target[:3, :3] = R_target
            T_target[:3, 3] = p_target_base

            T_step = _se3_rate_limited_step(
                T_base_ee_cur,
                T_target,
                dt=dt_target,
                max_lin_vel_m_s=float(cfg.max_lin_vel_m_s),
                max_ang_vel_deg_s=float(cfg.max_ang_vel_deg_s),
            )

            try:
                q_target = kin.inverse_kinematics(
                    joints,
                    T_step,
                    position_weight=float(cfg.ik_position_weight),
                    orientation_weight=float(cfg.ik_orientation_weight),
                )
            except Exception as e:
                logger.warning("[cvs-engine] IK failed (skip): %s", e)
                _sleep(loop_t, dt_target)
                continue
            q_target = np.asarray(q_target, dtype=np.float64).copy()

            if bool(cfg.pixel_feedback_enabled) and uv is not None:
                d_pan, d_tilt = _pixel_correction(
                    uv=uv, cx0=cx0, cy0=cy0, fx=fx, fy=fy, cfg=cfg
                )
                q_target[i_pan] += d_pan
                q_target[i_tilt] += d_tilt

            act = {f"{m}.pos": float(q_target[i]) for i, m in enumerate(ARM_MOTORS)}
            if "gripper.pos" in obs:
                act["gripper.pos"] = float(obs["gripper.pos"])
            try:
                robot.send_action(act)
            except Exception as e:
                logger.warning("[cvs-engine] send_action failed: %s", e)

            phase_label = f"servo[{servo_mode}]"
            if last_phase != phase_label:
                last_phase = phase_label
                logger.info(
                    "[cvs-engine] phase=%s (KF locked, approaching)", phase_label
                )
            if tick % log_every_n == 0:
                az, el, d_cam, d_ee = _bearing(p_obj_base, T_base_cam_cur, T_base_ee_cur)
                logger.info(
                    "[cvs-engine] tick=%d mode=%s bearing=(az=%+.1f°,el=%+.1f°) "
                    "d_cam_obj=%.3fm d_ee_obj=%.3fm d_meas=%.3fm d_bbox=%.3fm "
                    "p_cam=(%.3f,%.3f,%.3f) p_obj=(%.3f,%.3f,%.3f) "
                    "p_target=(%.3f,%.3f,%.3f)",
                    int(tick),
                    servo_mode,
                    az if az is not None else float("nan"),
                    el if el is not None else float("nan"),
                    d_cam if d_cam is not None else float("nan"),
                    d_ee if d_ee is not None else float("nan"),
                    float(d_meas) if d_meas is not None else float("nan"),
                    float(d_bbox) if d_bbox is not None else float("nan"),
                    float(T_base_cam_cur[0, 3]),
                    float(T_base_cam_cur[1, 3]),
                    float(T_base_cam_cur[2, 3]),
                    float(p_obj_base[0]),
                    float(p_obj_base[1]),
                    float(p_obj_base[2]),
                    float(p_target_base[0]),
                    float(p_target_base[1]),
                    float(p_target_base[2]),
                )

            _rerun_log(
                cfg=cfg,
                frame=tick,
                rgb=rgb,
                depth=depth,
                bbox_xyxy=bbox_xyxy,
                uv=uv,
                cx0=cx0,
                cy0=cy0,
                conf=float(getattr(det, "confidence", 0.0)) if det is not None else None,
                phase="servo",
                depth_m=d_meas,
                kin=kin,
                joints_deg=joints,
                T_base_ee=T_base_ee_cur,
                T_base_cam=T_base_cam_cur,
                p_obj_base=p_obj_base,
            )

            _sleep(loop_t, dt_target)
    except KeyboardInterrupt:
        logger.info("[cvs-engine] interrupted by user.")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


def _sleep(loop_t: float, dt_target: float) -> None:
    rem = dt_target - (time.time() - loop_t)
    if rem > 0:
        time.sleep(rem)
