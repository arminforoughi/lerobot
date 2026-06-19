# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Estimate-then-plan grasp engine for SO-101 + OAK-D eye-in-hand.

Physics-first rewrite of the reactive gaze/pbvs engines. The failure mode of
those engines was: they chase *pixels* tick-by-tick, so the instant the arm
moves enough that the object leaves the frame (gaze tilt saturates, or the
descend path sweeps it out the bottom) the feedback dies and the controller
drives blind. "Saw it, looked down, missed it."

This engine never chases pixels. It maintains the object as a **filtered 3D
point in the base frame**:

  ESTIMATE  Detect bbox -> back-project the bottom-center pixel onto the known
            object-height plane (ray ∩ plane). This needs only intrinsics +
            extrinsics + the surface height, so it is immune to OAK-D's garbage
            close-range stereo depth. The 3D point is EMA-filtered in the world
            frame, so arm motion (which cannot move a world point) does not
            disturb it, and a few dropped detections do not blind us.

  PLAN      Each tick, solve ONE full-pose IK: place the gripper tip at a
            standoff from the object along the chosen approach direction, with
            the camera optical axis pointed AT the object (look-at). The aim is
            baked into the IK orientation, so there is no per-step tilt cap to
            saturate. Rate-limit the joint step. The camera stays on the object
            for the whole approach by construction.

  GRASP     When the tip is within the trigger range, hand off to the shared
            ``run_grasp_sequence`` (open -> inch along optical axis -> current-
            sensed close -> lift confirm).

States: SEARCH -> APPROACH -> GRASP -> DONE. That is the whole machine.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.manipulation.visual_servo.cvs_engine import (
    _project_pixel_to_plane,
    approach_unit_vector,
    build_look_at_R,
)
from lerobot.manipulation.home_position import HomeFoldController, home_settings_from_config
from lerobot.manipulation.visual_servo.grasp_close import (
    GraspCloseConfig,
    run_grasp_sequence,
)
from lerobot.manipulation.yolo_track.depth import depth_from_bbox_size, median_depth_m
from lerobot.manipulation.yolo_track.math_utils import parse_tf_string
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.motion_executor import MotionExecutionConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@dataclass
class LookAtEngineConfig:
    # --- Robot / camera ---
    robot: RobotConfig = field(
        default_factory=lambda: SOFollowerRobotConfig(port="", cameras={})
    )
    urdf: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    camera_key: str = "front"
    # Camera pose in the EE frame: "x,y,z,rx,ry,rz" (rotvec radians).
    gripper_camera_tf: str = "0.04,0,0.09,-0.2690,0.2824,-1.6014"
    invert_gripper_camera_tf: bool = False
    # Fine pitch trim about the camera's own +X (image right); +deg tilts cam +Z down.
    gripper_camera_pitch_trim_deg: float = 0.0
    # Tip point along EE +Z (where the fingers close), metres.
    gripper_tip_offset_m: float = 0.10

    # --- Detection ---
    query: str = "red cube"
    model_path: str = "yolov8s-worldv2.pt"
    target_physical_size_m: float = 0.03
    min_detection_confidence: float = 0.20
    # Pixel row inside the bbox used as the tracked point: 0=top, 0.5=centre, 1=bottom.
    # MUST be 0.5 for an accurate 3D estimate — back-projecting any other row gives a
    # point off the object centre, and (critically) that offset GROWS as the object
    # gets closer/bigger, dragging the height estimate and inflating the centering
    # error. The gripper-tip-vs-camera offset already makes the object sit low in the
    # frame at grasp range; we do not need to bias the tracked pixel for that.
    bbox_grasp_v_frac: float = 0.5

    # --- Object world model ---
    # Known height of the object's centre above the robot base (m). If NaN, it is
    # auto-seeded from the first few clean (fully-in-frame) depth readings, then
    # LOCKED and used purely via plane projection — depth is never trusted after
    # that, which is what stops the height from drifting as the object nears the
    # frame edge. Pin it explicitly for best accuracy.
    object_center_z_m: float = float("nan")
    # Frames used to median-seed object_z in auto mode (taken while the bbox is
    # comfortably inside the frame, where the size→depth cue is most reliable).
    object_z_seed_frames: int = 8
    # EMA on the 3D world position estimate (0..1, higher = more responsive).
    pos_ema_alpha: float = 0.35
    # Reject a new estimate that jumps more than this from the filtered one (m).
    pos_outlier_reject_m: float = 0.12
    # Depth validity window for seeding / fusion (m).
    min_valid_depth_m: float = 0.06
    max_valid_depth_m: float = 1.2

    # --- Approach geometry ---
    # Direction FROM object TO end-effector: el=90 top-down, el=0 side-on,
    # az rotates about the object's vertical axis (0 = base side).
    approach_az_deg: float = 0.0
    approach_el_deg: float = 45.0
    # Tip standoff we drive toward during APPROACH (m). Keep below the grasp
    # trigger so the servo keeps pushing in until the trigger fires.
    final_standoff_m: float = 0.05
    # Hand off to the grasp sequence when the tip is within this range (m).
    grasp_trigger_tip_m: float = 0.10
    # Pre-open the gripper during APPROACH once the tip is within
    # (grasp_trigger + this) of the object, so it never drives closed fingers
    # into the object and shoves it before the grasp sequence starts.
    grasp_preopen_margin_m: float = 0.06
    # Floor guard, expressed RELATIVE to the estimated object centre height so
    # it works regardless of where the table sits vs the base (it can be well
    # below z=0). The tip is never commanded below ``object_z - this`` (m) —
    # i.e. it may descend to the object and a hair below, but not plough the
    # table. Set large to effectively disable.
    tip_floor_below_object_m: float = 0.03
    # Absolute hard safety floor (m), applied on top of the relative one. Only
    # bites if the object_z estimate itself is garbage. Keep well below the table.
    tip_floor_abs_z_m: float = -0.30

    # --- Control ---
    loop_hz: float = 25.0
    max_lin_vel_m_s: float = 0.05
    max_joint_step_deg: float = 3.0
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.30
    # Object must be this centered in the image before APPROACH->GRASP (px).
    grasp_center_pixel_err_px: float = 45.0
    # Detections needed to leave SEARCH; misses tolerated mid-approach.
    acquire_frames: int = 3
    lost_grace_ticks: int = 40
    # SEARCH pan sweep amplitude / speed (deg, deg per tick).
    search_pan_amplitude_deg: float = 35.0
    search_pan_step_deg: float = 1.2

    # --- Grasp ---
    grasp_enable: bool = True
    grasp_open_pct: float = 100.0
    grasp_close_pct: float = 0.0
    grasp_final_approach_m: float = 0.04
    # The final inch is sized dynamically to cover the tip-to-object gap measured at
    # handoff (d_tip - grasp_final_gap_m), capped at grasp_max_inch_m, so it actually
    # reaches the object instead of closing on air a fixed 4 cm in.
    grasp_final_gap_m: float = 0.012
    grasp_max_inch_m: float = 0.16
    grasp_final_approach_along_optical: bool = True
    grasp_post_contact_squeeze_pct: float = 8.0
    grasp_contact_delta_current_counts: float = 40.0
    grasp_lift_confirm: bool = True
    grasp_hold_after: bool = True

    # --- Home / fold-back ---
    home_config_path: str = "SO101/so101_home.yaml"
    fold_home_on_interrupt: bool = True
    live_home_keypress: bool = True
    home_joint_step_deg: float = 1.5
    home_inter_step_sleep_s: float = 0.04

    # --- Misc ---
    comm_error_max_consecutive: int = 25
    display_data: bool = True
    display_sim3d: bool = True
    log_every_n: int = 25


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _gripper_tip_base(T_base_ee: np.ndarray, tip_offset_m: float) -> np.ndarray:
    """Fingertip position in base frame (offset along gripper +Z in EE frame)."""
    T = np.asarray(T_base_ee, dtype=np.float64)
    return T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, float(tip_offset_m)])


def _backproject_depth_base(
    T_base_cam: np.ndarray, u: float, v: float, d_m: float, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Back-project pixel (u,v) at range d_m into the base frame."""
    p_cam = np.array(
        [(float(u) - cx) / fx * d_m, (float(v) - cy) / fy * d_m, d_m, 1.0],
        dtype=np.float64,
    )
    return (np.asarray(T_base_cam, dtype=np.float64) @ p_cam)[:3]


def _grasp_cfg(cfg: LookAtEngineConfig, final_approach_m: float | None = None) -> GraspCloseConfig:
    return GraspCloseConfig(
        enable=bool(cfg.grasp_enable),
        open_pct=float(cfg.grasp_open_pct),
        close_pct=float(cfg.grasp_close_pct),
        final_approach_m=float(
            final_approach_m if final_approach_m is not None else cfg.grasp_final_approach_m
        ),
        final_approach_along_optical=bool(cfg.grasp_final_approach_along_optical),
        post_contact_squeeze_pct=float(cfg.grasp_post_contact_squeeze_pct),
        contact_delta_current_counts=float(cfg.grasp_contact_delta_current_counts),
        lift_confirm=bool(cfg.grasp_lift_confirm),
    )


def _bbox_inside_frame(bbox, w: int, h: int, margin_px: float = 6.0) -> bool:
    """True if the bbox is comfortably inside the image (not clipped by an edge).

    A bbox touching an edge is truncated, so its apparent size under-reads and the
    pinhole range cue is unreliable — exactly when we must NOT seed object_z.
    """
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    return (
        x0 > margin_px and y0 > margin_px
        and x1 < (w - margin_px) and y1 < (h - margin_px)
    )


def _parse_detection(
    det, min_conf: float, v_frac: float
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float] | None, float]:
    """Return (bbox_xyxy, grasp_uv, conf). grasp_uv uses bbox_grasp_v_frac for v."""
    if det is None:
        return None, None, 0.0
    conf = float(getattr(det, "confidence", 0.0))
    if conf < float(min_conf):
        return None, None, conf
    x0, y0, x1, y1 = (float(det.xyxy[0]), float(det.xyxy[1]), float(det.xyxy[2]), float(det.xyxy[3]))
    u = 0.5 * (x0 + x1)
    v = y0 + float(np.clip(v_frac, 0.0, 1.0)) * (y1 - y0)
    return (x0, y0, x1, y1), (u, v), conf


def _try_init_rerun(cfg: LookAtEngineConfig) -> bool:
    if not (cfg.display_data or cfg.display_sim3d):
        return False
    try:
        from lerobot.utils.visualization_utils import init_rerun, send_agentic_rerun_blueprint

        init_rerun(session_name="lookat_engine")
        send_agentic_rerun_blueprint(
            show_camera_stream=bool(cfg.display_data),
            show_sim3d=bool(cfg.display_sim3d),
            camera_key=cfg.camera_key,
        )
        return True
    except Exception as e:
        logger.warning("[lookat-engine] rerun init failed: %s", e)
        return False


def _rerun_log(cfg, *, frame, rgb, depth, bbox, uv, cx, cy, conf, phase, depth_m, kin, joints, T_base_ee, T_base_cam, p_obj):
    if rgb is None or not (cfg.display_data or cfg.display_sim3d):
        return
    try:
        from lerobot.manipulation.yolo_track.rerun_viz import log_rerun_iter

        log_rerun_iter(
            frame=int(frame),
            camera_key=cfg.camera_key,
            rgb=np.asarray(rgb),
            depth=depth,
            bbox_xyxy=bbox,
            bbox_center_raw=uv,
            bbox_center_smoothed=uv,
            cx0=float(cx),
            cy0=float(cy),
            conf=conf,
            phase=phase,
            z_ee=float(T_base_ee[2, 3]) if T_base_ee is not None else None,
            depth_m=depth_m,
            kinematics=kin,
            joints_deg=np.asarray(joints, dtype=np.float64),
            T_base_ee=T_base_ee,
            T_base_cam=T_base_cam,
            p_target_base=p_obj,
            ee_trail=[],
            object_half_size_m=0.5 * float(cfg.target_physical_size_m),
            show_sim3d=bool(cfg.display_sim3d),
            show_camera=bool(cfg.display_data),
            object_semantic_label=str(cfg.query).strip() or None,
        )
    except Exception as e:
        logger.debug("[lookat-engine] rerun log failed: %s", e)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _measure_object_position(
    *,
    cfg: LookAtEngineConfig,
    T_base_cam: np.ndarray,
    uv: tuple[float, float],
    bbox: tuple[float, float, float, float],
    depth_map,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    object_z: float | None,
    prefer_plane: bool,
) -> tuple[np.ndarray | None, float | None]:
    """Best single-shot 3D object position in the base frame, plus a depth estimate.

    ``prefer_plane`` (set when the user pins ``object_center_z_m``): intersect the
    grasp-pixel ray with the known-height plane — geometry only, the most stable and
    accurate XYZ when the height is trusted.

    Otherwise: back-project the grasp pixel at the **known-size pinhole depth**. For a
    known object (a 3 cm cube) the apparent bbox size is a reliable range cue and,
    unlike OAK-D stereo at close range (which reads the table behind the cube), it
    does not blow up to garbage. Stereo is a fallback when pinhole is unavailable;
    ray ∩ last-known plane is the final fallback so a dropped depth keeps the
    lateral estimate.

    ``d_meas`` (pinhole-preferred range, for logging / object_z tracking) is always
    returned when measurable, independent of which position source is used.
    """
    u, v = float(uv[0]), float(uv[1])
    lo, hi = float(cfg.min_valid_depth_m), float(cfg.max_valid_depth_m)

    d_pin = depth_from_bbox_size(
        bbox, fx=fx, fy=fy, target_physical_size_m=float(cfg.target_physical_size_m)
    )
    d_meas = float(d_pin) if (d_pin is not None and lo <= d_pin <= hi) else None
    if d_meas is None and depth_map is not None:
        try:
            d_st = median_depth_m(
                np.asarray(depth_map), bbox, depth_scale=float(depth_scale),
                min_mm=lo * 1000.0, max_mm=hi * 1000.0,
            )
        except Exception:
            d_st = None
        if d_st is not None and lo <= d_st <= hi:
            d_meas = float(d_st)

    have_z = object_z is not None and math.isfinite(object_z)
    if prefer_plane and have_z:
        p_plane = _project_pixel_to_plane(
            T_base_cam, u=u, v_pix=v, fx=fx, fy=fy, cx0=cx, cy0=cy, plane_z_m=float(object_z)
        )
        if p_plane is not None:
            return np.asarray(p_plane, dtype=np.float64), d_meas
    if d_meas is not None:
        return _backproject_depth_base(T_base_cam, u, v, float(d_meas), fx, fy, cx, cy), d_meas
    if have_z:
        p_plane = _project_pixel_to_plane(
            T_base_cam, u=u, v_pix=v, fx=fx, fy=fy, cx0=cx, cy0=cy, plane_z_m=float(object_z)
        )
        if p_plane is not None:
            return np.asarray(p_plane, dtype=np.float64), d_meas
    return None, d_meas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@parser.wrap()
def lookat_engine_main(cfg: LookAtEngineConfig) -> None:
    run_lookat_engine(cfg)


def run_lookat_engine(cfg: LookAtEngineConfig) -> None:
    init_logging(console_level=os.environ.get("LEROBOT_LOG_LEVEL", "INFO"))
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.perception.yolo_world import YoloWorldDetector

    kin = RobotKinematics(urdf_path=cfg.urdf, target_frame_name=cfg.ee_frame, joint_names=ARM_MOTORS)
    detector = YoloWorldDetector(cfg.model_path)
    detector.set_query(str(cfg.query))

    T_ee_cam = parse_tf_string(cfg.gripper_camera_tf)
    if bool(cfg.invert_gripper_camera_tf):
        T_ee_cam = np.linalg.inv(np.asarray(T_ee_cam, dtype=np.float64))
    if abs(float(cfg.gripper_camera_pitch_trim_deg)) > 1e-6:
        from scipy.spatial.transform import Rotation as _R

        R_trim = _R.from_euler("x", float(cfg.gripper_camera_pitch_trim_deg), degrees=True).as_matrix()
        T_ee_cam[:3, :3] = T_ee_cam[:3, :3] @ R_trim

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    rerun_enabled = _try_init_rerun(cfg)

    intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0, "depth_scale": 0.001}
    cam = getattr(robot, "cameras", {}).get(cfg.camera_key)
    if cam is not None and hasattr(cam, "get_depth_intrinsics"):
        try:
            intrinsics = dict(cam.get_depth_intrinsics())
        except Exception as e:
            logger.warning("[lookat-engine] get_depth_intrinsics failed (%s); using fallback", e)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    depth_scale = float(intrinsics.get("depth_scale", 0.001))

    i_pan = ARM_MOTORS.index("shoulder_pan")
    dt_target = 1.0 / max(1.0, float(cfg.loop_hz))
    log_every_n = max(1, int(cfg.log_every_n))
    n_hat = approach_unit_vector(cfg.approach_az_deg, cfg.approach_el_deg)  # object -> EE
    tip_off = float(cfg.gripper_tip_offset_m)

    logger.info(
        "[lookat-engine] query=%r intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f rerun=%s",
        cfg.query, fx, fy, cx, cy, rerun_enabled,
    )
    logger.info(
        "[lookat-engine] approach az=%.0f el=%.0f n_hat(obj->EE)=(%+.2f,%+.2f,%+.2f) "
        "standoff=%.3fm grasp_trigger=%.3fm tip_offset=%.3fm",
        cfg.approach_az_deg, cfg.approach_el_deg, n_hat[0], n_hat[1], n_hat[2],
        cfg.final_standoff_m, cfg.grasp_trigger_tip_m, tip_off,
    )
    cam_z_ee = T_ee_cam[:3, 2]
    logger.info(
        "[lookat-engine] T_ee_cam origin_ee=(%.3f,%.3f,%.3f) cam+Z_in_EE=(%+.3f,%+.3f,%+.3f)",
        T_ee_cam[0, 3], T_ee_cam[1, 3], T_ee_cam[2, 3], cam_z_ee[0], cam_z_ee[1], cam_z_ee[2],
    )

    state = "SEARCH"
    p_obj_filt: np.ndarray | None = None
    # If the user pins the object-centre height, trust it (stable plane projection);
    # otherwise auto-track it from the live depth estimate.
    user_z = bool(math.isfinite(cfg.object_center_z_m))
    object_z: float | None = float(cfg.object_center_z_m) if user_z else None
    z_locked = user_z
    z_seed_samples: list[float] = []
    if user_z:
        logger.info("[lookat-engine] object_center_z pinned at %.3fm (plane projection)", object_z)
    det_streak = 0
    lost_ticks = 0
    d_tip_at_grasp = float(cfg.grasp_trigger_tip_m)
    search_pan_dir = 1.0
    search_pan_center: float | None = None
    comm_errors = 0
    tick = 0
    last_t = time.time()
    motion = MotionExecutionConfig()
    home_ctrl = HomeFoldController(home_settings_from_config(cfg), ARM_MOTORS)
    if home_ctrl.setup_standalone_keypress():
        pass  # log_status reports key listener state
    home_ctrl.log_status()
    interrupted = False

    def _read_obs():
        return robot.get_observation()

    try:
        while True:
            loop_t = time.time()
            tick += 1
            if home_ctrl.poll_and_fold(robot):
                break

            try:
                obs = _read_obs()
            except (ConnectionError, OSError, TimeoutError) as e:
                comm_errors += 1
                logger.warning("[lookat-engine] robot read failed (%d/%d): %s",
                               comm_errors, int(cfg.comm_error_max_consecutive), e)
                if comm_errors >= int(cfg.comm_error_max_consecutive):
                    raise
                _sleep(loop_t, dt_target)
                continue
            comm_errors = 0

            rgb = obs.get(cfg.camera_key)
            depth_map = obs.get(f"{cfg.camera_key}_depth")
            if rgb is None:
                _sleep(loop_t, dt_target)
                continue
            frame_h, frame_w = int(np.asarray(rgb).shape[0]), int(np.asarray(rgb).shape[1])

            joints = np.array([float(obs[f"{m}.pos"]) for m in ARM_MOTORS], dtype=np.float64)
            T_base_ee = np.asarray(kin.forward_kinematics(joints), dtype=np.float64)
            T_base_cam = T_base_ee @ T_ee_cam
            cam_eye = T_base_cam[:3, 3]

            det = detector.best_detection(np.asarray(rgb))
            bbox, uv, conf = _parse_detection(det, cfg.min_detection_confidence, cfg.bbox_grasp_v_frac)
            detected = bbox is not None

            # ---- Update the world model ----
            # Use plane projection once the height is known (pinned or locked); only
            # the brief auto-seeding window relies on depth back-projection.
            prefer_plane = user_z or z_locked
            p_meas, d_meas = (None, None)
            if detected:
                p_meas, d_meas = _measure_object_position(
                    cfg=cfg, T_base_cam=T_base_cam, uv=uv, bbox=bbox, depth_map=depth_map,
                    fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=depth_scale,
                    object_z=object_z, prefer_plane=prefer_plane,
                )

            if p_meas is not None:
                if p_obj_filt is None:
                    p_obj_filt = np.asarray(p_meas, dtype=np.float64)
                else:
                    jump = float(np.linalg.norm(np.asarray(p_meas) - p_obj_filt))
                    if jump <= float(cfg.pos_outlier_reject_m):
                        a = float(cfg.pos_ema_alpha)
                        p_obj_filt = (1.0 - a) * p_obj_filt + a * np.asarray(p_meas, dtype=np.float64)
                det_streak += 1
                lost_ticks = 0
                # Auto-seed object_z from the first few CLEAN (fully-in-frame) depth
                # readings, then LOCK it. After locking we switch to plane projection
                # and never touch depth again — this is what stops the height from
                # drifting down as the object nears the frame edge.
                if (
                    not user_z and not z_locked and d_meas is not None
                    and _bbox_inside_frame(bbox, frame_w, frame_h)
                ):
                    z_seed_samples.append(float(p_meas[2]))
                    if len(z_seed_samples) >= int(cfg.object_z_seed_frames):
                        object_z = float(np.median(z_seed_samples))
                        z_locked = True
                        logger.info(
                            "[lookat-engine] locked object_center_z=%.3fm "
                            "(median of %d clean frames) — plane projection from here",
                            object_z, len(z_seed_samples),
                        )
            else:
                det_streak = 0
                lost_ticks += 1

            # ---- Pixel error vs image centre (visibility gate for grasp) ----
            pixel_err = float(np.hypot(uv[0] - cx, uv[1] - cy)) if detected else float("inf")
            # Aim quality: how well the optical axis points at the world estimate
            # (1.0 = perfect look-at). The metric that tells us the object will
            # not leave the frame during approach.
            aim_dot = float("nan")
            d_tip_now = float("nan")
            if p_obj_filt is not None:
                p_tip_now = _gripper_tip_base(T_base_ee, tip_off)
                d_tip_now = float(np.linalg.norm(p_tip_now - p_obj_filt))
                to_obj = p_obj_filt - cam_eye
                n = float(np.linalg.norm(to_obj))
                if n > 1e-6:
                    aim_dot = float(np.dot(T_base_cam[:3, 2], to_obj / n))

            # ---- State machine ----
            q_cmd = joints.copy()

            if state == "SEARCH":
                if det_streak >= int(cfg.acquire_frames) and p_obj_filt is not None:
                    logger.info("[lookat-engine] SEARCH->APPROACH (locked, p_obj=(%.3f,%.3f,%.3f))",
                                *[float(x) for x in p_obj_filt])
                    state = "APPROACH"
                    search_pan_center = None
                else:
                    # Gentle pan sweep to bring an out-of-view object into the frame.
                    if search_pan_center is None:
                        search_pan_center = float(joints[i_pan])
                    nxt = float(q_cmd[i_pan]) + search_pan_dir * float(cfg.search_pan_step_deg)
                    if abs(nxt - search_pan_center) > float(cfg.search_pan_amplitude_deg):
                        search_pan_dir *= -1.0
                        nxt = float(q_cmd[i_pan]) + search_pan_dir * float(cfg.search_pan_step_deg)
                    q_cmd[i_pan] = nxt

            elif state == "APPROACH":
                if p_obj_filt is None or lost_ticks > int(cfg.lost_grace_ticks):
                    logger.info("[lookat-engine] APPROACH->SEARCH (lost %d ticks)", lost_ticks)
                    state = "SEARCH"
                else:
                    p_obj = p_obj_filt
                    p_tip = _gripper_tip_base(T_base_ee, tip_off)
                    d_tip = float(np.linalg.norm(p_tip - p_obj))

                    # Aim: camera optical axis -> object (look-at). Keeps it centred.
                    R_target = build_look_at_R(
                        T_base_ee[:3, :3], T_ee_cam, optical_axis_target_base=(p_obj - cam_eye)
                    )
                    # Tip goal: standoff from the object along the approach ray.
                    tip_goal = p_obj + float(cfg.final_standoff_m) * n_hat
                    # Floor guard, relative to the object height (the table can
                    # sit well below z=0) plus an absolute safety net.
                    floor_z = float(cfg.tip_floor_abs_z_m)
                    if object_z is not None and math.isfinite(object_z):
                        floor_z = max(floor_z, float(object_z) - float(cfg.tip_floor_below_object_m))
                    tip_goal[2] = max(float(tip_goal[2]), floor_z)
                    ee_pos_goal = tip_goal - R_target @ np.array([0.0, 0.0, tip_off])

                    T_goal = np.eye(4)
                    T_goal[:3, :3] = R_target
                    T_goal[:3, 3] = ee_pos_goal

                    q_des = None
                    for ow in (float(cfg.ik_orientation_weight), 0.1, 0.0):
                        try:
                            q_des = kin.inverse_kinematics(
                                joints, T_goal,
                                position_weight=float(cfg.ik_position_weight),
                                orientation_weight=float(ow),
                            )
                            break
                        except Exception as e:
                            if ow <= 0.0:
                                logger.warning("[lookat-engine] IK failed: %s", e)

                    if q_des is not None:
                        q_des = np.asarray(q_des, dtype=np.float64)
                        # Rate-limit: cap the largest single joint move this tick.
                        max_dq = float(cfg.max_joint_step_deg)
                        dq = q_des[: len(ARM_MOTORS)] - joints
                        peak = float(np.max(np.abs(dq))) if dq.size else 0.0
                        if max_dq > 0.0 and peak > max_dq:
                            dq = dq * (max_dq / peak)
                        q_cmd = joints + dq

                    centered = pixel_err <= float(cfg.grasp_center_pixel_err_px)
                    if d_tip <= float(cfg.grasp_trigger_tip_m) and (centered or not detected):
                        logger.info("[lookat-engine] APPROACH->GRASP (d_tip=%.3fm pixel_err=%.0fpx)",
                                    d_tip, pixel_err)
                        d_tip_at_grasp = float(d_tip)
                        state = "GRASP"

            # ---- Send action (SEARCH / APPROACH) ----
            if state in ("SEARCH", "APPROACH"):
                act = {f"{m}.pos": float(q_cmd[idx]) for idx, m in enumerate(ARM_MOTORS)}
                # Pre-open the gripper as we close in so we never drive closed
                # fingers into the object and push it before the grasp starts.
                preopen = (
                    state == "APPROACH"
                    and math.isfinite(d_tip_now)
                    and d_tip_now <= float(cfg.grasp_trigger_tip_m) + float(cfg.grasp_preopen_margin_m)
                )
                if preopen:
                    act["gripper.pos"] = float(cfg.grasp_open_pct)
                elif "gripper.pos" in obs:
                    act["gripper.pos"] = float(obs["gripper.pos"])
                try:
                    robot.send_action(act)
                except Exception as e:
                    logger.warning("[lookat-engine] send_action failed: %s", e)

            # ---- GRASP (blocking, with rerun pump) ----
            if state == "GRASP":
                def _pump():
                    # Keep detection LIVE during the (open-loop) grasp inch so the
                    # bbox stays visible and we can see whether the object is really
                    # under the gripper as the fingers close.
                    try:
                        o = robot.get_observation()
                        rgb_g = o.get(cfg.camera_key)
                        j_g = np.array([float(o[f"{m}.pos"]) for m in ARM_MOTORS])
                        T_ee_g = np.asarray(kin.forward_kinematics(j_g), dtype=np.float64)
                        T_cam_g = T_ee_g @ T_ee_cam
                        bbox_g = uv_g = None
                        conf_g = None
                        if rgb_g is not None:
                            d_g = detector.best_detection(np.asarray(rgb_g))
                            bbox_g, uv_g, conf_g = _parse_detection(
                                d_g, cfg.min_detection_confidence, cfg.bbox_grasp_v_frac
                            )
                        _rerun_log(
                            cfg, frame=tick, rgb=rgb_g,
                            depth=o.get(f"{cfg.camera_key}_depth"), bbox=bbox_g, uv=uv_g,
                            cx=cx, cy=cy, conf=conf_g, phase="GRASP", depth_m=None, kin=kin,
                            joints=j_g, T_base_ee=T_ee_g, T_base_cam=T_cam_g, p_obj=p_obj_filt,
                        )
                    except Exception:
                        pass

                # Size the final inch to actually cover the measured gap.
                inch_m = float(np.clip(
                    d_tip_at_grasp - float(cfg.grasp_final_gap_m),
                    0.01, float(cfg.grasp_max_inch_m),
                ))
                logger.info("[lookat-engine] grasp inch sized to %.3fm (d_tip=%.3fm)", inch_m, d_tip_at_grasp)
                outcome = run_grasp_sequence(
                    robot=robot, kin=kin, motor_names=ARM_MOTORS, T_base_cam=T_base_cam,
                    cfg=_grasp_cfg(cfg, final_approach_m=inch_m),
                    object_size_m=float(cfg.target_physical_size_m),
                    motion=motion, p_obj_base=p_obj_filt, on_tick=_pump if rerun_enabled else None,
                )
                logger.info("[lookat-engine] grasp outcome: success=%s — %s", outcome.success, outcome.message)
                if outcome.success and bool(cfg.grasp_hold_after):
                    logger.info("[lookat-engine] holding grasp — done.")
                    break
                state = "SEARCH"
                p_obj_filt = None
                det_streak = 0
                lost_ticks = 0

            # ---- Logging + viz ----
            if tick % log_every_n == 0:
                p_str = (
                    f"({p_obj_filt[0]:.3f},{p_obj_filt[1]:.3f},{p_obj_filt[2]:.3f})"
                    if p_obj_filt is not None else "n/a"
                )
                logger.info(
                    "[lookat-engine] tick=%d state=%s det=%s conf=%.2f pixel_err=%s "
                    "aim=%s d_tip=%s d_meas=%s p_obj=%s object_z=%s lost=%d",
                    tick, state, detected, conf,
                    f"{pixel_err:.0f}px" if math.isfinite(pixel_err) else "n/a",
                    f"{aim_dot:+.2f}" if math.isfinite(aim_dot) else "n/a",
                    f"{d_tip_now:.3f}m" if math.isfinite(d_tip_now) else "n/a",
                    f"{d_meas:.3f}m" if d_meas is not None else "n/a",
                    p_str,
                    f"{object_z:.3f}m" if object_z is not None else "unset",
                    lost_ticks,
                )
            if rerun_enabled:
                _rerun_log(
                    cfg, frame=tick, rgb=rgb, depth=depth_map, bbox=bbox, uv=uv,
                    cx=cx, cy=cy, conf=conf, phase=state, depth_m=d_meas, kin=kin,
                    joints=joints, T_base_ee=T_base_ee, T_base_cam=T_base_cam, p_obj=p_obj_filt,
                )

            _sleep(loop_t, dt_target)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("[lookat-engine] interrupted by user.")
    finally:
        home_ctrl.on_exit(robot, interrupted=interrupted)
        try:
            robot.disconnect()
        except Exception:
            pass


def _sleep(loop_t: float, dt_target: float) -> None:
    elapsed = time.time() - loop_t
    rem = dt_target - elapsed
    if rem > 0:
        time.sleep(rem)


if __name__ == "__main__":
    lookat_engine_main()
