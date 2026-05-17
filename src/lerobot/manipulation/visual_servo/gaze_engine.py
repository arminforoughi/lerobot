# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gaze-first visual servoing for SO-101 + OAK-D eye-in-hand.

Two decoupled loops:

  GAZE  (every tick, no IK): proportional control on bbox-center pixel error
        drives shoulder_pan (u) and wrist_flex (v) so the target never leaves
        the optical axis. Pixel error is a HARD priority, not a soft additive
        correction on top of a 6-DoF IK target.

  APPROACH (gated on small gaze error, position-only IK): once the gaze loop
        has the bbox centered, step the EE forward along the *current* optical
        axis. Depth comes from the bbox apparent size (pinhole, known physical
        size) — the only depth signal that survives inside OAK-D's stereo
        minimum range.

State machine:
  SEARCH    → scan with **shoulder_pan + wrist_flex only** (no whole-arm ramp);
              after a track loss, gaze at the last bbox center and retreat toward
              the last lock pose before sweeping. Optional multi-query rotation;
              on lock by default **TRACKING** (``preposition_from_search=False``).
              Set ``preposition_from_search=True`` for legacy SEARCH→PREPOSITIONING.
  PREPOSITIONING (optional) → sphere + look-at IK; **emergency gaze** when
              pixel error exceeds ``preposition_emergency_gaze_pixel_threshold_px``
              freezes orbit IK and re-centers so the target stays in view.
  TRACKING  → gaze only until centered, then APPROACHING (or live ``preposition``).
  APPROACHING → radial approach + visibility-regulated depth toward standoff.
  HOLD      → depth within tolerance of standoff; maintain gaze.

This is intentionally separate from ``cvs_engine`` — the goal is to never let
the object leave the FoV, which is the failure mode that motivated the
rewrite. Approach direction is always the current line of sight, not a
commanded ``(az, el)``.
"""

from __future__ import annotations

import logging
import math
import os
import select
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.manipulation.visual_servo.cvs_engine import (
    _se3_rate_limited_step,
    approach_unit_vector,
    build_look_at_R,
)
from lerobot.manipulation.yolo_track.depth import depth_from_bbox_size
from lerobot.manipulation.yolo_track.math_utils import (
    parse_tf_string,
    point_cam_to_base,
)
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@dataclass
class GazeEngineConfig:
    # Robot / camera
    robot: RobotConfig = field(
        default_factory=lambda: SOFollowerRobotConfig(port="", cameras={})
    )
    urdf: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    camera_key: str = "front"
    gripper_camera_tf: str = "0.04,0,0.02,0,-0.35,0"
    invert_gripper_camera_tf: bool = False
    # Frame-agnostic camera pitch trim. Applied as a rotation about the CAMERA's
    # own +X axis (image right), so positive degrees tilt the camera +Z (optical
    # axis) further DOWN in the world, regardless of the EE frame convention.
    # Use this to fine-tune an existing ``gripper_camera_tf`` rotvec without
    # having to recompute it. Default 0 (no trim).
    gripper_camera_pitch_trim_deg: float = 0.0
    # If True, the 3D visualization clamps a back-projected object position so
    # it cannot render below the workspace ground plane. Purely cosmetic — does
    # NOT affect the controller. Useful when the camera extrinsic is approximate
    # and the back-projected point dips slightly under the table in Rerun.
    viz_clamp_object_to_ground: bool = True
    viz_ground_plane_z_m: float = 0.0

    # Detection
    query: str = "red cube"
    model_path: str = "yolov8s-worldv2.pt"
    target_physical_size_m: float = 0.03
    min_detection_confidence: float = 0.25

    # Commanded approach geometry. The PREPOSITION state swings the EE to a
    # vantage point at ``preposition_initial_radius_m`` from the object along
    # ``n_hat(approach_az_deg, approach_el_deg)`` BEFORE switching to the
    # line-of-sight APPROACH. ``el=90`` is top-down (camera directly above the
    # object); ``el=0`` is side-on. ``az`` rotates around the object's vertical
    # axis: 0 means EE on the +X side of the object in base frame.
    approach_az_deg: float = 0.0
    # 90° is “top-down” and is often unreachable / unstable on SO-101 + eye-in-hand;
    # ~50–60° is a practical oblique vantage that still looks down onto the table.
    approach_el_deg: float = 55.0
    preposition_enabled: bool = True
    preposition_initial_radius_m: float = 0.18
    preposition_position_tolerance_m: float = 0.04
    preposition_max_lin_vel_m_s: float = 0.04
    preposition_max_ang_vel_deg_s: float = 80.0
    # PREPOSITION wants the camera pointed AT the object (look-at), so we want
    # orientation to matter more here than during APPROACHING.
    preposition_ik_orientation_weight: float = 1.5
    # If False, do not add gaze Δpan/Δtilt on top of preposition IK (they fight
    # look-at and often prevent pos_err from shrinking). Gaze resumes in APPROACHING.
    preposition_apply_gaze: bool = False
    # During PREPOSITIONING, if bbox center error (px) is >= this threshold,
    # **skip** preposition IK for that tick and apply **gaze only** so the target
    # stays in view. Prevents the arm from diving while the object drifts off-center.
    preposition_emergency_gaze_pixel_threshold_px: float = 48.0
    # If True, SEARCH goes straight to PREPOSITIONING after lock. If False
    # (default), SEARCH→TRACKING first so gaze centers the target before any
    # large orbit IK (much safer for eye-in-hand).
    preposition_from_search: bool = False
    # If False, PREPOSITIONING→APPROACHING only needs EE near the vantage pose
    # (pos_err), not a centered bbox — avoids deadlock when the arm cannot both
    # reach the commanded sphere point and keep the target in the image center.
    preposition_require_centered_bbox: bool = False
    # Separate joint clip for preposition (often needs larger steps than fine approach).
    preposition_max_joint_step_deg: float = 3.0
    # Multiplier on bbox-size depth (pinhole). Use >1 if measured range is farther
    # than the model (e.g. YOLO box padded → depth reads short).
    bbox_depth_scale: float = 1.0
    # Additive offset (m) applied AFTER ``bbox_depth_scale`` to compensate a
    # systematic underread (e.g. close-range pinhole bias). Positive pushes
    # the inferred depth farther from the camera; this also defers approach
    # completion since ``d_filt`` will read larger.
    bbox_depth_offset_m: float = 0.0
    # Clamp back-projected object Z in **base frame** (m) for IK / approach /
    # preposition. Stops chasing a phantom below the table when depth/extrinsics
    # are biased negative. Set to a large negative (e.g. -99) to disable.
    ik_object_floor_z_m: float = 0.015

    # Approach
    final_standoff_m: float = 0.10
    approach_done_tolerance_m: float = 0.012
    approach_kp: float = 0.6
    approach_max_lin_vel_m_s: float = 0.045
    # Hard per-joint per-tick cap on the IK output. With ``orientation_weight``
    # small the 5-DoF position-only IK has a nullspace the solver can explore,
    # producing big joint jumps for a 1 mm Cartesian step — which throws the
    # bbox off-frame and forces APPROACHING→TRACKING regression. The cap
    # clips |Δq_i| regardless of what the solver proposes. Set to 0 to disable.
    approach_max_joint_step_deg: float = 1.5
    # If True, APPROACH moves the EE along the **radial** vector from current
    # eye position toward the back-projected object point (not just along the
    # camera's optical axis). Keeps the trajectory directed at the actual
    # target even when look-at orientation is imperfect — eliminates the
    # "dive into floor" failure mode when T_ee_cam is mis-calibrated.
    approach_use_radial_to_object: bool = True
    # Soft visibility regulator. When pixel error exceeds this, the depth step
    # is scaled smoothly toward 0 (rather than waiting for the hard regress
    # threshold). 1.0 means no slowdown. Linear ramp between soft and regress.
    approach_fov_soft_threshold_px: float = 35.0

    # Gaze (P-control on pixel error → joint deltas, no IK)
    gaze_kp_pan: float = 0.45
    gaze_kp_tilt: float = 0.35
    gaze_max_step_pan_deg: float = 3.0
    gaze_max_step_tilt_deg: float = 3.0
    gaze_deadband_px: float = 6.0
    pan_sign: float = 1.0
    wrist_tilt_sign: float = 1.0
    # When bbox is already near the image center, scale down shoulder_pan gaze so
    # the arm re-orients mostly with shoulder_lift / elbow / wrist (joints 2–4)
    # via preposition IK, not endless left–right pan.
    gaze_pan_scale_when_aligned: float = 0.35
    gaze_pan_scale_aligned_enabled: bool = True

    # Live tuning (stdin and/or append-only file). Each non-empty line is a command.
    # Stdin: enable with ``live_control_stdin=True`` (type lines + Enter in the same
    # terminal as the process). File: set ``live_control_file`` to a path; append
    # lines with e.g. ``echo "el 88" >> /tmp/gaze_cmd.txt``.
    #
    # Commands (case-insensitive first token):
    #   el <deg> | elevation <deg>     — preposition / approach vantage elevation 0..90
    #   az <deg> | azimuth <deg>       — azimuth around object vertical axis
    #   depth <m> | standoff <m>       — target camera–object range (final standoff)
    #   radius <m>                     — preposition sphere radius
    #   top                            — el=90°, radius=max(standoff+0.14, 0.22), then PREPOSITION
    #   preposition | repo             — jump to PREPOSITIONING (if enabled + detected)
    #   pan <0..1> | pan auto          — manual pan gaze scale, or auto (aligned damping)
    #   up [deg] | down [deg]          — bump shoulder_lift trim (on-demand height)
    #   lift <deg> | lift reset        — set or clear shoulder_lift trim
    #   status | help                    — print current overrides / command list
    # If ``live_control_keypress`` and stdin is a TTY, stdin uses cbreak (no Enter):
    #   [ ] or ↓↑ — approach ``el`` ±``live_el_step_deg_default``;  , . — lift trim ±step
    live_control_stdin: bool = False
    live_control_file: str = ""
    live_control_keypress: bool = False
    # On-demand vertical bias: added to shoulder_lift (deg) on top of IK/gaze
    # every tick, clamped to ±live_lift_trim_max_deg. Controlled via stdin:
    #   up [deg]  |  down [deg]  |  lift <deg>  |  lift reset
    live_lift_trim_max_deg: float = 28.0
    live_lift_step_deg_default: float = 3.0
    live_el_step_deg_default: float = 5.0
    # Elevation clamp for keypress / `el` commands. 0..90 was too restrictive
    # for orbiting past top-down; widen as needed (negative = camera comes from
    # below the object's horizon, >90 = pole-flip toward the opposite side).
    live_el_min_deg: float = -45.0
    live_el_max_deg: float = 135.0
    live_radius_step_m_default: float = 0.03
    live_standoff_step_m_default: float = 0.01
    # ``-`` / ``=`` keys: composite "closeness" — adjusts both orbit radius and
    # final standoff by this much each press, so motion is visible in any state.
    live_closeness_step_m_default: float = 0.03
    # Key-repeat coalescing: at most this many logical steps per control tick
    # (comma/period/el/etc.), so holding a key nudges smoothly instead of
    # jumping the orbit target to the clamp in one frame.
    live_key_max_steps_per_tick: int = 2
    # Orbit radius/el used by IK slew toward the live target at this rate.
    live_radius_slew_m_s: float = 0.18
    live_standoff_slew_m_s: float = 0.04
    live_el_slew_deg_s: float = 45.0
    live_approach_boost_lin_vel_m_s: float = 0.06
    live_approach_boost_fov_scale_min: float = 0.85
    # Briefly raise preposition speed after a live key so motion keeps up with
    # the target (avoids "nothing… then jump").
    live_preposition_boost_duration_s: float = 0.55
    live_preposition_boost_lin_vel_m_s: float = 0.14
    live_preposition_boost_joint_step_deg: float = 9.0

    # State machine
    lock_required_frames: int = 4
    track_lost_frames: int = 12
    approach_pixel_threshold_px: float = 35.0
    approach_consecutive_centered_frames: int = 3
    approach_regress_pixel_threshold_px: float = 70.0

    # Depth filtering (EMA on bbox-size pinhole depth)
    depth_ema_alpha: float = 0.35

    # Search: pan (joint 1) + wrist tilt (joint 4) sweep only — shoulder_lift,
    # elbow, and wrist_roll stay at the seed pose. SO-101: +wrist_flex pitches
    # camera DOWN (table); less positive / negative pitches toward horizon/sky.
    # Table scan: start **low** (steep down at workspace), sweep **up** along the
    # table in front (wrist decreases toward ``search_wrist_flex_end_deg``), not
    # into the sky — keep ``search_wrist_flex_end_deg`` modest (≈0..+10).
    search_enabled: bool = True
    search_pan_amplitude_deg: float = 35.0
    search_pan_period_s: float = 6.0
    # If False, legacy whole-arm search also ramps shoulder_lift (not recommended).
    search_joints_pan_wrist_only: bool = True
    search_shoulder_lift_target_deg: float = -2.0
    search_wrist_flex_start_deg: float = 45.0
    search_wrist_flex_end_deg: float = 4.0
    search_look_up_period_s: float = 14.0
    search_pose_ramp_iters: int = 12
    # After track loss: gaze at last bbox center, retreat toward last lock joints,
    # then resume pan+wrist sweep from that pose.
    search_reacquire_on_lost: bool = True
    search_reacquire_max_frames: int = 18
    search_reacquire_retreat_iters: int = 10
    # Comma-separated extra YOLO-World prompts used only in SEARCH, rotated every
    # ``search_query_rotate_period_s`` so the sweep can lock under alternate wording.
    # Empty string => use ``query`` only (no rotation).
    search_queries: str = ""
    search_query_rotate_period_s: float = 3.0

    # IK weights for the approach step. Position dominates; a small nonzero
    # orientation weight anchors the 5-DoF nullspace (otherwise the solver
    # picks weird joint configurations to satisfy a 1 mm Cartesian step). The
    # gaze loop runs AFTER this IK and adds Δpan/Δtilt deltas, so it retains
    # final authority on camera pointing regardless of the orientation weight.
    ik_position_weight: float = 2.0
    ik_orientation_weight: float = 1.0

    # Loop
    loop_hz: float = 25.0

    # Display
    display_data: bool = False
    display_sim3d: bool = False
    log_every_n: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sleep(loop_t: float, dt_target: float) -> None:
    remaining = (loop_t + dt_target) - time.time()
    if remaining > 0.0:
        time.sleep(remaining)


def _init_live_runtime(cfg: GazeEngineConfig) -> dict:
    return {
        "el": float(cfg.approach_el_deg),
        "az": float(cfg.approach_az_deg),
        "standoff": float(cfg.final_standoff_m),
        "radius": float(cfg.preposition_initial_radius_m),
        "radius_target": float(cfg.preposition_initial_radius_m),
        "standoff_target": float(cfg.final_standoff_m),
        "el_target": float(cfg.approach_el_deg),
        "pan": None,  # None → auto (small pan when bbox centered); else 0..1 manual scale
        "lift_trim_deg": 0.0,
        "_file_offset": 0,
        "_goto_preposition": False,
        "_boost_until": 0.0,
        "_orbit_live_until": 0.0,
        "_kb_buf": b"",
        "_termios_old": None,
        "_stdin_keypress": False,
    }


def _apply_live_joint_trims(cfg: GazeEngineConfig, live: dict, q_deg: np.ndarray) -> np.ndarray:
    """Apply persistent shoulder_lift trim from live commands (on-demand up/down)."""
    q = np.asarray(q_deg, dtype=np.float64).copy()
    lim = float(getattr(cfg, "live_lift_trim_max_deg", 28.0))
    lt = float(np.clip(float(live.get("lift_trim_deg", 0.0)), -lim, lim))
    live["lift_trim_deg"] = lt
    i_lift = ARM_MOTORS.index("shoulder_lift")
    q[i_lift] = float(q[i_lift]) + lt
    return q


def _live_delta_lift_trim(live: dict, cfg: GazeEngineConfig, signed_step: float) -> None:
    lim = float(getattr(cfg, "live_lift_trim_max_deg", 28.0))
    cur = float(live.get("lift_trim_deg", 0.0))
    live["lift_trim_deg"] = float(np.clip(cur + float(signed_step), -lim, lim))
    logger.info(
        "[gaze-live] lift_trim=%+.1f° (%+.1f°, max ±%.0f°)",
        float(live["lift_trim_deg"]),
        float(signed_step),
        lim,
    )


def _live_arm_motion_boost(live: dict, cfg: GazeEngineConfig) -> None:
    dur = float(getattr(cfg, "live_preposition_boost_duration_s", 0.55))
    now = time.time()
    live["_boost_until"] = max(float(live.get("_boost_until", 0.0)), now + dur)
    # Orbit keys must run look-at IK even when bbox is off-center (emergency gaze
    # would otherwise block IK whenever pixel_err > threshold).
    live["_orbit_live_until"] = max(
        float(live.get("_orbit_live_until", 0.0)), now + dur
    )


def _live_orbit_keys_active(live: dict, loop_t: float) -> bool:
    return loop_t < float(live.get("_orbit_live_until", 0.0))


def _live_slew_orbit_targets(live: dict, cfg: GazeEngineConfig, dt: float) -> None:
    """Move commanded el/radius toward live targets at a bounded rate (smooth keys)."""
    dt_s = max(1e-3, float(dt))
    live["el_target"] = float(live.get("el_target", live["el"]))
    live["radius_target"] = float(live.get("radius_target", live["radius"]))
    live["standoff_target"] = float(live.get("standoff_target", live["standoff"]))
    el_rate = float(getattr(cfg, "live_el_slew_deg_s", 45.0))
    rad_rate = float(getattr(cfg, "live_radius_slew_m_s", 0.18))
    so_rate = float(getattr(cfg, "live_standoff_slew_m_s", 0.04))
    max_de = el_rate * dt_s
    max_dr = rad_rate * dt_s
    max_ds = so_rate * dt_s
    el = float(live["el"])
    el_t = float(live["el_target"])
    de = float(np.clip(el_t - el, -max_de, max_de))
    live["el"] = el + de
    rad = float(live["radius"])
    rad_t = float(live["radius_target"])
    dr = float(np.clip(rad_t - rad, -max_dr, max_dr))
    live["radius"] = rad + dr
    so = float(live["standoff"])
    so_t = float(live["standoff_target"])
    ds = float(np.clip(so_t - so, -max_ds, max_ds))
    live["standoff"] = so + ds


def _live_delta_el_deg(live: dict, cfg: GazeEngineConfig, signed_step: float) -> None:
    lo = float(getattr(cfg, "live_el_min_deg", -45.0))
    hi = float(getattr(cfg, "live_el_max_deg", 135.0))
    cur = float(live.get("el_target", live["el"]))
    new_el = float(np.clip(cur + float(signed_step), lo, hi))
    if abs(new_el - cur) < 1e-6:
        logger.info(
            "[gaze-live] approach_el_deg=%.1f° (clamped at %s%.0f°, widen with "
            "--live-el-min-deg/--live-el-max-deg)",
            new_el,
            "+" if signed_step > 0 else "",
            hi if signed_step > 0 else lo,
        )
        return
    live["el_target"] = new_el
    _live_arm_motion_boost(live, cfg)
    live["_goto_preposition"] = True
    logger.info(
        "[gaze-live] approach_el_deg → %.1f° (%+.1f°, slewing) PREPOSITION",
        new_el,
        float(signed_step),
    )


def _live_delta_radius_m(live: dict, cfg: GazeEngineConfig, signed_step: float) -> None:
    cur = float(live.get("radius_target", live.get("radius", 0.2)))
    new_r = float(np.clip(cur + float(signed_step), 0.05, 0.50))
    if abs(new_r - cur) < 1e-6:
        return
    live["radius_target"] = new_r
    _live_arm_motion_boost(live, cfg)
    live["_goto_preposition"] = True
    logger.info(
        "[gaze-live] orbit radius → %.3fm (%+.3fm, slewing) PREPOSITION",
        new_r,
        float(signed_step),
    )


def _live_delta_standoff_m(live: dict, cfg: GazeEngineConfig, signed_step: float) -> None:
    cur = float(live.get("standoff_target", live.get("standoff", 0.06)))
    new_s = float(np.clip(cur + float(signed_step), 0.02, 0.40))
    if abs(new_s - cur) < 1e-6:
        return
    live["standoff_target"] = new_s
    _live_arm_motion_boost(live, cfg)
    logger.info(
        "[gaze-live] goal depth (standoff) → %.3fm (%+.3fm, slewing)",
        new_s,
        float(signed_step),
    )


def _apply_live_line(raw: str, cfg: GazeEngineConfig, live: dict) -> None:
    s = str(raw).strip()
    if not s or s.startswith("#"):
        return
    parts = s.split()
    tok = parts[0].lower()
    rest = parts[1:]

    def _f(i: int) -> float:
        if i < len(rest):
            return float(rest[i])
        raise ValueError(f"missing numeric arg for {tok}")

    try:
        if tok in ("help", "?"):
            logger.info(
                "[gaze-live] commands: el <deg> | az <deg> | depth <m> | standoff <m> | "
                "radius <m> | top | preposition | pan <0..1>|pan auto | "
                "up [deg] | down [deg] | lift <deg>|lift reset | status | help"
                " — with --live-control-keypress: [ ] or arrows=el  , .=lift  ?=this"
            )
        elif tok == "status":
            logger.info(
                "[gaze-live] el=%.1f° az=%.1f° standoff=%.3fm radius=%.3fm lift_trim=%+.1f° pan_mode=%s",
                float(live["el"]),
                float(live["az"]),
                float(live["standoff"]),
                float(live["radius"]),
                float(live.get("lift_trim_deg", 0.0)),
                "manual(%.2f)" % float(live["pan"]) if live["pan"] is not None else "auto",
            )
        elif tok in ("el", "elevation"):
            v = float(np.clip(_f(0), 0.0, 90.0))
            live["el"] = v
            live["el_target"] = v
            _live_arm_motion_boost(live, cfg)
            logger.info("[gaze-live] approach_el_deg=%.1f°", v)
        elif tok in ("az", "azimuth"):
            live["az"] = float(_f(0))
            logger.info("[gaze-live] approach_az_deg=%.1f°", float(live["az"]))
        elif tok in ("depth", "standoff", "d"):
            v = max(0.02, float(_f(0)))
            live["standoff"] = v
            live["standoff_target"] = v
            _live_arm_motion_boost(live, cfg)
            logger.info("[gaze-live] goal depth (standoff)=%.3fm", v)
        elif tok in ("radius", "r"):
            v = max(0.05, float(_f(0)))
            live["radius"] = v
            live["radius_target"] = v
            _live_arm_motion_boost(live, cfg)
            logger.info("[gaze-live] preposition_initial_radius_m=%.3fm", v)
        elif tok == "top":
            live["el"] = 90.0
            live["radius"] = max(float(live["standoff"]) + 0.14, 0.22)
            live["_goto_preposition"] = True
            logger.info(
                "[gaze-live] TOP: el=90° radius=%.3fm → will PREPOSITION on next tick",
                float(live["radius"]),
            )
        elif tok in ("preposition", "repo", "orbit"):
            live["_goto_preposition"] = True
            logger.info("[gaze-live] PREPOSITION requested on next tick")
        elif tok == "pan":
            if not rest or rest[0].lower() == "auto":
                live["pan"] = None
                logger.info("[gaze-live] pan gaze scale = auto (weak when centered)")
            else:
                live["pan"] = float(np.clip(float(rest[0]), 0.0, 1.0))
                logger.info("[gaze-live] pan gaze scale = %.2f (manual)", float(live["pan"]))
        elif tok == "up":
            step = (
                float(rest[0])
                if rest
                else float(getattr(cfg, "live_lift_step_deg_default", 3.0))
            )
            _live_delta_lift_trim(live, cfg, step)
        elif tok == "down":
            step = (
                float(rest[0])
                if rest
                else float(getattr(cfg, "live_lift_step_deg_default", 3.0))
            )
            _live_delta_lift_trim(live, cfg, -step)
        elif tok == "lift":
            lim = float(getattr(cfg, "live_lift_trim_max_deg", 28.0))
            if not rest:
                raise ValueError("lift needs reset or <deg>")
            if rest[0].lower() == "reset":
                live["lift_trim_deg"] = 0.0
                logger.info("[gaze-live] lift_trim cleared")
            else:
                live["lift_trim_deg"] = float(np.clip(float(rest[0]), -lim, lim))
                logger.info("[gaze-live] lift_trim set to %+.1f°", float(live["lift_trim_deg"]))
        else:
            logger.warning("[gaze-live] unknown command: %r (type 'help')", s)
    except (ValueError, IndexError) as e:
        logger.warning("[gaze-live] bad line %r: %s", s, e)


def _install_stdin_keypress(live: dict) -> bool:
    """cbreak stdin so single keys work without Enter. Caller must restore via _restore_stdin_tty."""
    try:
        import termios as _termios
        import tty as _tty
    except ImportError:
        return False
    if not sys.stdin.isatty():
        return False
    fd = sys.stdin.fileno()
    try:
        live["_termios_old"] = _termios.tcgetattr(fd)
        _tty.setcbreak(fd)
    except (_termios.error, OSError, AttributeError):
        return False
    live["_stdin_keypress"] = True
    return True


def _restore_stdin_tty(live: dict) -> None:
    try:
        import termios as _termios
    except ImportError:
        live.pop("_termios_old", None)
        live["_stdin_keypress"] = False
        return
    old = live.get("_termios_old")
    if old is None:
        return
    try:
        _termios.tcsetattr(sys.stdin.fileno(), _termios.TCSADRAIN, old)
    except (_termios.error, OSError):
        pass
    live["_termios_old"] = None
    live["_stdin_keypress"] = False


def _drain_live_keypress(cfg: GazeEngineConfig, live: dict) -> None:
    if not live.get("_stdin_keypress"):
        return
    fd = sys.stdin.fileno()
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if not r:
                break
            more = os.read(fd, 256)
            if not more:
                break
            live["_kb_buf"] = bytes(live.get("_kb_buf", b"")) + more
    except (ValueError, OSError, BlockingIOError):
        pass

    buf = bytearray(live.get("_kb_buf", b""))
    if not buf:
        return

    el_step = float(getattr(cfg, "live_el_step_deg_default", 5.0))
    rad_step = float(getattr(cfg, "live_radius_step_m_default", 0.03))
    stand_step = float(getattr(cfg, "live_standoff_step_m_default", 0.03))
    max_steps = max(1, int(getattr(cfg, "live_key_max_steps_per_tick", 2)))

    net_rad = 0
    net_el = 0
    net_standoff = 0
    want_preposition = False
    want_help = False

    i = 0
    n = len(buf)
    while i < n:
        if buf[i] == 0x1B and i + 1 < n and buf[i + 1] == ord("[") and (i + 2) >= n:
            break
        if buf[i] == 0x1B and i + 1 >= n:
            break
        if (
            i + 2 < n
            and buf[i] == 0x1B
            and buf[i + 1] == ord("[")
            and buf[i + 2] in (ord("A"), ord("B"))
        ):
            net_el += 1 if buf[i + 2] == ord("A") else -1
            i += 3
            continue
        if buf[i] == 0x1B:
            i += 1
            continue

        c = buf[i]
        if c == ord("["):
            net_el -= 1
        elif c == ord("]"):
            net_el += 1
        # , / - = back (orbit radius +, goal depth +); . / = in (both −)
        elif c == ord(","):
            net_rad += 1
            net_standoff += 1
        elif c == ord("."):
            net_rad -= 1
            net_standoff -= 1
        elif c in (ord("-"), ord("_")):
            net_rad += 2
            net_standoff += 2
        elif c in (ord("="), ord("+")):
            net_rad -= 2
            net_standoff -= 2
        elif c == ord("p"):
            want_preposition = True
        elif c == ord("?"):
            want_help = True
        i += 1

    live["_kb_buf"] = bytes(buf[i:])

    def _cap(net: int) -> int:
        if net == 0:
            return 0
        sign = 1 if net > 0 else -1
        return sign * min(abs(net), max_steps)

    cr = _cap(net_rad)
    if cr != 0:
        _live_delta_radius_m(live, cfg, float(cr) * rad_step)
    cs = _cap(net_standoff)
    if cs != 0:
        _live_delta_standoff_m(live, cfg, float(cs) * stand_step)
    ce = _cap(net_el)
    if ce != 0:
        _live_delta_el_deg(live, cfg, float(ce) * el_step)
    if want_preposition:
        live["_goto_preposition"] = True
        logger.info("[gaze-live] PREPOSITION requested (key 'p')")
    if want_help:
        logger.info(
            "[gaze-live] keys: [ ]/↑↓=el ; ,/-=back (orbit+goal depth) ; ./=in ; "
            "max %d/tick ; p=preposition ; ? = help",
            max_steps,
        )


def _drain_live_commands(cfg: GazeEngineConfig, live: dict) -> None:
    _drain_live_keypress(cfg, live)
    lines: list[str] = []
    path = str(getattr(cfg, "live_control_file", "") or "").strip()
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                f.seek(int(live["_file_offset"]))
                chunk = f.read()
                live["_file_offset"] = int(f.tell())
            lines.extend(chunk.splitlines())
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.debug("[gaze-engine] live_control_file: %s", e)
    if bool(getattr(cfg, "live_control_stdin", False)) and not live.get(
        "_stdin_keypress"
    ):
        try:
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if not r:
                    break
                line = sys.stdin.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n\r"))
        except (ValueError, OSError):
            pass
    for ln in lines:
        _apply_live_line(ln, cfg, live)


def _maybe_floor_object_base(
    p: np.ndarray | None, z_floor_m: float
) -> np.ndarray | None:
    """Raise object Z in base frame for IK if it is implausibly below the table."""
    if p is None:
        return None
    zf = float(z_floor_m)
    if zf < -10.0:
        return np.asarray(p, dtype=np.float64).reshape(3).copy()
    out = np.asarray(p, dtype=np.float64).reshape(3).copy()
    if float(out[2]) < zf:
        out[2] = zf
    return out


def _search_vocab_queries(cfg: GazeEngineConfig) -> list[str]:
    extra = str(getattr(cfg, "search_queries", "") or "").strip()
    primary = str(cfg.query).strip()
    if not primary:
        raise ValueError("query must be non-empty")
    if not extra:
        return [primary]
    parts = [p.strip() for p in extra.split(",") if p.strip()]
    if not parts:
        return [primary]
    # Primary first, then extras (dedupe consecutive duplicates only).
    out: list[str] = [primary]
    for p in parts:
        if p != out[-1]:
            out.append(p)
    return out


def _gaze_joint_deltas(
    *,
    uv: tuple[float, float],
    cx0: float,
    cy0: float,
    fx: float,
    fy: float,
    cfg: GazeEngineConfig,
) -> tuple[float, float, float]:
    """Pure pixel-error P controller. Returns (Δpan_deg, Δtilt_deg, |err|_px)."""
    du = float(uv[0]) - float(cx0)
    dv = float(uv[1]) - float(cy0)
    err_px = float(math.hypot(du, dv))

    if abs(du) > float(cfg.gaze_deadband_px):
        d_pan_deg = math.degrees(math.atan2(du, max(1e-6, float(fx))))
        d_pan = float(
            np.clip(
                float(cfg.pan_sign) * float(cfg.gaze_kp_pan) * d_pan_deg,
                -float(cfg.gaze_max_step_pan_deg),
                +float(cfg.gaze_max_step_pan_deg),
            )
        )
    else:
        d_pan = 0.0

    if abs(dv) > float(cfg.gaze_deadband_px):
        d_tilt_deg = math.degrees(math.atan2(dv, max(1e-6, float(fy))))
        d_tilt = float(
            np.clip(
                float(cfg.wrist_tilt_sign) * float(cfg.gaze_kp_tilt) * d_tilt_deg,
                -float(cfg.gaze_max_step_tilt_deg),
                +float(cfg.gaze_max_step_tilt_deg),
            )
        )
    else:
        d_tilt = 0.0

    return d_pan, d_tilt, err_px


def _search_command(
    *,
    seed: np.ndarray,
    elapsed_s: float,
    ramp_alpha: float,
    pan_amp_deg: float,
    pan_period_s: float,
    lift_target_deg: float,
    wrist_start_deg: float,
    wrist_end_deg: float,
    look_up_period_s: float,
    pan_wrist_only: bool = True,
) -> np.ndarray:
    """Pan sweep + table tilt ramp for object acquisition.

    When ``pan_wrist_only`` (default), only shoulder_pan and wrist_flex move;
    the rest of the arm stays at ``seed``. Wrist follows an absolute table scan
    (start pitched down, sweep up along the workspace) — not blended from the
    startup wrist angle, which may already be looking at the ceiling.
    """
    look_frac = 0.0
    if float(look_up_period_s) > 1e-3:
        look_frac = float(
            np.clip(float(elapsed_s) / float(look_up_period_s), 0.0, 1.0)
        )
    wrist_cmd = float(wrist_start_deg) + look_frac * (
        float(wrist_end_deg) - float(wrist_start_deg)
    )
    a = float(np.clip(ramp_alpha, 0.0, 1.0))

    out = np.asarray(seed, dtype=np.float64).copy()
    if pan_period_s > 1e-3:
        omega = 2.0 * math.pi / float(pan_period_s)
        out[0] = float(seed[0]) + float(pan_amp_deg) * math.sin(omega * float(elapsed_s))
    out[3] = wrist_cmd
    if not bool(pan_wrist_only):
        target = seed.copy()
        target[1] = float(lift_target_deg)
        target[0] = out[0]
        target[3] = wrist_cmd
        return (1.0 - a) * seed + a * target
    return out


def _search_reacquire_command(
    *,
    joints_deg: np.ndarray,
    lock_joints_deg: np.ndarray,
    lock_uv: tuple[float, float],
    cx0: float,
    cy0: float,
    fx: float,
    fy: float,
    cfg: GazeEngineConfig,
    retreat_iter: int,
    retreat_iters: int,
) -> np.ndarray:
    """Gaze at last known bbox center; optionally blend all joints back to lock pose."""
    n_retreat = max(1, int(retreat_iters))
    ri = int(np.clip(int(retreat_iter), 0, n_retreat))
    alpha = float(ri) / float(n_retreat)
    q = (1.0 - alpha) * np.asarray(joints_deg, dtype=np.float64) + alpha * np.asarray(
        lock_joints_deg, dtype=np.float64
    )
    d_pan, d_tilt, _ = _gaze_joint_deltas(
        uv=lock_uv, cx0=cx0, cy0=cy0, fx=fx, fy=fy, cfg=cfg
    )
    q[0] = float(q[0]) + float(d_pan)
    q[3] = float(q[3]) + float(d_tilt)
    return q


def _preposition_q(
    *,
    joints_deg: np.ndarray,
    T_base_ee_cur: np.ndarray,
    T_ee_cam: np.ndarray,
    p_obj_base: np.ndarray,
    cfg: GazeEngineConfig,
    dt: float,
    kin,
    approach_az_deg: float | None = None,
    approach_el_deg: float | None = None,
    preposition_radius_m: float | None = None,
    max_lin_vel_m_s: float | None = None,
    max_joint_step_deg: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """Drive EE toward the commanded vantage point on a sphere around the object.

    Returns ``(q_new, p_eye_target, position_error_m)``. ``q_new`` is None on IK
    failure. The vantage point is ``p_obj + n_hat(az, el) · radius`` and the
    target rotation is a look-at from that point toward ``p_obj``. The SE(3)
    target is rate-limited so big swings happen over multiple ticks.
    """
    az = float(approach_az_deg if approach_az_deg is not None else cfg.approach_az_deg)
    el = float(approach_el_deg if approach_el_deg is not None else cfg.approach_el_deg)
    rad = float(
        preposition_radius_m
        if preposition_radius_m is not None
        else cfg.preposition_initial_radius_m
    )
    n_hat = approach_unit_vector(az, el)
    p_eye = np.asarray(p_obj_base, dtype=np.float64) + n_hat * rad
    look_dir = np.asarray(p_obj_base, dtype=np.float64) - p_eye
    ld_norm = float(np.linalg.norm(look_dir))
    optical_axis_target = (look_dir / ld_norm) if ld_norm > 1e-6 else -n_hat
    R_target = build_look_at_R(T_base_ee_cur[:3, :3], T_ee_cam, optical_axis_target)
    T_target = np.eye(4, dtype=np.float64)
    T_target[:3, :3] = R_target
    T_target[:3, 3] = p_eye

    lin_vel = float(
        max_lin_vel_m_s
        if max_lin_vel_m_s is not None
        else cfg.preposition_max_lin_vel_m_s
    )
    T_step = _se3_rate_limited_step(
        T_base_ee_cur,
        T_target,
        dt=float(dt),
        max_lin_vel_m_s=lin_vel,
        max_ang_vel_deg_s=float(cfg.preposition_max_ang_vel_deg_s),
    )
    pos_err = float(np.linalg.norm(T_base_ee_cur[:3, 3] - p_eye))
    q_new = None
    for ow in (
        float(cfg.preposition_ik_orientation_weight),
        0.15,
        0.0,
    ):
        try:
            q_new = kin.inverse_kinematics(
                joints_deg,
                T_step,
                position_weight=float(cfg.ik_position_weight),
                orientation_weight=float(ow),
            )
            break
        except Exception as e:
            if ow <= 0.0:
                logger.warning("[gaze-engine] preposition IK failed: %s", e)
                return None, p_eye, pos_err

    q_new = np.asarray(q_new, dtype=np.float64)
    max_dq = float(
        max_joint_step_deg
        if max_joint_step_deg is not None
        else cfg.preposition_max_joint_step_deg
    )
    if max_dq > 0.0:
        # Synchronized joint move: scale the WHOLE dq vector by a single
        # factor so the joint that needs the largest change uses max_dq and
        # every other joint reaches its target IN PROPORTION on the same tick.
        # Independent per-joint clipping makes the wrist (small dq) finish
        # while the elbow (large dq) is still moving — visually the wrist
        # snaps first, then the elbow catches up. This block fixes that.
        dq = q_new - np.asarray(joints_deg, dtype=np.float64)
        peak = float(np.max(np.abs(dq))) if dq.size else 0.0
        if peak > max_dq and peak > 1e-9:
            dq = dq * (max_dq / peak)
        q_new = np.asarray(joints_deg, dtype=np.float64) + dq
    pos_err = float(np.linalg.norm(T_base_ee_cur[:3, 3] - p_eye))
    return q_new, p_eye, pos_err


def _approach_q(
    *,
    joints_deg: np.ndarray,
    T_base_ee_cur: np.ndarray,
    T_base_cam_cur: np.ndarray,
    d_obj_m: float,
    p_obj_base: np.ndarray | None,
    pixel_err_px: float,
    cfg: GazeEngineConfig,
    dt: float,
    kin,
    final_standoff_m: float | None = None,
    max_lin_vel_m_s: float | None = None,
    max_joint_step_deg: float | None = None,
    fov_scale_min: float | None = None,
) -> tuple[np.ndarray | None, float, float]:
    """Step the EE radially toward the object (or along the optical axis as
    a fallback), with a soft visibility regulator that scales the depth step
    by pixel error.

    Returns (q_new, planned_step_m, fov_scale). ``q_new`` is None on IK
    failure. Radial mode keeps the trajectory directed at the actual
    back-projected object point even when the camera look-at is imperfect —
    that is what prevents the "dive into floor" failure mode.
    """
    eye_base = np.asarray(T_base_cam_cur[:3, 3], dtype=np.float64)
    if (
        bool(cfg.approach_use_radial_to_object)
        and p_obj_base is not None
        and float(np.linalg.norm(np.asarray(p_obj_base) - eye_base)) > 1e-3
    ):
        radial = np.asarray(p_obj_base, dtype=np.float64) - eye_base
        direction = radial / float(np.linalg.norm(radial))
    else:
        direction = np.asarray(T_base_cam_cur[:3, 2], dtype=np.float64)

    d_target = float(
        final_standoff_m if final_standoff_m is not None else cfg.final_standoff_m
    )
    err = float(d_obj_m) - d_target

    # Soft visibility regulator: smoothly cut depth velocity as pixel error
    # approaches the hard regress threshold.
    soft = float(cfg.approach_fov_soft_threshold_px)
    hard = float(cfg.approach_regress_pixel_threshold_px)
    if pixel_err_px <= soft or hard <= soft:
        fov_scale = 1.0
    else:
        fov_scale = max(0.0, 1.0 - (pixel_err_px - soft) / max(1e-3, hard - soft))
    if fov_scale_min is not None:
        fov_scale = max(float(fov_scale), float(fov_scale_min))

    lin_vel = float(
        max_lin_vel_m_s
        if max_lin_vel_m_s is not None
        else cfg.approach_max_lin_vel_m_s
    )
    max_step = lin_vel * float(dt)
    raw = float(cfg.approach_kp) * err * float(fov_scale)
    step = float(np.clip(raw, -max_step, +max_step))

    T_target = np.asarray(T_base_ee_cur, dtype=np.float64).copy()
    T_target[:3, 3] = T_base_ee_cur[:3, 3] + direction * step
    q_new = None
    for ow in (
        float(cfg.ik_orientation_weight),
        0.15,
        0.0,
    ):
        try:
            q_new = kin.inverse_kinematics(
                joints_deg,
                T_target,
                position_weight=float(cfg.ik_position_weight),
                orientation_weight=float(ow),
            )
            break
        except Exception as e:
            if ow <= 0.0:
                logger.warning("[gaze-engine] approach IK failed: %s", e)
                return None, step, float(fov_scale)

    q_new = np.asarray(q_new, dtype=np.float64)
    max_dq = float(
        max_joint_step_deg
        if max_joint_step_deg is not None
        else cfg.approach_max_joint_step_deg
    )
    if max_dq > 0.0:
        dq = q_new - np.asarray(joints_deg, dtype=np.float64)
        peak = float(np.max(np.abs(dq))) if dq.size else 0.0
        if peak > max_dq and peak > 1e-9:
            dq = dq * (max_dq / peak)
        q_new = np.asarray(joints_deg, dtype=np.float64) + dq
    return q_new, step, float(fov_scale)


def _try_init_rerun(cfg: GazeEngineConfig) -> bool:
    if not (cfg.display_data or cfg.display_sim3d):
        return False
    try:
        from lerobot.utils.visualization_utils import (
            init_rerun,
            send_agentic_rerun_blueprint,
        )

        init_rerun(session_name="gaze_engine")
        send_agentic_rerun_blueprint(
            show_camera_stream=bool(cfg.display_data),
            show_sim3d=bool(cfg.display_sim3d),
            camera_key=cfg.camera_key,
        )
        return True
    except Exception as e:
        logger.warning("[gaze-engine] rerun init failed: %s", e)
        return False


def _rerun_log(
    *,
    cfg: GazeEngineConfig,
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
    semantic_label: str | None = None,
) -> None:
    if rgb is None or not (cfg.display_data or cfg.display_sim3d):
        return
    try:
        from lerobot.manipulation.yolo_track.rerun_viz import log_rerun_iter
    except Exception:
        return
    try:
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
            object_half_size_m=0.5 * float(cfg.target_physical_size_m),
            show_sim3d=bool(cfg.display_sim3d),
            show_camera=bool(cfg.display_data),
            object_semantic_label=str(semantic_label or cfg.query).strip() or None,
        )
    except Exception as e:
        logger.debug("[gaze-engine] rerun log failed: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@parser.wrap()
def gaze_engine_main(cfg: GazeEngineConfig) -> None:
    run_gaze_engine(cfg)


def run_gaze_engine(cfg: GazeEngineConfig) -> None:
    init_logging(console_level=os.environ.get("LEROBOT_LOG_LEVEL", "INFO"))
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.perception.yolo_world import YoloWorldDetector

    kin = RobotKinematics(
        urdf_path=cfg.urdf, target_frame_name=cfg.ee_frame, joint_names=ARM_MOTORS
    )
    search_vocab_queries = _search_vocab_queries(cfg)
    detector = YoloWorldDetector(cfg.model_path)
    detector.set_query(search_vocab_queries[0])
    active_query_label = str(search_vocab_queries[0])
    search_vocab_idx = 0
    search_vocab_last_switch: float | None = None
    T_ee_cam = parse_tf_string(cfg.gripper_camera_tf)
    if bool(cfg.invert_gripper_camera_tf):
        T_ee_cam = np.linalg.inv(np.asarray(T_ee_cam, dtype=np.float64))
    # Frame-agnostic pitch trim: rotate about the camera's own +X (image right)
    # so positive degrees tilt the optical axis further DOWN in the world. This
    # does NOT clobber the calibrated rotvec; it only fine-tunes it.
    pitch_trim_deg = float(getattr(cfg, "gripper_camera_pitch_trim_deg", 0.0))
    if abs(pitch_trim_deg) > 1e-6:
        from scipy.spatial.transform import Rotation as _R

        R_camX_trim = _R.from_euler("x", float(pitch_trim_deg), degrees=True).as_matrix()
        T_ee_cam[:3, :3] = T_ee_cam[:3, :3] @ R_camX_trim
        logger.info(
            "[gaze-engine] applied pitch trim of %+.2f° about cam +X (image right)",
            float(pitch_trim_deg),
        )
    # Startup geometry sanity log: cam origin in EE frame, and cam +Z direction.
    cam_origin_ee = T_ee_cam[:3, 3]
    cam_z_ee = T_ee_cam[:3, 2]
    logger.info(
        "[gaze-engine] T_ee_cam: origin_ee=(%.3f, %.3f, %.3f) m  cam+Z_in_EE=(%+.3f, %+.3f, %+.3f)",
        float(cam_origin_ee[0]), float(cam_origin_ee[1]), float(cam_origin_ee[2]),
        float(cam_z_ee[0]), float(cam_z_ee[1]), float(cam_z_ee[2]),
    )

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    rerun_enabled = _try_init_rerun(cfg)

    intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0, "depth_scale": 0.001}
    cam = getattr(robot, "cameras", {}).get(cfg.camera_key)
    if cam is not None and hasattr(cam, "get_depth_intrinsics"):
        try:
            intrinsics = dict(cam.get_depth_intrinsics())
        except Exception as e:
            logger.warning("[gaze-engine] get_depth_intrinsics failed (%s); using fallback", e)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx0 = float(intrinsics["cx"])
    cy0 = float(intrinsics["cy"])

    logger.info(
        "[gaze-engine] query=%r search_vocab=%s final_standoff=%.3fm loop_hz=%.1f rerun=%s",
        cfg.query,
        search_vocab_queries,
        float(cfg.final_standoff_m),
        float(cfg.loop_hz),
        rerun_enabled,
    )
    logger.info(
        "[gaze-engine] intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f", fx, fy, cx0, cy0
    )
    logger.info(
        "[gaze-engine] preposition=%s el=%.1f° bbox_depth_scale=%.3f "
        "preposition_apply_gaze=%s require_centered_bbox=%s "
        "preposition_from_search=%s emergency_gaze_px=%.0f ik_floor_z=%.3fm",
        bool(cfg.preposition_enabled),
        float(cfg.approach_el_deg),
        float(cfg.bbox_depth_scale),
        bool(cfg.preposition_apply_gaze),
        bool(cfg.preposition_require_centered_bbox),
        bool(getattr(cfg, "preposition_from_search", False)),
        float(cfg.preposition_emergency_gaze_pixel_threshold_px),
        float(cfg.ik_object_floor_z_m),
    )

    i_pan = ARM_MOTORS.index("shoulder_pan")
    i_tilt = ARM_MOTORS.index("wrist_flex")
    dt_target = 1.0 / max(1.0, float(cfg.loop_hz))
    log_every_n = (
        max(1, int(cfg.log_every_n))
        if int(cfg.log_every_n) > 0
        else max(1, int(round(float(cfg.loop_hz))))
    )

    # State machine
    state = "SEARCH"
    detection_streak = 0
    miss_streak = 0
    centered_streak = 0

    # Depth EMA
    d_filt: float | None = None

    # Search bookkeeping
    search_seed: np.ndarray | None = None
    search_t0: float | None = None
    search_ramp = 0
    search_phase = "sweep"  # "reacquire" | "sweep"
    search_reacquire_frames = 0
    lock_uv: tuple[float, float] | None = None
    lock_joints: np.ndarray | None = None

    live = _init_live_runtime(cfg)
    want_keys = bool(getattr(cfg, "live_control_keypress", False)) and bool(
        cfg.live_control_stdin
    )
    if want_keys and _install_stdin_keypress(live):
        logger.info(
            "[gaze-live] keypress: [ ]/↑↓=el ; ,=back .=in ; -=back2x =in2x (orbit) ; "
            "p=preposition ; ? = help",
        )
    elif want_keys:
        logger.warning(
            "[gaze-live] --live-control-keypress ignored (stdin not a TTY); use a real terminal"
        )

    if bool(cfg.live_control_stdin) or str(getattr(cfg, "live_control_file", "") or "").strip():
        logger.info(
            "[gaze-live] enabled: stdin=%s keypress=%s file=%r — type `help` for line commands",
            bool(cfg.live_control_stdin),
            bool(live.get("_stdin_keypress")),
            str(getattr(cfg, "live_control_file", "") or ""),
        )

    last_t = time.time()
    tick = 0

    # One-time geometry sanity log at the current pose: where is the camera in
    # base frame and which way does cam +Z point? If this looks wrong vs your
    # physical setup, your --gripper-camera-tf is mis-calibrated.
    try:
        _initial_obs = robot.get_observation()
        _q0 = np.array(
            [float(_initial_obs[f"{m}.pos"]) for m in ARM_MOTORS], dtype=np.float64
        )
        _T0 = np.asarray(kin.forward_kinematics(_q0), dtype=np.float64) @ T_ee_cam
        _eye0 = _T0[:3, 3]
        _zb0 = _T0[:3, 2]
        _horiz = float(np.hypot(_zb0[0], _zb0[1])) + 1e-9
        _pitch_below_horiz_deg = math.degrees(math.atan2(-float(_zb0[2]), _horiz))
        logger.info(
            "[gaze-engine] startup camera pose (base frame): eye=(%.3f, %.3f, %.3f) m, "
            "cam+Z=(%+.3f, %+.3f, %+.3f), pitch_below_horizon=%+.1f°",
            float(_eye0[0]), float(_eye0[1]), float(_eye0[2]),
            float(_zb0[0]), float(_zb0[1]), float(_zb0[2]),
            _pitch_below_horiz_deg,
        )
    except Exception as e:
        logger.warning("[gaze-engine] startup geometry log failed: %s", e)

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
            T_base_ee_cur = np.asarray(kin.forward_kinematics(joints), dtype=np.float64)
            T_base_cam_cur = T_base_ee_cur @ T_ee_cam

            # Rotate open-vocabulary prompts during SEARCH before running YOLO.
            if state == "SEARCH" and len(search_vocab_queries) > 1:
                per = max(0.5, float(cfg.search_query_rotate_period_s))
                if search_vocab_last_switch is None:
                    search_vocab_last_switch = float(loop_t)
                elif float(loop_t) - float(search_vocab_last_switch) >= per:
                    search_vocab_idx = (int(search_vocab_idx) + 1) % len(search_vocab_queries)
                    sq = search_vocab_queries[int(search_vocab_idx)]
                    detector.set_query(sq)
                    active_query_label = str(sq)
                    search_vocab_last_switch = float(loop_t)
                    logger.info(
                        "[gaze-engine] SEARCH vocab %d/%d: %r",
                        int(search_vocab_idx) + 1,
                        len(search_vocab_queries),
                        sq,
                    )

            det = detector.best_detection(np.asarray(rgb))
            bbox_xyxy: tuple[float, float, float, float] | None = None
            uv: tuple[float, float] | None = None
            conf = 0.0
            if det is not None:
                conf = float(getattr(det, "confidence", 0.0))
                if conf >= float(cfg.min_detection_confidence):
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

            detected = bbox_xyxy is not None and uv is not None

            if detected and state != "SEARCH" and uv is not None:
                lock_uv = (float(uv[0]), float(uv[1]))
                lock_joints = np.asarray(joints, dtype=np.float64).copy()

            # Bbox-size pinhole depth (the depth signal we trust at close range)
            d_bbox: float | None = None
            if detected:
                d_bbox = depth_from_bbox_size(
                    bbox_xyxy,
                    fx=fx,
                    fy=fy,
                    target_physical_size_m=float(cfg.target_physical_size_m),
                )
                if d_bbox is not None:
                    sc = float(np.clip(float(cfg.bbox_depth_scale), 0.25, 4.0))
                    off = float(getattr(cfg, "bbox_depth_offset_m", 0.0))
                    d_bbox = max(0.005, float(d_bbox) * sc + off)
                    alpha = float(cfg.depth_ema_alpha)
                    d_filt = (
                        float(d_bbox)
                        if d_filt is None
                        else (1.0 - alpha) * d_filt + alpha * float(d_bbox)
                    )

            _drain_live_commands(cfg, live)
            _live_slew_orbit_targets(live, cfg, dt)
            if live.get("_goto_preposition", False):
                if not bool(cfg.preposition_enabled):
                    live["_goto_preposition"] = False
                elif detected:
                    if state != "PREPOSITIONING":
                        logger.info("[gaze-engine] LIVE→PREPOSITIONING (user command)")
                    state = "PREPOSITIONING"
                    centered_streak = 0
                    live["_goto_preposition"] = False

            # ---------- SEARCH ----------
            if state == "SEARCH":
                if detected:
                    detection_streak += 1
                else:
                    detection_streak = 0

                if not bool(cfg.search_enabled):
                    if tick % log_every_n == 0:
                        logger.info(
                            "[gaze-engine] tick=%d state=SEARCH search_enabled=false det=%s",
                            tick,
                            detected,
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
                        conf=conf,
                        phase="search",
                        depth_m=d_bbox,
                        kin=kin,
                        joints_deg=joints,
                        T_base_ee=T_base_ee_cur,
                        T_base_cam=T_base_cam_cur,
                        p_obj_base=None,
                        semantic_label=active_query_label,
                    )
                    _sleep(loop_t, dt_target)
                    continue

                if search_seed is None:
                    search_seed = joints.copy()
                    search_t0 = loop_t
                    search_ramp = 0
                search_ramp += 1
                ramp_alpha = min(
                    1.0, float(search_ramp) / max(1.0, float(cfg.search_pose_ramp_iters))
                )
                elapsed = float(loop_t - (search_t0 or loop_t))
                pan_wrist_only = bool(getattr(cfg, "search_joints_pan_wrist_only", True))
                in_reacquire = (
                    str(search_phase) == "reacquire"
                    and lock_uv is not None
                    and lock_joints is not None
                )
                if in_reacquire:
                    search_reacquire_frames += 1
                    q_search = _search_reacquire_command(
                        joints_deg=joints,
                        lock_joints_deg=lock_joints,
                        lock_uv=lock_uv,
                        cx0=cx0,
                        cy0=cy0,
                        fx=fx,
                        fy=fy,
                        cfg=cfg,
                        retreat_iter=search_reacquire_frames,
                        retreat_iters=int(cfg.search_reacquire_retreat_iters),
                    )
                    if search_reacquire_frames >= int(cfg.search_reacquire_max_frames):
                        search_phase = "sweep"
                        search_seed = np.asarray(lock_joints, dtype=np.float64).copy()
                        search_t0 = loop_t
                        search_ramp = 0
                        logger.info(
                            "[gaze-engine] SEARCH reacquire→sweep (no lock in %d frames, "
                            "last_uv=(%.0f,%.0f))",
                            int(cfg.search_reacquire_max_frames),
                            float(lock_uv[0]),
                            float(lock_uv[1]),
                        )
                else:
                    q_search = _search_command(
                        seed=search_seed,
                        elapsed_s=elapsed,
                        ramp_alpha=ramp_alpha,
                        pan_amp_deg=float(cfg.search_pan_amplitude_deg),
                        pan_period_s=float(cfg.search_pan_period_s),
                        lift_target_deg=float(cfg.search_shoulder_lift_target_deg),
                        wrist_start_deg=float(cfg.search_wrist_flex_start_deg),
                        wrist_end_deg=float(cfg.search_wrist_flex_end_deg),
                        look_up_period_s=float(cfg.search_look_up_period_s),
                        pan_wrist_only=pan_wrist_only,
                    )
                q_search = _apply_live_joint_trims(cfg, live, q_search)
                act = {f"{m}.pos": float(q_search[i]) for i, m in enumerate(ARM_MOTORS)}
                if "gripper.pos" in obs:
                    act["gripper.pos"] = float(obs["gripper.pos"])
                try:
                    robot.send_action(act)
                except Exception as e:
                    logger.warning("[gaze-engine] send_action failed (search): %s", e)

                if detection_streak >= int(cfg.lock_required_frames):
                    if bool(cfg.preposition_enabled) and bool(
                        getattr(cfg, "preposition_from_search", False)
                    ):
                        next_state = "PREPOSITIONING"
                    else:
                        next_state = "TRACKING"
                    logger.info(
                        "[gaze-engine] SEARCH→%s (locked on %d consecutive frames, conf=%.2f)",
                        next_state,
                        detection_streak,
                        conf,
                    )
                    state = next_state
                    detection_streak = 0
                    miss_streak = 0
                    centered_streak = 0
                    search_seed = None
                    search_t0 = None
                    search_ramp = 0
                    search_phase = "sweep"
                    search_reacquire_frames = 0

                if tick % log_every_n == 0:
                    logger.info(
                        "[gaze-engine] tick=%d state=SEARCH phase=%s det=%s conf=%.2f "
                        "streak=%d ramp=%.2f",
                        tick,
                        search_phase,
                        detected,
                        conf,
                        detection_streak,
                        ramp_alpha,
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
                    conf=conf,
                    phase="search",
                    depth_m=d_bbox,
                    kin=kin,
                    joints_deg=joints,
                    T_base_ee=T_base_ee_cur,
                    T_base_cam=T_base_cam_cur,
                    p_obj_base=None,
                    semantic_label=active_query_label,
                )
                _sleep(loop_t, dt_target)
                continue

            # ---------- TRACKING / APPROACH / HOLD share detection bookkeeping ----------
            if not detected:
                miss_streak += 1
                if miss_streak >= int(cfg.track_lost_frames):
                    logger.info(
                        "[gaze-engine] %s→SEARCH (lost target for %d frames)",
                        state,
                        miss_streak,
                    )
                    state = "SEARCH"
                    miss_streak = 0
                    centered_streak = 0
                    d_filt = None
                    search_vocab_idx = 0
                    search_vocab_last_switch = None
                    if search_vocab_queries:
                        detector.set_query(search_vocab_queries[0])
                        active_query_label = str(search_vocab_queries[0])
                    search_reacquire_frames = 0
                    search_ramp = 0
                    if (
                        bool(getattr(cfg, "search_reacquire_on_lost", True))
                        and lock_uv is not None
                        and lock_joints is not None
                    ):
                        search_phase = "reacquire"
                        search_seed = np.asarray(lock_joints, dtype=np.float64).copy()
                        search_t0 = loop_t
                        logger.info(
                            "[gaze-engine] SEARCH reacquire at last_uv=(%.0f,%.0f)",
                            float(lock_uv[0]),
                            float(lock_uv[1]),
                        )
                    else:
                        search_phase = "sweep"
                        search_seed = None
                        search_t0 = None
                _rerun_log(
                    cfg=cfg,
                    frame=tick,
                    rgb=rgb,
                    depth=depth,
                    bbox_xyxy=None,
                    uv=None,
                    cx0=cx0,
                    cy0=cy0,
                    conf=0.0,
                    phase=state.lower(),
                    depth_m=d_filt,
                    kin=kin,
                    joints_deg=joints,
                    T_base_ee=T_base_ee_cur,
                    T_base_cam=T_base_cam_cur,
                    p_obj_base=None,
                    semantic_label=active_query_label,
                )
                _sleep(loop_t, dt_target)
                continue
            miss_streak = 0

            d_pan, d_tilt, pixel_err = _gaze_joint_deltas(
                uv=uv, cx0=cx0, cy0=cy0, fx=fx, fy=fy, cfg=cfg
            )
            pan_mult = 1.0
            if live["pan"] is not None:
                pan_mult = float(np.clip(float(live["pan"]), 0.0, 1.0))
            elif bool(cfg.gaze_pan_scale_aligned_enabled) and state in (
                "TRACKING",
                "APPROACHING",
                "HOLD",
                "PREPOSITIONING",
            ):
                if pixel_err < float(cfg.approach_pixel_threshold_px):
                    pan_mult = float(cfg.gaze_pan_scale_when_aligned)
            d_pan = float(d_pan) * float(pan_mult)

            # Default action: gaze deltas only (used by TRACKING and HOLD)
            q_cmd = joints.copy()

            advance_to_approach = False
            regress_to_tracking = False
            regress_to_preposition = False
            preposition_done = False
            approach_done = False
            approach_step_m = 0.0
            approach_fov_scale = 1.0
            preposition_pos_err_m = float("nan")

            # Back-project bbox center with bbox-size depth into base frame —
            # used by PREPOSITIONING as p_obj, and by viz for everyone.
            p_obj_for_state: np.ndarray | None = None
            if d_filt is not None and uv is not None:
                try:
                    p_obj_for_state = point_cam_to_base(
                        T_base_cam_cur,
                        u=float(uv[0]),
                        v_pix=float(uv[1]),
                        depth_m=float(d_filt),
                        fx=fx,
                        fy=fy,
                        cx0=cx0,
                        cy0=cy0,
                    )
                except Exception:
                    p_obj_for_state = None
            p_obj_for_state = _maybe_floor_object_base(
                p_obj_for_state, float(cfg.ik_object_floor_z_m)
            )

            if state == "PREPOSITIONING":
                emergency_gaze = float(pixel_err) >= float(
                    cfg.preposition_emergency_gaze_pixel_threshold_px
                )
                orbit_live = _live_orbit_keys_active(live, loop_t)
                if orbit_live:
                    emergency_gaze = False
                if p_obj_for_state is not None and not emergency_gaze:
                    pre_lin: float | None = None
                    pre_dq: float | None = None
                    if loop_t < float(live.get("_boost_until", 0.0)):
                        pre_lin = float(cfg.live_preposition_boost_lin_vel_m_s)
                        pre_dq = float(cfg.live_preposition_boost_joint_step_deg)
                    q_pre, p_eye_target, pos_err = _preposition_q(
                        joints_deg=joints,
                        T_base_ee_cur=T_base_ee_cur,
                        T_ee_cam=T_ee_cam,
                        p_obj_base=p_obj_for_state,
                        cfg=cfg,
                        dt=dt_target,
                        kin=kin,
                        approach_az_deg=float(live["az"]),
                        approach_el_deg=float(live["el"]),
                        preposition_radius_m=float(live["radius"]),
                        max_lin_vel_m_s=pre_lin,
                        max_joint_step_deg=pre_dq,
                    )
                    preposition_pos_err_m = float(pos_err)
                    if q_pre is not None:
                        q_cmd = q_pre
                    if bool(cfg.preposition_require_centered_bbox):
                        ok_pre = (
                            pos_err < float(cfg.preposition_position_tolerance_m)
                            and pixel_err < float(cfg.approach_pixel_threshold_px)
                        )
                    else:
                        ok_pre = pos_err < float(cfg.preposition_position_tolerance_m)
                    if ok_pre:
                        centered_streak += 1
                        if centered_streak >= int(cfg.approach_consecutive_centered_frames):
                            preposition_done = True
                    else:
                        centered_streak = 0
                elif p_obj_for_state is not None:
                    preposition_pos_err_m = float(
                        np.linalg.norm(
                            T_base_ee_cur[:3, 3]
                            - (
                                np.asarray(p_obj_for_state, dtype=np.float64)
                                + approach_unit_vector(
                                    float(live["az"]), float(live["el"])
                                )
                                * float(live["radius"])
                            )
                        )
                    )

            elif state == "TRACKING":
                if pixel_err < float(cfg.approach_pixel_threshold_px):
                    centered_streak += 1
                    if centered_streak >= int(cfg.approach_consecutive_centered_frames):
                        advance_to_approach = True
                else:
                    centered_streak = 0

            elif state == "APPROACHING":
                if pixel_err > float(cfg.approach_regress_pixel_threshold_px):
                    if bool(cfg.preposition_enabled):
                        regress_to_preposition = True
                    else:
                        regress_to_tracking = True
                else:
                    if d_filt is not None:
                        apr_lin: float | None = None
                        apr_dq: float | None = None
                        apr_fov_min: float | None = None
                        if loop_t < float(live.get("_boost_until", 0.0)):
                            apr_lin = float(cfg.live_approach_boost_lin_vel_m_s)
                            apr_dq = float(cfg.live_preposition_boost_joint_step_deg)
                            apr_fov_min = float(cfg.live_approach_boost_fov_scale_min)
                        q_apr, planned_step, fov_scale = _approach_q(
                            joints_deg=joints,
                            T_base_ee_cur=T_base_ee_cur,
                            T_base_cam_cur=T_base_cam_cur,
                            d_obj_m=float(d_filt),
                            p_obj_base=p_obj_for_state,
                            pixel_err_px=float(pixel_err),
                            cfg=cfg,
                            dt=dt_target,
                            kin=kin,
                            final_standoff_m=float(live["standoff"]),
                            max_lin_vel_m_s=apr_lin,
                            max_joint_step_deg=apr_dq,
                            fov_scale_min=apr_fov_min,
                        )
                        approach_step_m = float(planned_step)
                        approach_fov_scale = float(fov_scale)
                        if q_apr is not None:
                            q_cmd = q_apr
                        if (
                            abs(float(d_filt) - float(live["standoff"]))
                            < float(cfg.approach_done_tolerance_m)
                        ):
                            approach_done = True

            elif state == "HOLD":
                # Maintain gaze, no forward motion
                pass

            # Gaze deltas (after any approach / preposition IK). During PREPOSITIONING
            # gaze is normally off so look-at IK is not fought — unless bbox error is
            # large (emergency) so we keep the target in view.
            allow_pre_gaze = bool(cfg.preposition_apply_gaze) or (
                state == "PREPOSITIONING"
                and float(pixel_err)
                >= float(cfg.preposition_emergency_gaze_pixel_threshold_px)
                and not _live_orbit_keys_active(live, loop_t)
            )
            if not (state == "PREPOSITIONING" and not allow_pre_gaze):
                q_cmd[i_pan] = float(q_cmd[i_pan]) + d_pan
                q_cmd[i_tilt] = float(q_cmd[i_tilt]) + d_tilt

            q_out = _apply_live_joint_trims(cfg, live, q_cmd)
            act = {f"{m}.pos": float(q_out[i]) for i, m in enumerate(ARM_MOTORS)}
            if "gripper.pos" in obs:
                act["gripper.pos"] = float(obs["gripper.pos"])
            try:
                robot.send_action(act)
            except Exception as e:
                logger.warning("[gaze-engine] send_action failed: %s", e)

            # State transitions (apply AFTER sending action so the log reflects
            # what we actually commanded this tick)
            if preposition_done:
                if bool(cfg.preposition_require_centered_bbox):
                    logger.info(
                        "[gaze-engine] PREPOSITIONING→APPROACHING "
                        "(pos_err=%.3fm < %.3fm, pixel_err=%.1fpx, d=%.3fm)",
                        preposition_pos_err_m,
                        float(cfg.preposition_position_tolerance_m),
                        pixel_err,
                        float(d_filt) if d_filt is not None else float("nan"),
                    )
                    state = "APPROACHING"
                else:
                    logger.info(
                        "[gaze-engine] PREPOSITIONING→TRACKING "
                        "(pos_err=%.3fm < %.3fm; centering before depth approach, d=%.3fm)",
                        preposition_pos_err_m,
                        float(cfg.preposition_position_tolerance_m),
                        float(d_filt) if d_filt is not None else float("nan"),
                    )
                    state = "TRACKING"
                centered_streak = 0
            elif advance_to_approach:
                logger.info(
                    "[gaze-engine] TRACKING→APPROACHING (centered %d frames, err=%.1fpx, d=%.3fm)",
                    centered_streak,
                    pixel_err,
                    float(d_filt) if d_filt is not None else float("nan"),
                )
                state = "APPROACHING"
                centered_streak = 0
            elif regress_to_preposition:
                logger.info(
                    "[gaze-engine] APPROACHING→PREPOSITIONING (pixel_err=%.1fpx > regress threshold)",
                    pixel_err,
                )
                state = "PREPOSITIONING"
                centered_streak = 0
            elif regress_to_tracking:
                logger.info(
                    "[gaze-engine] APPROACHING→TRACKING (pixel_err=%.1fpx > regress threshold)",
                    pixel_err,
                )
                state = "TRACKING"
                centered_streak = 0
            elif approach_done:
                logger.info(
                    "[gaze-engine] APPROACHING→HOLD (d=%.3fm within %.3fm of target)",
                    float(d_filt) if d_filt is not None else float("nan"),
                    float(cfg.approach_done_tolerance_m),
                )
                state = "HOLD"

            if tick % log_every_n == 0:
                d_tgt = float(live["standoff"])
                d_meas = float(d_filt) if d_filt is not None else float("nan")
                depth_err = (
                    float(d_meas - d_tgt)
                    if d_filt is not None
                    else float("nan")
                )
                pre_err_log = (
                    preposition_pos_err_m
                    if state == "PREPOSITIONING"
                    else float("nan")
                )
                logger.info(
                    "[gaze-engine] tick=%d state=%s det=True conf=%.2f "
                    "pixel_err=%.1fpx d_bbox=%.3fm d_filt=%.3fm d_target=%.3fm "
                    "depth_err=%+.3fm gaze=(Δpan=%+.2f°,Δtilt=%+.2f°) "
                    "approach_step=%+.4fm fov_scale=%.2f preposition_pos_err=%.3fm",
                    tick,
                    state,
                    conf,
                    pixel_err,
                    float(d_bbox) if d_bbox is not None else float("nan"),
                    d_meas,
                    d_tgt,
                    depth_err,
                    d_pan,
                    d_tilt,
                    approach_step_m,
                    approach_fov_scale,
                    pre_err_log,
                )

            # For viz: only render the object cube when we have a fresh
            # back-projection from THIS tick's detection. Avoids the "phantom
            # cube under the floor" symptom when uv/d_filt are stale.
            p_obj_viz: np.ndarray | None = None
            if detected and d_bbox is not None and uv is not None:
                try:
                    p_obj_viz = point_cam_to_base(
                        T_base_cam_cur,
                        u=float(uv[0]),
                        v_pix=float(uv[1]),
                        depth_m=float(d_filt) if d_filt is not None else float(d_bbox),
                        fx=fx,
                        fy=fy,
                        cx0=cx0,
                        cy0=cy0,
                    )
                    p_obj_viz = _maybe_floor_object_base(
                        p_obj_viz, float(cfg.ik_object_floor_z_m)
                    )
                    if bool(cfg.viz_clamp_object_to_ground) and p_obj_viz is not None:
                        gz = float(cfg.viz_ground_plane_z_m)
                        if float(p_obj_viz[2]) < gz:
                            p_obj_viz = np.asarray(p_obj_viz, dtype=np.float64).copy()
                            p_obj_viz[2] = gz
                except Exception:
                    p_obj_viz = None

            _rerun_log(
                cfg=cfg,
                frame=tick,
                rgb=rgb,
                depth=depth,
                bbox_xyxy=bbox_xyxy,
                uv=uv,
                cx0=cx0,
                cy0=cy0,
                conf=conf,
                phase=state.lower(),
                depth_m=d_filt,
                kin=kin,
                joints_deg=joints,
                T_base_ee=T_base_ee_cur,
                T_base_cam=T_base_cam_cur,
                p_obj_base=p_obj_viz,
                semantic_label=active_query_label,
            )

            _sleep(loop_t, dt_target)
    except KeyboardInterrupt:
        logger.info("[gaze-engine] interrupted by user.")
    finally:
        _restore_stdin_tty(live)
