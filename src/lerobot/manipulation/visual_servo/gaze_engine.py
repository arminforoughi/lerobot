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
  PAN_ALIGN → shoulder_pan (j1) only until the bbox is centered in u; then approach.
  TRACKING  → gaze only until centered, then PAN_ALIGN or APPROACHING.
  APPROACHING → radial approach + look-at IK (blocked until PAN_ALIGN done).
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
from lerobot.manipulation.yolo_track.depth import depth_from_bbox_size, median_depth_m
from lerobot.manipulation.yolo_track.math_utils import (
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
    # Look-at orientation during orbit IK. Keep low (0–0.1) so the solver moves
    # shoulder/elbow together; high values let the wrist satisfy rotation alone.
    preposition_ik_orientation_weight: float = 0.0
    # If False, skip orbit PREPOSITIONING on lock; go straight to depth APPROACHING
    # (continuous whole-arm servo). Orbit still available via live ``p`` / keys.
    preposition_use_orbit: bool = False
    # If False, do not add gaze Δpan/Δtilt on top of preposition IK.
    preposition_apply_gaze: bool = False
    # Legacy: emergency gaze disabled when preposition_apply_gaze is False and
    # fine_gaze_depth_err_m gates all gaze (see below).
    preposition_emergency_gaze_pixel_threshold_px: float = 9999.0
    preposition_emergency_ik_orientation_weight: float = 0.0
    # If orbit IK does not converge within this time, proceed to APPROACHING anyway.
    preposition_stuck_approach_s: float = 6.0
    # On entering PREPOSITIONING, set orbit radius from current bbox depth (clamped).
    preposition_sync_radius_to_depth: bool = True
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
    approach_max_joint_step_deg: float = 3.0
    # If True, APPROACH moves the EE along the **radial** vector from current
    # eye position toward the back-projected object point (not just along the
    # camera's optical axis). Keeps the trajectory directed at the actual
    # target even when look-at orientation is imperfect — eliminates the
    # "dive into floor" failure mode when T_ee_cam is mis-calibrated.
    approach_use_radial_to_object: bool = True
    # Set to 0 to disable. Non-zero was an experiment; original always used radial.
    approach_optical_only_depth_m: float = 0.0
    approach_steep_camera_down_dot: float = 0.88
    # Soft visibility regulator. When pixel error exceeds this, the depth step
    # is scaled smoothly toward 0 (rather than waiting for the hard regress
    # threshold). 1.0 means no slowdown. Linear ramp between soft and regress.
    approach_fov_soft_threshold_px: float = 50.0
    # Top-down: slow approach using horizontal error only (not vertical parallax).
    approach_fov_use_pan_err_px: bool = True
    approach_fov_min_scale: float = 0.22
    # Steep camera + large vertical bbox error: slide along view axis, not radial to floor point.
    approach_steep_vertical_optical_px: float = 55.0
    # When farther than this from goal depth, scale up approach linear velocity.
    approach_depth_boost_err_m: float = 0.05
    approach_depth_boost_lin_scale: float = 2.0
    # If True, ignore pixel-error slowdown while still far in depth (faster but can
    # drift off bbox). Default False = original visibility regulator at all ranges.
    approach_depth_priority: bool = False
    # While depth_err is above this, do not slow approach for pan misalignment.
    approach_depth_priority_min_err_m: float = 0.055
    # Minimum fraction of remaining depth error closed per tick (capped by max speed).
    approach_depth_min_fraction_per_tick: float = 0.22
    # Fuse OAK-D stereo ROI depth with bbox pinhole depth for approach.
    approach_use_stereo_depth: bool = True
    # When both exist, use max() so a shrinking bbox does not fake "too close" and retreat.
    approach_depth_pessimistic_fusion: bool = True
    # Only step backward if farther than standoff by at least this much (m).
    approach_retreat_min_err_m: float = 0.028
    # Cap how much filtered depth can increase per tick while APPROACHING (m).
    approach_depth_max_increase_per_tick_m: float = 0.010
    # Below this bbox depth (m), trust pinhole size only — stereo hits the table.
    approach_bbox_only_depth_max_m: float = 0.22
    # Reject stereo if it disagrees with bbox by more than this factor.
    approach_stereo_bbox_max_ratio: float = 1.35
    approach_stereo_bbox_min_ratio: float = 0.55
    # Pause forward translation only when already near standoff *and* off-center.
    approach_pause_depth_max_m: float = 0.20
    approach_pause_pixel_err_px: float = 55.0
    # Still move in if farther than this from goal depth (avoids gaze-only curl at ~15 cm).
    approach_pause_max_depth_err_m: float = 0.048
    # Do not declare approach done on depth alone — bbox must be reasonably centered.
    approach_require_centered_for_done: bool = True
    # Large pan error during coarse approach → re-run j1 PAN_ALIGN.
    approach_regress_to_pan_align: bool = False
    approach_regress_to_pan_align_px: float = 58.0
    approach_regress_to_pan_align_frames: int = 10
    approach_regress_to_pan_align_cooldown_s: float = 2.5
    # EMA on bbox center during APPROACHING (lower = smoother, less bbox twitch).
    approach_gaze_uv_ema_alpha: float = 0.22
    # Reject bbox-center jumps larger than this (px/tick) — YOLO flicker.
    gaze_uv_max_step_px: float = 45.0

    # Gaze (P-control on pixel error → joint deltas, no IK)
    gaze_kp_pan: float = 0.32
    gaze_kp_tilt: float = 0.22
    gaze_max_step_pan_deg: float = 1.8
    gaze_max_step_tilt_deg: float = 1.4
    gaze_deadband_px: float = 10.0
    # EMA on bbox center before gaze (reduces wrist/pan jitter from detector noise).
    gaze_uv_ema_alpha: float = 0.35
    pan_sign: float = 1.0
    wrist_tilt_sign: float = 1.0
    # When bbox is already near the image center, scale down shoulder_pan gaze so
    # the arm re-orients mostly with shoulder_lift / elbow / wrist (joints 2–4)
    # via preposition IK, not endless left–right pan.
    gaze_pan_scale_when_aligned: float = 0.35
    gaze_pan_scale_aligned_enabled: bool = True
    # During APPROACHING (only when depth is already near goal), scale gaze.
    gaze_scale_during_approach: float = 1.0
    gaze_tilt_scale_during_approach: float = 1.0
    gaze_pan_scale_during_close_approach: float = 1.0
    # PREPOSITION: wrist tilt only (shoulder_pan would fight orbit IK).
    gaze_scale_during_preposition: float = 0.0
    preposition_gaze_tilt_scale: float = 0.55
    # Allow wrist tilt while the bbox is off-center vertically, even when far in depth.
    coarse_gaze_tilt_pixel_err_px: float = 24.0
    # Pixel centering (pan/tilt) only when |d_filt - standoff| is below this (m).
    fine_gaze_depth_err_m: float = 0.045
    # Coarse approach: whole-arm look-at IK toward the back-projected object
    # (keeps the bbox in view; position-only IK leaves the camera pointing at
    # the floor while the arm translates).
    approach_coarse_look_at: bool = True
    approach_coarse_ik_orientation_weight: float = 0.45
    approach_coarse_max_ang_vel_deg_s: float = 45.0
    # While SEARCH is sweeping, nudge pan/wrist toward a live detection.
    search_detection_gaze_scale: float = 0.65

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
    # Legacy alias (stdin ``closeness`` command); keypress uses separate radius/standoff.
    live_closeness_step_m_default: float = 0.03
    # Key-repeat coalescing: at most this many logical steps per control tick.
    # Orbit radius/el used by IK slew toward the live target at this rate.
    live_radius_slew_m_s: float = 0.18
    live_standoff_slew_m_s: float = 0.04
    live_el_slew_deg_s: float = 32.0
    live_approach_boost_lin_vel_m_s: float = 0.06
    live_approach_boost_fov_scale_min: float = 0.85
    # Briefly raise preposition speed after a live key so motion keeps up with
    # the target (avoids "nothing… then jump").
    live_preposition_boost_duration_s: float = 0.55
    live_orbit_boost_duration_s: float = 1.25
    live_preposition_boost_lin_vel_m_s: float = 0.14
    live_preposition_boost_joint_step_deg: float = 9.0
    live_preposition_boost_ang_vel_deg_s: float = 120.0
    # During live [ ] , . - = : use look-at orientation like the original orbit tune.
    live_preposition_ik_orientation_weight: float = 1.2
    # If True, step toward full orbit pose each tick (can feel jumpy); else rate-limited.
    live_preposition_snap_se3: bool = False
    # When snap_se3: max fraction of pose gap closed per tick (0.25–0.5 = smooth).
    live_preposition_snap_alpha_max: float = 0.32
    # Keypress: jump el/radius/standoff targets immediately; False = slew toward target.
    live_keys_snap_targets: bool = False
    live_key_max_steps_per_tick: int = 2

    # State machine
    lock_required_frames: int = 4
    track_lost_frames: int = 12
    approach_pixel_threshold_px: float = 40.0
    approach_consecutive_centered_frames: int = 2
    approach_regress_pixel_threshold_px: float = 85.0
    # If still far in depth, allow APPROACHING with a looser centering gate.
    approach_depth_bypass_enabled: bool = True
    approach_depth_bypass_err_m: float = 0.07
    approach_depth_bypass_pixel_threshold_px: float = 52.0
    # After lock, jump to orbit IK when bbox depth is this far past goal standoff.
    preposition_on_lock_depth_err_m: float = 0.09
    # If TRACKING cannot center for this long while still far, auto PREPOSITION.
    tracking_stuck_preposition_s: float = 5.0
    # Require shoulder_pan (j1) to center the target in u before orbit/approach IK.
    require_pan_align: bool = True
    pan_align_threshold_px: float = 28.0
    pan_align_consecutive_frames: int = 4
    pan_align_kp_pan: float = 0.55
    pan_align_max_step_pan_deg: float = 2.8
    # Above this |u−cx|, use faster j1 steps and wrist tilt (not j1-only).
    pan_align_coarse_pan_err_px: float = 50.0
    pan_align_coarse_max_step_pan_deg: float = 5.0
    pan_align_coarse_allow_tilt: bool = True
    # If the target drifts off-center in u during coarse approach, pause IK and re-pan.
    pan_align_gate_approach: bool = True
    # After lock, always PAN_ALIGN before APPROACHING (do not skip if u is momentarily ok).
    pan_align_always_on_lock: bool = True
    # Brief YOLO dropouts during approach: keep last bbox for IK/gaze (frames).
    detection_hold_frames: int = 10
    # Do not drop to TRACKING on large pixel_err while still far (avoids bbox flicker loop).
    approach_regress_to_tracking: bool = False
    # Scale down approach IK speed when pan is not yet centered.
    approach_slowdown_pan_err_px: float = 35.0
    approach_slowdown_lin_scale: float = 0.35
    # Motor read failures before exiting the loop (comm glitches).
    comm_error_max_consecutive: int = 25

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
        "_goto_approaching": False,
        "_boost_until": 0.0,
        "_orbit_live_until": 0.0,
        "_approach_live_until": 0.0,
        "_kb_buf": b"",
        "_termios_old": None,
        "_stdin_keypress": False,
        "_uv_filt": None,
        "_tracking_enter_t": None,
        "_preposition_enter_t": None,
        "_held_det": None,
        "_det_hold_used": 0,
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


def _live_arm_motion_boost(
    live: dict,
    cfg: GazeEngineConfig,
    *,
    approach: bool = False,
    orbit: bool = False,
) -> None:
    dur = float(getattr(cfg, "live_preposition_boost_duration_s", 0.55))
    if orbit and not approach:
        dur = max(dur, float(getattr(cfg, "live_orbit_boost_duration_s", 1.25)))
    now = time.time()
    live["_boost_until"] = max(float(live.get("_boost_until", 0.0)), now + dur)
    if orbit:
        live["_preposition_stuck_pause_until"] = max(
            float(live.get("_preposition_stuck_pause_until", 0.0)), now + dur
        )
        live["_preposition_enter_t"] = None
    live["_orbit_live_until"] = max(
        float(live.get("_orbit_live_until", 0.0)), now + dur
    )
    if approach:
        live["_approach_live_until"] = max(
            float(live.get("_approach_live_until", 0.0)), now + dur
        )


def _live_orbit_keys_active(live: dict, loop_t: float) -> bool:
    return loop_t < float(live.get("_orbit_live_until", 0.0))


def _live_approach_keys_active(live: dict, loop_t: float) -> bool:
    return loop_t < float(live.get("_approach_live_until", 0.0))


def _live_motion_override_active(live: dict, loop_t: float) -> bool:
    """User pressed a live key recently — bypass pan-only gate so IK actually moves."""
    return _live_orbit_keys_active(live, loop_t) or _live_approach_keys_active(
        live, loop_t
    )


def _live_snap_el(live: dict, el_deg: float) -> None:
    live["el"] = float(el_deg)
    live["el_target"] = float(el_deg)


def _live_snap_radius(live: dict, radius_m: float) -> None:
    live["radius"] = float(radius_m)
    live["radius_target"] = float(radius_m)


def _live_snap_standoff(live: dict, standoff_m: float) -> None:
    live["standoff"] = float(standoff_m)
    live["standoff_target"] = float(standoff_m)


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
        now = time.time()
        if now - float(live.get("_el_clamp_log_t", 0.0)) > 0.6:
            live["_el_clamp_log_t"] = now
            logger.info(
                "[gaze-live] approach_el_deg=%.1f° (clamped at %s%.0f°, widen with "
                "--live-el-min-deg/--live-el-max-deg)",
                new_el,
                "+" if signed_step > 0 else "",
                hi if signed_step > 0 else lo,
            )
        return
    live["el_target"] = new_el
    if bool(getattr(cfg, "live_keys_snap_targets", True)):
        _live_snap_el(live, new_el)
    _live_arm_motion_boost(live, cfg, orbit=True)
    live["_goto_preposition"] = True
    logger.info(
        "[gaze-live] approach_el_deg → %.1f° (%+.1f°, %s) PREPOSITION",
        new_el,
        float(signed_step),
        "snap" if bool(getattr(cfg, "live_keys_snap_targets", True)) else "slewing",
    )


def _live_delta_radius_m(live: dict, cfg: GazeEngineConfig, signed_step: float) -> None:
    cur = float(live.get("radius_target", live.get("radius", 0.2)))
    new_r = float(np.clip(cur + float(signed_step), 0.05, 0.50))
    if abs(new_r - cur) < 1e-6:
        return
    live["radius_target"] = new_r
    if bool(getattr(cfg, "live_keys_snap_targets", True)):
        _live_snap_radius(live, new_r)
    _live_arm_motion_boost(live, cfg, orbit=True)
    live["_goto_preposition"] = True
    logger.info(
        "[gaze-live] orbit radius → %.3fm (%+.3fm, %s) PREPOSITION",
        new_r,
        float(signed_step),
        "snap" if bool(getattr(cfg, "live_keys_snap_targets", True)) else "slewing",
    )


def _live_delta_standoff_m(live: dict, cfg: GazeEngineConfig, signed_step: float) -> None:
    cur = float(live.get("standoff_target", live.get("standoff", 0.06)))
    new_s = float(np.clip(cur + float(signed_step), 0.02, 0.40))
    if abs(new_s - cur) < 1e-6:
        return
    live["standoff_target"] = new_s
    if bool(getattr(cfg, "live_keys_snap_targets", True)):
        _live_snap_standoff(live, new_s)
    _live_arm_motion_boost(live, cfg, approach=True)
    live["_goto_approaching"] = True
    logger.info(
        "[gaze-live] goal depth (standoff) → %.3fm (%+.3fm, %s) APPROACHING",
        new_s,
        float(signed_step),
        "snap" if bool(getattr(cfg, "live_keys_snap_targets", True)) else "slewing",
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
            _live_arm_motion_boost(live, cfg, approach=True)
            live["_goto_approaching"] = True
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
        # , . = orbit radius; - / = = radius 2× (standoff: stdin ``depth`` only)
        elif c == ord(","):
            net_rad += 1
        elif c == ord("."):
            net_rad -= 1
        elif c in (ord("-"), ord("_")):
            net_rad += 2
        elif c in (ord("="), ord("+")):
            net_rad -= 2
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
        _live_arm_motion_boost(live, cfg, orbit=True)
        live["_goto_preposition"] = True
        logger.info("[gaze-live] PREPOSITION requested (key 'p')")
    if want_help:
        logger.info(
            "[gaze-live] keys: [ ]/↑↓=el ; ,.=radius ; -=back +=in (2×) ; "
            "depth via stdin ; max %d/tick ; p=preposition ; ? = help",
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


def _pan_horizontal_err_px(uv: tuple[float, float], cx0: float) -> float:
    return abs(float(uv[0]) - float(cx0))


def _is_pan_aligned(uv: tuple[float, float], cx0: float, cfg: GazeEngineConfig) -> bool:
    return _pan_horizontal_err_px(uv, cx0) < float(
        getattr(cfg, "pan_align_threshold_px", 28.0)
    )


def _gaze_pan_only_delta(
    *,
    uv: tuple[float, float],
    cx0: float,
    fx: float,
    cfg: GazeEngineConfig,
    pan_err_px: float | None = None,
) -> float:
    """Shoulder_pan (j1) only — used before whole-arm approach IK."""
    du = float(uv[0]) - float(cx0)
    if abs(du) <= float(cfg.gaze_deadband_px):
        return 0.0
    d_pan_deg = math.degrees(math.atan2(du, max(1e-6, float(fx))))
    kp = float(getattr(cfg, "pan_align_kp_pan", 0.55))
    cap = float(getattr(cfg, "pan_align_max_step_pan_deg", 2.8))
    coarse_px = float(getattr(cfg, "pan_align_coarse_pan_err_px", 50.0))
    if pan_err_px is not None and float(pan_err_px) > coarse_px:
        cap = float(getattr(cfg, "pan_align_coarse_max_step_pan_deg", 5.0))
    return float(
        np.clip(
            float(cfg.pan_sign) * kp * d_pan_deg,
            -cap,
            +cap,
        )
    )


def _pan_align_required(cfg: GazeEngineConfig) -> bool:
    return bool(getattr(cfg, "require_pan_align", True))


def _states_hold_detection() -> tuple[str, ...]:
    return ("PAN_ALIGN", "APPROACHING", "PREPOSITIONING", "HOLD", "TRACKING")


def _apply_detection_hold(
    *,
    detected: bool,
    bbox_xyxy: tuple[float, float, float, float] | None,
    uv: tuple[float, float] | None,
    conf: float,
    state: str,
    live: dict,
    cfg: GazeEngineConfig,
) -> tuple[bool, tuple[float, float, float, float] | None, tuple[float, float] | None, float]:
    """Reuse last good detection for a few frames so approach IK does not stutter."""
    hold_n = max(0, int(getattr(cfg, "detection_hold_frames", 10)))
    if detected and bbox_xyxy is not None and uv is not None:
        live["_held_det"] = {
            "bbox": tuple(bbox_xyxy),
            "uv": (float(uv[0]), float(uv[1])),
            "conf": float(conf),
        }
        live["_det_hold_used"] = 0
        return True, bbox_xyxy, uv, float(conf)

    if (
        hold_n > 0
        and state in _states_hold_detection()
        and live.get("_held_det") is not None
        and int(live.get("_det_hold_used", 0)) < hold_n
    ):
        h = live["_held_det"]
        live["_det_hold_used"] = int(live.get("_det_hold_used", 0)) + 1
        return (
            True,
            tuple(h["bbox"]),
            (float(h["uv"][0]), float(h["uv"][1])),
            float(h["conf"]),
        )
    live["_det_hold_used"] = 0
    return False, None, None, 0.0


def _filter_bbox_uv(
    live: dict,
    uv: tuple[float, float],
    cfg: GazeEngineConfig,
    *,
    state: str = "",
) -> tuple[float, float]:
    if state == "APPROACHING":
        alpha = float(
            np.clip(
                float(getattr(cfg, "approach_gaze_uv_ema_alpha", 0.22)),
                0.05,
                1.0,
            )
        )
    else:
        alpha = float(
            np.clip(float(getattr(cfg, "gaze_uv_ema_alpha", 0.35)), 0.05, 1.0)
        )
    prev = live.get("_uv_filt")
    u_in, v_in = float(uv[0]), float(uv[1])
    if prev is not None and state not in ("PAN_ALIGN", "SEARCH", "APPROACHING"):
        jump = float(
            math.hypot(u_in - float(prev[0]), v_in - float(prev[1]))
        )
        if jump > float(getattr(cfg, "gaze_uv_max_step_px", 45.0)):
            u_in, v_in = float(prev[0]), float(prev[1])
    if prev is None:
        live["_uv_filt"] = (u_in, v_in)
    else:
        live["_uv_filt"] = (
            (1.0 - alpha) * float(prev[0]) + alpha * u_in,
            (1.0 - alpha) * float(prev[1]) + alpha * v_in,
        )
    f = live["_uv_filt"]
    return float(f[0]), float(f[1])


def _should_pause_approach_forward(
    *,
    d_bbox: float | None,
    pixel_err_px: float,
    depth_err_m: float | None,
    cfg: GazeEngineConfig,
) -> bool:
    """Pause translation only when near goal depth and off-center — not at ~15 cm."""
    if d_bbox is None:
        return False
    if float(pixel_err_px) <= float(
        getattr(cfg, "approach_pause_pixel_err_px", 55.0)
    ):
        return False
    if float(d_bbox) > float(getattr(cfg, "approach_pause_depth_max_m", 0.20)):
        return False
    if depth_err_m is not None and float(depth_err_m) > float(
        getattr(cfg, "approach_pause_max_depth_err_m", 0.048)
    ):
        return False
    return True


def _depth_err_m(d_filt: float | None, standoff_m: float) -> float | None:
    if d_filt is None:
        return None
    return float(d_filt) - float(standoff_m)


def _measure_object_depth_m(
    *,
    depth_map: np.ndarray | None,
    bbox_xyxy: tuple[float, float, float, float],
    fx: float,
    fy: float,
    cfg: GazeEngineConfig,
    depth_scale: float,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(d_fused, d_bbox, d_stereo)``. Fused depth is pessimistic (farther) when enabled."""
    d_bbox = depth_from_bbox_size(
        bbox_xyxy,
        fx=float(fx),
        fy=float(fy),
        target_physical_size_m=float(cfg.target_physical_size_m),
    )
    if d_bbox is not None:
        sc = float(np.clip(float(cfg.bbox_depth_scale), 0.25, 4.0))
        off = float(getattr(cfg, "bbox_depth_offset_m", 0.0))
        d_bbox = max(0.005, float(d_bbox) * sc + off)

    d_stereo: float | None = None
    if bool(getattr(cfg, "approach_use_stereo_depth", True)) and depth_map is not None:
        try:
            d_stereo = median_depth_m(
                np.asarray(depth_map),
                bbox_xyxy,
                depth_scale=float(depth_scale),
                min_mm=50.0,
                max_mm=3500.0,
            )
        except Exception:
            d_stereo = None

    close_max = float(getattr(cfg, "approach_bbox_only_depth_max_m", 0.22))
    if d_bbox is not None and float(d_bbox) < close_max:
        return float(d_bbox), d_bbox, d_stereo

    if d_stereo is not None and d_bbox is not None:
        ratio = float(d_stereo) / max(float(d_bbox), 1e-3)
        r_max = float(getattr(cfg, "approach_stereo_bbox_max_ratio", 1.35))
        r_min = float(getattr(cfg, "approach_stereo_bbox_min_ratio", 0.55))
        if ratio > r_max or ratio < r_min:
            d_fused = float(d_bbox)
        elif bool(getattr(cfg, "approach_depth_pessimistic_fusion", True)):
            d_fused = max(float(d_stereo), float(d_bbox))
        else:
            d_fused = 0.5 * float(d_stereo) + 0.5 * float(d_bbox)
    else:
        d_fused = d_stereo if d_stereo is not None else d_bbox
    return d_fused, d_bbox, d_stereo


def _filter_approach_depth(
    live: dict,
    d_meas: float,
    *,
    state: str,
    cfg: GazeEngineConfig,
    alpha: float,
    d_filt_prev: float | None,
) -> float:
    """EMA depth with monotonic guard while APPROACHING (bbox shrink must not trigger retreat)."""
    dm = float(d_meas)
    if state != "APPROACHING":
        live.pop("_d_approach_min", None)
        if d_filt_prev is None:
            return dm
        return float((1.0 - alpha) * d_filt_prev + alpha * dm)

    prev_min = live.get("_d_approach_min")
    max_up = float(getattr(cfg, "approach_depth_max_increase_per_tick_m", 0.010))
    if prev_min is None:
        live["_d_approach_min"] = dm
    else:
        pm = float(prev_min)
        if dm < pm:
            live["_d_approach_min"] = dm
        dm = min(dm, pm + max_up)

    if d_filt_prev is None:
        return dm
    return float((1.0 - alpha) * d_filt_prev + alpha * dm)


def _reset_approach_depth_filter(live: dict, *, d_bbox: float | None = None) -> None:
    live.pop("_d_approach_min", None)
    if d_bbox is not None:
        live["_d_approach_reseed"] = float(d_bbox)


def _depth_for_geometry(
    d_filt: float | None, d_bbox: float | None, cfg: GazeEngineConfig
) -> float | None:
    """Depth used for back-projection / IK (stable close-range bbox, else filtered)."""
    if d_bbox is not None and float(d_bbox) < float(
        getattr(cfg, "approach_bbox_only_depth_max_m", 0.22)
    ):
        return float(d_bbox)
    return d_filt


def _sync_preposition_radius_to_depth(
    live: dict, cfg: GazeEngineConfig, d_filt: float | None
) -> None:
    """Set orbit radius near current camera range so IK target is reachable."""
    if d_filt is None:
        return
    stand = float(live["standoff"])
    r = float(
        np.clip(
            float(d_filt),
            stand + 0.04,
            float(getattr(cfg, "preposition_initial_radius_m", 0.2)) + 0.15,
        )
    )
    live["radius"] = r
    live["radius_target"] = r


def _gaze_allow_pan(
    *,
    state: str,
    depth_err_m: float | None,
    cfg: GazeEngineConfig,
    pan_aligned: bool,
) -> bool:
    """When False, zero shoulder_pan gaze (orbit IK handles pan in PREPOSITION)."""
    if state == "PAN_ALIGN":
        return True
    if state == "SEARCH":
        return False
    if state == "PREPOSITIONING":
        return bool(cfg.preposition_apply_gaze)
    if state in ("APPROACHING", "TRACKING", "HOLD"):
        return True
    return False


def _camera_steep_top_down(T_base_cam: np.ndarray, cfg: GazeEngineConfig) -> bool:
    z_ax = np.asarray(T_base_cam[:3, 2], dtype=np.float64)
    zn = float(np.linalg.norm(z_ax))
    if zn < 1e-9:
        return False
    return abs(float(z_ax[2] / zn)) >= float(
        getattr(cfg, "approach_steep_camera_down_dot", 0.88)
    )


def _approach_fov_regulator_px(
    *,
    pixel_err_px: float,
    pan_err_px: float,
    T_base_cam_cur: np.ndarray,
    cfg: GazeEngineConfig,
) -> float:
    """Pixel error used to throttle forward speed (not full 2D when top-down)."""
    if bool(getattr(cfg, "approach_fov_use_pan_err_px", True)) and _camera_steep_top_down(
        T_base_cam_cur, cfg
    ):
        return float(pan_err_px)
    return float(pixel_err_px)


def _gaze_allow_tilt(
    *,
    state: str,
    depth_err_m: float | None,
    cfg: GazeEngineConfig,
    pan_aligned: bool,
    pixel_err_px: float,
    vertical_err_px: float,
) -> bool:
    """Wrist tilt to center the bbox in v — allowed even when pan is gated off."""
    if state == "PAN_ALIGN":
        if not bool(getattr(cfg, "pan_align_coarse_allow_tilt", True)):
            return False
        v_thresh = float(getattr(cfg, "coarse_gaze_tilt_pixel_err_px", 24.0))
        return abs(float(vertical_err_px)) > float(cfg.gaze_deadband_px) and (
            abs(float(vertical_err_px)) >= v_thresh
            or float(pixel_err_px) >= v_thresh
        )
    if state == "SEARCH":
        return False
    v_thresh = float(getattr(cfg, "coarse_gaze_tilt_pixel_err_px", 24.0))
    if abs(float(vertical_err_px)) > float(cfg.gaze_deadband_px) and (
        abs(float(vertical_err_px)) >= v_thresh
        or float(pixel_err_px) >= v_thresh
    ):
        if state in ("APPROACHING", "PREPOSITIONING", "TRACKING", "HOLD"):
            return True
    if state == "PREPOSITIONING":
        return bool(cfg.preposition_apply_gaze)
    fine = float(getattr(cfg, "fine_gaze_depth_err_m", 0.045))
    if depth_err_m is not None and float(depth_err_m) > fine:
        return False
    return state in ("TRACKING", "APPROACHING", "HOLD")


def _use_orbit_preposition(cfg: GazeEngineConfig) -> bool:
    return bool(cfg.preposition_enabled) and bool(
        getattr(cfg, "preposition_use_orbit", False)
    )


def _approach_centering_ok(
    *,
    pixel_err_px: float,
    depth_err_m: float | None,
    cfg: GazeEngineConfig,
) -> bool:
    tight = float(cfg.approach_pixel_threshold_px)
    if pixel_err_px < tight:
        return True
    if not bool(getattr(cfg, "approach_depth_bypass_enabled", True)):
        return False
    if depth_err_m is None:
        return False
    if float(depth_err_m) < float(getattr(cfg, "approach_depth_bypass_err_m", 0.07)):
        return False
    return pixel_err_px < float(
        getattr(cfg, "approach_depth_bypass_pixel_threshold_px", 52.0)
    )


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
    max_ang_vel_deg_s: float | None = None,
    ik_orientation_weight: float | None = None,
    snap_se3: bool = False,
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
    ang_vel = float(
        max_ang_vel_deg_s
        if max_ang_vel_deg_s is not None
        else cfg.preposition_max_ang_vel_deg_s
    )
    if bool(snap_se3):
        alpha_cap = float(
            np.clip(
                float(getattr(cfg, "live_preposition_snap_alpha_max", 0.32)), 0.05, 1.0
            )
        )
        T_step = interpolate_se3(T_base_ee_cur, T_target, alpha_cap)
    else:
        T_step = _se3_rate_limited_step(
            T_base_ee_cur,
            T_target,
            dt=float(dt),
            max_lin_vel_m_s=lin_vel,
            max_ang_vel_deg_s=ang_vel,
        )
    pos_err = float(np.linalg.norm(T_base_ee_cur[:3, 3] - p_eye))
    q_new = None
    ow_primary = float(
        ik_orientation_weight
        if ik_orientation_weight is not None
        else cfg.preposition_ik_orientation_weight
    )
    for ow in (ow_primary, 0.15, 0.0):
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
    pan_err_px: float = 0.0,
    vertical_err_px: float = 0.0,
    cfg: GazeEngineConfig,
    dt: float,
    kin,
    final_standoff_m: float | None = None,
    max_lin_vel_m_s: float | None = None,
    max_joint_step_deg: float | None = None,
    fov_scale_min: float | None = None,
    ik_orientation_weight: float | None = None,
    coarse_approach: bool = False,
    T_ee_cam: np.ndarray | None = None,
    pause_forward: bool = False,
) -> tuple[np.ndarray | None, float, float]:
    """Radial (or optical) step toward standoff + soft slowdown when bbox off-center.

    Matches the original gaze-engine: IK slides toward the back-projected point,
    joint gaze keeps tracking the bbox; ``fov_scale`` cuts forward speed when the
    target drifts in the image (closer range = smaller errors = easier tracking).
    """
    eye_base = np.asarray(T_base_cam_cur[:3, 3], dtype=np.float64)
    z_ax = np.asarray(T_base_cam_cur[:3, 2], dtype=np.float64)
    zn = float(np.linalg.norm(z_ax))
    if zn > 1e-9:
        z_ax = z_ax / zn
    v_lim = float(getattr(cfg, "approach_steep_vertical_optical_px", 55.0))
    use_optical = _camera_steep_top_down(T_base_cam_cur, cfg) and abs(
        float(vertical_err_px)
    ) >= v_lim
    if (
        not use_optical
        and bool(cfg.approach_use_radial_to_object)
        and p_obj_base is not None
        and float(np.linalg.norm(np.asarray(p_obj_base) - eye_base)) > 1e-3
    ):
        radial = np.asarray(p_obj_base, dtype=np.float64) - eye_base
        direction = radial / float(np.linalg.norm(radial))
    else:
        direction = z_ax if zn > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    d_target = float(
        final_standoff_m if final_standoff_m is not None else cfg.final_standoff_m
    )
    err = float(d_obj_m) - d_target

    reg_px = _approach_fov_regulator_px(
        pixel_err_px=float(pixel_err_px),
        pan_err_px=float(pan_err_px),
        T_base_cam_cur=T_base_cam_cur,
        cfg=cfg,
    )
    soft = float(cfg.approach_fov_soft_threshold_px)
    hard = float(cfg.approach_regress_pixel_threshold_px)
    if reg_px <= soft or hard <= soft:
        fov_scale = 1.0
    else:
        fov_scale = max(
            0.0, 1.0 - (reg_px - soft) / max(1e-3, hard - soft)
        )
    fov_floor = float(getattr(cfg, "approach_fov_min_scale", 0.22))
    fov_scale = max(float(fov_scale), fov_floor)
    if fov_scale_min is not None:
        fov_scale = max(float(fov_scale), float(fov_scale_min))
    if bool(getattr(cfg, "approach_depth_priority", False)) and err > float(
        getattr(cfg, "approach_depth_priority_min_err_m", 0.055)
    ):
        fov_scale = 1.0

    lin_vel = float(
        max_lin_vel_m_s
        if max_lin_vel_m_s is not None
        else cfg.approach_max_lin_vel_m_s
    )
    boost_err = float(getattr(cfg, "approach_depth_boost_err_m", 0.05))
    boost_scale = float(getattr(cfg, "approach_depth_boost_lin_scale", 2.0))
    if err > boost_err and boost_scale > 1.0:
        lin_vel *= boost_scale
    max_step = lin_vel * float(dt)
    raw = float(cfg.approach_kp) * err * float(fov_scale)
    step = float(np.clip(raw, -max_step, +max_step))
    retreat_min = float(getattr(cfg, "approach_retreat_min_err_m", 0.028))
    if step < 0.0 and err > -retreat_min:
        step = 0.0
    if (
        bool(getattr(cfg, "approach_depth_priority", False))
        and err > boost_err
        and not bool(pause_forward)
    ):
        frac = float(getattr(cfg, "approach_depth_min_fraction_per_tick", 0.22))
        step = max(step, min(float(err) * frac, max_step))
    if bool(pause_forward):
        step = 0.0

    T_target = np.asarray(T_base_ee_cur, dtype=np.float64).copy()
    T_target[:3, 3] = T_base_ee_cur[:3, 3] + direction * step
    T_ik = T_target

    q_new = None
    ow0 = float(
        ik_orientation_weight
        if ik_orientation_weight is not None
        else cfg.ik_orientation_weight
    )
    for ow in (ow0, 0.15, 0.0):
        try:
            q_new = kin.inverse_kinematics(
                joints_deg,
                T_ik,
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
        "[gaze-engine] approach: coarse_look_at=%s depth_priority=%s "
        "regress_pan_align=%s (APPROACHING is radial+joint-gaze; "
        "approach_el/az apply to PREPOSITION orbit only — use [ ] or p)",
        bool(getattr(cfg, "approach_coarse_look_at", True)),
        bool(getattr(cfg, "approach_depth_priority", True)),
        bool(getattr(cfg, "approach_regress_to_pan_align", False)),
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
    prev_state = "SEARCH"
    detection_streak = 0
    miss_streak = 0
    centered_streak = 0
    pan_align_streak = 0

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
            "[gaze-live] keypress: [ ]/↑↓=el ; ,.=radius ; -=back +=in (2×) ; "
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

    comm_errors = 0
    try:
        while True:
            loop_t = time.time()
            dt = max(1e-3, loop_t - last_t)
            last_t = loop_t
            tick += 1

            try:
                obs = robot.get_observation()
            except (ConnectionError, OSError, TimeoutError) as e:
                comm_errors += 1
                logger.warning(
                    "[gaze-engine] robot read failed (%d/%d): %s",
                    comm_errors,
                    int(getattr(cfg, "comm_error_max_consecutive", 25)),
                    e,
                )
                if comm_errors >= int(getattr(cfg, "comm_error_max_consecutive", 25)):
                    raise
                _sleep(loop_t, dt_target)
                continue
            comm_errors = 0

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
            detected, bbox_xyxy, uv, conf = _apply_detection_hold(
                detected=detected,
                bbox_xyxy=bbox_xyxy,
                uv=uv,
                conf=conf,
                state=state,
                live=live,
                cfg=cfg,
            )

            if detected and state != "SEARCH" and uv is not None:
                lock_uv = (float(uv[0]), float(uv[1]))
                lock_joints = np.asarray(joints, dtype=np.float64).copy()

            d_bbox: float | None = None
            d_stereo: float | None = None
            if detected and bbox_xyxy is not None:
                d_meas, d_bbox, d_stereo = _measure_object_depth_m(
                    depth_map=depth,
                    bbox_xyxy=bbox_xyxy,
                    fx=fx,
                    fy=fy,
                    cfg=cfg,
                    depth_scale=float(intrinsics.get("depth_scale", 0.001)),
                )
                if d_meas is not None:
                    reseed_d = live.pop("_d_approach_reseed", None)
                    if reseed_d is not None:
                        d_filt = float(reseed_d)
                    else:
                        alpha = float(cfg.depth_ema_alpha)
                        d_filt = _filter_approach_depth(
                            live,
                            float(d_meas),
                            state=state,
                            cfg=cfg,
                            alpha=alpha,
                            d_filt_prev=d_filt,
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
                    live["_preposition_enter_t"] = float(loop_t)
                    if bool(getattr(cfg, "preposition_sync_radius_to_depth", True)):
                        _sync_preposition_radius_to_depth(live, cfg, d_filt)
                    live["_goto_preposition"] = False
            if live.get("_goto_approaching", False):
                live["_goto_approaching"] = False
                if detected and state in (
                    "PREPOSITIONING",
                    "TRACKING",
                    "PAN_ALIGN",
                    "HOLD",
                ):
                    if state != "APPROACHING":
                        logger.info(
                            "[gaze-engine] LIVE→APPROACHING (standoff / depth key)"
                        )
                    state = "APPROACHING"
                    centered_streak = 0
                    live["_tracking_enter_t"] = None
                    live["_preposition_enter_t"] = None

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
                    if detected and uv is not None:
                        s_uv = _filter_bbox_uv(
                            live,
                            (float(uv[0]), float(uv[1])),
                            cfg,
                            state="SEARCH",
                        )
                        d_ps, d_ts, _ = _gaze_joint_deltas(
                            uv=s_uv,
                            cx0=cx0,
                            cy0=cy0,
                            fx=fx,
                            fy=fy,
                            cfg=cfg,
                        )
                        gs = float(
                            getattr(cfg, "search_detection_gaze_scale", 0.65)
                        )
                        q_search[0] = float(q_search[0]) + float(d_ps) * gs
                        q_search[3] = float(q_search[3]) + float(d_ts) * gs
                q_search = _apply_live_joint_trims(cfg, live, q_search)
                act = {f"{m}.pos": float(q_search[i]) for i, m in enumerate(ARM_MOTORS)}
                if "gripper.pos" in obs:
                    act["gripper.pos"] = float(obs["gripper.pos"])
                try:
                    robot.send_action(act)
                except Exception as e:
                    logger.warning("[gaze-engine] send_action failed (search): %s", e)

                if detection_streak >= int(cfg.lock_required_frames):
                    lock_depth_err = _depth_err_m(d_filt, float(live["standoff"]))
                    use_orbit = _use_orbit_preposition(cfg) and (
                        bool(getattr(cfg, "preposition_from_search", False))
                        or (
                            lock_depth_err is not None
                            and float(lock_depth_err)
                            > float(
                                getattr(cfg, "preposition_on_lock_depth_err_m", 0.09)
                            )
                        )
                    )
                    if use_orbit:
                        next_state = "PREPOSITIONING"
                    elif lock_depth_err is not None and float(lock_depth_err) > float(
                        cfg.approach_done_tolerance_m
                    ):
                        if _pan_align_required(cfg) and (
                            bool(getattr(cfg, "pan_align_always_on_lock", True))
                            or not (
                                uv is not None
                                and _is_pan_aligned((float(uv[0]), float(uv[1])), cx0, cfg)
                            )
                        ):
                            next_state = "PAN_ALIGN"
                        else:
                            next_state = "APPROACHING"
                    else:
                        next_state = "TRACKING"
                    logger.info(
                        "[gaze-engine] SEARCH→%s (locked on %d frames, conf=%.2f, depth_err=%s)",
                        next_state,
                        detection_streak,
                        conf,
                        f"{lock_depth_err:+.3f}m"
                        if lock_depth_err is not None
                        else "n/a",
                    )
                    state = next_state
                    detection_streak = 0
                    miss_streak = 0
                    centered_streak = 0
                    live["_tracking_enter_t"] = (
                        float(loop_t) if next_state == "TRACKING" else None
                    )
                    live["_preposition_enter_t"] = (
                        float(loop_t) if next_state == "PREPOSITIONING" else None
                    )
                    if next_state == "PREPOSITIONING" and bool(
                        getattr(cfg, "preposition_sync_radius_to_depth", True)
                    ):
                        _sync_preposition_radius_to_depth(live, cfg, d_filt)
                    if next_state == "APPROACHING":
                        logger.info(
                            "[gaze-engine] continuous approach: gaze off until "
                            "depth_err < %.3fm",
                            float(getattr(cfg, "fine_gaze_depth_err_m", 0.045)),
                        )
                    live["_uv_filt"] = None
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
                hold_lim = max(0, int(getattr(cfg, "detection_hold_frames", 10)))
                lost_lim = int(cfg.track_lost_frames)
                if state in _states_hold_detection() and hold_lim > 0:
                    lost_lim = lost_lim + hold_lim
                if miss_streak >= lost_lim:
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

            gaze_uv = _filter_bbox_uv(
                live,
                (float(uv[0]), float(uv[1])),
                cfg,
                state=state,
            )
            d_pan, d_tilt, pixel_err = _gaze_joint_deltas(
                uv=gaze_uv, cx0=cx0, cy0=cy0, fx=fx, fy=fy, cfg=cfg
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
            depth_err_m = _depth_err_m(d_filt, float(live["standoff"]))
            pan_err_px = _pan_horizontal_err_px(gaze_uv, cx0)
            pan_aligned = _is_pan_aligned(gaze_uv, cx0, cfg)
            motion_live = _live_motion_override_active(live, loop_t)
            vertical_err_px = float(gaze_uv[1]) - float(cy0)
            coarse_pan_px = float(getattr(cfg, "pan_align_coarse_pan_err_px", 50.0))
            coarse_v_px = float(getattr(cfg, "coarse_gaze_tilt_pixel_err_px", 24.0))
            pan_align_coarse = state == "PAN_ALIGN" and (
                float(pan_err_px) > coarse_pan_px
                or abs(float(vertical_err_px)) >= coarse_v_px
            )
            # j1-only in PAN_ALIGN when nearly centered; coarse misalignment uses full gaze.
            pan_only = (
                (not motion_live)
                and state == "PAN_ALIGN"
                and not pan_align_coarse
            )
            if pan_only:
                d_pan = _gaze_pan_only_delta(
                    uv=gaze_uv,
                    cx0=cx0,
                    fx=fx,
                    cfg=cfg,
                    pan_err_px=float(pan_err_px),
                )
                d_tilt = 0.0
            allow_pan = _gaze_allow_pan(
                state=state,
                depth_err_m=depth_err_m,
                cfg=cfg,
                pan_aligned=bool(pan_aligned),
            )
            allow_tilt = _gaze_allow_tilt(
                state=state,
                depth_err_m=depth_err_m,
                cfg=cfg,
                pan_aligned=bool(pan_aligned),
                pixel_err_px=float(pixel_err),
                vertical_err_px=vertical_err_px,
            )
            if not pan_only:
                if not allow_pan:
                    d_pan = 0.0
                if not allow_tilt:
                    d_tilt = 0.0
                coarse_v = float(
                    getattr(cfg, "coarse_gaze_tilt_pixel_err_px", 24.0)
                )
                if abs(vertical_err_px) >= coarse_v and state in (
                    "APPROACHING",
                    "PREPOSITIONING",
                ):
                    tilt_scale = 1.0
                elif state == "APPROACHING":
                    tilt_scale = float(
                        getattr(cfg, "gaze_scale_during_approach", 0.55)
                    )
                elif state == "PREPOSITIONING":
                    tilt_scale = float(
                        getattr(cfg, "preposition_gaze_tilt_scale", 0.55)
                    )
                else:
                    tilt_scale = 1.0
                if state == "PREPOSITIONING":
                    d_pan = 0.0
                elif state == "APPROACHING":
                    gs = float(getattr(cfg, "gaze_scale_during_approach", 1.0))
                    ts = float(getattr(cfg, "gaze_tilt_scale_during_approach", 1.0))
                    if abs(float(vertical_err_px)) > float(
                        getattr(cfg, "approach_steep_vertical_optical_px", 55.0)
                    ):
                        ts = max(ts, 1.0)
                    d_pan = float(d_pan) * gs
                    tilt_scale = min(float(tilt_scale), ts)
                d_tilt = float(d_tilt) * float(tilt_scale)

            # Default action: gaze deltas only (used by TRACKING and HOLD)
            q_cmd = joints.copy()

            advance_to_approach = False
            advance_from_pan_align = False
            regress_to_tracking = False
            regress_to_preposition = False
            regress_to_pan_align = False
            preposition_done = False
            approach_done = False
            approach_step_m = 0.0
            approach_fov_scale = 1.0
            preposition_pos_err_m = float("nan")

            # Back-project bbox center with bbox-size depth into base frame —
            # used by PREPOSITIONING as p_obj, and by viz for everyone.
            p_obj_for_state: np.ndarray | None = None
            d_geom = _depth_for_geometry(d_filt, d_bbox, cfg)
            if d_geom is not None and uv is not None:
                try:
                    p_obj_for_state = point_cam_to_base(
                        T_base_cam_cur,
                        u=float(uv[0]),
                        v_pix=float(uv[1]),
                        depth_m=float(d_geom),
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

            if state == "TRACKING":
                if live.get("_tracking_enter_t") is None:
                    live["_tracking_enter_t"] = float(loop_t)
                stuck_s = float(
                    getattr(cfg, "tracking_stuck_preposition_s", 5.0)
                )
                if (
                    depth_err_m is not None
                    and float(depth_err_m)
                    > float(getattr(cfg, "preposition_on_lock_depth_err_m", 0.09))
                    and stuck_s > 0.0
                    and (float(loop_t) - float(live["_tracking_enter_t"])) >= stuck_s
                ):
                    if _use_orbit_preposition(cfg):
                        logger.info(
                            "[gaze-engine] TRACKING→PREPOSITIONING (stuck %.1fs, "
                            "depth_err=%+.3fm)",
                            float(loop_t) - float(live["_tracking_enter_t"]),
                            float(depth_err_m),
                        )
                        state = "PREPOSITIONING"
                        live["_preposition_enter_t"] = float(loop_t)
                        if bool(
                            getattr(cfg, "preposition_sync_radius_to_depth", True)
                        ):
                            _sync_preposition_radius_to_depth(live, cfg, d_filt)
                    else:
                        logger.info(
                            "[gaze-engine] TRACKING→APPROACHING (stuck %.1fs, "
                            "depth_err=%+.3fm)",
                            float(loop_t) - float(live["_tracking_enter_t"]),
                            float(depth_err_m),
                        )
                        state = "APPROACHING"
                    centered_streak = 0
                    live["_tracking_enter_t"] = None
                    live["_uv_filt"] = None

            if state == "PREPOSITIONING":
                if live.get("_preposition_enter_t") is None:
                    live["_preposition_enter_t"] = float(loop_t)
                orbit_live = _live_orbit_keys_active(live, loop_t)
                if orbit_live:
                    pre_ow = float(
                        getattr(cfg, "live_preposition_ik_orientation_weight", 1.2)
                    )
                else:
                    pre_ow = float(cfg.preposition_ik_orientation_weight)
                if p_obj_for_state is not None and not pan_only:
                    pre_lin: float | None = None
                    pre_dq: float | None = None
                    pre_ang: float | None = None
                    snap_se3 = False
                    if loop_t < float(live.get("_boost_until", 0.0)):
                        pre_lin = float(cfg.live_preposition_boost_lin_vel_m_s)
                        pre_dq = float(cfg.live_preposition_boost_joint_step_deg)
                        pre_ang = float(
                            getattr(cfg, "live_preposition_boost_ang_vel_deg_s", 220.0)
                        )
                        if orbit_live and bool(
                            getattr(cfg, "live_preposition_snap_se3", True)
                        ):
                            snap_se3 = True
                    last_pre = float(live.get("_last_pre_pos_err", float("nan")))
                    if math.isnan(last_pre) or last_pre > 0.10:
                        base_lin = float(cfg.preposition_max_lin_vel_m_s)
                        pre_lin = max(
                            float(pre_lin) if pre_lin is not None else base_lin,
                            base_lin * 1.4,
                        )
                        pre_dq = max(
                            float(pre_dq)
                            if pre_dq is not None
                            else float(cfg.preposition_max_joint_step_deg),
                            float(cfg.preposition_max_joint_step_deg) * 1.2,
                        )
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
                        max_ang_vel_deg_s=pre_ang,
                        ik_orientation_weight=pre_ow,
                        snap_se3=snap_se3,
                    )
                    preposition_pos_err_m = float(pos_err)
                    live["_last_pre_pos_err"] = float(pos_err)
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
                    stuck_s = float(
                        getattr(cfg, "preposition_stuck_approach_s", 6.0)
                    )
                    stuck_paused = (
                        _live_orbit_keys_active(live, loop_t)
                        or time.time()
                        < float(live.get("_preposition_stuck_pause_until", 0.0))
                    )
                    if (
                        not preposition_done
                        and not stuck_paused
                        and stuck_s > 0.0
                        and live.get("_preposition_enter_t") is not None
                        and (float(loop_t) - float(live["_preposition_enter_t"]))
                        >= stuck_s
                        and preposition_pos_err_m
                        > float(cfg.preposition_position_tolerance_m)
                        and depth_err_m is not None
                        and float(depth_err_m)
                        > float(cfg.approach_done_tolerance_m)
                    ):
                        logger.warning(
                            "[gaze-engine] PREPOSITIONING stuck %.1fs "
                            "(pos_err=%.3fm, pixel_err=%.1fpx) → APPROACHING",
                            float(loop_t) - float(live["_preposition_enter_t"]),
                            preposition_pos_err_m,
                            pixel_err,
                        )
                        preposition_done = True

            elif state == "PAN_ALIGN":
                if pan_aligned:
                    pan_align_streak += 1
                    if pan_align_streak >= int(
                        getattr(cfg, "pan_align_consecutive_frames", 4)
                    ):
                        advance_from_pan_align = True
                else:
                    pan_align_streak = 0
                    stuck_n = int(live.get("_pan_align_stuck_ticks", 0)) + 1
                    live["_pan_align_stuck_ticks"] = stuck_n
                    if stuck_n == int(5.0 * float(cfg.loop_hz)):
                        logger.warning(
                            "[gaze-engine] PAN_ALIGN stuck: pan_err=%.1fpx "
                            "(need <%.0fpx for %d frames). j1-only=%s "
                            "(coarse if pan>%.0f or |v|>=%.0f).",
                            pan_err_px,
                            float(cfg.pan_align_threshold_px),
                            int(getattr(cfg, "pan_align_consecutive_frames", 4)),
                            pan_only,
                            coarse_pan_px,
                            coarse_v_px,
                        )
                if pan_aligned:
                    live["_pan_align_stuck_ticks"] = 0

            elif state == "TRACKING":
                if _pan_align_required(cfg) and not pan_aligned:
                    pass
                elif _approach_centering_ok(
                    pixel_err_px=float(pixel_err),
                    depth_err_m=depth_err_m,
                    cfg=cfg,
                ):
                    centered_streak += 1
                    if centered_streak >= int(cfg.approach_consecutive_centered_frames):
                        if _pan_align_required(cfg) and not pan_aligned:
                            pass
                        else:
                            advance_to_approach = True
                else:
                    centered_streak = 0

            elif state == "APPROACHING":
                coarse_apr = depth_err_m is not None and float(depth_err_m) > float(
                    getattr(cfg, "fine_gaze_depth_err_m", 0.045)
                )
                if (
                    bool(getattr(cfg, "approach_regress_to_tracking", False))
                    and pixel_err > float(cfg.approach_regress_pixel_threshold_px)
                    and not coarse_apr
                    and pan_aligned
                ):
                    if _use_orbit_preposition(cfg):
                        regress_to_preposition = True
                    else:
                        regress_to_tracking = True
                else:
                    reg_px = float(
                        getattr(cfg, "approach_regress_to_pan_align_px", 58.0)
                    )
                    reg_need = max(
                        int(getattr(cfg, "approach_regress_to_pan_align_frames", 10)),
                        1,
                    )
                    reg_cd = float(
                        getattr(cfg, "approach_regress_to_pan_align_cooldown_s", 2.5)
                    )
                    reg_ok = (
                        bool(getattr(cfg, "approach_regress_to_pan_align", False))
                        and coarse_apr
                        and not pan_aligned
                        and float(pan_err_px) > reg_px
                        and loop_t
                        >= float(live.get("_approach_enter_t", 0.0)) + reg_cd
                    )
                    if reg_ok:
                        live["_pan_regress_streak"] = (
                            int(live.get("_pan_regress_streak", 0)) + 1
                        )
                        if int(live["_pan_regress_streak"]) >= reg_need:
                            regress_to_pan_align = True
                    else:
                        live["_pan_regress_streak"] = 0
                if (
                    not regress_to_pan_align
                    and not regress_to_tracking
                    and not regress_to_preposition
                ):
                    if d_filt is not None:
                        apr_lin: float | None = None
                        apr_dq: float | None = None
                        apr_fov_min: float | None = None
                        apr_ow: float | None = None
                        if loop_t < float(live.get("_boost_until", 0.0)):
                            apr_lin = float(cfg.live_approach_boost_lin_vel_m_s)
                            apr_dq = float(cfg.live_preposition_boost_joint_step_deg)
                            apr_fov_min = float(cfg.live_approach_boost_fov_scale_min)
                        depth_pri = float(
                            getattr(cfg, "approach_depth_priority_min_err_m", 0.055)
                        )
                        pan_slow = float(
                            getattr(cfg, "approach_slowdown_pan_err_px", 35.0)
                        )
                        slow_for_pan = (
                            coarse_apr
                            and pan_err_px > pan_slow
                            and (
                                depth_err_m is None
                                or float(depth_err_m) <= depth_pri
                            )
                        )
                        if slow_for_pan and apr_lin is not None:
                            apr_lin = float(apr_lin) * float(
                                getattr(cfg, "approach_slowdown_lin_scale", 0.35)
                            )
                        elif slow_for_pan:
                            apr_lin = float(cfg.approach_max_lin_vel_m_s) * float(
                                getattr(cfg, "approach_slowdown_lin_scale", 0.35)
                            )
                        if (
                            coarse_apr
                            and depth_err_m is not None
                            and float(depth_err_m) > depth_pri
                        ):
                            base = float(cfg.approach_max_lin_vel_m_s)
                            apr_lin = max(
                                float(apr_lin) if apr_lin is not None else base,
                                base * 1.75,
                            )
                        if coarse_apr:
                            apr_ow = float(
                                getattr(
                                    cfg, "approach_coarse_ik_orientation_weight", 0.45
                                )
                            )
                            apr_dq = max(
                                float(apr_dq)
                                if apr_dq is not None
                                else float(cfg.approach_max_joint_step_deg),
                                float(cfg.approach_max_joint_step_deg) * 1.5,
                            )
                        pause_fwd = _should_pause_approach_forward(
                            d_bbox=d_bbox,
                            pixel_err_px=float(pixel_err),
                            depth_err_m=depth_err_m,
                            cfg=cfg,
                        )
                        d_cmd = (
                            float(d_geom)
                            if d_geom is not None
                            else (
                                float(d_filt)
                                if d_filt is not None
                                else float("nan")
                            )
                        )
                        q_apr, planned_step, fov_scale = _approach_q(
                            joints_deg=joints,
                            T_base_ee_cur=T_base_ee_cur,
                            T_base_cam_cur=T_base_cam_cur,
                            d_obj_m=float(d_cmd),
                            p_obj_base=p_obj_for_state,
                            pixel_err_px=float(pixel_err),
                            pan_err_px=float(pan_err_px),
                            vertical_err_px=float(vertical_err_px),
                            cfg=cfg,
                            dt=dt_target,
                            kin=kin,
                            final_standoff_m=float(live["standoff"]),
                            max_lin_vel_m_s=apr_lin,
                            max_joint_step_deg=apr_dq,
                            fov_scale_min=apr_fov_min,
                            ik_orientation_weight=apr_ow,
                            coarse_approach=bool(coarse_apr),
                            T_ee_cam=T_ee_cam,
                            pause_forward=bool(pause_fwd),
                        )
                        approach_step_m = float(planned_step)
                        approach_fov_scale = float(fov_scale)
                        if q_apr is not None:
                            q_cmd = q_apr
                        at_standoff = (
                            abs(float(d_filt) - float(live["standoff"]))
                            < float(cfg.approach_done_tolerance_m)
                        )
                        centered_ok = _approach_centering_ok(
                            pixel_err_px=float(pixel_err),
                            depth_err_m=depth_err_m,
                            cfg=cfg,
                        )
                        if at_standoff and (
                            not bool(
                                getattr(cfg, "approach_require_centered_for_done", True)
                            )
                            or centered_ok
                        ):
                            approach_done = True

            elif state == "HOLD":
                # Maintain gaze, no forward motion
                pass

            q_cmd[i_pan] = float(q_cmd[i_pan]) + float(d_pan)
            q_cmd[i_tilt] = float(q_cmd[i_tilt]) + float(d_tilt)

            q_out = _apply_live_joint_trims(cfg, live, q_cmd)
            act = {f"{m}.pos": float(q_out[i]) for i, m in enumerate(ARM_MOTORS)}
            if "gripper.pos" in obs:
                act["gripper.pos"] = float(obs["gripper.pos"])
            try:
                robot.send_action(act)
            except (ConnectionError, OSError, TimeoutError) as e:
                comm_errors += 1
                logger.warning(
                    "[gaze-engine] send_action failed (%d/%d): %s",
                    comm_errors,
                    int(getattr(cfg, "comm_error_max_consecutive", 25)),
                    e,
                )
                if comm_errors >= int(getattr(cfg, "comm_error_max_consecutive", 25)):
                    raise
            except Exception as e:
                logger.warning("[gaze-engine] send_action failed: %s", e)

            # State transitions (apply AFTER sending action so the log reflects
            # what we actually commanded this tick)
            if preposition_done:
                at_depth = (
                    depth_err_m is not None
                    and abs(float(depth_err_m))
                    < float(cfg.approach_done_tolerance_m)
                )
                need_center = bool(cfg.preposition_require_centered_bbox) and (
                    not _approach_centering_ok(
                        pixel_err_px=float(pixel_err),
                        depth_err_m=depth_err_m,
                        cfg=cfg,
                    )
                )
                if at_depth:
                    logger.info(
                        "[gaze-engine] PREPOSITIONING→HOLD "
                        "(pos_err=%.3fm, d=%.3fm at standoff)",
                        preposition_pos_err_m,
                        float(d_filt) if d_filt is not None else float("nan"),
                    )
                    state = "HOLD"
                elif need_center:
                    logger.info(
                        "[gaze-engine] PREPOSITIONING→TRACKING "
                        "(orbit ok, centering before approach, pixel_err=%.1fpx)",
                        pixel_err,
                    )
                    state = "TRACKING"
                    live["_tracking_enter_t"] = float(loop_t)
                elif _pan_align_required(cfg) and not pan_aligned:
                    logger.info(
                        "[gaze-engine] PREPOSITIONING→PAN_ALIGN "
                        "(orbit ok, pan_err=%.1fpx before approach)",
                        pan_err_px,
                    )
                    state = "PAN_ALIGN"
                    pan_align_streak = 0
                else:
                    logger.info(
                        "[gaze-engine] PREPOSITIONING→APPROACHING "
                        "(pos_err=%.3fm, pixel_err=%.1fpx, d=%.3fm, depth_err=%s)",
                        preposition_pos_err_m,
                        pixel_err,
                        float(d_filt) if d_filt is not None else float("nan"),
                        f"{depth_err_m:+.3f}m"
                        if depth_err_m is not None
                        else "n/a",
                    )
                    state = "APPROACHING"
                centered_streak = 0
                live["_uv_filt"] = None
            elif advance_from_pan_align:
                logger.info(
                    "[gaze-engine] PAN_ALIGN→APPROACHING "
                    "(j1 centered %d frames, pan_err=%.1fpx, depth_err=%s)",
                    pan_align_streak,
                    pan_err_px,
                    f"{depth_err_m:+.3f}m" if depth_err_m is not None else "n/a",
                )
                state = "APPROACHING"
                pan_align_streak = 0
                centered_streak = 0
                live["_uv_filt"] = None
            elif advance_to_approach:
                if _pan_align_required(cfg) and not pan_aligned:
                    logger.info(
                        "[gaze-engine] TRACKING→PAN_ALIGN (depth_err=%s, pan_err=%.1fpx)",
                        f"{depth_err_m:+.3f}m" if depth_err_m is not None else "n/a",
                        pan_err_px,
                    )
                    state = "PAN_ALIGN"
                else:
                    logger.info(
                        "[gaze-engine] TRACKING→APPROACHING "
                        "(centered %d frames, err=%.1fpx, depth_err=%s)",
                        centered_streak,
                        pixel_err,
                        f"{depth_err_m:+.3f}m" if depth_err_m is not None else "n/a",
                    )
                    state = "APPROACHING"
                centered_streak = 0
                live["_tracking_enter_t"] = None
                live["_uv_filt"] = None
            elif regress_to_pan_align:
                logger.info(
                    "[gaze-engine] APPROACHING→PAN_ALIGN "
                    "(pan_err=%.1fpx, pixel_err=%.1fpx — re-center j1)",
                    pan_err_px,
                    pixel_err,
                )
                state = "PAN_ALIGN"
                pan_align_streak = 0
                centered_streak = 0
                live["_uv_filt"] = None
            elif regress_to_preposition:
                logger.info(
                    "[gaze-engine] APPROACHING→PREPOSITIONING (pixel_err=%.1fpx > regress threshold)",
                    pixel_err,
                )
                state = "PREPOSITIONING"
                centered_streak = 0
                live["_uv_filt"] = None
            elif regress_to_tracking:
                logger.info(
                    "[gaze-engine] APPROACHING→TRACKING (pixel_err=%.1fpx > regress threshold)",
                    pixel_err,
                )
                state = "TRACKING"
                centered_streak = 0
                live["_tracking_enter_t"] = float(loop_t)
                live["_uv_filt"] = None
            elif approach_done:
                logger.info(
                    "[gaze-engine] APPROACHING→HOLD (d=%.3fm within %.3fm of target)",
                    float(d_filt) if d_filt is not None else float("nan"),
                    float(cfg.approach_done_tolerance_m),
                )
                state = "HOLD"

            if state == "APPROACHING" and prev_state != "APPROACHING":
                _reset_approach_depth_filter(live, d_bbox=d_bbox)
                live["_approach_enter_t"] = float(loop_t)
                live["_pan_regress_streak"] = 0
                if gaze_uv is not None:
                    live["_approach_uv_lock"] = (
                        float(gaze_uv[0]),
                        float(gaze_uv[1]),
                    )
            prev_state = state

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
                    "pixel_err=%.1fpx d_bbox=%.3fm d_stereo=%s d_filt=%.3fm d_target=%.3fm "
                    "depth_err=%+.3fm pan_err=%.1fpx gaze=(Δpan=%+.2f°,Δtilt=%+.2f°) "
                    "approach_step=%+.4fm fov_scale=%.2f preposition_pos_err=%.3fm",
                    tick,
                    state,
                    conf,
                    pixel_err,
                    float(d_bbox) if d_bbox is not None else float("nan"),
                    f"{float(d_stereo):.3f}"
                    if d_stereo is not None
                    else "n/a",
                    d_meas,
                    d_tgt,
                    depth_err,
                    pan_err_px,
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
                        depth_m=float(
                            _depth_for_geometry(d_filt, d_bbox, cfg)
                            or (d_filt if d_filt is not None else d_bbox)
                        ),
                        fx=fx,
                        fy=fy,
                        cx0=cx0,
                        cy0=cy0,
                    )
                    p_obj_viz = _maybe_floor_object_base(
                        p_obj_viz, float(cfg.ik_object_floor_z_m)
                    )
                    if state == "APPROACHING" and p_obj_viz is not None:
                        prev_p = live.get("_p_obj_viz")
                        if prev_p is not None:
                            beta = 0.35
                            p_obj_viz = (1.0 - beta) * np.asarray(
                                prev_p, dtype=np.float64
                            ) + beta * np.asarray(p_obj_viz, dtype=np.float64)
                        live["_p_obj_viz"] = np.asarray(p_obj_viz, dtype=np.float64)
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
