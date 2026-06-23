# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dataclass config for the YOLO-track approach pipeline (CLI surface).

This is deliberately kept as a single flat dataclass so draccus / ``@parser.wrap()`` continue to
expose the exact same ``--field-name`` CLI flags callers already rely on. Grouping the fields
into nested sub-configs would break those flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig  # noqa: F401 — registers oakd for CLI
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401 — registers opencv for CLI
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401 — registers realsense for CLI
from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig  # noqa: F401 — registers so100/so101_follower for CLI


@dataclass
class YoloTrackApproachConfig:
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
    detection_hold_s: float = 0.25
    """If YOLO misses briefly, keep using the last bbox for up to this many seconds.

    This prevents the approach logic from oscillating between "engaged" and "lost" due to
    single-frame dropouts or inference jitter. Set to 0 to disable.
    """

    # SAHI-style sliced inference for small tabletop objects (optional).
    sahi_enable: bool = False
    sahi_slice_w: int = 512
    sahi_slice_h: int = 512
    sahi_overlap: float = 0.20
    sahi_iou_threshold: float = 0.55
    sahi_include_full_image: bool = True

    # servo: image-centering + depth / bbox-area closing only.
    # umbrella: when centered in the image, also step along base Z (see umbrella_* fields).
    # plan_top: once the object's 3D position is locked in (depth + gripper_camera_tf), skip the
    #   umbrella lift/align entirely and execute ONE smooth pose waypoint to
    #   (x_target, y_target, z_target + top_approach_height_m) with the wrist pointing straight
    #   down, then straight-line descend (no visual servoing during the move, so the cube never
    #   leaves the FOV because of lateral wander).
    approach_style: str = "umbrella"
    kp_xy: float = 0.72
    kp_z: float = 0.55
    max_step_m: float = 0.024
    center_deadband_px: float = 22.0
    target_depth_m: float = 0.22
    min_depth_m: float = 0.11
    # Reject depth readings above this (meters); bad stereo often blows up and makes lateral gain huge.
    max_plausible_depth_m: float = 1.35
    # Stop forward motion when bbox covers this fraction of the image (no depth).
    area_close_frac: float = 0.18
    forward_if_no_depth_m: float = 0.012
    invert_lateral: bool = False

    # Extra base-frame Z per iteration when approach_style=umbrella and |pixel error| is in deadband.
    # SO arms usually use base Z up; negative sign moves the EE down. Tune or flip sign if inverted.
    umbrella_descent_step_m: float = 0.006
    umbrella_base_z_sign: float = -1.0

    # Phased umbrella: lift → planar align (XY only in base) → descend. Set false for legacy behavior.
    umbrella_phased: bool = True
    # How much higher (m) the EE Z (base frame) should be vs the pose at start before lateral servo.
    umbrella_clearance_z_m: float = 0.040
    umbrella_lift_step_m: float = 0.012
    # +1 => increase base Z to lift (typical URDF Z up). Flip to -1 if your model is inverted.
    umbrella_lift_z_sign: float = 1.0
    umbrella_lift_max_iters: int = 28
    # If True, keep lifting even when YOLO drops detections (not recommended for fixed cams).
    umbrella_lift_without_detection: bool = False
    # Require a stable detection before moving (prevents "lift then lose it"). YOLO-World on small
    # cubes typically returns confidences in the 0.30–0.55 range; keep the gate low enough that
    # the robot engages on the first reasonable hit instead of waiting for many frames in a row.
    engage_min_consecutive_detections: int = 2
    engage_min_conf: float = 0.30
    # During lift, also apply a small XY correction to keep the box in view.
    lift_track_xy: bool = True
    lift_track_xy_gain: float = 0.45
    # Umbrella phased ``lift``: one ``execute_gripper_nudge`` that merges XY tracking + look-at in a
    # single pose IK often jerks the wrist. When True, do a small translation-free look at the
    # filtered 3D point first, then an orientation-preserving Cartesian nudge (same idea as
    # ``gripper_maintain_aim`` for plan_top).
    umbrella_lift_decouple_look: bool = True
    # Max base-frame step (m) passed into lateral visual servo during lift (often tighter than
    # ``max_step_m``). ROI depth (when valid) is passed into the servo law so gain scales with range.
    lift_track_max_step_m: float = 0.012
    # Per-iteration rotation cap for the decoupled lift look-at (keep <= gripper_max_look_rot_step_deg).
    umbrella_lift_look_rot_step_deg: float = 7.0

    # Gripper-mounted camera: before engage, raise EE in base Z while there is no detection (wider FOV /
    # see the table). Stops at search_raise_max_m or search_raise_max_iters. Flip search_raise_z_sign if
    # the arm moves the wrong way.
    gripper_search_raise: bool = True
    search_raise_step_m: float = 0.010
    search_raise_z_sign: float = 1.0
    search_raise_max_m: float = 0.10
    search_raise_max_iters: int = 35

    # Generic search scan (works for any camera_mount). When no detection is found and the robot is
    # still in search phase, gently raise the EE and oscillate shoulder_pan left/right from the
    # folded/baseline pose so a gripper-mounted camera sweeps the workspace. This is the
    # mount-agnostic equivalent of ``gripper_search_raise`` (which only runs for camera_mount=gripper).
    search_scan_enabled: bool = True
    # Amplitude of shoulder_pan oscillation (degrees) around the pose captured at startup.
    search_pan_amplitude_deg: float = 25.0
    # Full sinusoid period in control iterations. Lower = faster scan. ~20 iters at 30 Hz is ≈1s
    # per sweep, fast enough to find a small object without jerking.
    search_pan_period_iters: int = 20
    # Cap of the shoulder_lift "look up" delta (degrees) added per scan step; negative = look up
    # on SO arms (adjust sign if your URDF is inverted).
    search_lift_up_delta_deg: float = -10.0
    search_lift_up_max_deg: float = -25.0
    # Wrist-pitch offset (deg) applied during search so the camera looks FORWARD into the
    # workspace instead of straight down. Must counteract any downward camera-mount pitch in
    # ``gripper_camera_tf`` (e.g. if the camera tf has pitch -20°, set this to -20° to level
    # the optical axis). Negative = wrist tilts up on SO arms. Kept as a baseline through the
    # lift/align phases so the cube stays in frame, then overridden by the tilt phase at the end.
    # Slightly stronger default so we "look higher" during search.
    search_wrist_up_delta_deg: float = -28.0
    # When a detection exists but we haven't "engaged" yet (still building confidence), do a small
    # base-yaw (shoulder_pan) correction to bring the bbox center toward the image center. This
    # prevents the search sinusoid from swinging past a target that is already visible in a corner.
    preengage_pan_center_enable: bool = False
    # Pixel deadband before yaw correction kicks in.
    preengage_pan_deadband_px: float = 18.0
    # Proportional gain applied to the approximate angular error atan((cx-cx0)/fx). 1.0 means
    # "turn by the full angular error" (still clamped by max step).
    preengage_pan_kp: float = 0.85
    # Max absolute yaw step (deg) applied per control iteration during pre-engage centering.
    preengage_pan_max_step_deg: float = 8.0
    # While pre-engage centering, also bias the wrist slightly up (deg). Negative = wrist tilts up
    # on SO arms; set to 0 to disable.
    preengage_wrist_up_deg: float = -6.0
    # Smoothness: joint micro-step magnitude and inter-step sleep when streaming search moves.
    # Bigger step + shorter sleep = faster search, at some cost in smoothness.
    search_joint_step_deg: float = 1.25
    search_joint_sleep_s: float = 0.022
    # While searching (no bbox), pitch the wrist so the camera looks horizontally in base XY instead of
    # staying at the fixed downward mount angle (avoids staring at the sky when the arm lifts).
    gripper_search_level_camera: bool = True
    gripper_search_horizon_dir: str = "0,-1,0"
    gripper_search_look_distance_m: float = 0.45
    # Once a bbox exists, aim the camera (+Z optical) at the deprojected 3D point while translating.
    gripper_point_at_target: bool = True
    gripper_preengage_aim: bool = True
    # Smaller per-step camera look during pre-engage (before ``engage``). Uses
    # ``min(gripper_max_look_rot_step_deg, this)`` when this is > 0; set very large to match global.
    gripper_preengage_max_look_rot_step_deg: float = 5.0
    gripper_ik_orientation_weight: float = 1.0
    gripper_aim_default_depth_m: float = 0.35
    # Fuse noisy backprojection into p_base (EMA). Aim can use the filtered point (see below).
    gripper_track_3d_enable: bool = True
    gripper_track_3d_ema_alpha: float = 0.55
    # If True, only run EMA when ROI depth is valid (avoids latching default-depth guesses).
    gripper_3d_ema_require_depth: bool = True
    gripper_aim_use_filtered_3d: bool = True
    # After a dropout, aim at the last p_base for this many control cycles (fixed-base static object).
    gripper_hold_target_frames: int = 18
    # Max end-effector rotation change per look-at command (deg); 0 = no limit.
    gripper_max_look_rot_step_deg: float = 14.0
    # plan_top: each outer-loop tick (see ``gripper_maintain_aim_period_iters``), first issue a
    # **translation-free** look-at the tracked 3D point so the camera keeps the object in view
    # while the phased lift/align/descend runs. Safer than ``plan_top_*_look_at_target`` (which
    # couples look + Cartesian in one solve and can thrash). Umbrella ``lift`` can decouple via
    # ``umbrella_lift_decouple_look``; servo still uses combined nudge+aim each iter — keep
    # ``gripper_maintain_aim_plan_top_only=true`` to avoid double commands on servo/align.
    gripper_maintain_aim_at_target: bool = True
    gripper_maintain_aim_plan_top_only: bool = True
    # 1 = every engaged tick; 2 = every second tick, etc.
    gripper_maintain_aim_period_iters: int = 1
    # During plan_top ``gather`` / ``lift`` / ``align`` / ``tilt`` / ``descend`` / ``center``,
    # skip per-tick maintain-aim: look-at IK shifts the EE (often backward in base) while Cartesian
    # phases command hover / Z / pixel centering. In ``center`` with no bbox, maintain-aim was
    # especially bad for losing the object. Set False to restore per-tick look during those phases.
    gripper_maintain_aim_skip_plan_top_lift_align_tilt: bool = True

    # Temporal smoothing (0..1]: 1 = off. Lower = smoother but slower to react.
    smooth_center_alpha: float = 0.38
    smooth_depth_alpha: float = 0.36
    smooth_delta_alpha: float = 0.32

    max_iters: int = 500
    loop_sleep_s: float = 0.03
    lost_patience: int = 25

    motion_cartesian_step_m: float = 0.012
    motion_min_steps_per_segment: int = 1
    motion_inter_step_sleep_s: float = 0.025
    motion_settle_timeout_s: float = 1.15
    motion_settle_threshold_deg: float = 3.0
    # False = stream joint targets without waiting for convergence each nudge (much smoother).
    motion_settle_last_step: bool = False

    # Plan-to-top: hover height (m) above the detected target before descent, and step size for the
    # straight-line descent after the top pose is reached. If depth is lost during descent, we keep
    # descending open-loop until ``top_descend_max_iters`` is reached or the hand hits
    # ``top_descend_min_z_m`` (safety floor).
    top_approach_height_m: float = 0.08
    top_descend_step_m: float = 0.006
    top_descend_max_iters: int = 40
    top_descend_min_z_m: float = -0.02
    # Known table-surface height in the robot base frame (m). When set (not None), the estimated
    # cube Z is clamped to this floor: bbox-depth combined with a pitched-down camera can project
    # the ray well below the actual table (Rerun will show the target marker below the grid and
    # the arm then aims into the table). Measure once: put the cube on the table, record the Z
    # your robot reports for the cube face, subtract the cube's half-height — that's your
    # ``table_z_m``. Typical SO-101 sitting on its base plate with a ~3 cm tall cube on the same
    # surface: ``--table-z-m=-0.02`` works well. Leave None to disable clamping.
    table_z_m: float | None = None
    # Extra safety clearance (m) between the fingertip and ``table_z_m`` during the plan_top
    # descend phase. The tip is never commanded below ``table_z_m + table_clearance_m``, even if
    # the cube depth estimate says otherwise.
    table_clearance_m: float = 0.005
    # Reject 3D targets that are farther than this from the base origin (m). Protects against
    # bogus stereo depth (e.g. 0.95 m reading on a cube that's really 15 cm away) before the arm
    # whips toward an unreachable point.
    top_max_reach_m: float = 0.32

    # --- Bounding-box size → depth fallback ---
    # Stereo cameras like the OAK-D have a minimum working distance (~0.35 m for OAK-D Lite / D /
    # Pro). Objects closer than that return either no depth or wildly wrong values. When the target
    # is a known-size object (e.g. a 3 cm cube) we can recover depth from the pinhole relation
    # ``d = fx * target_physical_size_m / bbox_pixel_width``. Enable this when stereo fails in the
    # near-field. Uses the larger side of the bbox (more robust to occlusion of one dimension).
    depth_from_bbox_enabled: bool = False
    # Longest physical side of the target in meters (e.g. 0.03 for a 3 cm cube face).
    target_physical_size_m: float = 0.03
    # If both stereo and bbox-size depth are available, how to combine them:
    #   "stereo_preferred": use stereo when valid, else bbox-size.
    #   "bbox_preferred":   use bbox-size (more reliable < 0.35 m), else stereo.
    #   "bbox_only":        ignore stereo entirely (use bbox-size always when enabled).
    depth_source_policy: str = "bbox_preferred"
    # Minimum number of good depth samples accumulated into p_target_base before committing the top
    # waypoint (so a single spurious depth reading cannot launch the arm across the table).
    top_min_good_depth_samples: int = 3
    # If True, always compute p_target_base from T_base_ee @ T_ee_cam (gripper_camera_tf), even when
    # camera_mount=fixed. Use this when the camera is physically mounted on the gripper but you want
    # the jerk-free translation-only execution path (camera_mount=fixed).
    target_from_gripper_tf: bool = False

    # --- Phased plan_top motion (umbrella style) ---
    # Instead of one giant waypoint commit, plan_top moves the EE through four phases, each one
    # small step per outer-loop iteration so the motion looks smooth and Rerun refreshes live:
    #   1) "lift"    : raise EE straight up by plan_top_lift_height_m (translation only, wrist unchanged)
    #   2) "align"   : translate EE in base XY to hover point (translation only, wrist unchanged)
    #   3) "tilt"    : gradually rotate wrist toward straight-down in plan_top_tilt_step_deg increments
    #   4) "descend" : straight-down descent toward the target (existing behavior)
    plan_top_lift_height_m: float = 0.06
    plan_top_xy_step_m: float = 0.015
    plan_top_lift_step_m: float = 0.012
    plan_top_tilt_step_deg: float = 5.0
    plan_top_align_tolerance_m: float = 0.012
    plan_top_tilt_tolerance_deg: float = 3.0
    # Fraction (0..1) of the rotation from the current wrist pose toward strict "wrist straight
    # down" that the tilt phase will command. Full wrist-down (1.0) is almost never reachable on
    # SO101 at the align XY/Z without the IK collapsing the arm (empirically the wrist drops
    # 10 cm in base Z while rotating, and the arm then folds backward into the base). A partial
    # tilt (e.g. 0.5 = halfway from current pose to wrist-down) keeps the IK inside a feasible
    # region, preserves z_ee, and is still "down enough" for the center phase's pixel→base-XY
    # math to produce meaningful nudges.
    plan_top_tilt_fraction: float = 0.5
    # The lift goal is straight-up base-Z; the IK picks a joint config for each commanded nudge
    # and real EE Z often oscillates a few mm around the goal (e.g. ±6 mm near a shoulder-elbow
    # near-singular pose). A tolerance tighter than that causes the phase to loop forever. Set
    # this >= plan_top_lift_step_m/2 and >= the observed oscillation amplitude.
    plan_top_lift_tolerance_m: float = 0.008
    # Safety caps: if a phase spins this many iterations without entering the next phase, we
    # bail out of the current phase (move on or stop). Prevents the "stuck at remain=0.006"
    # infinite loop we used to see when the IK oscillates around the goal.
    plan_top_lift_max_iters: int = 40
    plan_top_align_max_iters: int = 30
    plan_top_tilt_max_iters: int = 20
    # Optional: couple each Cartesian nudge with a look-at the frozen ``p_target_base``. This
    # often *destabilizes* numerical IK: translation + full orientation in one small step can
    # pick very different joint solutions every iteration (erratic Rerun / real motion). Default
    # is False — plain position nudges match “simple 3D goal” best. Turn on only for tuning.
    plan_top_lift_look_at_target: bool = False
    plan_top_align_look_at_target: bool = False
    plan_top_descend_recover_look_at_target: bool = False
    plan_top_descend_look_at_target: bool = False
    # ``shoulder_lift``: raise the arm in joint space (negative deg = up on SO-101, same sense as
    # ``search_lift_up_delta_deg``). Avoids base-frame +Z Cartesian recover that often drifts XY
    # backward on small arms. ``cartesian_z``: legacy pure +Z nudges.
    plan_top_vertical_recovery_mode: str = "shoulder_lift"
    plan_top_vertical_recovery_shoulder_lift_step_deg: float = -2.5
    plan_top_vertical_recovery_shoulder_lift_total_deg: float = -24.0
    plan_top_vertical_recovery_shoulder_joint_step_deg: float = 1.5
    plan_top_vertical_recovery_shoulder_joint_sleep_s: float = 0.02
    # Wrist-flex bias during descend recover fights shoulder lift recovery; keep off by default.
    plan_top_approach_wrist_flex_in_descend_recover: bool = False
    plan_top_center_xy_look_at_target: bool = False
    # When planar distance to the hover XY is below this, cap XY step size (reduces overshoot /
    # apparent retreat before descend).
    plan_top_align_near_xy_m: float = 0.04
    plan_top_align_near_xy_step_m: float = 0.008
    # After each Cartesian lift/align/descend step, nudge ``wrist_flex`` in joint space so the
    # gripper camera pitches toward the object instead of the table. Same sign as
    # ``search_wrist_up_delta_deg``: **negative** degrees = tilt camera up on SO-101.
    # ``total_deg`` caps cumulative flex change (negative). Set enable=false to disable.
    plan_top_approach_wrist_flex_bias_enable: bool = True
    plan_top_approach_wrist_flex_step_deg: float = -2.0
    plan_top_approach_wrist_flex_total_deg: float = -14.0
    plan_top_approach_wrist_flex_joint_step_deg: float = 1.5
    plan_top_approach_wrist_flex_joint_sleep_s: float = 0.02
    # If the bbox disappears in the center phase, re-aim the wrist at ``p_target_base`` (no
    # translation) — at most every ``plan_top_center_reacquire_every_n_iters`` when enabled.
    # Default off: per-frame re-aim thrashes IK when YOLO is already blind.
    plan_top_center_reacquire_look: bool = False
    plan_top_center_reacquire_every_n_iters: int = 5
    # If the base-frame progress over this window (m / rad) is below the threshold, the phase is
    # declared "stuck" and advances. Checked every iteration once we've run the minimum window.
    plan_top_stuck_window: int = 6
    plan_top_stuck_progress_m: float = 0.003
    plan_top_stuck_progress_rad: float = 0.03
    # Clearance (m) from the **top** of the object along base +Z to the fingertip after descend.
    # Object top is approximated as ``p_target_base.z + target_physical_size_m/2`` (same half
    # extent used for bbox-depth). For a 3 cm cube this is ~1.5 cm above the locked target Z.
    # Default 0.04 ≈ 3–4 cm air gap above the top face. Increase if the tip still crowds the object.
    plan_top_final_hover_m: float = 0.04
    plan_top_descend_step_m: float = 0.008
    # Offset (m) from the EE kinematic origin (wrist joint) to the gripper TIP along the EE +Z
    # axis. When the wrist points straight down, this much *additional* base-Z clearance is needed
    # above the cube to keep the fingers off the table. Measure it: with the gripper open and
    # fingers pointing down, distance from the wrist joint to the finger tips. SO-101 ≈ 0.10 m.
    plan_top_gripper_tip_offset_m: float = 0.10
    # --- Final visual centering phase (plan_top_center) ---
    # After descending to the hover height, re-detect the cube with the (now straight-down) camera
    # and issue small base-XY nudges until the bbox center sits within `center_tol_px` of the image
    # center. This corrects residual error from the open-loop plan (depth model, tilt compliance).
    plan_top_center_enable: bool = False
    plan_top_center_tol_px: float = 25.0
    plan_top_center_step_m: float = 0.006
    plan_top_center_max_iters: int = 30
    plan_top_center_min_conf: float = 0.25
    # While gathering depth samples, nudge shoulder_pan to reduce horizontal bbox offset so the
    # locked 3D target matches what you see in the image.
    plan_top_gather_pan_center_enable: bool = False
    # During plan_top lift/align/tilt/descend, keep the object in frame with a gentle smoothed
    # shoulder-pan correction. This does not update the frozen 3D target; it only keeps the
    # gripper camera from walking the bbox off to one side during the approach.
    plan_top_keep_in_frame_enable: bool = True
    plan_top_keep_in_frame_min_conf: float = 0.25
    plan_top_keep_in_frame_deadband_px: float = 26.0
    plan_top_keep_in_frame_kp: float = 0.45
    plan_top_keep_in_frame_max_step_deg: float = 2.5
    plan_top_keep_in_frame_period_iters: int = 1
    # After descend, correct residual horizontal error with shoulder_pan before / alongside XY.
    plan_top_center_pan_enable: bool = True
    plan_top_center_pan_deadband_px: float = 12.0
    plan_top_center_pan_kp: float = 0.95
    plan_top_center_pan_max_step_deg: float = 9.0
    # When center phase has no/low-conf bbox, nudge EE straight up in base +Z (orientation
    # preserved) every N iters so the camera clears the cube and YOLO can see again — instead of
    # relying on look-at IK. Set step_m=0 to disable.
    plan_top_center_no_det_lift_step_m: float = 0.006
    plan_top_center_no_det_lift_max_m: float = 0.04
    plan_top_center_no_det_recover_every_n: int = 4

    dry_run: bool = False
    dry_run_image: str = ""
    show_window: bool = False

    # Rerun live visualization. Opens a Rerun viewer and logs:
    #   - sim3d/*          robot URDF meshes, EE frame, target cube p_target_base, camera frustum, EE trail
    #   - observation/*    camera RGB (with YOLO bbox + center cross) and depth
    #   - scalars/*        z_ee, depth, conf, phase
    # See `--display-data` / `--display-sim3d`.
    display_data: bool = True
    display_sim3d: bool = True
    display_ip: str | None = None
    display_port: int | None = None
    # Assumed half-extent (m) of the YOLO target object for the 3D cube overlay.
    display_object_half_size_m: float = 0.025
