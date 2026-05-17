# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure math helpers used by the YOLO-track controller.

Nothing in this module touches robot hardware, detectors, or Rerun. Everything is deterministic
and unit-testable. Keep it free of heavy imports so callers don't pay for scipy/cv2 unless the
pipeline actually runs.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def parse_tf_string(tf_str: str) -> np.ndarray:
    """Parse a ``"x,y,z,rx,ry,rz"`` string into a 4x4 homogeneous transform (rotvec radians)."""
    parts = [float(v.strip()) for v in tf_str.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values (x,y,z,rx,ry,rz), got {len(parts)}: {tf_str}")
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 3] = parts[:3]
    if any(abs(v) > 1e-8 for v in parts[3:6]):
        tf[:3, :3] = Rotation.from_rotvec(parts[3:6]).as_matrix()
    return tf


def camera_opencv_to_robot_rotation(*, flip_lateral: bool = False) -> np.ndarray:
    """Rotation mapping OpenCV camera axes (X right, Y down, Z forward) to a typical robot base
    (X forward, Y left, Z up). ``flip_lateral`` inverts the Y sign for mirrored mounts."""
    R = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    if flip_lateral:
        R[1, 0] *= -1.0
    return R


def rotation_base_cam(T_base_cam: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation block of ``T_base_cam``."""
    return np.asarray(T_base_cam, dtype=np.float64)[:3, :3]


def visual_servo_delta_base(
    *,
    cx: float,
    cy: float,
    fx: float,
    fy: float,
    cx0: float,
    cy0: float,
    depth_m: float | None,
    R_base_cam: np.ndarray,
    kp_xy: float,
    kp_z: float,
    max_step_m: float,
    target_depth_m: float,
    min_depth_m: float,
    bbox_area_frac: float,
    area_close_frac: float,
    forward_if_no_depth_m: float,
    invert_lateral: bool,
) -> np.ndarray:
    """Camera-frame correction (meters) rotated into base; Z_cam is forward (OpenCV optical)."""
    ex_px = float(cx - cx0)
    ey_px = float(cy - cy0)
    sign = -1.0 if invert_lateral else 1.0

    d = float(depth_m) if depth_m is not None and depth_m > 1e-3 else 0.35
    sx = sign * kp_xy * (ex_px / max(fx, 1e-6)) * d
    sy = sign * kp_xy * (ey_px / max(fy, 1e-6)) * d
    sz = 0.0

    if depth_m is not None and depth_m <= min_depth_m:
        sz = 0.0
    elif depth_m is not None and depth_m > min_depth_m + 0.02:
        sz = float(np.clip(kp_z * (depth_m - target_depth_m), 0.0, max_step_m))
    elif depth_m is None and bbox_area_frac < area_close_frac:
        sz = min(forward_if_no_depth_m, max_step_m)

    delta_cam = np.array([sx, sy, sz], dtype=np.float64)
    delta_base = R_base_cam @ delta_cam
    n = float(np.linalg.norm(delta_base))
    if n > max_step_m and n > 1e-9:
        delta_base *= max_step_m / n
    return delta_base


def parse_horizon_dir_base(s: str) -> np.ndarray:
    """Parse a ``"vx,vy,vz"`` base-frame direction; projects to XY plane and normalizes."""
    parts = [float(v.strip()) for v in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values (vx,vy,vz), got {len(parts)}: {s}")
    v = np.array(parts[:3], dtype=np.float64)
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        v = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        v = v / n
    return v


def clamp_base_point_z_min(p_base: np.ndarray, z_min_m: float | None) -> np.ndarray:
    """Return a copy of ``p_base`` with ``z`` raised to at least ``z_min_m`` when set.

    Use when a known workspace floor (e.g. tabletop height in the URDF base frame)
    should bound noisy back-projections that otherwise sit slightly below the plane
    in Rerun or downstream planners.
    """
    out = np.asarray(p_base, dtype=np.float64).reshape(3).copy()
    if z_min_m is None:
        return out
    zm = float(z_min_m)
    if math.isfinite(zm):
        out[2] = max(float(out[2]), zm)
    return out


def clamp_base_point_z_max(p_base: np.ndarray, z_max_m: float | None) -> np.ndarray:
    """Return a copy of ``p_base`` with ``z`` lowered to at most ``z_max_m`` when set.

    Use when a tabletop workspace should never place the target above a small band
    (guards against wrong long-range depth / background rays that pull the apex into
    the air and make IK chase the ceiling).
    """
    out = np.asarray(p_base, dtype=np.float64).reshape(3).copy()
    if z_max_m is None:
        return out
    zm = float(z_max_m)
    if math.isfinite(zm):
        out[2] = min(float(out[2]), zm)
    return out


def cam_z_of_base_point(T_base_cam: np.ndarray, p_base: np.ndarray) -> float:
    """Return the Z coordinate of ``p_base`` in the OpenCV camera frame (+Z = optical forward).

    Used to reject back-projections that land behind the camera (wrong extrinsic or plane).
    """
    Tbc = np.asarray(T_base_cam, dtype=np.float64)
    p_b = np.asarray(p_base, dtype=np.float64).reshape(3)
    p_cam = Tbc[:3, :3].T @ (p_b - Tbc[:3, 3])
    return float(p_cam[2])


def point_cam_to_base(
    T_base_cam: np.ndarray,
    *,
    u: float,
    v_pix: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx0: float,
    cy0: float,
) -> np.ndarray:
    """Back-project pixel ``(u, v_pix)`` at ``depth_m`` into the base frame.

    OpenCV camera frame assumed (Z forward, X right, Y down).
    """
    x = (float(u) - cx0) / max(fx, 1e-6) * depth_m
    y = (float(v_pix) - cy0) / max(fy, 1e-6) * depth_m
    z = float(depth_m)
    p_h = np.array([x, y, z, 1.0], dtype=np.float64)
    return (np.asarray(T_base_cam, dtype=np.float64) @ p_h)[:3]


def project_base_point_to_cam_uv(
    T_base_cam: np.ndarray,
    p_base: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx0: float,
    cy0: float,
    z_min_m: float = 0.02,
) -> tuple[float, float] | None:
    """Project a base-frame 3D point into OpenCV camera pixels (Z forward).

    Returns ``None`` if the point lies on or behind the near clipping plane ``z_min_m``
    in the camera frame.
    """
    T = np.asarray(T_base_cam, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    p_b = np.asarray(p_base, dtype=np.float64).reshape(3)
    p_cam = R.T @ (p_b - t)
    zc = float(p_cam[2])
    if zc < float(z_min_m):
        return None
    u = float(fx) * float(p_cam[0]) / zc + float(cx0)
    v = float(fy) * float(p_cam[1]) / zc + float(cy0)
    return u, v


def ema_p_base(prev: np.ndarray | None, meas: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average on a 3D point; ``alpha`` near 1 = trust new measurement."""
    m = np.asarray(meas, dtype=np.float64).reshape(3)
    if prev is None or alpha >= 1.0 - 1e-9:
        return m.copy()
    p = np.asarray(prev, dtype=np.float64).reshape(3)
    return float(alpha) * m + (1.0 - float(alpha)) * p


def R_ee_step_clamp(R_curr_ee: np.ndarray, R_des_ee: np.ndarray, max_step_deg: float) -> np.ndarray:
    """Clamp the geodesic rotation from ``R_curr_ee`` to ``R_des_ee`` to at most ``max_step_deg``."""
    md = float(max_step_deg)
    if md <= 1e-6:
        return np.asarray(R_des_ee, dtype=np.float64)
    R0 = np.asarray(R_curr_ee, dtype=np.float64)[:3, :3]
    R1 = np.asarray(R_des_ee, dtype=np.float64)[:3, :3]
    r_rel = Rotation.from_matrix(R1 @ R0.T)
    rotvec = r_rel.as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    lim = np.deg2rad(md)
    if ang > lim and ang > 1e-9:
        rotvec = rotvec * (lim / ang)
    return Rotation.from_rotvec(rotvec).as_matrix() @ R0


def look_at_R_base_cam(
    eye_base: np.ndarray,
    target_base: np.ndarray,
    *,
    world_up: np.ndarray | None = None,
) -> np.ndarray | None:
    """Camera frame (+Z optical axis): build R_base_cam with columns = camera axes in base."""
    up0 = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if world_up is None
        else np.asarray(world_up, dtype=np.float64).reshape(3)
    )
    z_axis = target_base - eye_base
    zn = float(np.linalg.norm(z_axis))
    if zn < 1e-7:
        return None
    z_axis = z_axis / zn
    up = up0 / max(float(np.linalg.norm(up0)), 1e-9)
    if abs(float(np.dot(z_axis, up))) > 0.995:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    xn = float(np.linalg.norm(x_axis))
    if xn < 1e-7:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = np.cross(up, z_axis)
        xn = float(np.linalg.norm(x_axis))
    if xn < 1e-7:
        return None
    x_axis = x_axis / xn
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def wrist_down_R_base_ee(R_cur: np.ndarray | None = None) -> np.ndarray:
    """Build an R_base_ee with EE +Z pointing straight down while preserving current heading.

    The SO-101 is a 5-DOF arm without free wrist roll, so a rigid target like "X_ee = +X_base"
    is almost never reachable from a laterally-reached pose — IK either fails or barely moves.
    Instead we build the target from the *current* rotation: project current X_ee onto the
    horizontal plane, use that as target X_ee (preserves yaw), set Z_ee = -Z_base, derive Y_ee
    to keep the frame right-handed. This pure pitch-down motion is reachable.

    If ``R_cur`` is None (fallback), defaults to X_ee = +X_base.
    """
    if R_cur is None:
        x_flat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_cur = np.asarray(R_cur, dtype=np.float64)[:, 0]
        x_flat = np.array([x_cur[0], x_cur[1], 0.0], dtype=np.float64)
        n = float(np.linalg.norm(x_flat))
        if n < 1e-6:
            x_flat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            x_flat = x_flat / n
    z_tgt = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_tgt = np.cross(z_tgt, x_flat)
    y_tgt = y_tgt / float(np.linalg.norm(y_tgt))
    return np.column_stack([x_flat, y_tgt, z_tgt])


def rot_step_toward(
    R_cur: np.ndarray, R_tgt: np.ndarray, step_deg: float
) -> tuple[np.ndarray, float]:
    """Take a single small axis-angle step from ``R_cur`` toward ``R_tgt``.

    Returns ``(R_next, remaining_deg)``. When within ``step_deg`` of the target, snaps directly
    to ``R_tgt`` and returns ``0.0``.
    """
    R_cur = np.asarray(R_cur, dtype=np.float64)
    R_tgt = np.asarray(R_tgt, dtype=np.float64)
    R_rel = R_tgt @ R_cur.T
    cos_a = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos_a))
    if angle < 1e-6:
        return R_cur.copy(), 0.0
    sin_a = float(np.sin(angle))
    if abs(sin_a) < 1e-8:
        return R_tgt.copy(), 0.0
    axis = np.array(
        [
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sin_a)
    step = float(min(np.deg2rad(max(0.0, step_deg)), angle))
    if angle - step < np.deg2rad(0.5):
        return R_tgt.copy(), 0.0
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    R_step = np.eye(3, dtype=np.float64) + np.sin(step) * K + (1.0 - np.cos(step)) * (K @ K)
    R_next = R_step @ R_cur
    return R_next, float(np.degrees(angle - step))


def T_ee_aim_cam_at(
    T_base_ee: np.ndarray,
    T_ee_cam: np.ndarray,
    delta_base: np.ndarray,
    *,
    R_base_cam_desired: np.ndarray,
) -> np.ndarray:
    """Build a target EE pose that (a) translates by ``delta_base`` and (b) orients the camera so
    its +Z optical axis matches ``R_base_cam_desired``."""
    R_ee_cam = np.asarray(T_ee_cam, dtype=np.float64)[:3, :3]
    R_base_ee = np.asarray(R_base_cam_desired, dtype=np.float64) @ R_ee_cam.T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_base_ee
    T[:3, 3] = np.asarray(T_base_ee, dtype=np.float64)[:3, 3] + np.asarray(
        delta_base, dtype=np.float64
    ).reshape(3)
    return T


def gripper_pose_translate_only(T_base_ee: np.ndarray, delta_base: np.ndarray) -> np.ndarray:
    """Return a copy of ``T_base_ee`` with its translation shifted by ``delta_base`` (no rotation change)."""
    T = np.asarray(T_base_ee, dtype=np.float64).copy()
    T[:3, 3] = T[:3, 3] + np.asarray(delta_base, dtype=np.float64).reshape(3)
    return T


def build_top_pose(p_target_base: np.ndarray, *, hover_height_m: float) -> np.ndarray:
    """Pose with wrist pointing straight down, centered over ``p_target_base`` at +hover_height_m."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = wrist_down_R_base_ee()
    T[:3, 3] = np.asarray(p_target_base, dtype=np.float64).reshape(3) + np.array(
        [0.0, 0.0, float(hover_height_m)], dtype=np.float64
    )
    return T
