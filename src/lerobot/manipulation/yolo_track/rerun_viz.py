# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rerun live-visualization logging for the YOLO-track controller.

All logging is best-effort: if Rerun isn't installed or a sub-module fails, the controller keeps
running and just logs a debug message. Nothing here should ever raise back into the main loop.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Approximate axis-aligned half-extents (meters) in robot base frame for Rerun 3D proxies.
# Long/thin objects use the Z half as the “main” extent (typical tabletop: object long axis
# often closer to vertical than to base X/Y, but this is only a visualization hint).
_SEMANTIC_PROFILES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("smart phone", (0.075, 0.004, 0.038)),
    ("cell phone", (0.075, 0.004, 0.038)),
    ("smartphone", (0.075, 0.004, 0.038)),
    ("iphone", (0.075, 0.004, 0.038)),
    ("cellphone", (0.075, 0.004, 0.038)),
    ("phone", (0.075, 0.005, 0.038)),
    ("tablet", (0.12, 0.004, 0.085)),
    ("pencil", (0.004, 0.004, 0.09)),
    ("marker", (0.005, 0.005, 0.09)),
    ("pen", (0.004, 0.004, 0.09)),
    ("mug", (0.045, 0.045, 0.055)),
    ("cup", (0.045, 0.045, 0.055)),
    ("bottle", (0.035, 0.035, 0.12)),
    ("can", (0.033, 0.033, 0.065)),
    ("bowl", (0.08, 0.08, 0.035)),
    ("book", (0.12, 0.015, 0.09)),
    ("box", (0.06, 0.06, 0.06)),
)
# Longer phrases first so e.g. "pencil" does not match the substring "pen".
_SEMANTIC_PROFILES_MATCH_ORDER: tuple[tuple[str, tuple[float, float, float]], ...] = tuple(
    sorted(_SEMANTIC_PROFILES, key=lambda kv: -len(kv[0]))
)


def rerun_object_geometry_from_semantic(
    label: str | None, *, fallback_cube_half_m: float
) -> tuple[np.ndarray, str]:
    """Pick (hx, hy, hz) half-extents and a short name for Rerun paths from a free-text query."""
    h0 = float(fallback_cube_half_m)
    default = (np.array([h0, h0, h0], dtype=np.float64), "yolo_target")
    if label is None:
        return default
    raw = str(label).strip()
    if not raw:
        return default
    s = raw.lower()
    for key, halves in _SEMANTIC_PROFILES_MATCH_ORDER:
        if key in s:
            hx, hy, hz = (float(x) for x in halves)
            tag = key.replace(" ", "_")
            return (np.array([hx, hy, hz], dtype=np.float64), tag)
    return (np.array([h0, h0, h0], dtype=np.float64), raw[:48].replace(" ", "_"))


def log_rerun_iter(
    *,
    frame: int,
    camera_key: str,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    bbox_xyxy: tuple[float, float, float, float] | None,
    bbox_center_raw: tuple[float, float] | None,  # noqa: ARG001 — kept for signature parity
    bbox_center_smoothed: tuple[float, float] | None,
    cx0: float,
    cy0: float,
    conf: float | None,
    phase: str,
    z_ee: float | None,
    depth_m: float | None,
    kinematics,
    joints_deg: np.ndarray,
    T_base_ee: np.ndarray | None,  # noqa: ARG001 — reserved for future overlays
    T_base_cam: np.ndarray | None,
    p_target_base: np.ndarray | None,
    ee_trail: list[np.ndarray],
    object_half_size_m: float,
    show_sim3d: bool,
    show_camera: bool,
    object_semantic_label: str | None = None,
    sim3d_ground_plane_z_m: float = 0.0,
    bearing_az_deg: float | None = None,
    bearing_el_deg: float | None = None,
) -> None:
    """Best-effort Rerun logging. Silently no-ops if rerun not available."""
    try:
        import rerun as rr
    except Exception:
        return
    from lerobot.utils.visualization_utils import colorize_depth_mm_u16

    rr.set_time("frame", sequence=int(frame))

    if show_camera:
        try:
            vis = np.asarray(rgb).copy()
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if bbox_center_smoothed is not None:
                cv2.drawMarker(
                    vis,
                    (int(round(bbox_center_smoothed[0])), int(round(bbox_center_smoothed[1]))),
                    (255, 200, 0),
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    2,
                )
            cv2.drawMarker(
                vis,
                (int(round(cx0)), int(round(cy0))),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                18,
                2,
            )
            label = f"phase={phase}"
            if object_semantic_label:
                short = str(object_semantic_label).strip().replace("\n", " ")
                if len(short) > 36:
                    short = short[:33] + "..."
                label += f" | {short}"
            if conf is not None:
                label += f" conf={conf:.2f}"
            if depth_m is not None:
                label += f" d={depth_m:.2f}m"
            if bearing_az_deg is not None and bearing_el_deg is not None:
                label += f" az={bearing_az_deg:+.1f}° el={bearing_el_deg:+.1f}°"
            cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            rr.log(f"observation/{camera_key}", rr.Image(vis))
        except Exception as e:
            logger.debug("rerun camera log failed: %s", e)

        if depth is not None:
            try:
                d = np.asarray(depth)
                if np.issubdtype(d.dtype, np.floating):
                    d = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
                elif d.dtype != np.uint16:
                    d = d.astype(np.uint16)
                depth_rgb = cv2.cvtColor(colorize_depth_mm_u16(d), cv2.COLOR_BGR2RGB)
                rr.log(f"observation/{camera_key}_depth", rr.Image(depth_rgb))
            except Exception as e:
                logger.debug("rerun depth log failed: %s", e)

    if conf is not None:
        try:
            rr.log("scalars/yolo_confidence", rr.Scalars(float(conf)))
        except Exception:
            pass
    if depth_m is not None:
        try:
            rr.log("scalars/target_depth_m", rr.Scalars(float(depth_m)))
        except Exception:
            pass
    if z_ee is not None:
        try:
            rr.log("scalars/ee_z_m", rr.Scalars(float(z_ee)))
        except Exception:
            pass
    phase_code = {
        "lift": 1.0,
        "align": 2.0,
        "descend": 3.0,
        "vlm": 4.0,
        "vlm_descent": 4.5,
        "vlm_grasp": 5.0,
        "search": 10.0,
        "tracking": 11.0,
        "approaching": 12.0,
        "hold": 13.0,
    }.get(phase, 0.0)
    try:
        rr.log("scalars/phase_code", rr.Scalars(float(phase_code)))
    except Exception:
        pass

    # Force a flush so the viewer feels "live" even when the loop is bursty.
    try:
        rr.flush()
    except Exception:
        pass

    if not show_sim3d:
        return
    try:
        from lerobot.utils.manipulation_sim3d import log_manipulation_sim3d

        centers = None
        halves = None
        labels = None
        focus = None
        if p_target_base is not None:
            centers = np.asarray(p_target_base, dtype=np.float64).reshape(1, 3)
            halves_arr, tag = rerun_object_geometry_from_semantic(
                object_semantic_label, fallback_cube_half_m=float(object_half_size_m)
            )
            halves = halves_arr.reshape(1, 3)
            labels = [tag]
            focus = 0
        path_arr = None
        if ee_trail and len(ee_trail) >= 2:
            path_arr = np.asarray(ee_trail, dtype=np.float64)
        log_manipulation_sim3d(
            frame_sequence=frame,
            kinematics=kinematics,
            joint_deg=np.asarray(joints_deg, dtype=np.float64),
            object_centers_base=centers,
            object_half_sizes_base=halves,
            object_labels=labels,
            focus_object_index=focus,
            planned_ee_positions_base=path_arr,
            plan_summary=(
                f"phase={phase}"
                + (f" | {object_semantic_label}" if object_semantic_label else "")
            ),
            ground_plane_z_m=float(sim3d_ground_plane_z_m),
        )
    except Exception as e:
        logger.debug("rerun sim3d log failed: %s", e)

    if T_base_cam is not None:
        try:
            Tbc = np.asarray(T_base_cam, dtype=np.float64)
            eye = Tbc[:3, 3]
            z_cam_base = Tbc[:3, 2]
            tip = eye + 0.30 * z_cam_base
            rr.log(
                "sim3d/camera/optical_axis",
                rr.LineStrips3D(
                    strips=[np.stack([eye, tip], axis=0)],
                    radii=0.003,
                    colors=np.array([[255, 255, 80, 255]], dtype=np.uint8),
                ),
            )
            rr.log(
                "sim3d/camera/eye",
                rr.Points3D(
                    positions=eye.reshape(1, 3),
                    radii=0.012,
                    colors=np.array([[255, 220, 80, 255]], dtype=np.uint8),
                ),
            )
        except Exception as e:
            logger.debug("rerun camera axis log failed: %s", e)

    if p_target_base is not None and T_base_cam is not None:
        try:
            Tbc = np.asarray(T_base_cam, dtype=np.float64)
            eye = Tbc[:3, 3]
            p = np.asarray(p_target_base, dtype=np.float64).reshape(3)
            rr.log(
                "sim3d/aim_ray",
                rr.LineStrips3D(
                    strips=[np.stack([eye, p], axis=0)],
                    radii=0.0025,
                    colors=np.array([[255, 60, 200, 255]], dtype=np.uint8),
                ),
            )
        except Exception:
            pass
