# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Depth extraction helpers: stereo ROI median + pinhole fallback from bbox size."""

from __future__ import annotations

import numpy as np


def median_depth_m(
    depth: np.ndarray,
    xyxy: tuple[float, float, float, float],
    *,
    depth_scale: float,
    min_mm: float = 50.0,
    max_mm: float = 5000.0,
) -> float | None:
    """Median depth (meters) inside ``xyxy``.

    Supports both float depth maps (already in meters) and uint16 depth (``depth_scale`` converts
    a raw sample to meters, i.e. mm * 0.001 for typical stereo). Samples outside
    ``[min_mm, max_mm]`` are rejected. Returns ``None`` when fewer than 5 valid samples fall
    inside the ROI.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    h, w = depth.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = np.asarray(depth[y1 : y2 + 1, x1 : x2 + 1])
    if roi.size == 0:
        return None
    if np.issubdtype(roi.dtype, np.floating):
        d_m = roi.astype(np.float64).reshape(-1)
        valid = d_m[(d_m * 1000.0 > min_mm) & (d_m * 1000.0 < max_mm)]
        if valid.size < 5:
            return None
        return float(np.median(valid))
    raw = roi.astype(np.float64).reshape(-1)
    valid = raw[(raw * depth_scale * 1000.0 > min_mm) & (raw * depth_scale * 1000.0 < max_mm)]
    if valid.size < 5:
        return None
    return float(np.median(valid) * depth_scale)


def depth_from_bbox_size(
    xyxy: tuple[float, float, float, float],
    *,
    fx: float,
    fy: float,
    target_physical_size_m: float,
) -> float | None:
    """Estimate depth (m) from the larger bbox side assuming a known real object size.

    Pinhole model: ``d = f * S_real / W_px`` where f is the focal length along the matching axis
    and W_px is the pixel size of the object along that axis. Uses the larger bbox side and its
    corresponding focal length (more robust to partial occlusion of the other dimension).

    This is the primary depth source when stereo fails in the near field — typical OAK-D models
    have a minimum working distance of ~35 cm, but a known-size cube resolves well closer.
    """
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w_px = max(0.0, x2 - x1)
    h_px = max(0.0, y2 - y1)
    if w_px < 2.0 and h_px < 2.0:
        return None
    if w_px >= h_px:
        f_used = float(fx)
        s_px = w_px
    else:
        f_used = float(fy)
        s_px = h_px
    if s_px <= 1e-3 or f_used <= 1e-3 or target_physical_size_m <= 1e-6:
        return None
    return float(f_used) * float(target_physical_size_m) / float(s_px)
