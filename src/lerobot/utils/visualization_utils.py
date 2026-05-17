# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numbers
import os

import cv2
import numpy as np
import rerun as rr

from lerobot.processor import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR

logger = logging.getLogger(__name__)


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)
    try:
        rr.disable_batching()
    except Exception:
        pass


def send_agentic_rerun_blueprint(
    *,
    show_camera_stream: bool,
    show_sim3d: bool,
    camera_key: str | None = None,
) -> None:
    """Push a default blueprint: separate RGB + depth 2D views and the 3D sim view.

    When ``camera_key`` is provided we create two Spatial2DViews — one targeting
    ``observation/{camera_key}`` (RGB) and one targeting ``observation/{camera_key}_depth``
    (colorized depth) — so the viewer shows both side by side instead of picking just one.
    """
    if not show_camera_stream and not show_sim3d:
        return
    try:
        import rerun.blueprint as rrb
    except Exception as e:
        logger.warning("Rerun blueprint API unavailable: %s", e)
        return

    def _make_2d(name: str, contents: str):
        origin = contents.rstrip("/*")
        # Prefer origin= (pins the view to a specific entity so RGB and depth
        # render in distinct panels instead of getting de-duplicated by
        # auto-layout when both share the `observation/**` subtree).
        for kwargs in (
            {"name": name, "origin": origin, "contents": f"{origin}/**"},
            {"name": name, "origin": origin},
            {"name": name, "contents": contents},
        ):
            try:
                return rrb.Spatial2DView(**kwargs)
            except TypeError:
                continue
            except Exception as e:
                logger.debug("Spatial2DView(%s) failed: %s", kwargs, e)
                continue
        try:
            return rrb.Spatial2DView(name=name)
        except Exception:
            return None

    cam_views: list = []
    if show_camera_stream:
        if camera_key:
            rgb_v = _make_2d("Camera RGB", f"observation/{camera_key}")
            depth_v = _make_2d("Camera depth", f"observation/{camera_key}_depth")
            if rgb_v is not None:
                cam_views.append(rgb_v)
            if depth_v is not None:
                cam_views.append(depth_v)
        else:
            v = _make_2d("Camera (RGB + depth)", "observation/**")
            if v is not None:
                cam_views.append(v)

    v3d = None
    if show_sim3d:
        for kwargs in (
            {"name": "SO101 + scene (base frame)", "contents": "sim3d/**"},
            {"name": "SO101 + scene (base frame)", "origin": "sim3d"},
        ):
            try:
                v3d = rrb.Spatial3DView(**kwargs)
                break
            except TypeError:
                continue
        if v3d is None:
            try:
                v3d = rrb.Spatial3DView(name="SO101 + scene (base frame)")
            except Exception as e:
                logger.warning("Could not create Spatial3DView: %s", e)

    if not cam_views and v3d is None:
        return
    try:
        if cam_views and v3d is not None:
            if len(cam_views) > 1:
                try:
                    cam_panel = rrb.Vertical(*cam_views)
                except Exception:
                    try:
                        cam_panel = rrb.Tabs(*cam_views)
                    except Exception:
                        cam_panel = cam_views[0]
            else:
                cam_panel = cam_views[0]
            try:
                root = rrb.Horizontal(cam_panel, v3d, column_shares=[1, 2])
            except TypeError:
                root = rrb.Horizontal(cam_panel, v3d)
        elif cam_views:
            root = cam_views[0] if len(cam_views) == 1 else rrb.Vertical(*cam_views)
        else:
            root = v3d
        bp = rrb.Blueprint(root, auto_layout=True)
        rr.send_blueprint(bp, make_active=True, make_default=True)
        n_total = len(cam_views) + (1 if v3d is not None else 0)
        logger.info(
            "Rerun blueprint: opened %d view(s) including 3D (robot base frame). "
            "Use the 'SO101 + scene' panel if you do not see the arm.",
            n_total,
        )
    except Exception as e:
        logger.warning(
            "Could not send Rerun blueprint. Manually add View → Spatial 3D and set origin to `sim3d`: %s",
            e,
        )


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def colorize_depth_mm_u16(depth_mm: np.ndarray, max_range_mm: int = 2000) -> np.ndarray:
    """Convert uint16 depth in millimeters to BGR uint8 (e.g. for OpenCV or Rerun)."""
    depth = np.asarray(depth_mm)
    clipped = np.clip(depth.astype(np.float32), 0, max_range_mm)
    normalized = (clipped / max_range_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def log_stereo_pair_to_rerun(
    *,
    camera_key: str,
    rgb_hwc: np.ndarray,
    depth_u16_hw: np.ndarray,
    frame_sequence: int,
    compress_images: bool = False,
) -> None:
    """Log one RGB + colormapped depth pair on a Rerun timeline (agentic / MCP camera streams)."""
    if rgb_hwc.size == 0:
        return
    rr.set_time("frame", sequence=int(frame_sequence))
    h, w = int(rgb_hwc.shape[0]), int(rgb_hwc.shape[1])
    depth_bgr = colorize_depth_mm_u16(depth_u16_hw)
    depth_rgb = cv2.cvtColor(cv2.resize(depth_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    log_rerun_data(
        observation={
            camera_key: np.asarray(rgb_hwc),
            f"{camera_key}_depth_color": depth_rgb,
        },
        compress_images=compress_images,
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    static_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
        static_images: If True, log images as Rerun ``static`` data (does not advance with timelines — wrong for live cameras).
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=static_images)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
