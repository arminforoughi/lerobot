# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""YOLO-World text-query → detect → servo-approach pipeline.

The public entry point is :func:`yolo_track_approach` (thin wrapper used by the
``lerobot-yolo-track-approach`` CLI). Submodules expose the reusable primitives:

- :mod:`.math_utils`         TF parsing, rotation helpers, camera↔base projection
- :mod:`.depth`              depth extraction (stereo median + bbox-size pinhole fallback)
- :mod:`.motion_primitives`  joint streaming + gripper-aware Cartesian nudges
- :mod:`.rerun_viz`          live Rerun logging of frames, scalars, 3D scene
- :mod:`.visual_axis_track` design notes for ``approach_style=axis_track``
- :mod:`.config`             ``YoloTrackApproachConfig`` dataclass (CLI surface)
- :mod:`.runner`             main control loop (``yolo_track_approach``)

Top-level re-exports are lazy so that importing a single submodule (e.g. ``.depth``) does not
pull heavy transitive dependencies like ``scipy``/``torch`` until they are actually needed.
"""

from __future__ import annotations

__all__ = [
    "HybridGraspConfig",
    "MinimalTrackerConfig",
    "PlanTopHandoffState",
    "YoloTrackApproachConfig",
    "minimal_tracker",
    "run_minimal_tracker",
    "run_vlm_refine_and_grasp",
    "run_yolo_track_approach",
    "yolo_track_approach",
    "yolo_vlm_grasp",
]


def __getattr__(name: str):
    if name == "YoloTrackApproachConfig":
        from lerobot.manipulation.yolo_track.config import YoloTrackApproachConfig

        return YoloTrackApproachConfig
    if name in ("MinimalTrackerConfig", "minimal_tracker", "run_minimal_tracker"):
        from lerobot.manipulation.yolo_track import minimal_tracker as minimal_tracker_module

        return getattr(minimal_tracker_module, name)
    if name in ("yolo_track_approach", "run_yolo_track_approach", "PlanTopHandoffState"):
        from lerobot.manipulation.yolo_track import runner

        return getattr(runner, name)
    if name in ("HybridGraspConfig", "run_vlm_refine_and_grasp", "yolo_vlm_grasp"):
        from lerobot.manipulation.yolo_track import hybrid_grasp

        return getattr(hybrid_grasp, name)
    raise AttributeError(f"module 'lerobot.manipulation.yolo_track' has no attribute {name!r}")
