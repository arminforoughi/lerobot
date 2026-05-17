# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optional manipulation pipelines (e.g. DimOS-style pick sequences, YOLO-track approach).

Each sub-pipeline is imported lazily so a broken/optional dependency in one (e.g. a missing
perception helper) doesn't prevent the others from being usable. Callers should import the
submodule they need directly, e.g. ``from lerobot.manipulation.yolo_track import ...``.
"""

__all__ = [
    "DimOSStylePickConfig",
    "SplitPickPlan",
    "YoloTrackApproachConfig",
    "build_split_pick_plan",
    "grasp_strategy_from_task",
    "plan_dimos_style_pick",
    "yolo_track_approach",
]


def __getattr__(name: str):
    # Lazy re-exports so importing ``lerobot.manipulation`` never fails just because one
    # optional pipeline has a transitive import error.
    if name == "grasp_strategy_from_task":
        from lerobot.manipulation.approach_policy import grasp_strategy_from_task

        return grasp_strategy_from_task
    if name in ("DimOSStylePickConfig", "plan_dimos_style_pick"):
        from lerobot.manipulation import dimos_style

        return getattr(dimos_style, name)
    if name in ("SplitPickPlan", "build_split_pick_plan"):
        from lerobot.manipulation import manipulation_orchestrator

        return getattr(manipulation_orchestrator, name)
    if name in ("YoloTrackApproachConfig", "yolo_track_approach"):
        from lerobot.manipulation import yolo_track

        return getattr(yolo_track, name)
    raise AttributeError(f"module 'lerobot.manipulation' has no attribute {name!r}")
