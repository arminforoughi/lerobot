#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: ``lerobot-yolo-vlm-grasp``.

Thin wrapper around :func:`lerobot.manipulation.yolo_track.hybrid_grasp.yolo_vlm_grasp`.
Drives the existing YOLO ``plan_top`` approach to a hover above the target, then hands off
to a VLM fine-refinement loop + current-sensing gripper close. See
``--help`` for the full flag list (all ``lerobot-yolo-track-approach`` flags are accepted,
plus ``--vlm-*`` / ``--grasp-*`` / ``--hybrid-*``).
"""

from __future__ import annotations

from lerobot.manipulation.yolo_track.hybrid_grasp import yolo_vlm_grasp


def main() -> None:
    yolo_vlm_grasp()


if __name__ == "__main__":
    main()
