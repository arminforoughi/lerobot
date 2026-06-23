#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin CLI entrypoint for ``lerobot-yolo-track-approach``.

The actual implementation lives in :mod:`lerobot.manipulation.yolo_track`:

- :mod:`lerobot.manipulation.yolo_track.config` holds ``YoloTrackApproachConfig`` (CLI flags).
- :mod:`lerobot.manipulation.yolo_track.runner` holds the main control loop.
- Sibling modules (``math_utils``, ``depth``, ``motion_primitives``, ``rerun_viz``) hold the
  stateless helpers used by the runner.

See the package's ``__init__`` docstring or ``runner.yolo_track_approach`` for the high-level
behavior and per-phase semantics (search-scan → gather → lift → align → tilt → descend → center
for ``--approach-style=plan_top``; phased lift/align/descend for ``--approach-style=umbrella``;
continuous visual servo for ``--approach-style=servo``; and closed-loop **axis_track** (see
``visual_axis_track`` + ``axis_track_*`` config fields) for keep-center-while-closing behaviour.

Run ``lerobot-yolo-track-approach --help`` for all the flags (inherited verbatim from the
original monolithic script — no CLI surface changes were made during modularization).
"""

from __future__ import annotations

from lerobot.manipulation.yolo_track import yolo_track_approach


def main() -> None:
    yolo_track_approach()


if __name__ == "__main__":
    main()
