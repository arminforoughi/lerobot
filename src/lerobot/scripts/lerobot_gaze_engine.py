# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the gaze-first visual servoing engine (eye-in-hand)."""

from lerobot.manipulation.visual_servo.gaze_engine import gaze_engine_main


def main() -> None:
    gaze_engine_main()


if __name__ == "__main__":
    main()
