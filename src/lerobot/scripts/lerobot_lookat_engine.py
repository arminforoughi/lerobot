# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the estimate-then-plan look-at grasp engine (eye-in-hand)."""

from lerobot.manipulation.visual_servo.lookat_engine import lookat_engine_main


def main() -> None:
    lookat_engine_main()


if __name__ == "__main__":
    main()
