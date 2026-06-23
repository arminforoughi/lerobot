#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin CLI entrypoint for the minimal live tracker.

This script runs the reduced servo-only pipeline:
detect -> keep centered -> approach to depth standoff.
"""

from __future__ import annotations

from lerobot.manipulation.yolo_track.minimal_tracker import minimal_tracker


def main() -> None:
    minimal_tracker()


if __name__ == "__main__":
    main()
