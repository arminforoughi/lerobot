# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simplified eye-in-hand visual servoing for SO-101 + OAK-D.

This package intentionally re-implements the tracker from scratch rather than reusing
``manipulation/yolo_track``: the control law is a single proportional PBVS step per frame,
not a phased state machine.
"""
