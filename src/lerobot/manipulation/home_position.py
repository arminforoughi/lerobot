# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Saved resting pose and slow fold-home for approach / visual-servo engines."""

from __future__ import annotations

import logging
import os
import select
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lerobot.manipulation.yolo_track.motion_primitives import send_joint_target_smoothly

logger = logging.getLogger(__name__)

DEFAULT_HOME_CONFIG_PATH = "SO101/so101_home.yaml"
ARM_MOTORS_DEFAULT = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@dataclass
class HomePosition:
    motor_names: list[str]
    joints_deg: np.ndarray
    gripper_pct: float


@dataclass
class HomeMotionSettings:
    config_path: str = DEFAULT_HOME_CONFIG_PATH
    fold_on_interrupt: bool = True
    live_keypress: bool = True
    joint_step_deg: float = 1.5
    inter_step_sleep_s: float = 0.04


def home_settings_from_config(cfg: Any) -> HomeMotionSettings:
    """Build settings from engine configs that expose the standard home_* fields."""
    return HomeMotionSettings(
        config_path=str(getattr(cfg, "home_config_path", DEFAULT_HOME_CONFIG_PATH)),
        fold_on_interrupt=bool(getattr(cfg, "fold_home_on_interrupt", True)),
        live_keypress=bool(getattr(cfg, "live_home_keypress", True)),
        joint_step_deg=float(getattr(cfg, "home_joint_step_deg", 1.5)),
        inter_step_sleep_s=float(getattr(cfg, "home_inter_step_sleep_s", 0.04)),
    )


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    repo = Path(__file__).resolve().parents[3]
    candidate = repo / path
    if candidate.is_file():
        return candidate
    return p


def load_home_position(path: str = DEFAULT_HOME_CONFIG_PATH) -> HomePosition | None:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        logger.warning("[home] no config at %s", resolved)
        return None
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}
    names = list(data.get("motor_names") or ARM_MOTORS_DEFAULT)
    joints = np.asarray(data.get("joints_deg", []), dtype=np.float64)
    if joints.size != len(names):
        logger.warning("[home] invalid joints in %s (expected %d, got %d)", resolved, len(names), joints.size)
        return None
    gripper = float(data.get("gripper_pct", 100.0))
    return HomePosition(motor_names=names, joints_deg=joints, gripper_pct=gripper)


def capture_home_from_robot(robot, motor_names: list[str] | None = None) -> HomePosition:
    names = list(motor_names or ARM_MOTORS_DEFAULT)
    obs = robot.get_observation()
    joints = np.array([float(obs[f"{m}.pos"]) for m in names], dtype=np.float64)
    gripper = float(obs.get("gripper.pos", 100.0))
    return HomePosition(motor_names=names, joints_deg=joints, gripper_pct=gripper)


def save_home_position(home: HomePosition, path: str = DEFAULT_HOME_CONFIG_PATH) -> Path:
    resolved = _resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "motor_names": list(home.motor_names),
        "joints_deg": [round(float(v), 3) for v in home.joints_deg],
        "gripper_pct": round(float(home.gripper_pct), 3),
    }
    with open(resolved, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    return resolved


def fold_to_home(
    robot,
    home: HomePosition,
    *,
    joint_step_deg: float = 1.5,
    inter_step_sleep_s: float = 0.04,
) -> None:
    obs = robot.get_observation()
    current = np.array([float(obs[f"{m}.pos"]) for m in home.motor_names], dtype=np.float64)
    max_move = float(np.max(np.abs(home.joints_deg - current))) if current.size else 0.0
    if max_move < 0.5:
        logger.info("[home] already at home (max joint err %.2f°) — no motion needed", max_move)
        print(f"[home] already at saved pose (max joint error {max_move:.2f}°)")
        return
    logger.info(
        "[home] folding to saved pose (max move %.1f°, step=%.1f°)…",
        max_move,
        float(joint_step_deg),
    )
    send_joint_target_smoothly(
        robot,
        home.motor_names,
        current,
        home.joints_deg,
        step_deg=float(joint_step_deg),
        sleep_s=float(inter_step_sleep_s),
        gripper_open=home.gripper_pct >= 50.0,
        gripper_width_pct=float(home.gripper_pct),
    )
    logger.info("[home] fold complete.")


class HomeFoldController:
    """Poll 'h' (standalone or via :meth:`note_key`) and fold the arm to the saved pose."""

    def __init__(self, settings: HomeMotionSettings, motor_names: list[str] | None = None):
        self.settings = settings
        self.motor_names = list(motor_names or ARM_MOTORS_DEFAULT)
        self.home = load_home_position(settings.config_path)
        self.fold_requested = False
        self.folded = False
        self._stdin_keypress = False
        self._termios_old = None
        self._kb_buf = b""

    def log_status(self) -> None:
        if self.home is None:
            logger.warning(
                "[home] fold disabled — missing %s (run lerobot-save-home)",
                self.settings.config_path,
            )
        elif self.settings.live_keypress:
            if self._stdin_keypress:
                logger.info(
                    "[home] press 'h' to fold back slowly | Ctrl-C also folds when --fold-home-on-interrupt"
                )
            else:
                logger.warning(
                    "[home] keyboard disabled (stdin not a TTY) — use Ctrl-C to fold, "
                    "or run: lerobot-fold-home"
                )

    def setup_standalone_keypress(self) -> bool:
        if not self.settings.live_keypress:
            logger.info("[home] live keypress disabled (--live-home-keypress=false)")
            return False
        try:
            import termios as _termios
            import tty as _tty
        except ImportError:
            return False
        if not sys.stdin.isatty():
            return False
        fd = sys.stdin.fileno()
        try:
            self._termios_old = _termios.tcgetattr(fd)
            _tty.setcbreak(fd)
        except (_termios.error, OSError, AttributeError):
            return False
        self._stdin_keypress = True
        return True

    def teardown_standalone_keypress(self) -> None:
        if not self._stdin_keypress:
            return
        try:
            import termios as _termios
        except ImportError:
            return
        if self._termios_old is None:
            return
        try:
            _termios.tcsetattr(sys.stdin.fileno(), _termios.TCSADRAIN, self._termios_old)
        except (_termios.error, OSError):
            pass
        self._stdin_keypress = False

    def note_key(self, ch: int | str) -> None:
        if isinstance(ch, str):
            pressed = ch.lower() == "h"
        else:
            pressed = ch in (ord("h"), ord("H"))
        if pressed:
            self.fold_requested = True
            logger.info("[home] fold requested (h)")

    def poll_standalone(self) -> None:
        if not self._stdin_keypress:
            return
        fd = sys.stdin.fileno()
        try:
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if not r:
                    break
                more = os.read(fd, 256)
                if not more:
                    break
                self._kb_buf += more
        except (ValueError, OSError, BlockingIOError):
            return

        buf = bytearray(self._kb_buf)
        i = 0
        while i < len(buf):
            if buf[i] == 0x1B:
                if i + 2 < len(buf) and buf[i + 1] == ord("["):
                    i += 3
                    continue
                i += 1
                continue
            self.note_key(buf[i])
            i += 1
        self._kb_buf = b""

    def execute_fold(self, robot) -> None:
        if self.folded:
            return
        self.home = load_home_position(self.settings.config_path)
        if self.home is None:
            logger.warning("[home] fold skipped — no config at %s", self.settings.config_path)
            return
        self.folded = True
        self.fold_requested = False
        fold_to_home(
            robot,
            self.home,
            joint_step_deg=self.settings.joint_step_deg,
            inter_step_sleep_s=self.settings.inter_step_sleep_s,
        )

    def poll_and_fold(self, robot) -> bool:
        """Poll keys; fold if requested. Returns True when the main loop should stop."""
        if not self.settings.live_keypress:
            return False
        self.poll_standalone()
        if not self.fold_requested:
            return False
        self.execute_fold(robot)
        return True

    def on_exit(self, robot, *, interrupted: bool = False) -> None:
        self.teardown_standalone_keypress()
        if self.folded:
            return
        if interrupted and self.settings.fold_on_interrupt:
            logger.info("[home] folding to saved pose before exit…")
            self.execute_fold(robot)


# Shared config fields — copy into each engine's @dataclass.
HOME_CONFIG_FIELDS_DOC = """
    # Saved resting pose (SO101/so101_home.yaml). Press 'h' to fold back slowly.
    home_config_path: str = "SO101/so101_home.yaml"
    fold_home_on_interrupt: bool = True
    live_home_keypress: bool = True
    home_joint_step_deg: float = 1.5
    home_inter_step_sleep_s: float = 0.04
"""
