# Copyright 2024 The HuggingFace Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gripper open → final approach → current-sensed close for visual-servo engines."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.utils.motion_executor import MotionExecutionConfig, execute_cartesian_nudge_base

logger = logging.getLogger(__name__)


@dataclass
class GraspCloseConfig:
    enable: bool = True
    gripper_motor: str = "gripper"
    open_pct: float = 100.0
    close_pct: float = 0.0
    close_step_pct: float = 2.0
    close_step_wait_s: float = 0.08
    idle_sample_s: float = 0.5
    contact_delta_current_counts: float = 40.0
    contact_stall_steps: int = 3
    contact_min_pos_change_counts: float = 2.0
    hold_seconds: float = 0.6
    open_settle_s: float = 0.45
    # Small forward move along camera +Z after opening (bring cube between fingers).
    final_approach_m: float = 0.028
    final_approach_step_m: float = 0.007
    # Drive the final inch straight along the camera optical axis (+Z, i.e. where the
    # gripper is looking at the object). This is more robust than aiming at a radial
    # object point whose depth/floor estimate can push the move sideways or downward.
    final_approach_along_optical: bool = True
    # Contact must occur before gripper is this closed (empty grasp = near close_pct).
    min_contact_grip_pct: float = 6.0
    # After torque-sensed contact, keep closing this many extra % to firm up the grip
    # (clamped so we never drive past close_pct). Ensures a secure hold, not a light touch.
    post_contact_squeeze_pct: float = 8.0
    min_hold_delta_counts: float = 22.0
    lift_confirm: bool = True
    lift_height_m: float = 0.05
    lift_confirm_delta_counts: float = 18.0
    lift_resample_s: float = 0.35


@dataclass
class GraspOutcome:
    success: bool
    contact_detected: bool
    contact_reason: str
    mean_hold_delta_counts: float
    grip_pct_at_contact: float | None
    lift_confirmed: bool
    message: str


def read_present_current_counts(robot: Any, motor_name: str) -> float | None:
    if not hasattr(robot, "bus"):
        return None
    try:
        val = robot.bus.read("Present_Current", motor_name, normalize=False)
        return float(val) if val is not None else None
    except Exception as e:
        logger.debug("[grasp] Present_Current read failed: %s", e)
        return None


def sample_idle_current(
    robot: Any,
    motor_name: str,
    *,
    seconds: float,
    on_tick: Callable[[], None] | None = None,
) -> float:
    end_t = time.time() + max(0.0, float(seconds))
    samples: list[float] = []
    while time.time() < end_t:
        v = read_present_current_counts(robot, motor_name)
        if v is not None:
            samples.append(v)
        if on_tick is not None:
            on_tick()
        time.sleep(0.02)
    return float(np.mean(samples)) if samples else 0.0


def _gripper_write_pct(robot: Any, motor: str, pct: float) -> bool:
    try:
        robot.bus.write("Goal_Position", motor, float(pct), normalize=True)
        return True
    except Exception:
        try:
            act = {"gripper.pos": float(pct)}
            robot.send_action(act)
            return True
        except Exception as e:
            logger.warning("[grasp] gripper write %.1f%% failed: %s", pct, e)
            return False


def _grasp_forward_delta_base(
    T_base_cam: np.ndarray,
    approach_m: float,
    *,
    p_obj_base: np.ndarray | None = None,
) -> np.ndarray:
    """Base-frame translation toward the object along view ray (or radial to target).

    ``T_base_cam[:3, 2]`` is already camera +Z in base; do **not** multiply by R again.
    """
    T = np.asarray(T_base_cam, dtype=np.float64)
    eye = T[:3, 3]
    z_ax = np.asarray(T[:3, 2], dtype=np.float64)
    zn = float(np.linalg.norm(z_ax))
    if zn > 1e-9:
        z_ax = z_ax / zn
    else:
        z_ax = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    direction = z_ax
    if p_obj_base is not None:
        radial = np.asarray(p_obj_base, dtype=np.float64).reshape(3) - eye
        rn = float(np.linalg.norm(radial))
        if rn > 1e-3:
            direction = radial / rn
            if float(np.dot(direction, z_ax)) < 0.0:
                direction = -direction

    return direction * float(approach_m)


def _read_gripper_pct(robot: Any, motor: str) -> float | None:
    try:
        v = robot.bus.read("Present_Position", motor, normalize=True)
        return float(v) if v is not None else None
    except Exception:
        return None


def _firm_up_grip(robot: Any, motor: str, cfg: GraspCloseConfig, pct_at_contact: float) -> None:
    """Squeeze a few extra % past first contact so the object is held securely."""
    squeeze = float(getattr(cfg, "post_contact_squeeze_pct", 0.0))
    if squeeze <= 0.0:
        return
    target_pct = max(float(cfg.close_pct), float(pct_at_contact) - squeeze)
    if target_pct >= float(pct_at_contact):
        return
    _gripper_write_pct(robot, motor, target_pct)
    time.sleep(max(0.05, float(cfg.close_step_wait_s) * 2.0))
    logger.info(
        "[grasp] firm-up squeeze %.1f%% → %.1f%% (object held under torque)",
        float(pct_at_contact),
        target_pct,
    )


def close_gripper_with_current(
    robot: Any,
    cfg: GraspCloseConfig,
    *,
    i_idle: float,
    start_open_pct: float | None = None,
    on_tick: Callable[[], None] | None = None,
) -> tuple[bool, str, float, float | None]:
    motor = str(cfg.gripper_motor)
    stall_steps = 0
    contact_detected = False
    contact_reason = ""
    grip_at_contact: float | None = None
    pct = float(start_open_pct if start_open_pct is not None else cfg.open_pct)
    target = float(cfg.close_pct)
    step = float(cfg.close_step_pct)
    last_pos = None
    try:
        last_pos = robot.bus.read("Present_Position", motor, normalize=False)
    except Exception:
        pass

    while True:
        pct = max(target, pct - step)
        _gripper_write_pct(robot, motor, pct)
        if on_tick is not None:
            on_tick()
        time.sleep(max(0.0, float(cfg.close_step_wait_s)))

        cur = read_present_current_counts(robot, motor)
        try:
            pos = robot.bus.read("Present_Position", motor, normalize=False)
        except Exception:
            pos = None

        if cur is None or pos is None or last_pos is None:
            last_pos = pos
            if pct <= target:
                break
            continue

        delta_i = abs(float(cur) - float(i_idle))
        dpos = abs(float(pos) - float(last_pos))
        last_pos = pos

        if delta_i >= float(cfg.contact_delta_current_counts):
            contact_detected = True
            contact_reason = f"delta_current={delta_i:.1f}"
            grip_at_contact = _read_gripper_pct(robot, motor)
            _firm_up_grip(robot, motor, cfg, pct)
            break

        if dpos <= float(cfg.contact_min_pos_change_counts):
            stall_steps += 1
        else:
            stall_steps = 0

        if stall_steps >= int(cfg.contact_stall_steps):
            contact_detected = True
            contact_reason = f"stalled_{stall_steps}"
            grip_at_contact = _read_gripper_pct(robot, motor)
            _firm_up_grip(robot, motor, cfg, pct)
            break

        if pct <= target:
            break

    hold_end = time.time() + max(0.0, float(cfg.hold_seconds))
    hold_samples: list[float] = []
    while time.time() < hold_end:
        v = read_present_current_counts(robot, motor)
        if v is not None:
            hold_samples.append(v)
        if on_tick is not None:
            on_tick()
        time.sleep(0.02)
    mean_hold = (
        abs(float(np.mean(hold_samples)) - float(i_idle)) if hold_samples else 0.0
    )
    return contact_detected, contact_reason, mean_hold, grip_at_contact


def _lift_confirm(
    robot: Any,
    kin: Any,
    motor_names: list[str],
    cfg: GraspCloseConfig,
    *,
    i_idle: float,
    motion: MotionExecutionConfig | None,
    on_tick: Callable[[], None] | None = None,
) -> tuple[bool, float]:
    dz = float(cfg.lift_height_m)
    try:
        execute_cartesian_nudge_base(
            robot,
            kin,
            motor_names,
            np.array([0.0, 0.0, dz], dtype=np.float64),
            motion,
            on_tick=on_tick,
        )
    except Exception as e:
        logger.warning("[grasp] lift failed: %s", e)
        return False, 0.0

    end_t = time.time() + max(0.05, float(cfg.lift_resample_s))
    samples: list[float] = []
    while time.time() < end_t:
        v = read_present_current_counts(robot, str(cfg.gripper_motor))
        if v is not None:
            samples.append(v)
        if on_tick is not None:
            on_tick()
        time.sleep(0.02)
    if not samples:
        return False, 0.0
    delta = abs(float(np.mean(samples)) - float(i_idle))
    ok = delta >= float(cfg.lift_confirm_delta_counts)
    return ok, delta


def run_grasp_sequence(
    *,
    robot: Any,
    kin: Any,
    motor_names: list[str],
    T_base_cam: np.ndarray,
    cfg: GraspCloseConfig,
    object_size_m: float,
    motion: MotionExecutionConfig | None = None,
    p_obj_base: np.ndarray | None = None,
    on_tick: Callable[[], None] | None = None,
) -> GraspOutcome:
    """Open gripper, inch forward, close with current sensing, optional lift check.

    ``on_tick`` (optional) is called frequently during the (blocking) grasp so the
    caller can keep a live visualization / camera stream alive. It should be cheap and
    self-throttling.
    """
    if not bool(cfg.enable):
        return GraspOutcome(
            success=False,
            contact_detected=False,
            contact_reason="disabled",
            mean_hold_delta_counts=0.0,
            grip_pct_at_contact=None,
            lift_confirmed=False,
            message="grasp_enable=false",
        )
    if not hasattr(robot, "bus"):
        return GraspOutcome(
            success=False,
            contact_detected=False,
            contact_reason="no_bus",
            mean_hold_delta_counts=0.0,
            grip_pct_at_contact=None,
            lift_confirmed=False,
            message="robot has no motor bus for current sensing",
        )

    motor = str(cfg.gripper_motor)
    logger.info(
        "[grasp] sequence start (object_size=%.3fm): open → approach %.3fm → close",
        float(object_size_m),
        float(cfg.final_approach_m),
    )

    _gripper_write_pct(robot, motor, float(cfg.open_pct))
    _settle_end = time.time() + float(cfg.open_settle_s)
    while time.time() < _settle_end:
        if on_tick is not None:
            on_tick()
        time.sleep(0.02)

    i_idle = sample_idle_current(
        robot, motor, seconds=float(cfg.idle_sample_s), on_tick=on_tick
    )

    # Aim the final inch straight down the camera view by default; only fall back to
    # the radial-to-object vector if the caller explicitly disables optical approach.
    dir_obj = None if bool(cfg.final_approach_along_optical) else p_obj_base
    delta_base = _grasp_forward_delta_base(
        T_base_cam,
        float(cfg.final_approach_m),
        p_obj_base=dir_obj,
    )
    logger.info(
        "[grasp] final inch Δbase=(%.4f, %.4f, %.4f) m (|Δ|=%.4f) along=%s",
        float(delta_base[0]),
        float(delta_base[1]),
        float(delta_base[2]),
        float(np.linalg.norm(delta_base)),
        "optical+Z" if dir_obj is None else "radial→obj",
    )

    def _ee_pos() -> np.ndarray | None:
        try:
            obs = robot.get_observation()
            joints = np.array(
                [float(obs[f"{m}.pos"]) for m in motor_names], dtype=np.float64
            )
            return np.asarray(kin.forward_kinematics(joints)[:3, 3], dtype=np.float64)
        except Exception:
            return None

    ee_before = _ee_pos()
    n_steps = max(
        1, int(math.ceil(float(cfg.final_approach_m) / max(1e-4, float(cfg.final_approach_step_m))))
    )
    step_vec = delta_base / float(n_steps)
    for _ in range(n_steps):
        try:
            execute_cartesian_nudge_base(
                robot,
                kin,
                motor_names,
                step_vec,
                motion,
                on_tick=on_tick,
            )
        except Exception as e:
            logger.warning("[grasp] final approach step failed: %s", e)
            break
        if on_tick is not None:
            on_tick()
        time.sleep(0.04)
    ee_after = _ee_pos()
    if ee_before is not None and ee_after is not None:
        moved = ee_after - ee_before
        logger.info(
            "[grasp] EE moved Δ=(%.4f, %.4f, %.4f) m (|Δ|=%.4f, commanded %.4f) — "
            "if |Δ| << commanded the arm is hitting a limit/floor and cannot reach in",
            float(moved[0]),
            float(moved[1]),
            float(moved[2]),
            float(np.linalg.norm(moved)),
            float(np.linalg.norm(delta_base)),
        )

    start_pct = _read_gripper_pct(robot, motor)
    contact, reason, mean_hold, grip_pct = close_gripper_with_current(
        robot,
        cfg,
        i_idle=i_idle,
        start_open_pct=start_pct,
        on_tick=on_tick,
    )

    size_ok = True
    if contact and grip_pct is not None:
        # Cube inside fingers: stop above fully closed (not empty pinch).
        size_ok = float(grip_pct) >= float(cfg.min_contact_grip_pct)

    lift_ok = False
    lift_delta = 0.0
    if bool(cfg.lift_confirm) and contact and size_ok:
        lift_ok, lift_delta = _lift_confirm(
            robot, kin, motor_names, cfg, i_idle=i_idle, motion=motion, on_tick=on_tick
        )

    success = bool(contact and size_ok and mean_hold >= float(cfg.min_hold_delta_counts))
    if bool(cfg.lift_confirm):
        success = success and lift_ok

    msg_parts = [
        f"contact={contact} ({reason})",
        f"hold_I={mean_hold:.1f}",
        f"grip@contact={grip_pct:.1f}%" if grip_pct is not None else "grip@contact=?",
    ]
    if bool(cfg.lift_confirm):
        msg_parts.append(f"lift={lift_ok} (ΔI={lift_delta:.1f})")
    message = ", ".join(msg_parts)

    logger.info("[grasp] sequence done: success=%s %s", success, message)

    return GraspOutcome(
        success=success,
        contact_detected=contact,
        contact_reason=reason,
        mean_hold_delta_counts=mean_hold,
        grip_pct_at_contact=grip_pct,
        lift_confirmed=lift_ok,
        message=message,
    )