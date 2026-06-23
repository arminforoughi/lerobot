# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smart hybrid of YOLO approach + VLM fine-interaction + current-sensing grasp.

Pipeline (per attempt):

1. The existing ``yolo_track`` ``plan_top`` approach brings the gripper to a hover above the
   target with the wrist partially tilted toward the object (coarse phase).
2. Runner hands off to :func:`_hybrid_refine_grasp` via ``on_plan_top_descend_done``.
3. VLM fine-refinement loop (:class:`_GeminiRefiner`) iteratively suggests small camera-XY
   nudges and wrist-roll tweaks until it says ``close_now`` (subject to confidence / depth /
   pixel vetoes), or safety caps are hit.
4. Final Cartesian descent to the pre-close tip gap.
5. Gripper closes in small percent-steps while we watch ``Present_Current`` on the gripper
   motor (Feetech STS3215). Contact is declared on current spike or stall.
6. Lift-and-confirm: raise a few cm, re-sample current. If current stays elevated, the object
   is held. Otherwise we report an empty grasp.

The VLM is used ONLY for fine decisions (dx, dy, dz along the view ray, dyaw, close_now).
YOLO still drives the macro motion. Current sensing is the ground-truth grasp-success
signal. The observe → act micro-step rhythm matches the spirit of
``lerobot-agentic-manipulate`` (see ``lerobot_agentic_manipulate.py``), but here the
policy is JSON deltas instead of a full LLM plan.

When ``hybrid_rerun_log`` is enabled (PBVS passes ``display_data`` through
``_hybrid_cfg_from_pbvs``), each VLM iteration, ray-descent step, and the pre-close
``vlm_grasp`` frame are sent to Rerun alongside the end-effector trail.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lerobot.configs import parser
from lerobot.manipulation.yolo_track.config import YoloTrackApproachConfig
from lerobot.manipulation.yolo_track.depth import depth_from_bbox_size, median_depth_m
from lerobot.manipulation.yolo_track.math_utils import point_cam_to_base
from lerobot.manipulation.yolo_track.runner import (
    SO100_MOTOR_NAMES,
    PlanTopHandoffState,
    run_yolo_track_approach,
)
from lerobot.utils.motion_executor import execute_cartesian_nudge_base
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def _parse_partial_json_object(s: str) -> dict[str, Any] | None:
    """Recover numeric fields from truncated JSON (Gemini often cuts mid-object).

    Complements :func:`json.loads` when the stream ends before ``close_now`` / ``reason``.
    """
    s = (s or "").strip()
    if "{" not in s:
        return None
    out: dict[str, Any] = {}
    patterns: list[tuple[str, str]] = [
        ("dx_mm", r'"dx_mm"\s*:\s*([-+]?\d+(?:\.\d+)?)'),
        ("dy_mm", r'"dy_mm"\s*:\s*([-+]?\d+(?:\.\d+)?)'),
        ("dz_mm", r'"dz_mm"\s*:\s*([-+]?\d+(?:\.\d+)?)'),
        ("d_yaw_deg", r'"d_yaw_deg"\s*:\s*([-+]?\d+(?:\.\d+)?)'),
        ("d_wrist_flex_deg", r'"d_wrist_flex_deg"\s*:\s*([-+]?\d+(?:\.\d+)?)'),
    ]
    for key, pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r'"close_now"\s*:\s*(true|false)', s, flags=re.IGNORECASE)
    if m:
        out["close_now"] = m.group(1).lower() == "true"
    m = re.search(r'"reason"\s*:\s*"([^"]*)"', s, flags=re.IGNORECASE)
    if m:
        out["reason"] = m.group(1)
    if any(k in out for k in ("dx_mm", "dy_mm", "dz_mm", "d_yaw_deg", "d_wrist_flex_deg")):
        return out
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HybridGraspConfig(YoloTrackApproachConfig):
    """YOLO-approach config + VLM-refinement + current-sensing grasp params.

    Inherits every ``lerobot-yolo-track-approach`` flag (``--query``, ``--urdf``, camera
    mounting, etc.). The new hybrid phases are disabled with conservative defaults so the
    first run should be reasonable for a 3–5 cm cube grasp.
    """

    # Force plan_top; disable the runner's built-in center phase — we replace it with VLM.
    approach_style: str = "plan_top"
    plan_top_center_enable: bool = False
    aim_u_offset_px: float = 0.0
    """Horizontal pixel offset (px) applied to target centering during hybrid VLM/descent.

    This matches PBVS ``aim_u_offset_px`` when hybrid is launched from PBVS. Use it when the
    gripper camera is biased toward the left finger: we want the object slightly to the **right**
    of the image principal point so the fingers, not the camera centerline, align with the object.
    """

    # -- VLM backend (uses the OpenAI-compatible Gemini endpoint, same as perception.vlm_detector)
    vlm_backend: str = "gemini"  # "gemini" | "cloud" | "claude"
    # NOTE: this is passed to the Gemini OpenAI-compatible endpoint. Use a model name that exists there.
    vlm_model: str = "gemini-2.5-flash"
    vlm_api_key: str = ""
    vlm_base_url: str = ""
    # Timeout for each VLM call, seconds. Keep short; loop will just continue on failure.
    vlm_timeout_s: float = 8.0

    # -- VLM refinement loop (runs after plan_top descend, before gripper close)
    vlm_max_refine_iters: int = 12
    # Successful VLM parses required before honoring ``close_now`` (stops one-shot "centered").
    vlm_min_refine_iters_before_close: int = 3
    # If pixel error exceeds this, never allow ``close_now`` (server-side), when enabled.
    vlm_require_pixel_motion: bool = True
    # If model says close but bbox center is farther than this from the principal point, veto.
    vlm_geom_close_tol_px: float = 22.0
    vlm_geom_fallback_enable: bool = True
    # Re-run YOLO on each refine iter when ``state.yolo_detector`` is set (PBVS handoff).
    vlm_redetect_each_iter: bool = True
    vlm_max_output_tokens: int = 1024
    vlm_open_gripper_pick_geometry_prompt: bool = True
    vlm_forbid_negative_dz_iters: int = 16
    vlm_clamp_negative_dz_mm: bool = True
    # Scale XY/ray per-step caps from tracking error (bigger moves when far, finer when near).
    vlm_adaptive_step_enable: bool = True
    vlm_adaptive_step_px_ref: float = 35.0
    """When max(|du|,|dv|) reaches this many pixels, XY/ray caps hit ``vlm_adaptive_step_scale_max`` (blend below)."""
    vlm_adaptive_depth_err_ref_m: float = 0.035
    """When |depth − goal| reaches this many metres, boost caps similarly (along with pixel term)."""
    vlm_adaptive_step_scale_max: float = 2.0
    """Upper bound on the multiplier applied to ``vlm_nudge_step_m`` and ``vlm_ray_step_max_m``."""
    # Max per-step base-XY Cartesian nudge (m). Safety clamp applied to any VLM suggestion.
    vlm_nudge_step_m: float = 0.022
    # Total base-XY travel budget across all refine steps (m). Stops runaway drift.
    vlm_nudge_total_budget_m: float = 0.14
    # Max per-step wrist-roll delta (deg) and total budget (deg).
    vlm_wrist_roll_step_deg: float = 15.0
    vlm_wrist_roll_total_deg: float = 45.0
    # Along camera optical axis in base frame (same sense as PBVS approach); per-step / budget (m).
    vlm_ray_step_max_m: float = 0.022
    vlm_ray_total_budget_m: float = 0.22
    # After XY+ray+wrist moves, brief pause before fresh obs + Rerun (keep small; VLM is the slow part).
    vlm_post_motion_settle_s: float = 0.05
    vlm_wrist_flex_step_deg: float = 8.0
    vlm_wrist_flex_total_deg: float = 24.0
    # If true, annotate the image (bbox + target crosshair) before sending to the VLM.
    vlm_annotate_image: bool = True
    # If True, draw the red crosshair at the bbox center (matches PBVS du/dv), not the principal point.
    vlm_annotate_crosshair_at_bbox: bool = True
    vlm_annotate_bbox_left_edge: bool = True
    """If True, draw a vertical line on the bbox **left edge** (pick cue for the fixed finger)."""
    # Fallback auto-close if VLM doesn't answer or budget is exhausted.
    vlm_auto_close_on_exhaustion: bool = True
    # If True, abort the grasp if every VLM iteration fails (prevents one-shot descent/close).
    vlm_abort_if_all_iters_fail: bool = True
    # If the image-center pixel error is already below this (and VLM agrees), close immediately.
    vlm_center_autoclose_tol_px: float = 14.0
    # Do not honor VLM ``close_now`` when YOLO confidence is below this (reduces closes on junk boxes).
    vlm_min_detection_conf_for_close: float = 0.48
    # Do not honor ``close_now`` when measured depth at the bbox is beyond this (m) — background
    # false positives often read 0.5–0.8 m while the real cube is < ~0.35 m in the refine phase.
    vlm_max_depth_at_close_m: float = 0.36

    # -- Visual auto-close (non-VLM): close when the target looks "in the gripper"
    # This is a pragmatic guardrail for near-field failure modes: the VLM can be conservative and
    # never emit close_now even when the object is already between the fingers.
    hybrid_visual_autoclose_enable: bool = True
    hybrid_visual_autoclose_min_iters: int = 2
    """Require at least this many VLM iterations before visual auto-close can trigger."""
    hybrid_visual_autoclose_max_px_err: float = 26.0
    """Max max(|du|,|dv|) allowed to auto-close (pixels)."""
    hybrid_visual_autoclose_max_depth_m: float = 0.14
    """If estimated depth is <= this, consider it 'near/in gripper'."""
    hybrid_visual_autoclose_min_bbox_area_frac: float = 0.020
    """Min bbox area fraction (bbox_area / image_area) to consider the object close enough."""
    hybrid_visual_autoclose_min_conf: float = 0.45
    """Min detection confidence for visual auto-close."""

    # -- Close-loop ray descent (after VLM refine; replaces single open-loop Z drop when enabled)
    hybrid_iterative_descent_enable: bool = True
    hybrid_descent_step_m: float = 0.007
    hybrid_descent_max_iters: int = 18
    hybrid_descent_total_budget_m: float = 0.06
    """Absolute cap on total descent translation (m). Prevents runaway push on bad depth/detections."""
    hybrid_descent_depth_tol_m: float = 0.006
    hybrid_descent_kp: float = 0.55
    hybrid_descent_pause_s: float = 0.03
    # Depth sampling during hybrid phase (align with PBVS when set from ``_hybrid_cfg_from_pbvs``).
    hybrid_pbvs_depth_policy: str = "bbox_preferred"
    hybrid_min_valid_depth_m: float = 0.08
    hybrid_max_valid_depth_m: float = 0.85

    # -- Final descent to pre-grasp height (tip gap above cube top)
    # Before closing, descend so that the gripper tip is this far above the cube top.
    hybrid_final_tip_gap_m: float = 0.012

    # -- Gripper close / contact detection (mirrors lerobot-gripper-current-test knobs)
    grasp_gripper_motor: str = "gripper"
    grasp_open_pct: float = 100.0
    grasp_close_pct: float = 0.0
    grasp_close_step_pct: float = 2.0
    grasp_close_step_wait_s: float = 0.08
    grasp_idle_sample_seconds: float = 0.6
    # Contact thresholds on |Present_Current - idle|; raw Feetech counts.
    grasp_contact_delta_current_counts: float = 40.0
    grasp_contact_stall_steps: int = 3
    grasp_contact_min_pos_change_counts: float = 2.0
    grasp_hold_seconds: float = 0.8
    grasp_preopen_enable: bool = False
    """If True, open to `grasp_open_pct` right before closing. Default False to avoid ejecting a near-captured object."""

    # -- Lift-and-confirm
    grasp_lift_height_m: float = 0.06
    # After lift, re-sample current and check |I - idle| stays above this (object still held).
    grasp_lift_confirm_delta_counts: float = 20.0
    grasp_lift_resample_seconds: float = 0.4

    # -- JSON log
    hybrid_log_jsonl: str = ""
    # Rerun (same session as PBVS / yolo-track when enabled from handoff config).
    hybrid_rerun_log: bool = False
    hybrid_rerun_log_sim3d: bool = False
    hybrid_rerun_object_half_size_m: float = 0.015
    hybrid_rerun_frame_base: int = 1_000_000

    # -- Near-field depth stabilization knobs (used by `_hybrid_stabilize_depth`)
    hybrid_depth_jump_abs_m: float = 0.12
    hybrid_depth_stereo_bbox_disagree_m: float = 0.10
    hybrid_depth_near_field_m: float = 0.28
    hybrid_depth_ema_alpha: float = 0.30


# ---------------------------------------------------------------------------
# Gemini refinement client
# ---------------------------------------------------------------------------


_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_CLAUDE_BASE_URL = "https://api.anthropic.com/v1/"


@dataclass
class _RefineSuggestion:
    dx_mm: float  # +right in image
    dy_mm: float  # +down in image
    dz_mm: float  # +forward along camera optical axis (camera frame +Z)
    d_yaw_deg: float  # +cw looking down from above (wrist_roll)
    d_wrist_flex_deg: float  # wrist_flex delta (deg)
    close_now: bool
    reason: str


class _GeminiRefiner:
    """Single-call VLM refinement: image in, JSON action out.

    Keeps the interaction intentionally cheap: one multi-modal chat completion per loop step.
    """

    def __init__(self, cfg: HybridGraspConfig):
        backend = (cfg.vlm_backend or "gemini").lower()
        if backend not in ("gemini", "cloud"):
            # Claude support would require a different client path; fall back to Gemini and warn.
            logger.warning(
                "[hybrid-grasp] VLM backend %r not supported for refinement; using 'gemini'.",
                backend,
            )
            backend = "gemini"
        self.backend = backend
        self.model = cfg.vlm_model or ("gemini-2.5-flash" if backend == "gemini" else "gpt-4o")
        self.timeout_s = float(cfg.vlm_timeout_s)

        if backend == "gemini":
            self.api_key = cfg.vlm_api_key or os.environ.get("GEMINI_API_KEY", "")
            self.base_url = (cfg.vlm_base_url or _GEMINI_BASE_URL).strip()
            env_hint = "GEMINI_API_KEY"
        else:
            self.api_key = cfg.vlm_api_key or os.environ.get("OPENAI_API_KEY", "")
            self.base_url = (cfg.vlm_base_url or "").strip() or None
            env_hint = "OPENAI_API_KEY"

        if not self.api_key:
            raise ValueError(
                f"VLM refiner requires {env_hint} (or --vlm-api-key). "
                "Either export the env var or pass --vlm-api-key=..."
            )
        self.max_output_tokens = int(getattr(cfg, "vlm_max_output_tokens", 768))
        self._pick_geom = bool(getattr(cfg, "vlm_open_gripper_pick_geometry_prompt", True))

    @staticmethod
    def _encode_jpeg_b64(rgb: np.ndarray) -> str:
        from PIL import Image

        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def suggest(
        self,
        rgb: np.ndarray,
        *,
        task_hint: str,
        px_per_mm_at_object: float,
        refine_context: dict[str, Any] | None = None,
    ) -> _RefineSuggestion | None:
        """Ask the VLM for a tiny adjustment. Returns None on failure.

        Args:
            rgb: annotated RGB image (HxWx3 uint8). The model sees whatever we draw.
            task_hint: short object label ("red cube", "cup", etc.).
            px_per_mm_at_object: image pixels per mm at the object plane (approx), only used
                to echo scale in the prompt so the VLM's dx/dy mm are grounded.
            refine_context: optional telemetry (iter, depth, pixel error, last action) for agentic steps.
        """
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover - optional dep
            logger.warning("[hybrid-grasp] openai client import failed: %s", e)
            return None

        t_vlm0 = time.perf_counter()
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        b64 = self._encode_jpeg_b64(rgb)
        h, w = rgb.shape[:2]
        ctx = refine_context or {}
        ctx_lines: list[str] = []
        if ctx:
            for k in (
                "agent_step",
                "depth_ema_m",
                "du_px",
                "dv_px",
                "min_steps_before_close",
                "last_action",
            ):
                if k in ctx and ctx[k] is not None:
                    ctx_lines.append(f"{k}={ctx[k]}")
        ctx_block = (" State: " + " ".join(ctx_lines)) if ctx_lines else ""

        # Gemini's OpenAI-compatible endpoint sometimes ignores response_format=json_object.
        system_prompt = (
            "You are a close-range robot gripper controller (eye-in-hand camera).\n"
            "Return ONLY valid JSON. No markdown. No code fences. No preface.\n"
            "Output must be a single JSON object starting with '{' and ending with '}'.\n"
            "Required keys: dx_mm, dy_mm, dz_mm, d_yaw_deg, d_wrist_flex_deg, close_now, reason.\n"
            "dx_mm/dy_mm: small translation in the image plane (+x right, +y down) mapped to base XY.\n"
            "dz_mm: move along the camera optical axis (+ = toward what the camera sees, forward in OpenCV cam +Z).\n"
            "d_yaw_deg: wrist_roll. d_wrist_flex_deg: wrist pitch.\n"
            "Per-step bounds: |dx_mm|,|dy_mm|,|dz_mm|<=18, |d_yaw_deg|<=15, |d_wrist_flex_deg|<=8.\n"
            "Priority for PICK: (1) lateral alignment, (2) then advance. Do NOT alternate large +/− dx_mm/dy_mm "
            "each step—use small consistent corrections.\n"
            "close_now: true ONLY when **both** fingertips visibly bracket the object at **near** range "
            "(use bbox depth / image: if the target still looks far away or the box is ambiguous, "
            "close_now=false). Never close on a box that sits on empty table while the cube is elsewhere.\n"
            "Example (approach along +Z toward the object):\n"
            '{"dx_mm":0,"dy_mm":0,"dz_mm":5,"d_yaw_deg":0,"d_wrist_flex_deg":0,"close_now":false,"reason":"advance"}'
        )
        if self._pick_geom:
            system_prompt += (
                "\n\nOPEN-GRIPPER PICK GEOMETRY:\n"
                "- The gripper is commanded OPEN (~100%). Often only the **fixed (left) finger** is clearly visible; "
                "the moving jaw may be off-screen.\n"
                "- **First objective:** align that **visible left finger** with the **left side** of the target "
                "(use the green bbox **left edge** in the image as the vertical reference). Use small dx_mm/dy_mm "
                "so the finger pad sits beside that edge (not centered on the whole bbox unless the finger is "
                "already on the left).\n"
                "- **Second objective:** use **positive dz_mm** (toward the scene along +Z_cam) so the **finger "
                "tips move forward into / past the front face** of the object for a power grasp. Avoid negative "
                "dz_mm (backing away along the optical axis) unless you clearly see an imminent collision.\n"
                "- Keep `reason` short and name what you aligned (e.g. \"left finger to bbox left edge\").\n"
            )
        user_text = (
            f"Image {w}x{h} px. Scale≈{px_per_mm_at_object:.2f} px/mm at object plane. "
            f"Task: {task_hint!r}.{ctx_block}"
        )

        def _call(messages: list[dict[str, Any]]) -> Any | None:
            try:
                return client.chat.completions.create(
                    model=self.model,
                    timeout=self.timeout_s,
                    messages=messages,
                    max_tokens=int(self.max_output_tokens),
                    temperature=0.0,
                )
            except Exception as e:
                logger.warning("[hybrid-grasp] VLM call failed: %s", e)
                return None

        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ]

        response = _call(base_messages)
        if response is None:
            return None

        msg = response.choices[0].message
        # Some OpenAI-compatible backends (incl. Gemini) may return structured JSON via tool/function calls.
        content = getattr(msg, "content", None)
        text = ""
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            # Some clients/backends represent message content as a list of parts.
            parts: list[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    # OpenAI-style: {"type":"text","text":"..."}
                    t = p.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            text = "\n".join([t for t in parts if t]).strip()
        else:
            text = str(content).strip() if content is not None else ""

        if os.getenv("LEROBOT_VLM_DEBUG", "").strip():
            try:
                tc = getattr(msg, "tool_calls", None)
                logger.warning(
                    "[hybrid-grasp][vlm-debug] message fields: content_type=%s content_str_len=%s tool_calls=%s",
                    type(content).__name__,
                    None if content is None else len(str(content)),
                    "none" if not tc else f"{len(tc)}",
                )
                logger.warning(
                    "[hybrid-grasp][vlm-debug] content (first 600 chars): %s",
                    (str(content)[:600] if content is not None else "<None>"),
                )
                if tc:
                    try:
                        fn = getattr(tc[0], "function", None)
                        args = getattr(fn, "arguments", None) if fn is not None else None
                        logger.warning(
                            "[hybrid-grasp][vlm-debug] tool_call[0].function.arguments (first 600 chars): %s",
                            (str(args)[:600] if args is not None else "<None>"),
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        if not text:
            try:
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    fn = getattr(tool_calls[0], "function", None)
                    args = getattr(fn, "arguments", None) if fn is not None else None
                    if isinstance(args, str) and args.strip():
                        text = args.strip()
            except Exception:
                pass
        def _extract_json_object(s: str) -> str | None:
            """Extract the first JSON object from a possibly chatty response.

            Handles common Gemini/OpenAI-style wrappers like:
            - "Here is the JSON requested:\n{...}"
            - fenced blocks: ```json\n{...}\n```
            """
            s = (s or "").strip()
            if not s:
                return None
            # Prefer a fenced JSON block if present anywhere.
            m_fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
            if m_fence:
                return m_fence.group(1).strip()
            # Otherwise, find the first balanced {...} region (brace counting, quote-aware).
            start = s.find("{")
            if start < 0:
                return None
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1].strip()
            return None

        def _parse_kv_fallback(s: str) -> dict | None:
            """Fallback when the model returns 'dx_mm: 2, ...' without braces."""
            s = (s or "").strip()
            if not s:
                return None
            # Try to extract each field independently.
            out: dict[str, object] = {}
            m = re.search(r"\bdx_mm\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
            if m:
                out["dx_mm"] = float(m.group(1))
            m = re.search(r"\bdy_mm\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
            if m:
                out["dy_mm"] = float(m.group(1))
            m = re.search(r"\bdz_mm\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
            if m:
                out["dz_mm"] = float(m.group(1))
            m = re.search(r"\bd[_\s-]?yaw[_\s-]?deg\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
            if m:
                out["d_yaw_deg"] = float(m.group(1))
            m = re.search(
                r"\bd[_\s-]?wrist[_\s-]?flex[_\s-]?deg\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
                s,
                flags=re.IGNORECASE,
            )
            if m:
                out["d_wrist_flex_deg"] = float(m.group(1))
            m = re.search(r"\bclose_now\b\s*[:=]\s*(true|false|1|0)", s, flags=re.IGNORECASE)
            if m:
                out["close_now"] = m.group(1).lower() in ("true", "1")
            m = re.search(r"\breason\b\s*[:=]\s*\"([^\"]+)\"", s, flags=re.IGNORECASE)
            if m:
                out["reason"] = m.group(1)
            return out if out else None

        def _loads_flexible(s: str) -> dict | None:
            raw = _extract_json_object(s)
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        o, _e = json.JSONDecoder().raw_decode(raw)
                        return o if isinstance(o, dict) else None
                    except json.JSONDecodeError:
                        pass
            st = (s or "").find("{")
            if st < 0:
                return None
            try:
                o, _end = json.JSONDecoder().raw_decode(s[st:])
                return o if isinstance(o, dict) else None
            except json.JSONDecodeError:
                return None

        obj = _loads_flexible(text)
        if obj is None:
            obj = _parse_kv_fallback(text)
        if obj is None:
            obj = _parse_partial_json_object(text)
        if obj is None:
            # One immediate, explicit retry. This fixes the common Gemini failure mode:
            # replying with "Here is the JSON ..." but not actually including it.
            retry_messages = list(base_messages)
            retry_messages.append(
                {
                    "role": "assistant",
                    "content": (text or "")[:400],
                }
            )
            retry_messages.append(
                {
                    "role": "user",
                    "content": "You did not output JSON. Output ONLY the JSON object now, no other text.",
                }
            )
            response2 = _call(retry_messages)
            if response2 is None:
                return None
            msg2 = response2.choices[0].message
            text2 = getattr(msg2, "content", "") or ""
            if not isinstance(text2, str):
                text2 = str(text2)
            text2 = text2.strip()
            raw_json2 = _extract_json_object(text2)
            obj2: dict | None = None
            if raw_json2:
                try:
                    obj2 = json.loads(raw_json2)
                except Exception:
                    try:
                        o2, _ = json.JSONDecoder().raw_decode(raw_json2)
                        obj2 = o2 if isinstance(o2, dict) else None
                    except Exception:
                        obj2 = None
            if obj2 is None:
                obj2 = _loads_flexible(text2)
            if obj2 is None:
                obj2 = _parse_kv_fallback(text2)
            if obj2 is None:
                obj2 = _parse_partial_json_object(text2)
            if obj2 is None:
                logger.warning(
                    "[hybrid-grasp] VLM returned unparseable text (len=%d): %s",
                    len(text),
                    text[:220],
                )
                return None
            obj = obj2

        dt_vlm = time.perf_counter() - t_vlm0
        try:
            out = _RefineSuggestion(
                dx_mm=float(obj.get("dx_mm", 0) or 0),
                dy_mm=float(obj.get("dy_mm", 0) or 0),
                dz_mm=float(obj.get("dz_mm", 0) or 0),
                d_yaw_deg=float(obj.get("d_yaw_deg", 0) or 0),
                d_wrist_flex_deg=float(obj.get("d_wrist_flex_deg", 0) or 0),
                close_now=bool(obj.get("close_now", False)),
                reason=str(obj.get("reason", ""))[:80],
            )
            logger.info(
                "[hybrid-grasp] VLM round-trip %.2fs (model=%s)",
                dt_vlm,
                self.model,
            )
            return out
        except Exception as e:
            logger.warning("[hybrid-grasp] VLM response coercion failed: %s | %s", e, obj)
            return None


# ---------------------------------------------------------------------------
# Image annotation for the VLM prompt
# ---------------------------------------------------------------------------


def _annotate_for_vlm(
    rgb: np.ndarray,
    *,
    bbox_xyxy: tuple[float, float, float, float] | None,
    image_center: tuple[float, float],
    label: str = "",
    show_bbox_left_edge: bool = False,
) -> np.ndarray:
    """Return a BGR-or-RGB copy with the object bbox + gripper-tip crosshair drawn on it."""
    img = np.asarray(rgb).copy()
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if show_bbox_left_edge:
            cv2.line(img, (x1, y1), (x1, y2), (255, 0, 255), 2)
        if label:
            cv2.putText(
                img, label, (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
    cu, cv_ = int(round(image_center[0])), int(round(image_center[1]))
    cv2.drawMarker(img, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(img, (cu, cv_), 14, (0, 0, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Motion helpers (base-XY nudge from camera-plane dx/dy + wrist-roll tweak)
# ---------------------------------------------------------------------------


def _image_mm_to_base_xy(
    T_base_ee: np.ndarray,
    T_ee_cam: np.ndarray,
    dx_mm: float,
    dy_mm: float,
) -> np.ndarray:
    """Map a small (dx, dy) offset in the camera image plane (mm) to a base-frame XY delta.

    Camera frame convention (OpenCV): +x right in image, +y down, +z forward. With the wrist
    pointing down, +z_cam ≈ -z_base. We only need the XY projection, so we apply T_base_cam
    to the [dx, dy, 0] vector and keep the XY components.
    """
    T_bc = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(T_ee_cam, dtype=np.float64)
    R_bc = T_bc[:3, :3]
    v_cam = np.array([dx_mm * 1e-3, dy_mm * 1e-3, 0.0], dtype=np.float64)
    v_base = R_bc @ v_cam
    return np.array([v_base[0], v_base[1], 0.0], dtype=np.float64)


def _apply_wrist_roll_delta(robot: Any, d_deg: float) -> None:
    """Add ``d_deg`` degrees to the current wrist_roll position (direct bus write).

    Small moves only — no trajectory streaming. Relies on the motor's internal trapezoid to
    reach the goal smoothly.
    """
    try:
        cur = robot.bus.read("Present_Position", "wrist_roll", normalize=True)
        target = float(cur) + float(d_deg)
        robot.bus.write("Goal_Position", "wrist_roll", target, normalize=True)
    except Exception as e:
        logger.warning("[hybrid-grasp] wrist-roll nudge failed: %s", e)


def _apply_wrist_flex_delta(robot: Any, d_deg: float) -> None:
    """Add ``d_deg`` to wrist_flex (same pattern as wrist_roll)."""
    try:
        cur = robot.bus.read("Present_Position", "wrist_flex", normalize=True)
        target = float(cur) + float(d_deg)
        robot.bus.write("Goal_Position", "wrist_flex", target, normalize=True)
    except Exception as e:
        logger.warning("[hybrid-grasp] wrist-flex nudge failed: %s", e)


def _ray_mm_to_base_delta(
    T_base_ee: np.ndarray,
    T_ee_cam: np.ndarray,
    dz_mm: float,
) -> np.ndarray:
    """Map dz along camera +Z (OpenCV optical axis) to a base-frame translation (3-vector)."""
    T_bc = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(T_ee_cam, dtype=np.float64)
    R_bc = T_bc[:3, :3]
    v_cam = np.array([0.0, 0.0, float(dz_mm) * 1e-3], dtype=np.float64)
    return (R_bc @ v_cam).astype(np.float64)


def _hybrid_select_depth(
    policy: str,
    depth_map: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    *,
    fx: float,
    fy: float,
    depth_scale: float,
    object_size_m: float,
    dmin: float,
    dmax: float,
) -> tuple[float | None, float | None, float | None]:
    """Same policy semantics as PBVS ``_select_depth`` (bbox_preferred, etc.)."""
    d_stereo: float | None = None
    if depth_map is not None:
        stereo_min_mm = float(max(12.0, (dmin * 1000.0) * 0.5))
        d_raw = median_depth_m(
            np.asarray(depth_map),
            bbox,
            depth_scale=depth_scale,
            min_mm=stereo_min_mm,
        )
        if d_raw is not None and dmin < float(d_raw) < dmax:
            d_stereo = float(d_raw)

    d_bbox: float | None = None
    if object_size_m > 1e-4:
        d_bb = depth_from_bbox_size(tuple(bbox), fx=fx, fy=fy, target_physical_size_m=object_size_m)
        if d_bb is not None and dmin < float(d_bb) < dmax:
            d_bbox = float(d_bb)

    p = (policy or "bbox_preferred").lower().strip()
    if p == "stereo_only":
        return d_stereo, d_stereo, d_bbox
    if p == "bbox_only":
        return d_bbox, d_stereo, d_bbox
    if p == "stereo_preferred":
        return (d_stereo if d_stereo is not None else d_bbox), d_stereo, d_bbox
    return (d_bbox if d_bbox is not None else d_stereo), d_stereo, d_bbox


def _hybrid_stabilize_depth(
    state: PlanTopHandoffState,
    cfg: HybridGraspConfig,
    *,
    d_used: float | None,
    d_stereo: float | None,
    d_bbox: float | None,
    dmin: float,
    dmax: float,
) -> float | None:
    """Stabilize near-field depth to prevent background jumps.

    OAK-D stereo frequently fails when the object is very close / partially occluded by the gripper,
    causing ROI median depth to jump to the table/background (e.g. 0.67 m) and then back. That can
    trigger runaway "advance" steps that push the object.

    Heuristic:
    - Keep an EMA + last-valid depth in ``state``.
    - When stereo and bbox disagree strongly in the near field, prefer bbox.
    - Rate-limit per-tick depth jumps; if an abrupt jump happens, hold last-valid (or bbox).
    """
    # Configurable knobs (kept conservative; can be exposed later if needed).
    jump_abs_m = float(getattr(cfg, "hybrid_depth_jump_abs_m", 0.12) or 0.12)
    stereo_bbox_disagree_m = float(getattr(cfg, "hybrid_depth_stereo_bbox_disagree_m", 0.10) or 0.10)
    near_field_m = float(getattr(cfg, "hybrid_depth_near_field_m", 0.28) or 0.28)
    ema_alpha = float(getattr(cfg, "hybrid_depth_ema_alpha", 0.30) or 0.30)

    last_valid = getattr(state, "_hybrid_last_valid_depth_m", None)
    try:
        last_valid_f = float(last_valid) if last_valid is not None else None
    except Exception:
        last_valid_f = None

    d = float(d_used) if d_used is not None else None
    ds = float(d_stereo) if d_stereo is not None else None
    db = float(d_bbox) if d_bbox is not None else None

    # Prefer bbox in near field when stereo likely latched onto background.
    if db is not None and ds is not None:
        if db < near_field_m and (ds - db) > stereo_bbox_disagree_m:
            d = db

    # If current choice is invalid, fall back to bbox or last-valid.
    if d is None or not (dmin < d < dmax):
        if db is not None and (dmin < db < dmax):
            d = db
        elif last_valid_f is not None and (dmin < last_valid_f < dmax):
            d = last_valid_f
        else:
            return None

    # Rate-limit abrupt jumps (most common failure: jump from near to far).
    if last_valid_f is not None:
        if abs(float(d) - float(last_valid_f)) > jump_abs_m:
            # Prefer bbox if it agrees with last-valid; otherwise just hold last-valid.
            if db is not None and abs(float(db) - float(last_valid_f)) <= stereo_bbox_disagree_m:
                d = float(db)
            else:
                d = float(last_valid_f)

    # Update last-valid and EMA on state.
    state._hybrid_last_valid_depth_m = float(d)
    prev_ema = getattr(state, "depth_ema_m", None)
    try:
        prev_ema_f = float(prev_ema) if prev_ema is not None else None
    except Exception:
        prev_ema_f = None
    if prev_ema_f is None:
        state.depth_ema_m = float(d)
    else:
        state.depth_ema_m = float((1.0 - ema_alpha) * prev_ema_f + ema_alpha * float(d))
    return float(d)


def _sync_T_base_ee_from_robot(state: PlanTopHandoffState) -> None:
    """Refresh ``state.T_base_ee`` from live joint encoders (gripper mount)."""
    if state.robot is None or state.cfg.dry_run:
        return
    try:
        obs = state.robot.get_observation()
        joints = np.array(
            [float(obs[f"{m}.pos"]) for m in SO100_MOTOR_NAMES],
            dtype=np.float64,
        )
        T = state.kinematics.forward_kinematics(joints)
        state.T_base_ee[:, :] = np.asarray(T, dtype=np.float64)
    except Exception as e:
        logger.debug("[hybrid-grasp] FK sync failed: %s", e)


def _hybrid_try_rerun_log(
    state: PlanTopHandoffState,
    cfg: HybridGraspConfig,
    *,
    frame_seq: int,
    phase: str,
    ee_trail: list[np.ndarray],
    bbox: tuple[float, float, float, float] | None,
    bbox_center: tuple[float, float] | None,
    depth_m: float | None,
    label: str,
    robot_obs: dict[str, Any] | None = None,
) -> None:
    """Log one hybrid-grasp frame to Rerun (same blueprint as PBVS when viewer is open).

    If ``robot_obs`` is provided, reuse it instead of calling ``get_observation()`` again
    (cuts latency when logging right after a read used for depth / RGB).
    """
    if not bool(getattr(cfg, "hybrid_rerun_log", False)) and not bool(
        getattr(cfg, "hybrid_rerun_log_sim3d", False)
    ):
        return
    try:
        from lerobot.manipulation.yolo_track.rerun_viz import log_rerun_iter
    except Exception:
        return
    if state.robot is None or cfg.dry_run:
        return
    try:
        obs = robot_obs if robot_obs is not None else state.robot.get_observation()
        rgb = obs.get(cfg.camera_key)
        if rgb is None:
            return
        depth = obs.get(f"{cfg.camera_key}_depth")
        joints = np.array(
            [float(obs[f"{m}.pos"]) for m in SO100_MOTOR_NAMES],
            dtype=np.float64,
        )
        T_base_ee = state.kinematics.forward_kinematics(joints)
        T_base_cam = np.asarray(T_base_ee, dtype=np.float64) @ np.asarray(
            state.T_ee_cam, dtype=np.float64,
        )
        ee_now = np.asarray(T_base_ee, dtype=np.float64)[:3, 3].copy()
        if not ee_trail or float(np.linalg.norm(ee_now - ee_trail[-1])) > 1e-4:
            ee_trail.append(ee_now)
            if len(ee_trail) > 500:
                ee_trail.pop(0)
        p_tgt = None
        if depth_m is not None and bbox_center is not None:
            cx_ref = float(state.cx0) + float(getattr(cfg, "aim_u_offset_px", 0.0) or 0.0)
            p_tgt = point_cam_to_base(
                T_base_cam,
                u=float(bbox_center[0]),
                v_pix=float(bbox_center[1]),
                depth_m=float(depth_m),
                fx=float(state.fx),
                fy=float(state.fy),
                cx0=cx_ref,
                cy0=float(state.cy0),
            )
        conf = None
        det = getattr(state, "det", None)
        if bbox is not None and det is not None:
            conf = float(getattr(det, "confidence", 0.0) or 0.0)
        br = bbox_center[0] if bbox_center else None
        bv = bbox_center[1] if bbox_center else None
        log_rerun_iter(
            frame=int(frame_seq),
            camera_key=str(cfg.camera_key),
            rgb=np.asarray(rgb),
            depth=depth,
            bbox_xyxy=bbox,
            bbox_center_raw=(float(br), float(bv)) if br is not None else None,
            bbox_center_smoothed=bbox_center,
            cx0=float(state.cx0) + float(getattr(cfg, "aim_u_offset_px", 0.0) or 0.0),
            cy0=float(state.cy0),
            conf=conf,
            phase=phase,
            z_ee=float(ee_now[2]),
            depth_m=depth_m,
            kinematics=state.kinematics,
            joints_deg=joints,
            T_base_ee=np.asarray(T_base_ee, dtype=np.float64),
            T_base_cam=T_base_cam,
            p_target_base=p_tgt,
            ee_trail=ee_trail,
            object_half_size_m=float(getattr(cfg, "hybrid_rerun_object_half_size_m", 0.015)),
            show_sim3d=bool(getattr(cfg, "hybrid_rerun_log_sim3d", False)),
            show_camera=bool(getattr(cfg, "hybrid_rerun_log", False)),
            object_semantic_label=(label[:80] if label else None),
        )
    except Exception as e:
        logger.debug("[hybrid-grasp] rerun log failed: %s", e)


def _iterative_descent_to_depth_goal(
    state: PlanTopHandoffState,
    cfg: HybridGraspConfig,
    *,
    depth_goal_m: float,
    ee_trail: list[np.ndarray] | None = None,
    rerun_frame_counter: list[int] | None = None,
    rerun_label: str = "",
) -> float:
    """Closed-loop steps along the camera→target ray until measured depth matches ``depth_goal_m``.

    Returns total norm of base-frame translation applied (m).
    """
    if not bool(cfg.hybrid_iterative_descent_enable):
        planned_drop = float(cfg.plan_top_final_hover_m) - float(cfg.hybrid_final_tip_gap_m)
        if planned_drop > 0.001 and not cfg.dry_run and state.robot is not None:
            logger.info(
                "[hybrid-grasp] open-loop final descent: -%.3f m (iterative disabled).",
                planned_drop,
            )
            try:
                execute_cartesian_nudge_base(
                    state.robot,
                    state.kinematics,
                    SO100_MOTOR_NAMES,
                    np.array([0.0, 0.0, -planned_drop], dtype=np.float64),
                    state.motion,
                )
                state.T_base_ee[2, 3] -= planned_drop
            except Exception as e:
                logger.warning("[hybrid-grasp] open-loop descent failed: %s", e)
            return float(abs(planned_drop))
        return 0.0

    total_m = 0.0
    depth_scale = float(state.intrinsics.get("depth_scale", 0.001))
    policy = str(getattr(cfg, "hybrid_pbvs_depth_policy", "bbox_preferred"))
    dmin = float(cfg.hybrid_min_valid_depth_m)
    dmax = float(cfg.hybrid_max_valid_depth_m)
    obj_size = float(getattr(cfg, "target_physical_size_m", 0.03) or 0.03)
    tol = float(cfg.hybrid_descent_depth_tol_m)
    max_step = float(cfg.hybrid_descent_step_m)
    kp = float(cfg.hybrid_descent_kp)
    cam_key = str(cfg.camera_key)

    for step_i in range(int(cfg.hybrid_descent_max_iters)):
        if cfg.dry_run or state.robot is None:
            break
        # Hard safety cap: never advance indefinitely on a bad depth lock (common near-field failure).
        if total_m >= float(getattr(cfg, "hybrid_descent_total_budget_m", 0.06) or 0.06):
            logger.warning(
                "[hybrid-grasp] descent: budget exhausted (moved=%.3f >= cap=%.3f). Stopping before pushing further.",
                float(total_m),
                float(getattr(cfg, "hybrid_descent_total_budget_m", 0.06) or 0.06),
            )
            break
        _sync_T_base_ee_from_robot(state)
        try:
            obs = state.robot.get_observation()
        except Exception as e:
            logger.warning("[hybrid-grasp] descent iter %d: obs failed %s", step_i, e)
            break
        rgb, det = _fresh_observation(state, robot_obs=obs)
        bbox = getattr(det, "xyxy", None) if det is not None else None
        if bbox is None:
            logger.warning("[hybrid-grasp] descent iter %d: no bbox; stopping.", step_i)
            break
        depth = obs.get(f"{cam_key}_depth")
        if depth is None:
            logger.warning("[hybrid-grasp] descent iter %d: no depth map; stopping.", step_i)
            break
        d_used, d_stereo, d_bbox = _hybrid_select_depth(
            policy,
            depth,
            bbox,
            fx=float(state.fx),
            fy=float(state.fy),
            depth_scale=depth_scale,
            object_size_m=obj_size,
            dmin=dmin,
            dmax=dmax,
        )
        d_used = _hybrid_stabilize_depth(
            state,
            cfg,
            d_used=d_used,
            d_stereo=d_stereo,
            d_bbox=d_bbox,
            dmin=dmin,
            dmax=dmax,
        )
        if d_used is None:
            logger.warning("[hybrid-grasp] descent iter %d: no depth; stopping.", step_i)
            break
        err = float(d_used) - float(depth_goal_m)
        if abs(err) <= tol:
            logger.info(
                "[hybrid-grasp] iterative descent done at step %d: d=%.3f goal=%.3f err=%.4f",
                step_i,
                float(d_used),
                float(depth_goal_m),
                err,
            )
            break
        step_mag = float(np.clip(kp * err, -max_step, max_step))
        if abs(step_mag) < 1e-6:
            break
        x1, y1, x2, y2 = (float(v) for v in bbox)
        uc = 0.5 * (x1 + x2)
        vc = 0.5 * (y1 + y2)
        cx_ref = float(state.cx0) + float(getattr(cfg, "aim_u_offset_px", 0.0) or 0.0)
        x_c = (uc - cx_ref) / max(float(state.fx), 1e-6) * float(d_used)
        y_c = (vc - float(state.cy0)) / max(float(state.fy), 1e-6) * float(d_used)
        z_c = float(d_used)
        p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        pn = float(np.linalg.norm(p_cam))
        n_cam = p_cam / pn if pn > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        step_cam = n_cam * step_mag
        T_base_ee = np.asarray(state.T_base_ee, dtype=np.float64)
        T_ee_cam = np.asarray(state.T_ee_cam, dtype=np.float64)
        R_base_cam = (T_base_ee @ T_ee_cam)[:3, :3]
        delta_base = (R_base_cam @ step_cam).astype(np.float64)
        try:
            execute_cartesian_nudge_base(
                state.robot,
                state.kinematics,
                SO100_MOTOR_NAMES,
                delta_base,
                state.motion,
            )
            state.T_base_ee[:3, 3] += delta_base
            step_norm = float(np.linalg.norm(delta_base))
            total_m += step_norm
            logger.info(
                "[hybrid-grasp] descent iter %d: d_meas=%.4f goal=%.4f err=%+.4f step|Δ|=%.4fm",
                step_i,
                float(d_used),
                float(depth_goal_m),
                float(d_used) - float(depth_goal_m),
                step_norm,
            )
        except Exception as e:
            logger.warning("[hybrid-grasp] descent iter %d nudge failed: %s", step_i, e)
            break
        if ee_trail is not None and rerun_frame_counter is not None:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            uc = 0.5 * (x1 + x2)
            vc = 0.5 * (y1 + y2)
            _hybrid_try_rerun_log(
                state,
                cfg,
                frame_seq=int(rerun_frame_counter[0]),
                phase="vlm_descent",
                ee_trail=ee_trail,
                bbox=bbox,
                bbox_center=(uc, vc),
                depth_m=float(d_used),
                label=rerun_label,
                robot_obs=obs,
            )
            rerun_frame_counter[0] += 1
        time.sleep(max(0.0, float(getattr(cfg, "hybrid_descent_pause_s", 0.03))))
    return total_m


# ---------------------------------------------------------------------------
# Current-sensing grasp close + lift-confirm
# ---------------------------------------------------------------------------


def _read_present_current_counts(robot: Any, motor_name: str) -> float | None:
    try:
        val = robot.bus.read("Present_Current", motor_name, normalize=False)
        return float(val) if val is not None else None
    except Exception as e:
        logger.debug("[hybrid-grasp] read Present_Current failed: %s", e)
        return None


def _sample_idle_current(robot: Any, motor_name: str, *, seconds: float, dt: float = 0.02) -> float:
    """Sample the motor's idle current for ``seconds`` and return the mean (counts)."""
    end_t = time.time() + max(0.0, float(seconds))
    samples: list[float] = []
    while time.time() < end_t:
        v = _read_present_current_counts(robot, motor_name)
        if v is not None:
            samples.append(v)
        time.sleep(dt)
    if not samples:
        logger.warning("[hybrid-grasp] no idle current samples; using 0.")
        return 0.0
    return float(np.mean(samples))


@dataclass
class _GraspResult:
    contact_detected: bool
    contact_reason: str
    mean_hold_delta_counts: float
    lift_confirmed: bool
    lift_delta_counts: float


def _grasp_close_with_current(
    robot: Any,
    cfg: HybridGraspConfig,
    *,
    i_idle: float,
    start_open_pct: float | None = None,
) -> _GraspResult:
    """Incrementally close the gripper; declare contact on current spike / stall.

    Returns a :class:`_GraspResult` describing whether contact was detected.
    """
    motor = cfg.grasp_gripper_motor

    def w(pct: float) -> bool:
        try:
            robot.bus.write("Goal_Position", motor, float(pct), normalize=True)
            return True
        except Exception as e:
            logger.warning("[hybrid-grasp] gripper write %.1f%% failed: %s", pct, e)
            return False

    if bool(getattr(cfg, "grasp_preopen_enable", False)):
        logger.info(
            "[hybrid-grasp] opening gripper to %.0f%% before grasp.", cfg.grasp_open_pct,
        )
        w(float(cfg.grasp_open_pct))
        time.sleep(0.4)

    last_pos = None
    try:
        last_pos = robot.bus.read("Present_Position", motor, normalize=False)
    except Exception:
        pass

    stall_steps = 0
    contact_detected = False
    contact_reason = ""
    pct = float(start_open_pct) if start_open_pct is not None else float(cfg.grasp_open_pct)
    step = float(cfg.grasp_close_step_pct)
    target = float(cfg.grasp_close_pct)

    while True:
        pct = max(target, pct - step)
        w(pct)
        time.sleep(max(0.0, float(cfg.grasp_close_step_wait_s)))

        cur = _read_present_current_counts(robot, motor)
        try:
            pos = robot.bus.read("Present_Position", motor, normalize=False)
        except Exception:
            pos = None

        if cur is None or pos is None or last_pos is None:
            last_pos = pos
            if pct <= target:
                break
            continue

        delta_i = abs(float(cur) - i_idle)
        dpos = abs(float(pos) - float(last_pos))
        last_pos = pos

        if delta_i >= float(cfg.grasp_contact_delta_current_counts):
            contact_detected = True
            contact_reason = f"delta_current={delta_i:.1f}>=thresh"
            break

        if dpos <= float(cfg.grasp_contact_min_pos_change_counts):
            stall_steps += 1
        else:
            stall_steps = 0

        if stall_steps >= int(cfg.grasp_contact_stall_steps):
            contact_detected = True
            contact_reason = f"stalled_steps={stall_steps}"
            break

        if pct <= target:
            break

    # Hold + sample current under load
    hold_end = time.time() + max(0.0, float(cfg.grasp_hold_seconds))
    hold_samples: list[float] = []
    while time.time() < hold_end:
        v = _read_present_current_counts(robot, motor)
        if v is not None:
            hold_samples.append(v)
        time.sleep(0.02)
    if hold_samples:
        mean_hold_delta = abs(float(np.mean(hold_samples)) - i_idle)
    else:
        mean_hold_delta = 0.0

    logger.info(
        "[hybrid-grasp] close done: contact=%s reason=%r mean|I-Iidle|=%.1f counts",
        contact_detected, contact_reason, mean_hold_delta,
    )

    return _GraspResult(
        contact_detected=contact_detected,
        contact_reason=contact_reason,
        mean_hold_delta_counts=mean_hold_delta,
        lift_confirmed=False,
        lift_delta_counts=0.0,
    )


def _lift_and_confirm(
    state: PlanTopHandoffState,
    cfg: HybridGraspConfig,
    *,
    i_idle: float,
    motor: str,
) -> tuple[bool, float]:
    """Lift the arm straight up; re-sample gripper current. Returns (confirmed, delta)."""
    if cfg.dry_run or state.robot is None:
        return False, 0.0

    dz = float(cfg.grasp_lift_height_m)
    logger.info("[hybrid-grasp] lifting +%.3f m to confirm grasp.", dz)
    try:
        execute_cartesian_nudge_base(
            state.robot,
            state.kinematics,
            SO100_MOTOR_NAMES,
            np.array([0.0, 0.0, dz], dtype=np.float64),
            state.motion,
        )
    except Exception as e:
        logger.warning("[hybrid-grasp] lift failed: %s", e)
        return False, 0.0

    end_t = time.time() + max(0.05, float(cfg.grasp_lift_resample_seconds))
    samples: list[float] = []
    while time.time() < end_t:
        v = _read_present_current_counts(state.robot, motor)
        if v is not None:
            samples.append(v)
        time.sleep(0.02)
    if not samples:
        return False, 0.0

    delta = abs(float(np.mean(samples)) - i_idle)
    confirmed = delta >= float(cfg.grasp_lift_confirm_delta_counts)
    logger.info(
        "[hybrid-grasp] lift sample mean |I-Iidle|=%.1f counts → confirmed=%s",
        delta, confirmed,
    )
    return confirmed, delta


# ---------------------------------------------------------------------------
# Main callback: VLM refine → descend → close → confirm
# ---------------------------------------------------------------------------


def _write_jsonl(path: str, record: dict) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.debug("[hybrid-grasp] jsonl write failed: %s", e)


def _vlm_step_adaptive_multiplier(
    cfg: HybridGraspConfig,
    *,
    px_err: float,
    depth_live: float | None,
    depth_goal_m: float | None,
) -> float:
    """Larger per-step XY/ray caps when pixel or depth error is large; 1.0 when already aligned."""
    if not bool(getattr(cfg, "vlm_adaptive_step_enable", True)):
        return 1.0
    smax = float(getattr(cfg, "vlm_adaptive_step_scale_max", 2.0))
    pref = max(1e-3, float(getattr(cfg, "vlm_adaptive_step_px_ref", 35.0)))
    m = 1.0 + (smax - 1.0) * min(1.0, float(max(0.0, px_err)) / pref)
    if depth_live is not None and depth_goal_m is not None:
        dref = max(1e-4, float(getattr(cfg, "vlm_adaptive_depth_err_ref_m", 0.035)))
        de = abs(float(depth_live) - float(depth_goal_m))
        m = max(m, 1.0 + (smax - 1.0) * min(1.0, de / dref))
    return float(min(smax, max(1.0, m)))


def _visual_autoclose_should_trigger(
    cfg: HybridGraspConfig,
    *,
    bbox: tuple[float, float, float, float] | None,
    img_wh: tuple[int, int],
    du_px: float,
    dv_px: float,
    det_conf: float,
    depth_m: float | None,
) -> tuple[bool, str]:
    """Heuristic close trigger when the object appears near/in the gripper.

    Returns (trigger, reason).
    """
    if not bool(getattr(cfg, "hybrid_visual_autoclose_enable", True)):
        return False, ""
    if bbox is None:
        return False, ""
    w, h = int(img_wh[0]), int(img_wh[1])
    if w <= 1 or h <= 1:
        return False, ""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area_frac = (bw * bh) / max(1.0, float(w * h))
    px_err = max(abs(float(du_px)), abs(float(dv_px)))
    if det_conf + 1e-9 < float(getattr(cfg, "hybrid_visual_autoclose_min_conf", 0.45)):
        return False, ""
    if area_frac + 1e-12 < float(getattr(cfg, "hybrid_visual_autoclose_min_bbox_area_frac", 0.02)):
        return False, ""
    if px_err > float(getattr(cfg, "hybrid_visual_autoclose_max_px_err", 26.0)):
        return False, ""
    dmax = float(getattr(cfg, "hybrid_visual_autoclose_max_depth_m", 0.14))
    if depth_m is not None and float(depth_m) > dmax:
        return False, ""
    if depth_m is None:
        # If we don't have depth, still allow on a very large box.
        if area_frac < 0.06:
            return False, ""
    return True, f"visual_autoclose(area={area_frac:.3f} px_err={px_err:.1f} depth={depth_m if depth_m is not None else 'unknown'})"


def _fresh_observation(
    state: PlanTopHandoffState,
    *,
    robot_obs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, Any]:
    """Pull a fresh RGB frame + detection; optionally re-run YOLO when ``state.yolo_detector`` is set.

    If ``robot_obs`` is the result of ``robot.get_observation()``, it is reused so callers can avoid
    a duplicate camera read in the same control tick.
    """
    rgb = state.rgb
    det = state.det
    if rgb is None:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    if state.cfg.dry_run or state.robot is None:
        return np.asarray(rgb), det
    try:
        obs = robot_obs if robot_obs is not None else state.robot.get_observation()
        new_rgb = obs.get(state.cfg.camera_key)
        if new_rgb is not None:
            rgb = np.asarray(new_rgb)
    except Exception as e:
        logger.debug("[hybrid-grasp] get_observation failed: %s", e)
    rgb = np.asarray(rgb)
    if bool(getattr(state.cfg, "vlm_redetect_each_iter", False)) and getattr(
        state, "yolo_detector", None
    ) is not None:
        try:
            nd = state.yolo_detector.best_detection(rgb)
            if nd is not None:
                from types import SimpleNamespace

                x1, y1, x2, y2 = nd.xyxy
                det = SimpleNamespace(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(nd.confidence),
                    label=str(getattr(nd, "label", None) or state.cfg.query or "target"),
                )
        except Exception as e:
            logger.debug("[hybrid-grasp] YOLO re-detect failed: %s", e)
    return rgb, det


def _hybrid_refine_grasp(state: PlanTopHandoffState, cfg: HybridGraspConfig) -> None:
    """Callback invoked once plan_top finishes descend. Owns the final interaction.

    Flow:
    1. VLM refine loop (up to ``vlm_max_refine_iters``):
       - grab fresh image, annotate with the cached bbox + image-center crosshair
       - ask Gemini for (dx, dy, dyaw, close_now)
       - if close_now: break
       - else apply clamped base-XY nudge + wrist-roll delta
    2. Cartesian descent to ``hybrid_final_tip_gap_m`` above the cube top.
    3. Current-sensing gripper close.
    4. Lift & confirm.
    """

    try:
        refiner = _GeminiRefiner(cfg)
    except Exception as e:
        logger.warning(
            "[hybrid-grasp] VLM init failed (%s); skipping refinement, closing on YOLO-only pose.",
            e,
        )
        refiner = None

    motor = str(cfg.grasp_gripper_motor)
    jsonl = str(cfg.hybrid_log_jsonl or "")
    ee_trail: list[np.ndarray] = []
    rerun_fc = [int(getattr(cfg, "hybrid_rerun_frame_base", 1_000_000))]
    label = getattr(state.det, "label", None) or state.cfg.query or "target"

    # Seed near-field depth stabilizer from PBVS / approach EMA so a later stereo/background jump
    # (e.g. 0.11m -> 0.70m) gets rejected.
    try:
        if getattr(state, "_hybrid_last_valid_depth_m", None) is None:
            d0 = getattr(state, "depth_ema_m", None)
            if d0 is not None:
                d0f = float(d0)
                dmin0 = float(getattr(cfg, "hybrid_min_valid_depth_m", 0.08))
                dmax0 = float(getattr(cfg, "hybrid_max_valid_depth_m", 0.85))
                if dmin0 < d0f < dmax0:
                    state._hybrid_last_valid_depth_m = d0f
    except Exception:
        pass

    # --- Idle-current sample BEFORE the fingers touch anything.
    i_idle = 0.0
    if not cfg.dry_run and state.robot is not None:
        # Open the gripper first so the idle sample is taken with fingers free.
        try:
            state.robot.bus.write(
                "Goal_Position", motor, float(cfg.grasp_open_pct), normalize=True,
            )
            time.sleep(0.4)
        except Exception as e:
            logger.warning("[hybrid-grasp] pre-idle gripper open failed: %s", e)
        i_idle = _sample_idle_current(
            state.robot, motor, seconds=float(cfg.grasp_idle_sample_seconds),
        )
        logger.info("[hybrid-grasp] idle gripper current = %.1f counts", i_idle)

    # One Rerun frame at handoff (PBVS timeline continues in ``frame`` ≥ hybrid_rerun_frame_base).
    try:
        rgb0, det0 = _fresh_observation(state)
        bb0 = getattr(det0, "xyxy", None) if det0 is not None else None
        if bb0 is not None:
            x1, y1, x2, y2 = (float(v) for v in bb0)
            c0 = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
            d0 = getattr(state, "depth_ema_m", None)
            d0f = float(d0) if d0 is not None else None
            _hybrid_try_rerun_log(
                state,
                cfg,
                frame_seq=int(rerun_fc[0]),
                phase="vlm",
                ee_trail=ee_trail,
                bbox=bb0,
                bbox_center=c0,
                depth_m=d0f,
                label=str(label),
            )
            rerun_fc[0] += 1
    except Exception:
        pass

    # --- VLM refinement loop (observe → VLM JSON act → log; same *idea* as ``lerobot-agentic-manipulate`` micro-steps).
    total_xy = 0.0
    total_yaw_deg = 0.0
    total_ray_m = 0.0
    total_flex_deg = 0.0
    close_now = False
    gripper_close_trigger = "none"
    last_reason = ""
    saw_any_vlm = False
    successful_vlm_rounds = 0

    # Scale hint for the VLM: at hover_h above the cube, 1 px ≈ hover_h/fx meters.
    ee_z = float(state.T_base_ee[2, 3])
    obj_z = float(np.asarray(state.p_target_base).reshape(3)[2])
    hover_h = max(0.02, ee_z - obj_z)
    mm_per_px = hover_h / max(1.0, state.fx) * 1000.0
    px_per_mm = 1.0 / max(1e-6, mm_per_px)
    last_action_summary = "start"

    if refiner is None:
        close_now = True
        last_reason = "vlm_disabled"
        gripper_close_trigger = "vlm_disabled"

    for i in range(int(cfg.vlm_max_refine_iters)):
        if close_now:
            break

        rob_obs: dict[str, Any] | None = None
        if not cfg.dry_run and state.robot is not None:
            try:
                rob_obs = state.robot.get_observation()
            except Exception as e:
                logger.debug("[hybrid-grasp] get_observation failed: %s", e)
                rob_obs = None

        rgb, det = _fresh_observation(state, robot_obs=rob_obs)
        mag = 0.0
        ray_mag = 0.0
        d_yaw = 0.0
        d_flex = 0.0
        bbox = getattr(det, "xyxy", None) if det is not None else None
        cx_draw = float(state.cx0) + float(getattr(cfg, "aim_u_offset_px", 0.0) or 0.0)
        cy_draw = float(state.cy0)
        if bool(getattr(cfg, "vlm_annotate_crosshair_at_bbox", True)) and bbox is not None:
            x1a, y1a, x2a, y2a = (float(v) for v in bbox)
            cx_draw = 0.5 * (x1a + x2a)
            cy_draw = 0.5 * (y1a + y2a)
        annotated = (
            _annotate_for_vlm(
                rgb,
                bbox_xyxy=bbox,
                image_center=(cx_draw, cy_draw),
                label=str(label),
                show_bbox_left_edge=bool(getattr(cfg, "vlm_annotate_bbox_left_edge", True)),
            )
            if cfg.vlm_annotate_image
            else rgb
        )

        du_live = 0.0
        dv_live = 0.0
        depth_live: float | None = None
        cx_ref = float(state.cx0) + float(getattr(cfg, "aim_u_offset_px", 0.0) or 0.0)
        if bbox is not None:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            uc = 0.5 * (x1 + x2)
            vc = 0.5 * (y1 + y2)
            du_live = float(uc - cx_ref)
            dv_live = float(vc - float(state.cy0))
            try:
                obs_d = rob_obs
                if obs_d is None and not cfg.dry_run and state.robot is not None:
                    obs_d = state.robot.get_observation()
                if obs_d is not None:
                    depth_map = obs_d.get(f"{cfg.camera_key}_depth")
                    ds = float(state.intrinsics.get("depth_scale", 0.001))
                    d_used, d_stereo, d_bbox = _hybrid_select_depth(
                        str(cfg.hybrid_pbvs_depth_policy),
                        depth_map,
                        bbox,
                        fx=float(state.fx),
                        fy=float(state.fy),
                        depth_scale=ds,
                        object_size_m=float(getattr(cfg, "target_physical_size_m", 0.03) or 0.03),
                        dmin=float(cfg.hybrid_min_valid_depth_m),
                        dmax=float(cfg.hybrid_max_valid_depth_m),
                    )
                    depth_live = _hybrid_stabilize_depth(
                        state,
                        cfg,
                        d_used=d_used,
                        d_stereo=d_stereo,
                        d_bbox=d_bbox,
                        dmin=float(cfg.hybrid_min_valid_depth_m),
                        dmax=float(cfg.hybrid_max_valid_depth_m),
                    )
            except Exception:
                pass
        if depth_live is None and getattr(state, "depth_ema_m", None) is not None:
            depth_live = float(state.depth_ema_m)

        det_conf = float(getattr(det, "confidence", 0.0) or 0.0) if det is not None else 0.0
        dg_ctx = getattr(state, "depth_goal_m", None)

        # If the object already appears "in the gripper" (near + big bbox + centered), close now
        # even if the VLM never emits close_now.
        if i >= int(getattr(cfg, "hybrid_visual_autoclose_min_iters", 0) or 0):
            trig, why = _visual_autoclose_should_trigger(
                cfg,
                bbox=bbox,
                img_wh=(int(rgb.shape[1]), int(rgb.shape[0])),
                du_px=du_live,
                dv_px=dv_live,
                det_conf=det_conf,
                depth_m=depth_live,
            )
            if trig:
                close_now = True
                gripper_close_trigger = "visual_autoclose"
                last_reason = why
                logger.info("[hybrid-grasp] visual auto-close at iter %d: %s", i, why)
                break

        refine_ctx = {
            "agent_step": i,
            "depth_ema_m": f"{depth_live:.3f}" if depth_live is not None else "unknown",
            "depth_goal_m": f"{float(dg_ctx):.3f}" if dg_ctx is not None else "unknown",
            "det_conf": f"{det_conf:.2f}",
            "du_px": f"{du_live:+.1f}",
            "dv_px": f"{dv_live:+.1f}",
            "min_steps_before_close": int(cfg.vlm_min_refine_iters_before_close),
            "last_action": last_action_summary[:120],
        }

        sug = refiner.suggest(
            annotated,
            task_hint=str(label),
            px_per_mm_at_object=px_per_mm,
            refine_context=refine_ctx,
        ) if refiner is not None else None
        if sug is None:
            logger.warning("[hybrid-grasp] VLM iter %d returned nothing; retrying next step.", i)
            continue
        saw_any_vlm = True
        successful_vlm_rounds += 1

        dz_cap_n = int(getattr(cfg, "vlm_forbid_negative_dz_iters", 0) or 0)
        if bool(getattr(cfg, "vlm_clamp_negative_dz_mm", True)) and dz_cap_n > 0 and i < dz_cap_n:
            if sug.dz_mm < 0.0:
                logger.info(
                    "[hybrid-grasp] VLM iter %d: clamped dz_mm %.2f -> 0.0 (no optical retreat for first %d steps)",
                    i,
                    sug.dz_mm,
                    dz_cap_n,
                )
                sug.dz_mm = 0.0

        cn = bool(sug.close_now)
        px_err = max(abs(du_live), abs(dv_live))
        if cn and successful_vlm_rounds < int(cfg.vlm_min_refine_iters_before_close):
            logger.info(
                "[hybrid-grasp] VLM iter %d: veto close_now (successful rounds %d < min %d)",
                i,
                successful_vlm_rounds,
                int(cfg.vlm_min_refine_iters_before_close),
            )
            cn = False
        if cn and bool(cfg.vlm_require_pixel_motion):
            if px_err > float(cfg.vlm_center_autoclose_tol_px):
                logger.info(
                    "[hybrid-grasp] VLM iter %d: veto close_now (pixel err %.1f > tol %.1f px)",
                    i,
                    px_err,
                    float(cfg.vlm_center_autoclose_tol_px),
                )
                cn = False
        if cn and bool(cfg.vlm_geom_fallback_enable):
            if px_err > float(cfg.vlm_geom_close_tol_px):
                logger.info(
                    "[hybrid-grasp] VLM iter %d: veto close_now (geom |du|,|dv| %.1f > %.1f px)",
                    i,
                    px_err,
                    float(cfg.vlm_geom_close_tol_px),
                )
                cn = False
        min_dc = float(getattr(cfg, "vlm_min_detection_conf_for_close", 0.0) or 0.0)
        if cn and min_dc > 1e-6 and det_conf + 1e-9 < min_dc:
            logger.info(
                "[hybrid-grasp] VLM iter %d: veto close_now (det_conf=%.2f < min_for_close %.2f)",
                i,
                det_conf,
                min_dc,
            )
            cn = False
        max_dc = float(getattr(cfg, "vlm_max_depth_at_close_m", 0.0) or 0.0)
        if cn and max_dc > 1e-6 and depth_live is not None and float(depth_live) > max_dc:
            logger.info(
                "[hybrid-grasp] VLM iter %d: veto close_now (depth=%.3f m > max_for_close %.3f — likely far/background)",
                i,
                float(depth_live),
                max_dc,
            )
            cn = False

        sug_close = cn
        step_scale = _vlm_step_adaptive_multiplier(
            cfg,
            px_err=px_err,
            depth_live=depth_live,
            depth_goal_m=getattr(state, "depth_goal_m", None),
        )
        logger.info(
            "[hybrid-grasp] VLM iter %d: dx=%.1fmm dy=%.1fmm dz=%.1fmm dyaw=%.1f° dflex=%.1f° close=%s (%s) "
            "step_scale=%.2f",
            i,
            sug.dx_mm,
            sug.dy_mm,
            sug.dz_mm,
            sug.d_yaw_deg,
            sug.d_wrist_flex_deg,
            sug_close,
            sug.reason,
            step_scale,
        )
        last_reason = sug.reason
        _write_jsonl(jsonl, {
            "phase": "vlm_refine", "iter": i,
            "dx_mm": sug.dx_mm, "dy_mm": sug.dy_mm, "dz_mm": sug.dz_mm,
            "d_yaw_deg": sug.d_yaw_deg, "d_wrist_flex_deg": sug.d_wrist_flex_deg,
            "close_now": sug_close, "reason": sug.reason,
            "total_xy_m": total_xy, "total_yaw_deg": total_yaw_deg, "total_ray_m": total_ray_m,
        })

        if sug_close:
            close_now = True
            gripper_close_trigger = "vlm_close_now"
            last_action_summary = f"close_now:{sug.reason}"[:120]
            break

        # -- Clamp & apply the XY nudge (adaptive caps: faster when |du|,|dv| or depth error is large)
        step_cap_m = float(cfg.vlm_nudge_step_m) * step_scale
        dx_m = float(np.clip(sug.dx_mm * 1e-3, -step_cap_m, step_cap_m))
        dy_m = float(np.clip(sug.dy_mm * 1e-3, -step_cap_m, step_cap_m))
        delta_base_xy = _image_mm_to_base_xy(
            state.T_base_ee, state.T_ee_cam, dx_m * 1000.0, dy_m * 1000.0,
        )
        remaining_xy = max(0.0, float(cfg.vlm_nudge_total_budget_m) - total_xy)
        mag = float(np.linalg.norm(delta_base_xy[:2]))
        if mag > remaining_xy + 1e-12:
            if remaining_xy <= 1e-9:
                delta_base_xy = np.zeros(3, dtype=np.float64)
                mag = 0.0
            else:
                scale = remaining_xy / max(1e-9, mag)
                delta_base_xy = delta_base_xy * scale
                mag = remaining_xy
        if mag > 1e-4 and not cfg.dry_run and state.robot is not None:
            try:
                execute_cartesian_nudge_base(
                    state.robot, state.kinematics, SO100_MOTOR_NAMES,
                    delta_base_xy, state.motion,
                )
                total_xy += mag
                state.T_base_ee[:3, 3] += delta_base_xy
            except Exception as e:
                logger.warning("[hybrid-grasp] XY nudge failed: %s", e)

        # -- Ray (camera +Z) nudge in base frame
        ray_cap = float(cfg.vlm_ray_step_max_m) * step_scale
        dz_m = float(np.clip(sug.dz_mm * 1e-3, -ray_cap, ray_cap))
        delta_ray = _ray_mm_to_base_delta(state.T_base_ee, state.T_ee_cam, dz_m * 1000.0)
        ray_mag = float(np.linalg.norm(delta_ray))
        remaining_ray = max(0.0, float(cfg.vlm_ray_total_budget_m) - total_ray_m)
        if ray_mag > remaining_ray + 1e-12:
            if remaining_ray <= 1e-9:
                delta_ray = np.zeros(3, dtype=np.float64)
                ray_mag = 0.0
            else:
                delta_ray *= remaining_ray / max(1e-9, ray_mag)
                ray_mag = remaining_ray
        if ray_mag > 1e-5 and not cfg.dry_run and state.robot is not None:
            try:
                execute_cartesian_nudge_base(
                    state.robot, state.kinematics, SO100_MOTOR_NAMES,
                    delta_ray, state.motion,
                )
                total_ray_m += ray_mag
                state.T_base_ee[:3, 3] += delta_ray
            except Exception as e:
                logger.warning("[hybrid-grasp] ray nudge failed: %s", e)

        # -- Wrist roll
        yaw_step_cap = float(cfg.vlm_wrist_roll_step_deg)
        yaw_total_cap = float(cfg.vlm_wrist_roll_total_deg)
        d_yaw = float(np.clip(sug.d_yaw_deg, -yaw_step_cap, yaw_step_cap))
        remaining_yaw = yaw_total_cap - abs(total_yaw_deg)
        if abs(d_yaw) > remaining_yaw:
            d_yaw = math.copysign(remaining_yaw, d_yaw)
        if abs(d_yaw) >= 0.5 and not cfg.dry_run and state.robot is not None:
            _apply_wrist_roll_delta(state.robot, d_yaw)
            total_yaw_deg += d_yaw

        # -- Wrist flex
        flex_cap = float(cfg.vlm_wrist_flex_step_deg)
        flex_total_cap = float(cfg.vlm_wrist_flex_total_deg)
        d_flex = float(np.clip(sug.d_wrist_flex_deg, -flex_cap, flex_cap))
        rem_flex = flex_total_cap - abs(total_flex_deg)
        if abs(d_flex) > rem_flex:
            d_flex = math.copysign(rem_flex, d_flex)
        if abs(d_flex) >= 0.5 and not cfg.dry_run and state.robot is not None:
            _apply_wrist_flex_delta(state.robot, d_flex)
            total_flex_deg += d_flex

        last_action_summary = (
            f"xy={mag:.4f}m ray={ray_mag:.4f}m yaw={d_yaw:.1f}° flex={d_flex:.1f}°"
        )[:120]
        rem_xy_b = float(cfg.vlm_nudge_total_budget_m) - total_xy
        rem_ray_b = float(cfg.vlm_ray_total_budget_m) - total_ray_m
        logger.info(
            "[hybrid-grasp] after VLM iter %d: total_xy=%.4fm total_ray=%.4fm | budget_left xy=%.4fm ray=%.4fm",
            i,
            total_xy,
            total_ray_m,
            max(0.0, rem_xy_b),
            max(0.0, rem_ray_b),
        )
        time.sleep(max(0.0, float(getattr(cfg, "vlm_post_motion_settle_s", 0.05))))
        _sync_T_base_ee_from_robot(state)
        try:
            bc: tuple[float, float] | None = None
            if bbox is not None:
                x1r, y1r, x2r, y2r = (float(v) for v in bbox)
                bc = (0.5 * (x1r + x2r), 0.5 * (y1r + y2r))
            rob_obs_log: dict[str, Any] | None = None
            if not cfg.dry_run and state.robot is not None:
                try:
                    rob_obs_log = state.robot.get_observation()
                except Exception:
                    rob_obs_log = None
            _hybrid_try_rerun_log(
                state,
                cfg,
                frame_seq=int(rerun_fc[0]),
                phase="vlm",
                ee_trail=ee_trail,
                bbox=bbox,
                bbox_center=bc,
                depth_m=depth_live,
                label=str(label),
                robot_obs=rob_obs_log,
            )
            rerun_fc[0] += 1
        except Exception:
            pass

    if gripper_close_trigger == "none":
        if close_now:
            gripper_close_trigger = "vlm_close_now"
        elif bool(cfg.vlm_auto_close_on_exhaustion):
            gripper_close_trigger = "refine_exhausted_auto_close"

    # If the VLM never produced a usable suggestion, do not proceed to the "one-shot" grasp.
    if refiner is not None and bool(cfg.vlm_abort_if_all_iters_fail):
        if not saw_any_vlm:
            logger.warning(
                "[hybrid-grasp] VLM failed to return JSON for all %d iter(s); aborting before descent/close.",
                int(cfg.vlm_max_refine_iters),
            )
            _write_jsonl(jsonl, {"phase": "abort", "reason": "vlm_all_iters_failed"})
            return

    if not close_now and not bool(cfg.vlm_auto_close_on_exhaustion):
        logger.warning(
            "[hybrid-grasp] refine budget exhausted and auto-close disabled; aborting grasp.",
        )
        _write_jsonl(jsonl, {"phase": "abort", "reason": "refine_exhausted_no_autoclose"})
        return

    # --- Final approach: closed-loop ray steps to ``depth_goal_m`` (PBVS), else legacy Z drop.
    depth_goal = getattr(state, "depth_goal_m", None)
    if depth_goal is None and getattr(state, "depth_ema_m", None) is not None:
        depth_goal = float(state.depth_ema_m)
    if depth_goal is None:
        depth_goal = float(cfg.plan_top_final_hover_m)
    descent_moved = _iterative_descent_to_depth_goal(
        state,
        cfg,
        depth_goal_m=float(depth_goal),
        ee_trail=ee_trail,
        rerun_frame_counter=rerun_fc,
        rerun_label=str(label),
    )
    logger.info(
        "[hybrid-grasp] post-VLM descent moved ≈%.4f m (iterative=%s goal_depth=%.3f)",
        float(descent_moved),
        bool(cfg.hybrid_iterative_descent_enable),
        float(depth_goal),
    )
    try:
        o_e: dict[str, Any] | None = None
        if not cfg.dry_run and state.robot is not None:
            try:
                o_e = state.robot.get_observation()
            except Exception:
                o_e = None
        rgb_e, det_e = _fresh_observation(state, robot_obs=o_e)
        bb_e = getattr(det_e, "xyxy", None) if det_e is not None else None
        bc_e = None
        if bb_e is not None:
            x1e, y1e, x2e, y2e = (float(v) for v in bb_e)
            bc_e = (0.5 * (x1e + x2e), 0.5 * (y1e + y2e))
        d_e = None
        if bb_e is not None and o_e is not None:
            try:
                dm_e = o_e.get(f"{cfg.camera_key}_depth")
                ds_e = float(state.intrinsics.get("depth_scale", 0.001))
                d_e, _, _ = _hybrid_select_depth(
                    str(cfg.hybrid_pbvs_depth_policy),
                    dm_e,
                    bb_e,
                    fx=float(state.fx),
                    fy=float(state.fy),
                    depth_scale=ds_e,
                    object_size_m=float(getattr(cfg, "target_physical_size_m", 0.03) or 0.03),
                    dmin=float(cfg.hybrid_min_valid_depth_m),
                    dmax=float(cfg.hybrid_max_valid_depth_m),
                )
            except Exception:
                pass
        _hybrid_try_rerun_log(
            state,
            cfg,
            frame_seq=int(rerun_fc[0]),
            phase="vlm_grasp",
            ee_trail=ee_trail,
            bbox=bb_e,
            bbox_center=bc_e,
            depth_m=float(d_e) if d_e is not None else None,
            label=str(label),
            robot_obs=o_e,
        )
        rerun_fc[0] += 1
    except Exception:
        pass

    # --- Current-sensing grasp close
    if cfg.dry_run or state.robot is None:
        logger.info("[hybrid-grasp] dry-run: skipping gripper close.")
        return

    logger.info(
        "[hybrid-grasp] closing gripper (current sensing): trigger=%s last_reason=%r depth_goal=%.3fm",
        gripper_close_trigger,
        (last_reason or "")[:160],
        float(depth_goal),
    )
    start_pct = None
    try:
        # Start from current gripper opening unless we explicitly pre-open.
        if not bool(getattr(cfg, "grasp_preopen_enable", False)):
            v = state.robot.bus.read("Present_Position", motor, normalize=True)
            start_pct = float(v) if v is not None else None
    except Exception:
        start_pct = None

    grasp = _grasp_close_with_current(
        state.robot,
        cfg,
        i_idle=i_idle,
        start_open_pct=start_pct,
    )

    # --- Lift & confirm
    lift_ok, lift_delta = _lift_and_confirm(
        state, cfg, i_idle=i_idle, motor=motor,
    )
    grasp.lift_confirmed = lift_ok
    grasp.lift_delta_counts = lift_delta

    success = bool(grasp.contact_detected) and bool(lift_ok)
    logger.info(
        "[hybrid-grasp] RESULT: success=%s contact=%s hold_delta=%.1f lift_delta=%.1f reason=%s",
        success, grasp.contact_detected, grasp.mean_hold_delta_counts,
        grasp.lift_delta_counts, grasp.contact_reason or last_reason,
    )
    
    _write_jsonl(jsonl, {
        "phase": "result",
        "success": success,
        "contact_detected": grasp.contact_detected,
        "contact_reason": grasp.contact_reason,
        "mean_hold_delta_counts": grasp.mean_hold_delta_counts,
        "lift_confirmed": grasp.lift_confirmed,
        "lift_delta_counts": grasp.lift_delta_counts,
        "total_xy_m": total_xy,
        "total_yaw_deg": total_yaw_deg,
        "total_ray_m": total_ray_m,
        "total_flex_deg": total_flex_deg,
        "last_vlm_reason": last_reason,
    })

    print("PHASE: result")
    print(f"SUCCESS: {success}")
    print(f"CONTACT_DETECTED: {grasp.contact_detected}")
    print(f"CONTACT_REASON: {grasp.contact_reason}")
    print(f"MEAN_HOLD_DELTA_COUNTS: {grasp.mean_hold_delta_counts}")
    print(f"LIFT_CONFIRMED: {grasp.lift_confirmed}")
    print(f"LIFT_DELTA_COUNTS: {grasp.lift_delta_counts}")
    print(f"TOTAL_XY_M: {total_xy}")
    print(f"TOTAL_YAW_DEG: {total_yaw_deg}")
    print(f"TOTAL_RAY_M: {total_ray_m}")
    print(f"TOTAL_FLEX_DEG: {total_flex_deg}")
    print(f"LAST_VLM_REASON: {last_reason}")

    if not success:
        # Release the (possibly missed) object to avoid dragging it.
        try:
            state.robot.bus.write(
                "Goal_Position", motor, float(cfg.grasp_open_pct), normalize=True,
            )
        except Exception:
            pass


# Public alias for programmatic handoffs (e.g. PBVS) that build ``PlanTopHandoffState``
# directly and then want to run the same hybrid refine+grasp routine.
run_vlm_refine_and_grasp = _hybrid_refine_grasp


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@parser.wrap()
def yolo_vlm_grasp(cfg: HybridGraspConfig) -> None:
    """CLI: ``lerobot-yolo-vlm-grasp``.

    Drives the existing ``plan_top`` YOLO approach to a hover, then hands off to the VLM
    refinement + current-sensing gripper close. All ``lerobot-yolo-track-approach`` flags
    are accepted unchanged; new flags live under ``--vlm-*``, ``--grasp-*``, and
    ``--hybrid-*``.
    """
    init_logging()
    logger.info(
        "[hybrid-grasp] starting: query=%r backend=%s model=%s", cfg.query, cfg.vlm_backend, cfg.vlm_model,
    )
    run_yolo_track_approach(
        cfg,
        on_plan_top_descend_done=lambda st: _hybrid_refine_grasp(st, cfg),
    )


def main() -> None:  # pragma: no cover - thin wrapper
    yolo_vlm_grasp()


if __name__ == "__main__":  # pragma: no cover
    main()
