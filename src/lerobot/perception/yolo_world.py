# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""YOLO-World open-vocabulary detection for simple visual tracking (text query → bbox)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class YoloWorldDetection:
    """Single detection from YOLO-World."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "YOLO-World tracking requires `ultralytics`. Install with:\n"
            "  pip install 'ultralytics>=8.3.0'\n"
            "or: pip install -e '.[yolo-world]'"
        ) from e


class YoloWorldDetector:
    """Thin wrapper: load a World checkpoint, set text classes, run inference."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        conf: float = 0.25,
        sahi_enable: bool = False,
        sahi_slice_wh: tuple[int, int] = (512, 512),
        sahi_overlap: float = 0.20,
        sahi_iou_threshold: float = 0.55,
        sahi_include_full_image: bool = True,
    ) -> None:
        _require_ultralytics()
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        if device:
            self._model.to(device)
        self.conf = float(conf)
        self._classes_key: tuple[str, ...] = ()
        self.sahi_enable = bool(sahi_enable)
        self.sahi_slice_wh = (int(sahi_slice_wh[0]), int(sahi_slice_wh[1]))
        self.sahi_overlap = float(sahi_overlap)
        self.sahi_iou_threshold = float(sahi_iou_threshold)
        self.sahi_include_full_image = bool(sahi_include_full_image)

    def set_query(self, text: str) -> None:
        """Set the open-vocabulary class name(s) for the next ``predict`` calls."""
        _require_ultralytics()
        q = str(text).strip()
        if not q:
            raise ValueError("query must be non-empty")
        classes = [q]
        key = tuple(classes)
        if key != self._classes_key:
            self._model.set_classes(classes)
            self._classes_key = key

    @staticmethod
    def _xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = (float(a[i]) for i in range(4))
        bx1, by1, bx2, by2 = (float(b[i]) for i in range(4))
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 1e-12 else 0.0

    @classmethod
    def _nms_xyxy(
        cls,
        dets: Iterable[YoloWorldDetection],
        *,
        iou_thr: float,
    ) -> list[YoloWorldDetection]:
        """Greedy NMS on xyxy boxes (single-class friendly)."""
        items = sorted(list(dets), key=lambda d: float(d.confidence), reverse=True)
        if not items:
            return []
        kept: list[YoloWorldDetection] = []
        for d in items:
            box_d = np.array(d.xyxy, dtype=np.float64)
            suppress = False
            for k in kept:
                if cls._xyxy_iou(box_d, np.array(k.xyxy, dtype=np.float64)) >= float(iou_thr):
                    suppress = True
                    break
            if not suppress:
                kept.append(d)
        return kept

    def _predict_bgr(self, bgr: np.ndarray) -> list[YoloWorldDetection]:
        results = self._model.predict(
            bgr,
            conf=self.conf,
            verbose=False,
        )
        out: list[YoloWorldDetection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(np.int32)
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = (float(xyxy[i, j]) for j in range(4))
                out.append(
                    YoloWorldDetection(
                        xyxy=(x1, y1, x2, y2),
                        confidence=float(confs[i]),
                        class_id=int(cls_ids[i]),
                    )
                )
        return out

    @staticmethod
    def _tile_origins(
        *,
        full_w: int,
        full_h: int,
        tile_w: int,
        tile_h: int,
        overlap: float,
    ) -> list[tuple[int, int, int, int]]:
        """Return (x1,y1,x2,y2) tiles covering the image with overlap."""
        tile_w = max(32, int(tile_w))
        tile_h = max(32, int(tile_h))
        ov = float(np.clip(overlap, 0.0, 0.9))
        sx = max(1, int(round(tile_w * (1.0 - ov))))
        sy = max(1, int(round(tile_h * (1.0 - ov))))
        xs = list(range(0, max(1, full_w - tile_w + 1), sx))
        ys = list(range(0, max(1, full_h - tile_h + 1), sy))
        if not xs or xs[-1] != max(0, full_w - tile_w):
            xs.append(max(0, full_w - tile_w))
        if not ys or ys[-1] != max(0, full_h - tile_h):
            ys.append(max(0, full_h - tile_h))
        tiles: list[tuple[int, int, int, int]] = []
        for y1 in ys:
            for x1 in xs:
                x2 = min(full_w, x1 + tile_w)
                y2 = min(full_h, y1 + tile_h)
                tiles.append((int(x1), int(y1), int(x2), int(y2)))
        return tiles

    def predict_rgb(self, rgb: np.ndarray) -> list[YoloWorldDetection]:
        """Run detection on an RGB uint8 image (H, W, 3). Returns boxes sorted by confidence.

        When ``sahi_enable`` is True, runs SAHI-style tiled inference with overlap and merges
        detections with NMS in the full-image coordinate frame.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB HxWx3, got {rgb.shape}")
        if not self._classes_key:
            raise RuntimeError("call set_query() before predict_rgb()")

        # ultralytics expects BGR for OpenCV-style images
        import cv2

        bgr_full = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if not bool(self.sahi_enable):
            out = self._predict_bgr(bgr_full)
            out.sort(key=lambda d: d.confidence, reverse=True)
            return out

        h, w = int(bgr_full.shape[0]), int(bgr_full.shape[1])
        tw, th = self.sahi_slice_wh
        overlap = float(self.sahi_overlap)
        tiles = self._tile_origins(full_w=w, full_h=h, tile_w=tw, tile_h=th, overlap=overlap)
        all_dets: list[YoloWorldDetection] = []

        if bool(self.sahi_include_full_image):
            all_dets.extend(self._predict_bgr(bgr_full))

        for x1, y1, x2, y2 in tiles:
            crop = bgr_full[y1:y2, x1:x2]
            dets = self._predict_bgr(crop)
            for d in dets:
                bx1, by1, bx2, by2 = d.xyxy
                all_dets.append(
                    YoloWorldDetection(
                        xyxy=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                        confidence=float(d.confidence),
                        class_id=int(d.class_id),
                    )
                )

        merged = self._nms_xyxy(all_dets, iou_thr=float(self.sahi_iou_threshold))
        merged.sort(key=lambda d: d.confidence, reverse=True)
        return merged

    def best_detection(self, rgb: np.ndarray) -> YoloWorldDetection | None:
        dets = self.predict_rgb(rgb)
        return dets[0] if dets else None
