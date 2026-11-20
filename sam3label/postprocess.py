"""掩码后处理：编码、去重、过滤、多边形化、可视化。

所有掩码在内存里统一用 HxW 的 bool numpy 表示；
落盘时用 COCO RLE（pycocotools 压缩格式）编码，体积小、可无损还原。
"""

from __future__ import annotations

import base64
from typing import Iterable

import numpy as np
from pycocotools import mask as mask_utils

try:  # opencv 只用于多边形化与可视化，缺失时降级
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------- 编解码
def encode_mask(mask: np.ndarray) -> dict:
    """bool / uint8 掩码 -> COCO RLE dict（counts 为 str，可直接 json 序列化）。"""
    mask = np.asarray(mask)
    if mask.ndim == 3:  # 兼容 [1,H,W] / [H,W,1]
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def decode_mask(seg: dict, height: int | None = None, width: int | None = None) -> np.ndarray:
    """COCO RLE dict -> bool 掩码。"""
    rle = {"size": list(seg["size"]), "counts": seg["counts"].encode("ascii")}
    mask = mask_utils.decode(rle).astype(bool)
    if height is not None and width is not None and mask.shape != (height, width):
        out = np.zeros((height, width), dtype=bool)
        h = min(height, mask.shape[0])
        w = min(width, mask.shape[1])
        out[:h, :w] = mask[:h, :w]
        mask = out
    return mask


def encode_mask_png(mask: np.ndarray) -> str:
    """掩码 -> base64 PNG（单通道，供前端直接贴图）。"""
    if cv2 is None:
        raise RuntimeError("需要 opencv 才能把掩码编码成 PNG")
    ok, buf = cv2.imencode(".png", (mask.astype(np.uint8) * 255))
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------- 度量
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0.0, 0.0, 0.0, 0.0]
    y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return [float(x0), float(y0), float(x1 + 1), float(y1 + 1)]


# --------------------------------------------------------------------- 过滤/去重
def filter_instances(
    instances: Iterable,
    min_area: int = 0,
    dedup_iou: float = 0.75,
    class_agnostic: bool = True,
):
    """按面积过滤 + 贪心掩码 NMS（保留分数高的）。

    SAM3 的检测头已经做过一次内部 NMS，但同一物体仍可能同时被
    "car" / "vehicle" 两个提示词命中，这里做一次跨提示词的兜底去重。
    """
    items = [i for i in instances if min_area <= 0 or i.area >= min_area]
    items.sort(key=lambda i: i.score, reverse=True)
    kept: list = []
    for cand in items:
        if dedup_iou >= 1.0:
            kept.append(cand)
            continue
        duplicate = False
        for prev in kept:
            if not class_agnostic and prev.class_name != cand.class_name:
                continue
            # 先用 bbox 粗筛，再算掩码 IoU
            if not _bbox_overlap(prev.box, cand.box):
                continue
            if mask_iou(prev.mask, cand.mask) > dedup_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def _bbox_overlap(a: list[float], b: list[float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return False
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return False
    inter = iw * ih
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return smaller > 0 and inter / smaller > 0.3


# --------------------------------------------------------------------- 多边形
def mask_to_polygons(mask: np.ndarray, epsilon_ratio: float = 0.0025) -> list[list[float]]:
    """掩码 -> 多边形列表（每个多边形是扁平 [x0,y0,x1,y1,...]，图像像素坐标）。

    用于 YOLO-seg / LabelMe 等需要矢量轮廓的格式；COCO 用 RLE 更保真。
    """
    if cv2 is None:
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[list[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1:
            continue
        eps = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        polygons.append([float(v) for v in approx.reshape(-1)])
    return polygons


# --------------------------------------------------------------------- 可视化
PALETTE = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
    (175, 82, 222), (255, 204, 0), (90, 200, 250), (255, 45, 85),
    (88, 86, 214), (0, 199, 190), (142, 142, 147), (162, 132, 94),
    (50, 173, 230), (255, 105, 180), (60, 179, 113), (218, 112, 214),
]


def color_for(index: int) -> tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b)。"""
    v = value.strip().lstrip("#")
    if len(v) != 6:
        raise ValueError(f"非法颜色：{value}")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def render_overlay(
    image_rgb: np.ndarray,
    instances: list[dict],
    alpha: float = 0.45,
    draw_boxes: bool = True,
    class_colors: dict | None = None,
) -> np.ndarray:
    """在 RGB 图上叠加半透明实例掩码 + 边框，返回新的 RGB 图。"""
    canvas = image_rgb.astype(np.float32).copy()
    color_layer = np.zeros_like(canvas)
    weight = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    def _color(idx: int, inst: dict) -> tuple[int, int, int]:
        name = inst.get("class_name") or ""
        custom = (class_colors or {}).get(name)
        if custom:
            try:
                return hex_to_rgb(custom)
            except Exception:
                pass
        return color_for(idx)

    for idx, inst in enumerate(instances):
        mask = inst.get("_mask")
        if mask is None:
            continue
        color = np.array(_color(idx, inst), dtype=np.float32)
        color_layer[mask] = color
        weight[mask] = 1.0
    if weight.any():
        m = (weight > 0)[..., None]
        canvas = np.where(m, canvas * (1 - alpha) + color_layer * alpha, canvas)
    out = np.clip(canvas, 0, 255).astype(np.uint8)

    if cv2 is not None:
        for idx, inst in enumerate(instances):
            color = _color(idx, inst)
            if draw_boxes:
                x0, y0, x1, y1 = [int(round(v)) for v in inst["box"]]
                cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            label = inst.get("class_name") or ""
            if label:
                score = inst.get("score")
                text = f"{label} {score:.2f}" if isinstance(score, float) else str(label)
                x0, y0 = int(round(inst["box"][0])), int(round(inst["box"][1]))
                y = max(y0 - 6, 12)
                cv2.putText(
                    out, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    out, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
    return out
