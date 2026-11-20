"""标注结果导出：COCO / YOLO-seg / PNG 掩码 / LabelMe / 预览图。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import postprocess as pp
from .store import Project, timestamp


def _iter_kept(project: Project, only_reviewed: bool = False, entries=None):
    """遍历有标注结果且包含有效实例的图片。

    entries 给定时只在这些图片里选（按分组/勾选导出的情况）。
    """
    for entry in (entries if entries is not None else project.images):
        if only_reviewed and entry.status != "reviewed":
            continue
        annot = project.load_annotation(entry)
        if not annot:
            continue
        kept = [i for i in annot.get("instances", []) if i.get("kept", True)]
        if not kept:
            continue
        yield entry, annot, kept


# --------------------------------------------------------------------- COCO
def export_coco(
    project: Project,
    out_path: str | Path | None = None,
    only_reviewed: bool = False,
    polygons: bool = False,
    entries=None,
) -> Path:
    out_path = Path(
        out_path
        or project.root / "exports" / f"coco-{timestamp()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    classes = project.classes
    cat_index: dict[str, int] = {}
    for entry, annot, _ in _iter_kept(project, only_reviewed, entries):
        for c in annot.get("classes") or []:
            if c and c not in classes:
                classes.append(c)
    for c in classes:
        cat_index[c] = len(cat_index) + 1

    images_out = []
    annotations_out = []
    ann_id = 1
    for entry, annot, kept in _iter_kept(project, only_reviewed, entries):
        width = annot.get("width") or entry.width
        height = annot.get("height") or entry.height
        images_out.append(
            {
                "id": _numeric_id(entry.id),
                "file_name": entry.relpath,
                "width": width,
                "height": height,
            }
        )
        for inst in kept:
            cls = inst.get("class_name", "?")
            if cls not in cat_index:
                cat_index[cls] = len(cat_index) + 1
                classes.append(cls)
            seg = inst["segmentation"]
            if polygons:
                mask = pp.decode_mask(seg, height, width)
                polys = pp.mask_to_polygons(mask)
                segmentation = polys
            else:
                segmentation = seg
            annotations_out.append(
                {
                    "id": ann_id,
                    "image_id": _numeric_id(entry.id),
                    "category_id": cat_index[cls],
                    "segmentation": segmentation,
                    "area": float(inst.get("area", 0.0)),
                    "bbox": _xywh(inst.get("box", [0, 0, 0, 0])),
                    "iscrowd": 0,
                    "score": float(inst.get("score", 1.0)),
                    "source": inst.get("source", "text"),
                }
            )
            ann_id += 1

    coco = {
        "info": {
            "description": f"{project.meta.get('name')} exported by sam3label",
            "date_created": timestamp(),
            "version": "1.0",
        },
        "licenses": [],
        "images": images_out,
        "annotations": annotations_out,
        "categories": [
            {"id": cat_index[c], "name": c, "supercategory": "object"} for c in classes
        ],
    }
    out_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return out_path


def _numeric_id(image_id: str) -> int:
    """COCO 的 image_id 需要整数；把 12 位 hex 映射成稳定整数。"""
    return int(image_id, 16) % (10**12)


def _xywh(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    return [float(x0), float(y0), float(max(x1 - x0, 0)), float(max(y1 - y0, 0))]


# --------------------------------------------------------------------- YOLO-seg
def export_yolo(
    project: Project,
    out_dir: str | Path | None = None,
    only_reviewed: bool = False,
    copy_images: bool = True,
    entries=None,
) -> Path:
    out_dir = Path(out_dir or project.root / "exports" / f"yolo-seg-{timestamp()}")
    labels_dir = out_dir / "labels"
    images_dir = out_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    classes = project.classes
    for entry, annot, _ in _iter_kept(project, only_reviewed, entries):
        for c in annot.get("classes") or []:
            if c and c not in classes:
                classes.append(c)
    cls_to_id = {c: i for i, c in enumerate(classes)}

    n = 0
    for entry, annot, kept in _iter_kept(project, only_reviewed, entries):
        width = annot.get("width") or entry.width
        height = annot.get("height") or entry.height
        if not width or not height:
            continue
        lines: list[str] = []
        for inst in kept:
            cls = inst.get("class_name", "?")
            cid = cls_to_id.setdefault(cls, len(cls_to_id))
            mask = pp.decode_mask(inst["segmentation"], height, width)
            for poly in pp.mask_to_polygons(mask):
                pts = []
                for i in range(0, len(poly) - 1, 2):
                    pts.append(f"{poly[i] / width:.6f}")
                    pts.append(f"{poly[i + 1] / height:.6f}")
                if len(pts) >= 6:
                    lines.append(f"{cid} " + " ".join(pts))
        rel = Path(entry.relpath)
        stem = rel.stem
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if copy_images:
            src = project.image_path(entry)
            dst = images_dir / f"{stem}{src.suffix}"
            if not dst.exists():
                shutil.copy2(src, dst)
        n += 1

    yaml = ["path: .", "train: images", "val: images", "names:"]
    for i, c in enumerate(classes):
        yaml.append(f"  {i}: {json.dumps(c, ensure_ascii=False)}")
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        f"YOLO-seg 格式：{n} 张图，{len(classes)} 类\n"
        "labels/*.txt 为归一化多边形（class_id x1 y1 x2 y2 ...），可直接用于 "
        "ultralytics 的 segment 训练。\n",
        encoding="utf-8",
    )
    return out_dir


# --------------------------------------------------------------------- PNG 掩码
def export_masks(
    project: Project,
    out_dir: str | Path | None = None,
    only_reviewed: bool = False,
    instance_png: bool = False,
    entries=None,
) -> Path:
    """导出语义掩码（uint8，像素值=类别 id+1，0 为背景）+ 调色板。"""
    out_dir = Path(out_dir or project.root / "exports" / f"masks-{timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = project.classes
    for entry, annot, _ in _iter_kept(project, only_reviewed, entries):
        for c in annot.get("classes") or []:
            if c and c not in classes:
                classes.append(c)
    cls_to_id = {c: i for i, c in enumerate(classes)}

    n = 0
    for entry, annot, kept in _iter_kept(project, only_reviewed, entries):
        width = annot.get("width") or entry.width
        height = annot.get("height") or entry.height
        semantic = np.zeros((height, width), dtype=np.uint8)
        for inst in kept:
            cid = cls_to_id.get(inst.get("class_name", "?"), 0) + 1
            mask = pp.decode_mask(inst["segmentation"], height, width)
            semantic[mask] = np.uint8(min(cid, 255))
            if instance_png:
                inst_dir = out_dir / "instances"
                inst_dir.mkdir(exist_ok=True)
                name = f"{Path(entry.relpath).stem}_{inst.get('id', 0)}_{inst.get('class_name','x')}.png"
                Image.fromarray((mask.astype(np.uint8) * 255)).save(inst_dir / name)
        Image.fromarray(semantic).save(out_dir / f"{Path(entry.relpath).stem}.png")
        n += 1

    palette = {i + 1: {"class": c, "rgb": list(pp.color_for(i))} for i, c in enumerate(classes)}
    (out_dir / "palette.json").write_text(
        json.dumps({"background": 0, "classes": palette}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


# --------------------------------------------------------------------- LabelMe
def export_labelme(
    project: Project,
    out_dir: str | Path | None = None,
    only_reviewed: bool = False,
    entries=None,
) -> Path:
    """LabelMe / X-AnyLabeling 可打开的 JSON（多边形），方便人工二次修正。"""
    out_dir = Path(out_dir or project.root / "exports" / f"labelme-{timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry, annot, kept in _iter_kept(project, only_reviewed, entries):
        width = annot.get("width") or entry.width
        height = annot.get("height") or entry.height
        shapes = []
        for inst in kept:
            mask = pp.decode_mask(inst["segmentation"], height, width)
            for poly in pp.mask_to_polygons(mask):
                pts = [[round(poly[i], 2), round(poly[i + 1], 2)] for i in range(0, len(poly) - 1, 2)]
                shapes.append(
                    {
                        "label": inst.get("class_name", "?"),
                        "points": pts,
                        "group_id": inst.get("id"),
                        "shape_type": "polygon",
                        "flags": {"score": float(inst.get("score", 1.0))},
                    }
                )
        payload = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": entry.relpath,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        name = f"{Path(entry.relpath).stem}.json"
        (out_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out_dir


# --------------------------------------------------------------------- 预览图
def export_previews(
    project: Project,
    out_dir: str | Path | None = None,
    only_reviewed: bool = False,
    max_size: int = 1600,
    entries=None,
) -> Path:
    out_dir = Path(out_dir or project.root / "exports" / f"previews-{timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry, annot, kept in _iter_kept(project, only_reviewed, entries):
        src = project.image_path(entry)
        image = Image.open(src).convert("RGB")
        image.load()
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale))
            )
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]
        instances = []
        for inst in kept:
            mask = pp.decode_mask(
                inst["segmentation"], annot.get("height") or h, annot.get("width") or w
            )
            if mask.shape != (h, w):
                mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h))
                ).astype(bool)
            item = dict(inst)
            item["_mask"] = mask
            instances.append(item)
        overlay = pp.render_overlay(rgb, instances, class_colors=project.class_colors)
        Image.fromarray(overlay).save(out_dir / f"{Path(entry.relpath).stem}.jpg", quality=90)
    return out_dir


def export_all(project: Project, out_dir: str | Path) -> dict:
    """一次性导出全部格式，返回各格式路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "coco": str(export_coco(project, out_dir / "instances_coco.json")),
        "yolo": str(export_yolo(project, out_dir / "yolo_seg")),
        "masks": str(export_masks(project, out_dir / "masks")),
        "labelme": str(export_labelme(project, out_dir / "labelme")),
        "previews": str(export_previews(project, out_dir / "previews")),
    }
