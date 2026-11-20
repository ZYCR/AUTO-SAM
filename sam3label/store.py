"""项目 / 数据集状态管理。

目录结构（每个项目一个文件夹）：

    data/projects/<name>/
        project.json              # 项目元信息 + 类别表 + 默认参数
        images.json               # 图片清单（id / 相对路径 / 状态 / 实例数）
        annotations/<id>.json     # 每张图的标注（RLE 掩码 + 可选软删除）
        exports/                  # 导出结果
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import config


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _image_id(relpath: str) -> str:
    """稳定的图片 id：路径哈希前缀，增删图片不影响已有 id。"""
    return hashlib.md5(relpath.encode("utf-8")).hexdigest()[:12]


@dataclass
class ImageEntry:
    id: str
    relpath: str
    status: str = "pending"  # pending | done | reviewed
    n_instances: int = 0
    width: int = 0
    height: int = 0
    updated: str = ""
    # 外部引用（不复制到项目里）时的绝对路径；为空表示用 image_dir/relpath
    abs_path: str = ""
    # 视频来源帧才有：所属视频名、帧序号、时间戳（秒）
    video: str = ""
    frame_index: int = 0
    ts: float = 0.0
    # 左侧列表分组用：group=显示名，group_path=唯一键（video:<名> 或文件夹绝对路径）
    group: str = ""
    group_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Project:
    root: Path
    meta: dict = field(default_factory=dict)
    images: list[ImageEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ 创建/打开
    @classmethod
    def create(
        cls,
        root: str | Path,
        name: str | None = None,
        classes: list[str] | None = None,
        image_dir: str | Path | None = None,
    ) -> "Project":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(exist_ok=True)
        (root / "exports").mkdir(exist_ok=True)
        meta = {
            "name": name or root.name,
            "classes": classes or [],
            "image_dir": str(Path(image_dir).resolve()) if image_dir else "",
            "created": _now(),
            "settings": {
                "confidence": config.CONFIDENCE,
                "min_area": config.MIN_AREA,
                "dedup_iou": config.DEDUP_IOU,
                "resolution": config.RESOLUTION,
            },
        }
        (root / "project.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        proj = cls(root=root, meta=meta, images=[])
        proj.save_images()
        return proj

    @classmethod
    def open(cls, root: str | Path) -> "Project":
        root = Path(root)
        meta_path = root / "project.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{root} 不是有效的项目目录（缺少 project.json）")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        (root / "annotations").mkdir(exist_ok=True)
        (root / "exports").mkdir(exist_ok=True)
        images_path = root / "images.json"
        images = []
        if images_path.exists():
            data = json.loads(images_path.read_text(encoding="utf-8"))
            images = [ImageEntry(**d) for d in data.get("images", [])]
        # 老数据迁移：早期导入的视频帧只有 video 字段，补上分组信息
        changed = False
        for e in images:
            if e.video and not e.group_path:
                e.group = e.video
                e.group_path = f"video:{e.video}"
                changed = True
        proj = cls(root=root, meta=meta, images=images)
        if changed:
            proj.save_images()
        return proj

    # ------------------------------------------------------------------ 落盘
    def save_meta(self) -> None:
        (self.root / "project.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_images(self) -> None:
        payload = {"updated": _now(), "images": [i.to_dict() for i in self.images]}
        (self.root / "images.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ 图片
    @property
    def classes(self) -> list[str]:
        return list(self.meta.get("classes", []))

    @property
    def excluded(self) -> list[str]:
        return list(self.meta.get("excluded", []))

    def _excluded_set(self) -> set[str]:
        return set(self.meta.get("excluded", []))

    def _mark_excluded(self, paths: list[str]) -> None:
        ex = self._excluded_set()
        ex.update(paths)
        self.meta["excluded"] = sorted(ex)
        self.save_meta()

    def _unmark_excluded(self, paths: list[str]) -> None:
        ex = self._excluded_set()
        for p in paths:
            ex.discard(p)
        self.meta["excluded"] = sorted(ex)
        self.save_meta()

    @property
    def class_conf(self) -> dict[str, float]:
        """每个类别的置信度阈值（未配置的用全局）。"""
        raw = self.meta.get("settings", {}).get("class_conf", {}) or {}
        return {str(k): float(v) for k, v in raw.items()}

    @property
    def class_colors(self) -> dict[str, str]:
        """每个类别自定义颜色 {类别: "#rrggbb"}，没配的用默认调色板。"""
        raw = self.meta.get("settings", {}).get("class_colors", {}) or {}
        out = {}
        for k, v in raw.items():
            v = str(v).strip()
            if len(v) == 7 and v.startswith("#"):
                out[str(k)] = v.lower()
        return out

    def set_class_colors(self, colors: dict[str, str]) -> None:
        clean = {}
        for k, v in (colors or {}).items():
            v = str(v).strip()
            if len(v) == 7 and v.startswith("#"):
                clean[str(k).strip()] = v.lower()
        self.meta.setdefault("settings", {})["class_colors"] = clean
        self.save_meta()

    def color_for_class(self, class_name: str, index: int = 0):
        """优先用自定义颜色，否则按类别序号取默认调色板。"""
        from .postprocess import color_for, hex_to_rgb

        custom = self.class_colors.get(class_name)
        if custom:
            try:
                return hex_to_rgb(custom)
            except Exception:
                pass
        return color_for(index)

    def set_class_conf(self, conf: dict[str, float]) -> None:
        clean = {}
        for k, v in (conf or {}).items():
            k = str(k).strip()
            if not k:
                continue
            try:
                clean[k] = round(min(0.999, max(0.01, float(v))), 3)
            except (TypeError, ValueError):
                continue
        self.meta.setdefault("settings", {})["class_conf"] = clean
        self.save_meta()

    def threshold_for(self, class_name: str, default: float | None = None) -> float:
        conf = self.class_conf
        if class_name in conf:
            return conf[class_name]
        if default is not None:
            return default
        return float(self.meta.get("settings", {}).get("confidence", 0.5))

    def sync_classes_from_annotations(self) -> list[str]:
        """把标注里出现但项目类别表里没有的类别补进去（顺序：先原有，再按出现频次）。"""
        import collections

        counts = collections.Counter()
        for entry in self.images:
            annot = self.load_annotation(entry)
            if not annot:
                continue
            for inst in annot.get("instances", []):
                if inst.get("kept", True):
                    counts[inst.get("class_name") or "?"] += 1
        merged = list(self.classes)
        for name, _ in counts.most_common():
            if name not in merged:
                merged.append(name)
        if merged != self.classes:
            self.set_classes(merged)
        return merged

    def refilter(self, apply: bool = False) -> dict:
        """按"类别阈值"重新判定已有实例的 kept 标记。

        - 手动改过 kept 的实例（manual=True）一律跳过，尊重人工决定
        - apply=False 只统计影响（预览）；apply=True 会写回标注文件
        """
        conf = self.class_conf
        default = float(self.meta.get("settings", {}).get("confidence", 0.5))
        stats: dict[str, dict] = {}
        for entry in self.images:
            annot = self.load_annotation(entry)
            if not annot:
                continue
            changed = False
            for inst in annot.get("instances", []):
                cls = inst.get("class_name") or "?"
                thr = conf.get(cls, default)
                score = float(inst.get("score", 0.0))
                bucket = stats.setdefault(
                    cls, {"threshold": thr, "above": 0, "below": 0, "changed": 0, "manual": 0}
                )
                bucket["threshold"] = thr
                above = score >= thr
                if inst.get("manual"):
                    bucket["manual"] += 1
                    bucket["above" if inst.get("kept", True) else "below"] += 1
                    continue
                bucket["above" if above else "below"] += 1
                if bool(inst.get("kept", True)) != above:
                    bucket["changed"] += 1
                    if apply:
                        inst["kept"] = above
                        changed = True
            if apply and changed:
                self.save_annotation(entry, annot)
        return stats

    def set_classes(self, classes: list[str]) -> None:
        self.meta["classes"] = classes
        self.save_meta()

    def image_dir(self) -> Path:
        d = self.meta.get("image_dir") or ""
        return Path(d) if d else self.root / "images"

    def add_images(self, paths: list[Path], base: Path | None = None) -> int:
        """把图片加入清单（按相对路径去重）。返回新增数量。"""
        base = base or self.image_dir()
        known = {i.relpath for i in self.images}
        excluded = self._excluded_set()
        added = 0
        for p in paths:
            p = Path(p)
            if str(p.resolve()) in excluded:
                continue
            try:
                rel = str(p.resolve().relative_to(Path(base).resolve()))
            except ValueError:
                rel = p.name
            if rel in known:
                continue
            entry = ImageEntry(id=_image_id(rel), relpath=rel)
            self._fill_size(entry)
            self.images.append(entry)
            known.add(rel)
            added += 1
        if added:
            self.save_images()
        return added

    # ------------------------------------------------------- 外部导入（引用/复制）
    def _existing_abs(self) -> set[str]:
        out = set()
        for i in self.images:
            try:
                out.add(str(self.image_path(i).resolve()))
            except Exception:
                continue
        return out

    def add_external(
        self,
        paths: list[str | Path],
        copy_to: str | Path | None = None,
        recursive: bool = False,
    ) -> dict:
        """把本机图片/文件夹加入项目。

        - 传文件夹时**只取该文件夹这一层**的直接图片文件，不会递归到子目录
          （需要连子目录一起加就显式传 recursive=True）
        - 传单个图片文件时只加这一张
        - copy_to 为 None：只登记绝对路径（引用，不占额外磁盘，原图不动）
        - copy_to 给目录：复制进去（同名自动加序号）

        返回 {"added": [...], "skipped": [...], "missing": [...]}
        """
        import shutil

        resolved: list[Path] = []
        folder_group: dict[str, tuple[str, str]] = {}
        missing: list[str] = []
        bad: list[str] = []
        for raw in paths:
            p = Path(str(raw)).expanduser()
            if not p.exists():
                missing.append(str(raw))
                continue
            if p.is_dir():
                walker = p.rglob("*") if recursive else p.iterdir()
                batch = sorted(
                    x for x in walker
                    if x.is_file() and x.suffix.lower() in config.IMAGE_EXTS
                )
                resolved.extend(batch)
                for x in batch:
                    folder_group[str(x.resolve())] = (p.name, str(p.resolve()))
            elif p.is_file() and p.suffix.lower() in config.IMAGE_EXTS:
                resolved.append(p)
            else:
                bad.append(str(raw))

        known = self._existing_abs()
        excluded = self._excluded_set()
        added: list[str] = []
        skipped: list[str] = []
        dest_dir = Path(copy_to).expanduser() if copy_to else None
        if dest_dir is not None:
            dest_dir.mkdir(parents=True, exist_ok=True)

        for src in resolved:
            real = str(src.resolve())
            if real in known:
                skipped.append(str(src))
                continue
            excluded.discard(real)  # 之前"移除但不删文件"的，这次显式导入就恢复
            if dest_dir is not None:
                target = dest_dir / src.name
                n = 1
                while target.exists():
                    target = dest_dir / f"{src.stem}_{n}{src.suffix}"
                    n += 1
                shutil.copy2(src, target)
                entry = ImageEntry(
                    id=_image_id(str(target.resolve())),
                    relpath=target.name,
                    abs_path=str(target.resolve()),
                )
            else:
                entry = ImageEntry(
                    id=_image_id(real),
                    relpath=src.name,
                    abs_path=real,
                )
            grp = folder_group.get(str(src.resolve()))
            if grp:
                entry.group, entry.group_path = grp[0], grp[1]
            # id 冲突兜底
            if any(i.id == entry.id for i in self.images):
                entry.id = _image_id(real + "#dup")
            self._fill_size(entry)
            self.images.append(entry)
            known.add(str(self.image_path(entry).resolve()))
            added.append(str(src.resolve()))

        if added:
            self.save_images()
            self._unmark_excluded(added)
        return {"added": added, "skipped": skipped, "missing": missing, "bad": bad}

    def add_video_frames(self, video_name: str, frames: list[dict]) -> list[ImageEntry]:
        """把抽出来的视频帧注册成项目图片（relpath 带视频名前缀，避免重名）。"""
        known = self._existing_abs()
        added: list[ImageEntry] = []
        for fr in frames:
            path = Path(fr["path"])
            real = str(path.resolve())
            if real in known:
                continue
            rel = f"{video_name}/{path.name}"
            entry = ImageEntry(
                id=_image_id(rel),
                relpath=rel,
                abs_path=real,
                width=int(fr.get("width") or 0),
                height=int(fr.get("height") or 0),
                video=video_name,
                frame_index=int(fr.get("index") or 0),
                ts=float(fr.get("time") or 0.0),
                group=video_name,
                group_path=f"video:{video_name}",
            )
            if any(i.id == entry.id for i in self.images):
                entry.id = _image_id(rel + "#dup")
            self.images.append(entry)
            known.add(real)
            added.append(entry)
        if added:
            self.save_images()
            self._unmark_excluded([i.abs_path for i in added])
        return added

    def groups(self) -> list[dict]:
        """按来源分组统计（视频 / 文件夹批量导入的才成组）。"""
        out: dict[str, dict] = {}
        for e in self.images:
            if not e.group_path:
                continue
            g = out.setdefault(
                e.group_path,
                {"key": e.group_path, "name": e.group, "kind": "video" if e.group_path.startswith("video:") else "folder",
                 "count": 0, "ids": []},
            )
            g["count"] += 1
            g["ids"].append(e.id)
        return list(out.values())

    def entries_of_groups(self, keys: list[str]) -> list[ImageEntry]:
        """按分组键取图片（keys 里可以有 "video:xxx" 或文件夹绝对路径）。"""
        want = set(keys or [])
        return [i for i in self.images if i.group_path in want]

    def frames_of_video(self, video_name: str) -> list[ImageEntry]:
        return sorted(
            (i for i in self.images if i.video == video_name),
            key=lambda i: i.frame_index,
        )

    def scan(self, recursive: bool = False) -> int:
        """扫描 image_dir，把新图片加进清单（默认只看当前层）。"""
        base = self.image_dir()
        if not base.exists():
            return 0
        walker = base.rglob("*") if recursive else base.iterdir()
        found = sorted(
            p for p in walker if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS
        )
        return self.add_images(found, base=base)

    def _fill_size(self, entry: ImageEntry) -> None:
        try:
            from PIL import Image

            with Image.open(self.image_path(entry)) as im:
                entry.width, entry.height = im.size
        except Exception:
            entry.width = entry.height = 0

    def image_path(self, entry: ImageEntry | str) -> Path:
        if isinstance(entry, str):
            entry = self.get(entry)
        if entry.abs_path:
            p = Path(entry.abs_path)
            if p.exists():
                return p
        return self.image_dir() / entry.relpath

    def get(self, image_id: str) -> ImageEntry:
        for i in self.images:
            if i.id == image_id:
                return i
        raise KeyError(f"图片 id 不存在: {image_id}")

    # ------------------------------------------------------------------ 标注
    def annotation_path(self, entry: ImageEntry | str) -> Path:
        image_id = entry if isinstance(entry, str) else entry.id
        return self.root / "annotations" / f"{image_id}.json"

    def load_annotation(self, entry: ImageEntry | str) -> dict | None:
        path = self.annotation_path(entry)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_annotation(self, entry: ImageEntry | str, annot: dict) -> None:
        image_id = entry if isinstance(entry, str) else entry.id
        annot = dict(annot)
        annot["image_id"] = image_id
        annot["updated"] = _now()
        path = self.annotation_path(image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(
                json.dumps(annot, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, path)
            # 注意：任务线程可能持有被 refresh 换掉的旧 ImageEntry，
            # 所以按 id 找"当前清单里"的那一条来更新，避免状态写丢。
            target = next((i for i in self.images if i.id == image_id), None)
            if target is None and not isinstance(entry, str):
                target = entry
            if target is not None:
                target.n_instances = sum(
                    1 for i in annot.get("instances", []) if i.get("kept", True)
                )
                target.status = annot.get("status", target.status)
                target.updated = annot["updated"]
            self.save_images()

    def delete_annotation(self, entry: ImageEntry | str) -> None:
        path = self.annotation_path(entry)
        if path.exists():
            path.unlink()
        if not isinstance(entry, str):
            entry.status = "pending"
            entry.n_instances = 0
            self.save_images()

    # ------------------------------------------------------------------ 统计
    def missing_entries(self) -> list[ImageEntry]:
        """原图已被移动/删除的条目（引用模式下常见）。"""
        out = []
        for entry in self.images:
            if not self.image_path(entry).exists():
                out.append(entry)
        return out

    def prune_missing(self) -> list[str]:
        gone = self.missing_entries()
        if not gone:
            return []
        ids = {i.id for i in gone}
        self.images = [i for i in self.images if i.id not in ids]
        self.save_images()
        return [i.relpath for i in gone]

    def remove_images(self, ids: list[str], delete_files: bool = False) -> dict:
        """把勾选的图片从工作区移除。

        - delete_files=False：只删清单条目 + 标注，磁盘文件保留，
          并把路径记进 project.json 的 excluded，避免下次 scan 又加回来
        - delete_files=True：位于项目目录内的副本直接删掉；
          项目外的引用文件**不会删**（那是你的原图），只移除引用并记墓碑
        """
        ids = [i for i in dict.fromkeys(ids) if i]
        id_set = set(ids)
        targets = [i for i in self.images if i.id in id_set]
        projects_root = self.root.resolve()
        removed, deleted, kept_external, tomb = [], [], [], []
        for entry in targets:
            path = self.image_path(entry)
            real = str(path.resolve()) if path.exists() else (entry.abs_path or "")
            if delete_files and path.exists():
                try:
                    inside = path.resolve().is_relative_to(projects_root)
                except AttributeError:  # py<3.9
                    inside = str(path.resolve()).startswith(str(projects_root))
                if inside:
                    try:
                        path.unlink()
                        deleted.append(real)
                    except OSError:
                        kept_external.append(real)
                        tomb.append(real)
                else:
                    kept_external.append(real)  # 项目外的原图，不动
                    tomb.append(real)
            elif real:
                tomb.append(real)
            ap = self.annotation_path(entry)
            if ap.exists():
                ap.unlink()
            removed.append(entry.relpath)
        self.images = [i for i in self.images if i.id not in id_set]
        self.save_images()
        if tomb:
            self._mark_excluded(tomb)
        # 清掉这些 id 的缩略图缓存
        thumb_dir = self.root / ".cache" / "thumbs"
        if thumb_dir.exists():
            for f in thumb_dir.glob("*.jpg"):
                if f.name.split("_")[0] in id_set:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        return {
            "removed": removed,
            "deleted_files": deleted,
            "kept_external": kept_external,
            "total": len(self.images),
        }

    def resync_manifest(self) -> dict:
        """用 annotations/*.json 回填清单里的 status / n_instances。

        任务并发、或早期版本把状态写丢时用来修复；也可当一致性自检用。
        """
        changed = 0
        have = set()
        for entry in self.images:
            have.add(entry.id)
            annot = self.load_annotation(entry)
            if annot is None:
                continue
            kept = sum(1 for i in annot.get("instances", []) if i.get("kept", True))
            status = annot.get("status") or entry.status
            if entry.status != status or entry.n_instances != kept:
                entry.status = status
                entry.n_instances = kept
                entry.updated = annot.get("updated", entry.updated)
                changed += 1
        if changed:
            self.save_images()
        orphans = [
            p.stem for p in (self.root / "annotations").glob("*.json")
            if p.stem not in have
        ]
        return {"changed": changed, "total": len(self.images), "orphans": len(orphans)}

    def stats(self) -> dict:
        by_class: dict[str, int] = {}
        for entry in self.images:
            annot = self.load_annotation(entry)
            if not annot:
                continue
            for inst in annot.get("instances", []):
                if not inst.get("kept", True):
                    continue
                cls = inst.get("class_name", "?")
                by_class[cls] = by_class.get(cls, 0) + 1
        return {
            "name": self.meta.get("name"),
            "images": len(self.images),
            "missing": len(self.missing_entries()),
            "done": sum(1 for i in self.images if i.status == "done"),
            "reviewed": sum(1 for i in self.images if i.status == "reviewed"),
            "pending": sum(1 for i in self.images if i.status == "pending"),
            "instances": sum(by_class.values()),
            "by_class": by_class,
            "classes": self.classes,
            "image_dir": str(self.image_dir()),
            "root": str(self.root),
        }


def parse_classes(text: str | list[str] | None) -> list[str]:
    if not text:
        return []
    if isinstance(text, list):
        return [c.strip() for c in text if c.strip()]
    return [c.strip() for c in text.replace("，", ",").split(",") if c.strip()]


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
