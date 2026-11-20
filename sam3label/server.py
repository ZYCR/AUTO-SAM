"""FastAPI 复核服务：浏览器里跑批、看掩码、改类别、删误检、导出。

设计要点：
- 模型只加载一次，所有 GPU 调用用全局锁串行化（FastAPI 会在线程池里跑同步端点）。
- 掩码着色的 PNG 由服务端渲染（前端只负责叠图和画框），省得在 JS 里解 RLE。
- 所有改动立即落盘到 annotations/<id>.json，不依赖前端内存状态。
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import config, exporters
from . import postprocess as pp
from .store import Project

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class JobCanceled(Exception):
    """任务被取消时从 checkpoint 抛出。"""


class Job:
    """一个后台任务：可暂停 / 继续 / 取消，带百分比进度。"""

    def __init__(self, job_id: str, name: str) -> None:
        self.id = job_id
        self.name = name
        self.state = "running"  # running | paused | finished | failed | canceled
        self.message = ""
        self.done = 0
        self.total = 0
        self.progress = 0.0  # 0~1，前端优先用它算百分比
        self.started = time.time()
        self.results: list = []
        self._pause = threading.Event()
        self._pause.set()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 控制
    def pause(self) -> None:
        with self._lock:
            if self.state == "running":
                self.state = "paused"
                self._pause.clear()

    def resume(self) -> None:
        with self._lock:
            if self.state == "paused":
                self.state = "running"
                self._pause.set()

    def cancel(self) -> None:
        with self._lock:
            if self.state in ("running", "paused"):
                self.state = "canceling"
                self.message = "取消中…（等当前帧处理完）"
            self._stop.set()
            self._pause.set()

    def checkpoint(self) -> None:
        """在任务的安全点调用：被取消则抛异常，被暂停则阻塞等待。"""
        if self._stop.is_set():
            raise JobCanceled()
        if not self._pause.is_set():
            with self._lock:
                if self.state == "running":
                    self.state = "paused"
            while not self._pause.wait(timeout=0.2):
                if self._stop.is_set():
                    raise JobCanceled()
            with self._lock:
                if self.state == "paused":
                    self.state = "running"
        if self._stop.is_set():
            raise JobCanceled()

    def set_progress(self, message: str | None = None, frac: float | None = None) -> None:
        with self._lock:
            if message is not None:
                self.message = message
            if frac is not None:
                self.progress = max(0.0, min(1.0, float(frac)))

    def set_done(self, done: int, total: int | None = None) -> None:
        with self._lock:
            self.done = done
            if total is not None:
                self.total = total
            if self.total:
                self.progress = max(0.0, min(1.0, self.done / self.total))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "state": self.state,
                "message": self.message,
                "done": self.done,
                "total": self.total,
                "progress": round(self.progress, 4),
                "elapsed": round(time.time() - self.started, 1),
                "results": list(self.results),
            }


class JobManager:
    """后台任务表：批标注 / 视频导入都跑在线程里，前端轮询。"""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, name: str) -> Job:
        job_id = f"job{int(time.time() * 1000) % 10**8}"
        job = Job(job_id, name)
        with self._lock:
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return job

    def remove(self, job_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job.state in ("running", "paused", "canceling"):
                raise HTTPException(400, "任务还在运行，先取消再删除")
            self.jobs.pop(job_id, None)
            return True

    def get(self, job_id: str) -> dict:
        return self.get_job(job_id).snapshot()


def create_app(
    project_path: str | Path,
    preload: bool = True,
    confidence: float | None = None,
    scan: bool = True,
    host: str = "127.0.0.1",
    port: int = 8099,
    serve_flags: list[str] | None = None,
) -> FastAPI:
    proj = Project.open(project_path)
    if scan:
        proj.scan()  # --no-scan 时启动不往清单里加任何新文件
    app = FastAPI(title="SAM3 标注复核", docs_url="/api/docs")
    jobs = JobManager()
    gpu_lock = threading.Lock()
    overlay_cache: dict[tuple, bytes] = {}
    undo_buf: dict[str, dict] = {}  # image_id -> 最近删除的实例（撤销用）
    decode_cache: dict[str, tuple[str, list]] = {}  # image_id -> (版本, [(id, 类别, 掩码)])
    labelmap_cache: dict[tuple, bytes] = {}  # (id, 版本, 过滤条件, 尺寸) -> PNG

    def build_labelmap(items: list, width: int, height: int, skip_ids: set, skip_classes: set) -> np.ndarray:
        """把实例 id 编进 RGB 通道，用于前端按像素做命中测试。

        掩码有重叠时，面积小的后写（覆盖面积大的），
        于是"点在重叠处"会选中更内层/更小的那个实例。
        """
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        order = sorted(items, key=lambda t: int(t[2].sum()), reverse=True)
        for inst_id, cname, mask in order:
            if inst_id in skip_ids or cname in skip_classes:
                continue
            rows = np.any(mask, axis=1)
            if not rows.any():
                continue
            cols = np.any(mask, axis=0)
            y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
            x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
            roi = arr[y0 : y1 + 1, x0 : x1 + 1]
            m = mask[y0 : y1 + 1, x0 : x1 + 1]
            roi[m] = (
                inst_id & 255,
                (inst_id >> 8) & 255,
                (inst_id >> 16) & 255,
            )
        return arr

    def decoded_masks(entry, annot: dict) -> list:
        version = annot.get("updated", "")
        hit = decode_cache.get(entry.id)
        if hit and hit[0] == version:
            return hit[1]
        height = annot.get("height") or entry.height
        width = annot.get("width") or entry.width
        items = [
            (i.get("id"), i.get("class_name") or "?", pp.decode_mask(i["segmentation"], height, width))
            for i in annot.get("instances", [])
            if i.get("kept", True)
        ]
        decode_cache[entry.id] = (version, items)
        while len(decode_cache) > 2:  # 只留最近两张图的解码结果
            decode_cache.pop(next(iter(decode_cache)))
        return items
    thumb_dir = proj.root / ".cache" / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    state: dict = {"engine": None, "error": None, "load_seconds": None}

    def refresh_manifest() -> None:
        """服务运行期间 CLI 可能改过清单/类别：按 images.json 的 mtime 重新加载。"""
        path = proj.root / "images.json"
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if state.get("manifest_mtime") == mtime:
            return
        state["manifest_mtime"] = mtime
        fresh = Project.open(proj.root)
        proj.images = fresh.images
        proj.meta = fresh.meta

    def get_engine():
        if state["engine"] is None:
            from .engine import SAM3Engine

            if state["error"]:
                raise HTTPException(500, f"模型加载失败: {state['error']}")
            eng = SAM3Engine(confidence=confidence)
            state["engine"] = eng
            state["load_seconds"] = round(eng.load_seconds, 2)
        return state["engine"]

    if preload:
        try:
            get_engine()
        except Exception as exc:  # pragma: no cover
            state["error"] = str(exc)

    # ------------------------------------------------------------- 辅助
    def entry_or_404(image_id: str):
        refresh_manifest()
        try:
            return proj.get(image_id)
        except KeyError:
            raise HTTPException(404, "图片不存在")

    def load_or_init(image_id: str) -> dict:
        entry = entry_or_404(image_id)
        annot = proj.load_annotation(entry)
        if annot is None:
            annot = {
                "image_id": entry.id,
                "path": entry.relpath,
                "width": entry.width,
                "height": entry.height,
                "classes": proj.classes,
                "status": entry.status,
                "params": dict(proj.meta.get("settings", {})),
                "instances": [],
                "next_id": 1,
            }
        annot.setdefault("next_id", len(annot.get("instances", [])) + 1)
        return annot

    def public_annotation(annot: dict) -> dict:
        """去掉 RLE，只留前端需要的字段（掩码走 PNG 接口）。"""
        out = {k: v for k, v in annot.items() if k != "instances"}
        out["instances"] = [
            {
                "id": i.get("id"),
                "class_name": i.get("class_name"),
                "score": i.get("score"),
                "box": i.get("box"),
                "area": i.get("area"),
                "source": i.get("source"),
                "edited": bool(i.get("edited")),
                "kept": i.get("kept", True),
            }
            for i in annot.get("instances", [])
        ]
        return out

    def bump_cache(image_id: str) -> None:
        overlay_cache.pop(image_id, None)

    def save(image_id: str, annot: dict, status: str | None = None) -> None:
        entry = entry_or_404(image_id)
        if status:
            annot["status"] = status
        proj.save_annotation(entry, annot)
        bump_cache(image_id)

    def render_masks(
        annot: dict,
        width: int,
        height: int,
        highlight: int | None = None,
        items: list | None = None,
        skip_ids: set | None = None,
        skip_classes: set | None = None,
    ) -> np.ndarray:
        """渲一张 RGBA 掩码叠加层。

        highlight 指定某个实例 id 时：该实例提亮 + 白色粗描边，
        其余实例压暗，方便在密集场景里一眼认出选中的是哪一个。
        """
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        classes = proj.classes
        shown = 0
        dimmed = highlight is not None
        if items is None:
            items = [
                (i.get("id"), i.get("class_name") or "?", pp.decode_mask(i["segmentation"], height, width))
                for i in annot.get("instances", [])
                if i.get("kept", True)
            ]
        skip_ids = skip_ids or set()
        skip_classes = skip_classes or set()
        for inst_id, cname, mask in items:
            if inst_id in skip_ids or cname in skip_classes:
                continue
            idx = classes.index(cname) if cname in classes else len(classes)
            custom = cname in proj.class_colors
            color = proj.color_for_class(cname, idx)  # 支持每类自定义颜色
            # 自定义颜色不加明度抖动，保证和用户选的颜色一致
            jitter = 0 if custom else (shown % 3) * 18 - 18
            base = tuple(int(np.clip(c + jitter, 24, 255)) for c in color)
            is_hi = highlight is not None and inst_id == highlight

            rows = np.any(mask, axis=1)
            if not rows.any():
                continue
            cols = np.any(mask, axis=0)
            y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
            x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
            ys, ye = max(0, y0 - 2), min(height, y1 + 3)
            xs, xe = max(0, x0 - 2), min(width, x1 + 3)

            # 只在实例的外接框内做合成，避免整图 4 通道花式索引（那是最慢的一步）
            roi = rgba[ys:ye, xs:xe]
            m = mask[ys:ye, xs:xe]
            m_erode = _erode(m)

            if is_hi:
                fill = tuple(int(np.clip(c * 0.50 + 255 * 0.50, 0, 255)) for c in base)
                roi[m] = (fill[0], fill[1], fill[2], 200)
                edge = _dilate(_dilate(m)) & ~m_erode  # 3px 白边，缩到 88% 也看得清
                roi[edge] = (255, 255, 255, 255)
            else:
                roi[m] = (base[0], base[1], base[2], 46 if dimmed else 78)
                edge = m & ~m_erode
                ec = (
                    int(np.clip(base[0] + 50, 0, 255)),
                    int(np.clip(base[1] + 50, 0, 255)),
                    int(np.clip(base[2] + 50, 0, 255)),
                )
                roi[edge] = (ec[0], ec[1], ec[2], 120 if dimmed else 215)
            shown += 1
        return rgba

    # ------------------------------------------------------------- 基础接口
    @app.get("/api/state")
    def api_state():
        refresh_manifest()
        stats = proj.stats()
        engine = state["engine"]
        return {
            "project": stats,
            "classes": proj.classes,
            "settings": proj.meta.get("settings", {}),
            "class_colors": proj.class_colors,
            "image_dir": str(proj.image_dir()),
            "engine": (engine.info() if engine is not None else None),
            "load_error": state["error"],
        }

    @app.get("/api/projects")
    def api_projects():
        """列出 data/projects 下的所有项目（供界面切换）。"""
        refresher = None
        root = config.PROJECTS_DIR
        out = []
        if root.exists():
            for d in sorted(root.iterdir(), key=lambda x: x.name.lower()):
                if not (d / "project.json").exists():
                    continue
                try:
                    meta = json.loads((d / "project.json").read_text(encoding="utf-8"))
                    images_path = d / "images.json"
                    n_img = 0
                    if images_path.exists():
                        payload = json.loads(images_path.read_text(encoding="utf-8"))
                        n_img = len(payload.get("images", []))
                except Exception:
                    continue
                out.append(
                    {
                        "name": meta.get("name") or d.name,
                        "dir": str(d.resolve()),
                        "rel": f"data/projects/{d.name}",
                        "images": n_img,
                        "current": d.resolve() == proj.root.resolve(),
                    }
                )
        return {"projects": out, "current": str(proj.root.resolve())}

    def restart_with(target: Path) -> str:
        """后台重启本服务，指向 target 项目；返回命令行（便于回显）。"""
        import os
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "sam3label.cli", "serve",
            "--project", str(target),
            "--host", host,
            "--port", str(port),
        ] + list(serve_flags or [])

        def restart() -> None:
            time.sleep(0.6)
            log_path = config.ROOT / "logs" / "serve.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "ab") as log:
                subprocess.Popen(
                    cmd, cwd=str(config.ROOT), stdout=log, stderr=log, start_new_session=True
                )
            os._exit(0)

        threading.Thread(target=restart, daemon=True).start()
        return " ".join(cmd)

    reg_new = {"route": True}

    @app.post("/api/project/create")
    def api_project_create(
        name: str = Body(..., embed=True),
        source_dir: str = Body("", embed=True),
        classes: list[str] | None = Body(None, embed=True),
        do_copy: bool = Body(False, embed=True),
        recursive: bool = Body(False, embed=True),
        switch: bool = Body(True, embed=True),
    ):
        """从文件夹新建一个项目（可选复制图片），建好后可直接切过去。"""
        safe = "".join(c for c in (name or "").strip() if c not in "/\\:*?\"'<>|").strip()
        if not safe:
            raise HTTPException(400, "项目名不能为空")
        target = (config.PROJECTS_DIR / safe).resolve()
        if (target / "project.json").exists():
            raise HTTPException(400, f"项目已存在：{target}")
        target.mkdir(parents=True, exist_ok=True)
        proj_new = Project.create(
            target, name=safe, classes=classes or [], image_dir=target / "images"
        )
        added = 0
        missing: list[str] = []
        if source_dir.strip():
            src = Path(source_dir).expanduser()
            if not src.exists():
                raise HTTPException(400, f"文件夹不存在：{src}")
            res = proj_new.add_external(
                [src],
                copy_to=(target / "images") if do_copy else None,
                recursive=recursive,
            )
            added = len(res["added"])
            missing = res["missing"]
        state.pop("manifest_mtime", None)
        cmd = restart_with(target) if switch else None
        return {
            "ok": True,
            "project": safe,
            "dir": str(target),
            "images": len(proj_new.images),
            "added": added,
            "missing": missing,
            "restarting": bool(switch),
            "cmd": cmd,
        }

    @app.post("/api/project/switch")
    def api_switch_project(path: str = Body(..., embed=True)):
        """切换到另一个项目：后台重启本服务指向新项目，前端轮询后自动刷新。"""
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = (config.ROOT / target)
        target = target.resolve()
        if not (target / "project.json").exists():
            raise HTTPException(400, f"不是有效的项目目录：{target}")
        if target == proj.root.resolve():
            return {"ok": True, "restarting": False, "project": str(target)}
        cmd = restart_with(target)
        return {"ok": True, "restarting": True, "project": str(target), "cmd": cmd}

    @app.get("/api/stats")
    def api_stats():
        refresh_manifest()
        return proj.stats()

    @app.post("/api/thresholds")
    def api_set_thresholds(conf: dict[str, float] = Body(..., embed=True)):
        """保存每个类别的置信度阈值。"""
        refresh_manifest()
        proj.set_class_conf(conf)
        return {"class_conf": proj.class_conf, "global": proj.meta.get("settings", {}).get("confidence")}

    @app.post("/api/colors")
    def api_set_colors(colors: dict[str, str] = Body(..., embed=True)):
        """保存每个类别的显示颜色（#rrggbb）。"""
        refresh_manifest()
        proj.set_class_colors(colors)
        for key in list(overlay_cache):
            overlay_cache.pop(key, None)
        return {"class_colors": proj.class_colors}

    @app.post("/api/refilter")
    def api_refilter(
        apply: bool = Body(False, embed=True),
    ):
        """按类别阈值重筛已有实例（apply=false 只预览影响）。"""
        refresh_manifest()
        stats = proj.refilter(apply=apply)
        changed = sum(v["changed"] for v in stats.values())
        if apply:
            for key in list(overlay_cache):
                overlay_cache.pop(key, None)
            for key in list(labelmap_cache):
                labelmap_cache.pop(key, None)
            decode_cache.clear()
            state.pop("manifest_mtime", None)
        return {"apply": apply, "classes": stats, "changed": changed}

    @app.post("/api/classes")
    def api_set_classes(classes: list[str] = Body(..., embed=True)):
        proj.set_classes([c.strip() for c in classes if c.strip()])
        return {"classes": proj.classes}

    @app.get("/api/images")
    def api_images(
        status: str = Query("all"),
        q: str = Query(""),
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=2000),
    ):
        refresh_manifest()
        items = proj.images
        if status != "all":
            items = [i for i in items if i.status == status]
        if q:
            items = [i for i in items if q.lower() in i.relpath.lower()]
        total = len(items)
        # 分组统计基于"过滤后的全部图片"，不受分页影响
        gstats: dict[str, dict] = {}
        for i in items:
            key = i.group_path or ""
            g = gstats.setdefault(
                key,
                {
                    "key": key,
                    "name": i.group or "",
                    "kind": "video" if key.startswith("video:") else ("folder" if key else ""),
                    "count": 0,
                },
            )
            g["count"] += 1
        page = items[offset : offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "groups": list(gstats.values()),
            "items": [
                {
                    "id": i.id,
                    "relpath": i.relpath,
                    "status": i.status,
                    "n_instances": i.n_instances,
                    "width": i.width,
                    "height": i.height,
                    "group": i.group,
                    "group_path": i.group_path,
                    "group_kind": "video" if i.group_path.startswith("video:") else ("folder" if i.group_path else ""),
                    "thumb": f"/media/thumb/{i.id}",
                }
                for i in page
            ],
        }

    @app.get("/api/images/{image_id}")
    def api_image(image_id: str):
        refresh_manifest()
        entry = entry_or_404(image_id)
        annot = load_or_init(image_id)
        body = public_annotation(annot)
        body["relpath"] = entry.relpath
        body["image_url"] = f"/media/image/{image_id}"
        body["overlay_url"] = f"/media/overlay/{image_id}?v={annot.get('updated', '')}"
        return body

    @app.post("/api/images/{image_id}/classes")
    def api_image_classes(image_id: str, classes: list[str] = Body(..., embed=True)):
        """把类别清单写到当前这张图的标注里（前端"保存类别"用）。"""
        annot = load_or_init(image_id)
        annot["classes"] = [c.strip() for c in classes if c and c.strip()]
        save(image_id, annot)
        return {"classes": annot["classes"]}

    @app.post("/api/images/{image_id}/status")
    def api_status(image_id: str, status: str = Body(..., embed=True)):
        if status not in {"pending", "done", "reviewed"}:
            raise HTTPException(400, "非法状态")
        annot = load_or_init(image_id)
        save(image_id, annot, status=status)
        return {"status": status}

    # ------------------------------------------------------------- 推理
    @app.post("/api/images/{image_id}/segment")
    def api_segment(
        image_id: str,
        classes: list[str] | None = Body(None, embed=True),
        confidence: float | None = Body(None, embed=True),
        min_area: int | None = Body(None, embed=True),
        replace: bool = Body(True, embed=True),
    ):
        """用一组类别提示词重跑整图（默认覆盖旧结果）。"""
        entry = entry_or_404(image_id)
        annot = load_or_init(image_id)
        prompt_list = [c.strip() for c in (classes or proj.classes) if c.strip()]
        if not prompt_list:
            raise HTTPException(400, "没有类别提示词")
        settings = proj.meta.get("settings", {})
        conf = confidence if confidence is not None else settings.get("confidence", config.CONFIDENCE)
        marea = min_area if min_area is not None else settings.get("min_area", config.MIN_AREA)
        with gpu_lock:
            engine = get_engine()
            image = _load_rgb(proj.image_path(entry))
            st = engine.set_image(image, key=entry.id)
            result = engine.segment_text(
                st,
                prompt_list,
                confidence=conf,
                min_area=marea,
                dedup_iou=settings.get("dedup_iou", config.DEDUP_IOU),
                class_conf=proj.class_conf,
            )
        next_id = 1 if replace else annot.get("next_id", 1)
        instances = [] if replace else list(annot.get("instances", []))
        for inst in result.instances:
            rec = {"id": next_id, "kept": True, **inst.to_record()}
            instances.append(rec)
            next_id += 1
        annot.update(
            {
                "width": image.width,
                "height": image.height,
                "classes": prompt_list,
                "instances": instances,
                "next_id": next_id,
                "params": {
                    "confidence": conf,
                    "min_area": marea,
                    "dedup_iou": settings.get("dedup_iou", config.DEDUP_IOU),
                },
                "status": "done",
            }
        )
        if prompt_list:
            merged = list(proj.classes)
            for c in prompt_list:
                if c not in merged:
                    merged.append(c)
            if merged != proj.classes:
                proj.set_classes(merged)
        save(image_id, annot)
        return {"added": result.count, "total": len(instances), "timings": result.timings}

    @app.post("/api/images/{image_id}/prompt")
    def api_prompt(
        image_id: str,
        text: str = Body(..., embed=True),
        confidence: float | None = Body(None, embed=True),
        min_area: int = Body(0, embed=True),
    ):
        """追加一个文本提示词的结果（不清空已有实例）。"""
        entry = entry_or_404(image_id)
        annot = load_or_init(image_id)
        with gpu_lock:
            engine = get_engine()
            image = _load_rgb(proj.image_path(entry))
            st = engine.set_image(image, key=entry.id)
            result = engine.segment_text(
                st, [text], confidence=confidence, min_area=min_area, dedup_iou=1.01,
                class_conf=proj.class_conf,
            )
        next_id = annot.get("next_id", 1)
        for inst in result.instances:
            annot.setdefault("instances", []).append(
                {"id": next_id, "kept": True, **inst.to_record()}
            )
            next_id += 1
        annot["next_id"] = next_id
        if text not in proj.classes:
            proj.set_classes(proj.classes + [text])
        save(image_id, annot)
        return {"added": result.count, "next_id": next_id}

    @app.post("/api/images/{image_id}/box_prompt")
    def api_box_prompt(
        image_id: str,
        box: list[float] = Body(..., embed=True),
        class_name: str = Body("object", embed=True),
        text_hint: str | None = Body(None, embed=True),
        confidence: float | None = Body(None, embed=True),
    ):
        entry = entry_or_404(image_id)
        annot = load_or_init(image_id)
        image = _load_rgb(proj.image_path(entry))
        with gpu_lock:
            engine = get_engine()
            st = engine.set_image(image, key=entry.id)
            cc = proj.class_conf
            thr = float(cc.get(text_hint or class_name, confidence or 0.5))
            result = engine.segment_with_box(
                st,
                box,
                (image.width, image.height),
                class_name=class_name,
                text_hint=text_hint,
                confidence=thr,
            )
        next_id = annot.get("next_id", 1)
        for inst in result.instances:
            annot.setdefault("instances", []).append(
                {"id": next_id, "kept": True, **inst.to_record()}
            )
            next_id += 1
        annot["next_id"] = next_id
        save(image_id, annot)
        return {"added": result.count}

    # ------------------------------------------------------------- 编辑
    @app.post("/api/images/{image_id}/instances/{inst_id}")
    def api_edit_instance(
        image_id: str,
        inst_id: int,
        kept: bool | None = Body(None, embed=True),
        class_name: str | None = Body(None, embed=True),
    ):
        annot = load_or_init(image_id)
        found = None
        for inst in annot.get("instances", []):
            if inst.get("id") == inst_id:
                found = inst
                break
        if found is None:
            raise HTTPException(404, "实例不存在")
        if kept is not None:
            found["kept"] = bool(kept)
            found["manual"] = True  # 人工决定，重筛时跳过
        if class_name:
            found["class_name"] = class_name
            if class_name not in proj.classes:
                proj.set_classes(proj.classes + [class_name])
        save(image_id, annot)
        return {"ok": True}

    @app.delete("/api/images/{image_id}/instances/{inst_id}")
    def api_delete_instance(image_id: str, inst_id: int):
        annot = load_or_init(image_id)
        before = len(annot.get("instances", []))
        removed = [i for i in annot.get("instances", []) if i.get("id") == inst_id]
        annot["instances"] = [
            i for i in annot.get("instances", []) if i.get("id") != inst_id
        ]
        if len(annot["instances"]) == before:
            raise HTTPException(404, "实例不存在")
        undo_buf[image_id] = removed[0]  # 只保留最近一次删除，供 Ctrl+Z 撤销
        save(image_id, annot)
        return {"ok": True, "undo_available": True, "class_name": removed[0].get("class_name")}

    @app.post("/api/images/{image_id}/undelete")
    def api_undelete_instance(image_id: str):
        """撤销上一次删除（Ctrl+Z）。"""
        rec = undo_buf.pop(image_id, None)
        if rec is None:
            raise HTTPException(404, "没有可撤销的删除")
        annot = load_or_init(image_id)
        next_id = annot.get("next_id", len(annot.get("instances", [])) + 1)
        rec = dict(rec)
        rec["id"] = next_id
        annot.setdefault("instances", []).append(rec)
        annot["next_id"] = next_id + 1
        save(image_id, annot)
        return {"ok": True, "id": next_id}

    @app.post("/api/images/{image_id}/clear")
    def api_clear(image_id: str):
        annot = load_or_init(image_id)
        annot["instances"] = []
        annot["next_id"] = 1
        annot["status"] = "pending"
        save(image_id, annot)
        return {"ok": True}

    # ------------------------------------------------------------- 掩码精修
    @app.get("/api/images/{image_id}/labelmap")
    def api_labelmap(
        image_id: str,
        hide: list[str] = Query(default=[]),
        hcls: list[str] = Query(default=[]),
        max_size: int = Query(2400, ge=0, le=12000),
    ):
        """实例 id 标签图（RGB 编码 id= r + g*256 + b*65536），前端按像素命中。

        过滤条件和 overlay 一致：视图层隐藏的实例/类别不会出现在标签图里。
        """
        entry = entry_or_404(image_id)
        annot = proj.load_annotation(entry)
        if not annot:
            raise HTTPException(404, "还没有标注")
        skip_ids = set()
        for raw in hide:
            for part in raw.split(","):
                part = part.strip()
                if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
                    skip_ids.add(int(part))
        skip_classes = {c.strip() for raw in hcls for c in raw.split(",") if c.strip()}
        width = annot.get("width") or entry.width
        height = annot.get("height") or entry.height
        key = (
            image_id,
            annot.get("updated", ""),
            tuple(sorted(skip_ids)),
            tuple(sorted(skip_classes)),
            max_size,
        )
        cached = labelmap_cache.get(key)
        if cached is not None:
            return Response(cached, media_type="image/png")
        items = decoded_masks(entry, annot)
        arr = build_labelmap(items, width, height, skip_ids, skip_classes)
        img = Image.fromarray(arr, mode="RGB")
        if max_size and max(width, height) > max_size:
            scale = max_size / max(width, height)
            img = img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        data = buf.getvalue()
        labelmap_cache[key] = data
        while len(labelmap_cache) > 8:
            labelmap_cache.pop(next(iter(labelmap_cache)))
        return Response(data, media_type="image/png")

    @app.get("/api/images/{image_id}/instances/{inst_id}/mask")
    def api_instance_mask(
        image_id: str,
        inst_id: int,
        color: str = Query("255,255,255"),
        alpha: int = Query(255, ge=0, le=255),
        max_size: int = Query(0, ge=0, le=12000),
    ):
        """单独取一个实例的掩码（RGBA，掩码外透明），画笔精修/高亮用。"""
        entry = entry_or_404(image_id)
        annot = proj.load_annotation(entry)
        if not annot:
            raise HTTPException(404, "还没有标注")
        inst = next((i for i in annot.get("instances", []) if i.get("id") == inst_id), None)
        if inst is None:
            raise HTTPException(404, "实例不存在")
        height = annot.get("height") or entry.height
        width = annot.get("width") or entry.width
        mask = pp.decode_mask(inst["segmentation"], height, width)
        if max_size and max(width, height) > max_size:
            scale = max_size / max(width, height)
            mask = _resize_mask(mask, max(1, round(width * scale)), max(1, round(height * scale)))
        try:
            rgb = tuple(int(round(float(x))) for x in color.split(",")[:3])
            if len(rgb) != 3:
                raise ValueError
            rgb = tuple(min(255, max(0, v)) for v in rgb)
        except ValueError:
            rgb = (255, 255, 255)
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = alpha
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
        return Response(buf.getvalue(), media_type="image/png")

    @app.post("/api/images/{image_id}/instances/{inst_id}/mask")
    def api_edit_mask(image_id: str, inst_id: int, payload: dict = Body(...)):
        """画笔精修：payload = {w, h, add: PNG dataURL, sub: PNG dataURL}。

        前端把涂抹层以 PNG dataURL 传上来，服务端缩放到原图分辨率后
        做 mask |= add、mask &= ~sub，再重算面积/外框并落盘。
        """
        entry = entry_or_404(image_id)
        annot = load_or_init(image_id)
        inst = next((i for i in annot.get("instances", []) if i.get("id") == inst_id), None)
        if inst is None:
            raise HTTPException(404, "实例不存在")
        height = annot.get("height") or entry.height
        width = annot.get("width") or entry.width
        mask = pp.decode_mask(inst["segmentation"], height, width)

        add_png = payload.get("add")
        sub_png = payload.get("sub")
        if not add_png and not sub_png:
            raise HTTPException(400, "没有涂抹内容")
        bw = int(payload.get("w") or width)
        bh = int(payload.get("h") or height)

        if add_png:
            layer = _decode_mask_png(add_png, bw, bh, width, height)
            mask = mask | layer
        if sub_png:
            layer = _decode_mask_png(sub_png, bw, bh, width, height)
            mask = mask & ~layer
        if not mask.any():
            raise HTTPException(400, "掩码被擦空了；如果不要这个实例，请直接删除它")

        inst["segmentation"] = pp.encode_mask(mask)
        inst["area"] = int(mask.sum())
        inst["box"] = pp.bbox_from_mask(mask)
        inst["source"] = (inst.get("source") or "text") + "+brush" if "brush" not in str(inst.get("source")) else inst.get("source")
        inst["edited"] = True
        save(image_id, annot)
        return {
            "ok": True,
            "area": inst["area"],
            "box": inst["box"],
        }

    # ------------------------------------------------------------- 批处理
    @app.post("/api/annotate")
    def api_annotate(
        classes: list[str] | None = Body(None, embed=True),
        confidence: float | None = Body(None, embed=True),
        min_area: int | None = Body(None, embed=True),
        overwrite: bool = Body(False, embed=True),
        status: str = Body("pending", embed=True),
        limit: int = Body(0, embed=True),
    ):
        refresh_manifest()
        prompt_list = [c.strip() for c in (classes or proj.classes) if c.strip()]
        if not prompt_list:
            raise HTTPException(400, "没有类别提示词")
        settings = proj.meta.get("settings", {})
        conf = confidence if confidence is not None else settings.get("confidence", config.CONFIDENCE)
        marea = min_area if min_area is not None else settings.get("min_area", config.MIN_AREA)
        targets = [i for i in proj.images if status == "all" or i.status == status]
        if not overwrite:
            targets = [i for i in targets if not proj.annotation_path(i).exists()]
        if limit:
            targets = targets[:limit]
        if not targets:
            raise HTTPException(400, "没有待处理的图片")
        job = jobs.create(f"批量标注 {len(targets)} 张")
        job.set_done(0, len(targets))

        def worker() -> None:
            done = 0
            total_inst = 0
            try:
                if gpu_lock.locked():
                    job.set_progress("排队中…（等前一个 GPU 任务结束）")
                with gpu_lock:
                    engine = get_engine()
                    for idx, entry in enumerate(targets, 1):
                        job.checkpoint()
                        try:
                            image = _load_rgb(proj.image_path(entry))
                            st = engine.set_image(image, key=entry.id)
                            result = engine.segment_text(
                                st,
                                prompt_list,
                                confidence=conf,
                                min_area=marea,
                                dedup_iou=settings.get("dedup_iou", config.DEDUP_IOU),
                                class_conf=proj.class_conf,
                            )
                            annot = {
                                "image_id": entry.id,
                                "path": entry.relpath,
                                "width": image.width,
                                "height": image.height,
                                "classes": prompt_list,
                                "status": "done",
                                "params": {
                                    "confidence": conf,
                                    "min_area": marea,
                                    "dedup_iou": settings.get("dedup_iou", config.DEDUP_IOU),
                                },
                                "instances": [
                                    {"id": i + 1, "kept": True, **inst.to_record()}
                                    for i, inst in enumerate(result.instances)
                                ],
                                "next_id": result.count + 1,
                            }
                            proj.save_annotation(entry, annot)
                            bump_cache(entry.id)
                            done += 1
                            total_inst += result.count
                            job.set_done(done, len(targets))
                            job.set_progress(f"{entry.relpath} · {result.count} 实例 · 累计 {total_inst}")
                        except JobCanceled:
                            raise
                        except Exception as exc:
                            job.set_progress(f"{entry.relpath} 失败: {exc}")
                merged = list(proj.classes)
                for c in prompt_list:
                    if c not in merged:
                        merged.append(c)
                if merged != proj.classes:
                    proj.set_classes(merged)
                job.set_done(done, len(targets))
                job.set_progress(f"完成 {done} 张，{total_inst} 实例", 1.0)
                with job._lock:
                    job.state = "finished"
            except JobCanceled:
                job.set_progress(f"已取消（完成 {done} 张，{total_inst} 实例）")
                with job._lock:
                    job.state = "canceled"
            except Exception as exc:  # pragma: no cover
                job.set_progress(str(exc))
                with job._lock:
                    job.state = "failed"

        threading.Thread(target=worker, daemon=True).start()
        return {"job_id": job.id, "total": len(targets)}

    @app.get("/api/video/probe")
    def api_video_probe(path: str = Query(..., description="视频文件路径")):
        """读取视频基本信息（分辨率/帧率/时长），前端选完视频先看一眼。"""
        from .video import probe_video

        p = Path(path).expanduser()
        if not p.exists():
            raise HTTPException(404, f"文件不存在：{p}")
        if p.suffix.lower() not in config.VIDEO_EXTS:
            raise HTTPException(400, f"不是支持的视频格式：{p.suffix}")
        try:
            return probe_video(p)
        except Exception as exc:
            raise HTTPException(400, f"读取失败：{exc}")

    @app.post("/api/video")
    def api_video(
        path: str = Body(..., embed=True),
        classes: list[str] | None = Body(None, embed=True),
        fps: float = Body(2.0, embed=True),
        max_frames: int = Body(300, embed=True),
        max_size: int = Body(1600, embed=True),
        start: float = Body(0.0, embed=True),
        end: float | None = Body(None, embed=True),
        mode: str = Body("track", embed=True),
        confidence: float | None = Body(None, embed=True),
        min_area: int | None = Body(None, embed=True),
        annotate: bool = Body(True, embed=True),
        chunk_size: int = Body(0, embed=True),
    ):
        """导入视频：抽帧 + 自动标注（后台任务，进度走 /api/jobs/{id}）。"""
        refresh_manifest()
        video_path = Path(path).expanduser()
        if not video_path.exists():
            raise HTTPException(400, f"视频不存在：{video_path}")
        settings = proj.meta.get("settings", {})
        prompt_list = [c.strip() for c in (classes or proj.classes) if c.strip()]
        if annotate and not prompt_list:
            raise HTTPException(400, "自动标注需要至少一个类别提示词")
        conf = confidence if confidence is not None else settings.get("confidence", config.CONFIDENCE)
        marea = min_area if min_area is not None else settings.get("min_area", config.MIN_AREA)
        mode = mode if mode in ("track", "frame") else "track"

        job = jobs.create(f"视频 {video_path.name}")
        job.set_progress("准备中…", 0.0)

        def worker() -> None:
            res = None
            try:
                from .video import run_video_import

                if gpu_lock.locked():
                    job.set_progress("排队中…（等前一个 GPU 任务结束）", 0.0)
                with gpu_lock:
                    # 视频模型要占 7~8GB 显存，先把图像模型释放掉，跑完再懒加载回来
                    job.set_progress("释放图像模型显存…", 0.005)
                    job.checkpoint()
                    eng = state.get("engine")
                    state["engine"] = None
                    if eng is not None:
                        del eng
                    import gc as _gc

                    import torch as _torch

                    _gc.collect()
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                        free_b, total_b = _torch.cuda.mem_get_info()
                        need_b = 7.0 * 1024**3
                        if free_b < need_b:
                            raise RuntimeError(
                                f"可用显存只有 {free_b / 1024**3:.1f} GB，"
                                f"视频标注约需 7 GB。请关闭占用显卡的程序（如 UE 仿真），"
                                f"或把「最多帧数 / 长边」调小后重试"
                            )
                        job.set_progress(
                            f"显存可用 {free_b / 1024**3:.1f} GB，开始抽帧…", 0.01
                        )

                    def progress(msg: str, frac: float | None = None) -> None:
                        job.set_progress(msg, frac)

                    res = run_video_import(
                        proj,
                        video_path,
                        prompt_list,
                        fps=fps,
                        max_frames=max_frames,
                        max_size=max_size,
                        start_time=start,
                        end_time=end,
                        mode=mode,
                        confidence=conf,
                        min_area=marea,
                        annotate=annotate,
                        progress=progress,
                        checkpoint=job.checkpoint,
                        class_conf=proj.class_conf,
                        chunk_size=(chunk_size or None),
                    )
                state.pop("manifest_mtime", None)
                errs = res.get("chunk_errors") or []
                job.set_progress(
                    f"完成：{res.get('sampled')} 帧 / {res.get('instances', 0)} 实例 / "
                    f"{res.get('mode')} / {res.get('seconds')}s"
                    + (f"｜{len(errs)} 段因显存不足被跳过（建议提高对应类别阈值或减少帧数）" if errs else ""),
                    1.0,
                )
                with job._lock:
                    job.state = "finished"
                    job.results = [res]
                    job.total = max(1, res.get("sampled", 1))
                    job.done = job.total
            except JobCanceled:
                done_n = (res or {}).get("annotated", 0)
                job.set_progress(f"已取消（已写入 {done_n} 帧）")
                with job._lock:
                    job.state = "canceled"
            except Exception as exc:  # pragma: no cover
                import traceback

                tb = traceback.format_exc()
                print(tb, flush=True)  # 写进服务日志
                msg = f"{type(exc).__name__}: {exc}"
                frames_tb = [ln.strip() for ln in tb.strip().splitlines()[-6:]]
                msg += " | " + " <- ".join(reversed(frames_tb))[:400]
                if "out of memory" in str(exc).lower():
                    msg = (
                        "显存不足（CUDA OOM）：视频标注需要约 7 GB 空闲显存。"
                        "请关闭占用显卡的程序（如 UE 仿真），或把「最多帧数 / 长边」调小后重试。"
                        "已抽出的帧会保留，可以只跑标注。"
                    )
                job.set_progress(msg)
                with job._lock:
                    job.state = "failed"
            finally:
                # 视频模型不常驻：无论成功/失败/取消都释放，避免显存泄漏拖垮下一个任务
                try:
                    from .video import free_video_tracker

                    free_video_tracker()
                except Exception:
                    pass
                import gc as _gc2

                import torch as _torch2

                _gc2.collect()
                if _torch2.cuda.is_available():
                    _torch2.cuda.empty_cache()

        threading.Thread(target=worker, daemon=True).start()
        return {"job_id": job.id, "video": video_path.name}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        return jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/pause")
    def api_job_pause(job_id: str):
        job = jobs.get_job(job_id)
        job.pause()
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/resume")
    def api_job_resume(job_id: str):
        job = jobs.get_job(job_id)
        job.resume()
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/cancel")
    def api_job_cancel(job_id: str):
        job = jobs.get_job(job_id)
        job.cancel()
        return job.snapshot()

    @app.delete("/api/jobs/{job_id}")
    def api_job_remove(job_id: str):
        return {"removed": jobs.remove(job_id)}

    @app.get("/api/jobs")
    def api_jobs():
        with jobs._lock:
            return {"jobs": [j.snapshot() for j in jobs.jobs.values()]}

    @app.post("/api/export")
    def api_export(
        formats: list[str] = Body(default=["coco", "yolo", "masks"], embed=True),
        only_reviewed: bool = Body(False, embed=True),
        out: str | None = Body(None, embed=True),
        groups: list[str] | None = Body(None, embed=True),
        images: list[str] | None = Body(None, embed=True),
    ):
        """导出标注。groups=分组键列表（如 video:xxx），images=勾选的图片 id，用于局部导出。"""
        if out:
            out_path = Path(out).expanduser()
            if not out_path.is_absolute():
                out_path = proj.root / out_path
        else:
            out_path = proj.root / "exports" / f"web-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            out_path.mkdir(parents=True, exist_ok=True)
            probe = out_path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise HTTPException(400, f"导出目录不可写：{out_path} ({exc})")
        entries = None
        if images:
            wanted = set(images)
            entries = [i for i in proj.images if i.id in wanted]
        elif groups:
            entries = proj.entries_of_groups(groups)
        if entries is not None and not entries:
            raise HTTPException(400, "选定范围内没有图片")

        written: dict[str, str] = {}
        if "coco" in formats:
            written["coco"] = str(
                exporters.export_coco(proj, out_path / "instances_coco.json", only_reviewed=only_reviewed, entries=entries)
            )
        if "yolo" in formats:
            written["yolo"] = str(exporters.export_yolo(proj, out_path / "yolo_seg", only_reviewed=only_reviewed, entries=entries))
        if "masks" in formats:
            written["masks"] = str(exporters.export_masks(proj, out_path / "masks", only_reviewed=only_reviewed, entries=entries))
        if "labelme" in formats:
            written["labelme"] = str(exporters.export_labelme(proj, out_path / "labelme", only_reviewed=only_reviewed, entries=entries))
        if "previews" in formats:
            written["previews"] = str(exporters.export_previews(proj, out_path / "previews", only_reviewed=only_reviewed, entries=entries))
        return {"out": str(out_path.resolve()), "written": written}

    # ------------------------------------------------------------- 导入图片
    @app.get("/api/browse")
    def api_browse(dir: str = Query("~")):
        """给"添加图片文件夹"用的极简目录浏览（只列子目录 + 可识别的图片数）。"""
        base = Path(dir).expanduser()
        if not base.exists():
            base = Path.home()
        base = base.resolve()
        try:
            entries = sorted(
                [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")],
                key=lambda p: p.name.lower(),
            )
        except PermissionError:
            raise HTTPException(403, f"没有权限读取 {base}")
        dirs = [
            {"name": p.name, "path": str(p)} for p in entries if not p.is_symlink() or p.exists()
        ]
        imgs = sorted(
            (p for p in base.iterdir() if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS),
            key=lambda p: p.name.lower(),
        )
        vids = sorted(
            (p for p in base.iterdir() if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS),
            key=lambda p: p.name.lower(),
        )
        return {
            "path": str(base),
            "parent": str(base.parent) if base.parent != base else "",
            "dirs": dirs[:400],
            # 只列当前层的图片（不进子目录），前端可逐张勾选
            "n_images": len(imgs),
            "images": [
                {"name": p.name, "path": str(p), "size": p.stat().st_size}
                for p in imgs[:800]
            ],
            "n_videos": len(vids),
            "videos": [
                {"name": p.name, "path": str(p), "size": p.stat().st_size}
                for p in vids[:400]
            ],
        }

    @app.post("/api/import")
    def api_import(
        paths: list[str] = Body(..., embed=True),
        do_copy: bool = Body(False, embed=True),
        dest: str | None = Body(None, embed=True),
        recursive: bool = Body(False, embed=True),
    ):
        """把本机路径（文件夹或图片）加入工作区。

        传文件夹时只取该层直接包含的图片（recursive=true 才递归子目录）；
        可以一次传多个路径（单张图片/多张图片/文件夹混合）。
        do_copy=false：只登记绝对路径，原图不动、不占额外磁盘；
        do_copy=true ：复制到 dest（默认 <project>/images）。
        """
        refresh_manifest()
        if not paths:
            raise HTTPException(400, "没有给出路径")
        copy_to = None
        if do_copy:
            copy_to = Path(dest).expanduser() if dest else (proj.root / "images")
            if not copy_to.is_absolute():
                copy_to = proj.root / copy_to
        res = proj.add_external(paths, copy_to=copy_to, recursive=recursive)
        state.pop("manifest_mtime", None)  # 强制下次刷新
        return {
            "added": len(res["added"]),
            "skipped": len(res["skipped"]),
            "missing": res["missing"],
            "bad": res["bad"],
            "total": len(proj.images),
        }

    @app.post("/api/prune")
    def api_prune():
        """清理原图已丢失的条目。"""
        refresh_manifest()
        removed = proj.prune_missing()
        state.pop("manifest_mtime", None)
        return {"removed": len(removed), "names": removed[:50], "total": len(proj.images)}

    @app.post("/api/images/remove")
    def api_remove_images(
        ids: list[str] = Body(..., embed=True),
        delete_files: bool = Body(False, embed=True),
    ):
        """批量移除勾选的图片。

        delete_files=false：只移除清单条目 + 标注，磁盘文件保留（并记入 excluded 防止被重新扫回）；
        delete_files=true ：项目目录内的副本一并删除；**项目外的原图不删**，只解除引用。
        """
        refresh_manifest()
        if not ids:
            raise HTTPException(400, "没有勾选图片")
        # 保护：某个视频任务正在跑时，不允许删它的帧（否则任务会读不到文件而失败）
        running = [j for j in jobs.jobs.values() if j.state in ("running", "paused", "canceling")]
        if running:
            wanted = {i.group for i in proj.images if i.id in set(ids) and i.group}
            for j in running:
                for name in wanted:
                    if name and name in j.name:
                        raise HTTPException(
                            400,
                            f"「{name}」的视频任务还在跑（{j.name}），等它完成或先取消，再删除这些帧",
                        )
        res = proj.remove_images(ids, delete_files=delete_files)
        # 把这些 id 从各种缓存里踢掉
        for key in list(overlay_cache):
            if key[0] in set(ids):
                overlay_cache.pop(key, None)
        for key in list(labelmap_cache):
            if key[0] in set(ids):
                labelmap_cache.pop(key, None)
        for iid in ids:
            decode_cache.pop(iid, None)
        state.pop("manifest_mtime", None)
        res["kept_external"] = res.get("kept_external", [])[:50]
        return res

    @app.post("/api/upload")
    async def api_upload(files: list[UploadFile] = File(...)):
        """浏览器选择文件夹/多张图片后上传，落到 <project>/images 并登记。"""
        refresh_manifest()
        dest = proj.root / "images"
        dest.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for f in files:
            name = Path(f.filename or "upload").name
            if Path(name).suffix.lower() not in config.IMAGE_EXTS:
                continue
            target = dest / name
            n = 1
            while target.exists():
                target = dest / f"{Path(name).stem}_{n}{Path(name).suffix}"
                n += 1
            with open(target, "wb") as fh:
                while chunk := await f.read(1 << 20):
                    fh.write(chunk)
            saved.append(str(target))
        if not saved:
            raise HTTPException(400, "没有可识别的图片文件")
        res = proj.add_external(saved, copy_to=None)
        state.pop("manifest_mtime", None)
        return {"added": len(res["added"]), "total": len(proj.images)}


    # ------------------------------------------------------------- 媒体
    @app.get("/media/image/{image_id}")
    def media_image(image_id: str, max_size: int = Query(2000, ge=200, le=8000)):
        entry = entry_or_404(image_id)
        path = proj.image_path(entry)
        if not path.exists():
            raise HTTPException(404, "原图文件缺失")
        img = _load_rgb(path)
        if max(img.size) > max_size:
            scale = max_size / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.get("/media/thumb/{image_id}")
    def media_thumb(image_id: str, size: int = Query(360, ge=64, le=1024)):
        entry = entry_or_404(image_id)
        thumb_dir.mkdir(parents=True, exist_ok=True)
        cache = thumb_dir / f"{entry.id}_{size}.jpg"
        path = proj.image_path(entry)
        if not path.exists():
            # 引用模式下原图可能被移走：返回占位图，避免界面出现碎图标
            buf = io.BytesIO()
            Image.new("RGB", (size, int(size * 0.75)), (32, 36, 46)).save(buf, format="JPEG", quality=70)
            return Response(buf.getvalue(), media_type="image/jpeg")
        # 源文件比缓存新（视频重新抽帧会复用同名文件）→ 重新生成
        stale = (not cache.exists()) or cache.stat().st_mtime < path.stat().st_mtime
        if stale:
            try:
                img = _load_rgb(path)
                img.thumbnail((size, size))
                img.save(cache, format="JPEG", quality=80)
                try:
                    import os as _os

                    _os.utime(cache, (path.stat().st_atime, path.stat().st_mtime))
                except OSError:
                    pass
            except Exception:
                raise HTTPException(500, "缩略图生成失败")
        return FileResponse(
            cache, media_type="image/jpeg", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/media/overlay/{image_id}")
    def media_overlay(
        image_id: str,
        v: str = Query(""),
        hi: str = Query(""),
        hide: list[str] = Query(default=[]),
        hcls: list[str] = Query(default=[]),
    ):
        """掩码叠加层。

        hi=<实例id> 高亮该实例并把其它实例压暗；
        hide=<实例id>（可重复）/ hcls=<类别名>（可重复）用于视图层隐藏，不影响标注数据。
        """
        entry = entry_or_404(image_id)
        annot = proj.load_annotation(entry)
        if not annot:
            raise HTTPException(404, "还没有标注")
        try:
            highlight = int(hi) if hi else None
        except ValueError:
            highlight = None
        if highlight is not None and not any(
            i.get("id") == highlight and i.get("kept", True)
            for i in annot.get("instances", [])
        ):
            highlight = None  # 实例不存在或被隐藏，回落成普通渲染
        skip_ids = set()
        for raw in hide:
            for part in raw.split(","):
                part = part.strip()
                if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
                    skip_ids.add(int(part))
        skip_classes = {c.strip() for raw in hcls for c in raw.split(",") if c.strip()}
        if highlight in skip_ids or any(
            i.get("id") == highlight and (i.get("class_name") or "?") in skip_classes
            for i in annot.get("instances", [])
        ):
            highlight = None
        key = (
            image_id,
            annot.get("updated", ""),
            highlight,
            tuple(sorted(skip_ids)),
            tuple(sorted(skip_classes)),
        )
        cached = overlay_cache.get(key)
        if cached is not None:
            return Response(cached, media_type="image/png")
        width = annot.get("width") or entry.width
        height = annot.get("height") or entry.height
        items = decoded_masks(entry, annot)
        rgba = render_masks(
            annot, width, height, highlight=highlight, items=items,
            skip_ids=skip_ids, skip_classes=skip_classes,
        )
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=3)
        data = buf.getvalue()
        overlay_cache[key] = data
        while len(overlay_cache) > 16:
            overlay_cache.pop(next(iter(overlay_cache)))
        return Response(data, media_type="image/png")

    # ------------------------------------------------------------- 静态页面
    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        # 给静态资源带上 mtime 版本号，改完前端不用清缓存
        for name in ("app.js", "style.css"):
            f = WEB_DIR / name
            if f.exists():
                ver = str(int(f.stat().st_mtime))
                html = html.replace(f'/static/{name}"', f'/static/{name}?v={ver}"')
        js = WEB_DIR / "app.js"
        if js.exists():
            tag = f'<meta name="app-ver" content="{int(js.stat().st_mtime)}">'
            html = html.replace("</head>", tag + "</head>")
        return HTMLResponse(
            content=html,
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    return app


# --------------------------------------------------------------------- 小工具
def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        img = im.convert("RGB")
        img.load()
    return img


def _erode(mask: np.ndarray) -> np.ndarray:
    """4 邻域腐蚀，只用于画掩码描边。"""
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    """4 邻域膨胀，用于给高亮实例描一圈更粗的白边。"""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """最近邻缩放 bool 掩码。"""
    if (mask.shape[1], mask.shape[0]) == (width, height):
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(img.resize((width, height), Image.NEAREST)) > 127


def _decode_mask_png(data_url: str, bw: int, bh: int, width: int, height: int) -> np.ndarray:
    """把前端传的 mask PNG dataURL 解码成原图分辨率的 bool 掩码。"""
    import base64

    raw = data_url.split(",", 1)[-1]
    try:
        data = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(400, f"涂抹层解码失败: {exc}")
    with Image.open(io.BytesIO(data)) as im:
        arr = np.asarray(im.convert("L")) > 127
    if arr.shape != (bh, bw):
        arr = _resize_mask(arr, bw, bh)
    if arr.shape != (height, width):
        arr = _resize_mask(arr, width, height)
    return arr

