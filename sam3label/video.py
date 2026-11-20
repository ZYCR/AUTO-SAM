"""视频导入 + 自动标注。

两条路子：
  track  —— 用 SAM3 视频模型：在首帧用文本提示找到目标，再跨帧跟踪传播，
            同一目标在所有帧里共用同一个 obj_id（时间上连贯，推荐）
  frame  —— 逐帧当独立图像跑 SAM3 图像模型（简单，但相邻帧的实例 id 不连续，
            目标移动快时容易漏检）

帧抽取统一走 cv2，抽出的 JPEG 帧会注册进项目，后续复核/导出与普通图片完全一致。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- 探测
def probe_video(path: str | Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = (count / fps) if fps > 0 else 0.0
    return {
        "path": str(Path(path).resolve()),
        "fps": round(float(fps), 3),
        "frames": count,
        "width": width,
        "height": height,
        "duration": round(float(duration), 2),
    }


# --------------------------------------------------------------------- 抽帧
def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float | None = None,
    max_frames: int = 300,
    max_size: int = 1600,
    start_time: float = 0.0,
    end_time: float | None = None,
    jpeg_quality: int = 92,
    progress=None,
    checkpoint=None,
) -> dict:
    """按 fps 抽帧存成 JPEG，返回帧列表与元信息。"""
    import cv2

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path)
    src_fps = info["fps"] or 25.0
    step = 1 if not fps or fps <= 0 else max(1, int(round(src_fps / fps)))
    start_frame = max(0, int(round(start_time * src_fps)))
    end_frame = int(round(end_time * src_fps)) if end_time else info["frames"] or 10**9

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video_path}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 预估要抽多少帧，用于进度百分比
    est = int(max(1, min(max_frames, ((end_frame - start_frame) / step) if step else max_frames)))
    frames: list[dict] = []
    idx = start_frame
    scale_applied = None
    t0 = time.perf_counter()
    while True:
        if checkpoint:
            checkpoint()
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= end_frame:
            break
        if (idx - start_frame) % step == 0:
            height, width = frame.shape[:2]
            if max_size and max(width, height) > max_size:
                scale = max_size / max(width, height)
                frame = cv2.resize(
                    frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                scale_applied = round(scale, 4)
            n = len(frames) + 1
            name = f"{n:06d}.jpg"  # <int>.jpg：SAM3 视频加载器按数字排序，不会告警
            out = out_dir / name
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            frames.append(
                {
                    "index": n,
                    "source_frame": idx,
                    "time": round(idx / src_fps, 3),
                    "path": str(out),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                }
            )
            if progress and len(frames) % 5 == 0:
                progress(f"抽帧 {len(frames)} 帧", min(1.0, len(frames) / est) * 0.15)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    meta = {
        **info,
        "sample_fps": round(src_fps / step, 3),
        "step": step,
        "max_size": max_size,
        "scale": scale_applied,
        "n_extracted": len(frames),
        "extract_seconds": round(time.perf_counter() - t0, 2),
        "frames": frames,
    }
    (out_dir.parent / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


# --------------------------------------------------------------------- 跟踪标注
@dataclass
class TrackResult:
    frames: list[dict]        # [{frame_index(1-based), instances:[...]}]
    per_class: dict[str, int]  # 每个类别跟踪到的目标数
    seconds: float
    errors: list[str] = field(default_factory=list)  # 分段失败信息（不致命）


class JobCanceledError(Exception):
    """任务被取消（由上层传入的 checkpoint 抛出，这里只做类型对齐）。"""


_TRACKER: "VideoTracker | None" = None


def get_video_tracker() -> "VideoTracker":
    """全局只保留一个视频模型实例（显存贵，绝不重复加载）。"""
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = VideoTracker()
    return _TRACKER


def free_video_tracker() -> None:
    global _TRACKER
    if _TRACKER is not None:
        _TRACKER.free()


class VideoTracker:
    """SAM3 视频模型封装（单例，显存占用大，用完记得 free）。"""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        bpe_path: str | Path | None = None,
    ) -> None:
        from . import config

        self.checkpoint = str(checkpoint or config.CHECKPOINT)
        self.bpe_path = str(bpe_path or config.BPE_PATH)
        self.predictor = None
        self.load_seconds = 0.0

    def load(self) -> None:
        from sam3.model_builder import build_sam3_video_predictor

        t0 = time.perf_counter()
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=self.checkpoint,
            bpe_path=self.bpe_path,
        )
        self.load_seconds = time.perf_counter() - t0

    def free(self) -> None:
        import gc

        import torch

        self.predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def run(
        self,
        frames: list[dict],
        classes: list[str],
        frame_index: int = 0,
        confidence: float = 0.5,
        min_area: int = 100,
        progress=None,
        checkpoint=None,
        progress_base: float = 0.15,
        progress_span: float = 0.85,
        class_conf: dict[str, float] | None = None,
        chunk_size: int | None = None,
    ) -> TrackResult:
        """逐类别、逐段跑「首帧文本提示 → 段内跟踪」。

        为什么要分段：SAM3 的检测-跟踪关联要算 N_det × N_track × H×W 的 IoU 矩阵，
        长视频里实例一多就会吃掉几 GB 显存（实测 OOM 在 mask_iou）。
        按 chunk 切开、每段独立 session，能把峰值压住；代价是 track id 在段边界会重置。

        为什么每类要单独 session：SAM3 的文本提示是语义提示，add_prompt 内部会
        reset_state，多个类别不能在一次 session 里叠加。
        """
        import os as _os

        import torch as _torch

        if self.predictor is None:
            if progress:
                progress(
                    f"加载视频模型（约 10~20s，之后开始跟踪 {len(frames)} 帧 × {len(classes)} 类）",
                    progress_base,
                )
            self.load()
        vp = self.predictor
        frames_dir = str(Path(frames[0]["path"]).parent)
        per_frame: dict[int, list[dict]] = {k: [] for k in range(len(frames))}
        per_class: dict[str, int] = {}
        errors: list[str] = []
        t0 = time.perf_counter()
        n_cls = max(1, len(classes))

        chunk = int(chunk_size or _os.environ.get("SAM3_VIDEO_CHUNK", "80"))
        chunk = max(10, chunk)
        chunks = [frames[i : i + chunk] for i in range(0, len(frames), chunk)] or [frames]
        chunk_dirs = self._chunk_dirs(frames_dir, chunks) if len(chunks) > 1 else [frames_dir]
        if progress and len(chunks) > 1:
            progress(f"分 {len(chunks)} 段跟踪（每段 ≤{chunk} 帧）", progress_base)
        total_units = max(1, n_cls * len(chunks))
        done_units = 0

        for ci, cls in enumerate(classes):
            n_obj = 0
            thr_cls = float((class_conf or {}).get(cls, confidence))
            for gi, part in enumerate(chunks):
                if checkpoint:
                    checkpoint()
                offset = gi * chunk
                resp = vp.handle_request(
                    {"type": "start_session", "resource_path": chunk_dirs[gi]}
                )
                session_id = resp["session_id"]
                try:
                    vp.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": cls,
                        }
                    )
                    for step in vp.handle_stream_request(
                        {"type": "propagate_in_video", "session_id": session_id}
                    ):
                        fi_local = int(step["frame_index"])
                        fi = offset + fi_local
                        if checkpoint:
                            checkpoint()
                        if not (0 <= fi < len(frames)):
                            continue
                        if progress and (fi + 1) % 5 == 0:
                            frac = (done_units + (fi_local + 1) / max(1, len(part))) / total_units
                            seg = f"，第 {gi + 1}/{len(chunks)} 段" if len(chunks) > 1 else ""
                            progress(
                                f"「{cls}」跟踪 {fi + 1}/{len(frames)} 帧"
                                f"（第 {ci + 1}/{n_cls} 类{seg}）",
                                progress_base + progress_span * frac,
                            )
                        out = step["outputs"] or {}
                        obj_ids = np.asarray(out.get("out_obj_ids", []))
                        probs = np.asarray(out.get("out_probs", []))
                        masks = np.asarray(out.get("out_binary_masks", []))
                        for k, obj_id in enumerate(obj_ids.tolist()):
                            if k >= len(masks):
                                continue
                            prob = float(probs[k]) if k < len(probs) else 0.0
                            if prob < thr_cls:
                                continue
                            mask = masks[k]
                            if mask.ndim != 2:
                                continue
                            area = int(mask.sum())
                            if area < min_area:
                                continue
                            ys, xs = np.where(mask)
                            per_frame[fi].append(
                                {
                                    "track_id": int(obj_id) + gi * 100000,
                                    "class_name": cls,
                                    "score": prob,
                                    "area": area,
                                    "box": [
                                        float(xs.min()), float(ys.min()),
                                        float(xs.max() + 1), float(ys.max() + 1),
                                    ],
                                    "mask": mask,
                                }
                            )
                            n_obj = max(n_obj, int(obj_id) + 1)
                except Exception as exc:  # 单段失败不影响其它段/类别
                    if type(exc).__name__ in ("JobCanceled", "JobCanceledError"):
                        raise
                    errors.append(
                        f"{cls} 第{gi + 1}段（{len(part)} 帧）失败：{type(exc).__name__}"
                        + ("（该段目标数过多，建议提高此类别阈值或减小「每段帧数」）"
                           if "out of memory" in str(exc).lower() else f"：{exc}"[:120])
                    )
                    if progress:
                        progress(
                            f"「{cls}」第 {gi + 1} 段失败（{type(exc).__name__}），继续下一段",
                            progress_base + progress_span * (done_units + 1) / total_units,
                        )
                finally:
                    try:
                        vp.handle_request({"type": "close_session", "session_id": session_id})
                    except Exception:
                        pass
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                done_units += 1
            per_class[cls] = n_obj
            if progress:
                frac = (ci + 1) / n_cls
                progress(
                    f"「{cls}」跟踪完成：{n_obj} 个目标",
                    progress_base + progress_span * frac,
                )

        ordered = [
            {"frame_index": k + 1, "instances": per_frame.get(k, [])}
            for k in range(len(frames))
        ]
        return TrackResult(
            frames=ordered,
            per_class=per_class,
            seconds=time.perf_counter() - t0,
            errors=errors,
        )

    @staticmethod
    def _chunk_dirs(frames_dir: str, chunks: list[list[dict]]) -> list[str]:
        """为每段建一个只含该段帧的目录（优先软链接），SAM3 的 session 只认目录。"""
        import shutil

        base = Path(frames_dir).parent / "chunks"
        base.mkdir(parents=True, exist_ok=True)
        out: list[str] = []
        for i, part in enumerate(chunks):
            d = base / f"chunk_{i + 1:03d}"
            d.mkdir(exist_ok=True)
            for old in d.glob("*.jpg"):
                old.unlink()
            for k, p in enumerate(part):
                src = Path(p["path"]).resolve()
                link = d / f"{k + 1:06d}.jpg"
                try:
                    link.symlink_to(src)
                except OSError:
                    shutil.copy2(src, link)
            out.append(str(d))
        return out

# --------------------------------------------------------------------- 逐帧标注
def annotate_per_image(
    frames: list[dict],
    classes: list[str],
    confidence: float = 0.5,
    min_area: int = 100,
    progress=None,
    checkpoint=None,
    class_conf: dict[str, float] | None = None,
) -> TrackResult:
    """退路：把每帧当独立图像跑图像模型。"""
    from PIL import Image

    from .engine import SAM3Engine

    engine = SAM3Engine(confidence=confidence)
    out: list[dict] = []
    t0 = time.perf_counter()
    for k, frame in enumerate(frames):
        if checkpoint:
            checkpoint()
        image = Image.open(frame["path"]).convert("RGB")
        image.load()
        state = engine.set_image(image, key=f"fr{k}")
        res = engine.segment_text(
            state, classes, confidence=confidence, min_area=min_area, class_conf=class_conf
        )
        instances = []
        for inst in res.instances:
            instances.append(
                {
                    "track_id": -1,
                    "class_name": inst.class_name,
                    "score": float(inst.score),
                    "area": inst.area,
                    "box": inst.box,
                    "mask": inst.mask,
                }
            )
        out.append({"frame_index": k + 1, "instances": instances})
        if progress and (k + 1) % 5 == 0:
            progress(f"逐帧标注 {k + 1}/{len(frames)}", 0.15 + 0.85 * (k + 1) / len(frames))
    per_class = {}
    for fr in out:
        for inst in fr["instances"]:
            per_class[inst["class_name"]] = per_class.get(inst["class_name"], 0) + 1
    return TrackResult(frames=out, per_class=per_class, seconds=time.perf_counter() - t0)


# --------------------------------------------------------------------- 完整流程
def unique_video_name(project, stem: str) -> str:
    """同一个视频重复导入时给新分组编号：xxx、xxx（2）、xxx（3）…"""
    used = {i.group for i in project.images if i.group}
    vids_dir = Path(project.root) / "videos"
    if vids_dir.exists():
        used |= {d.name for d in vids_dir.iterdir() if d.is_dir()}
    if stem not in used:
        return stem
    n = 2
    while f"{stem}（{n}）" in used:
        n += 1
    return f"{stem}（{n}）"


def run_video_import(
    project,
    video_path: str | Path,
    classes: list[str],
    fps: float | None = 2.0,
    max_frames: int = 300,
    max_size: int = 1600,
    start_time: float = 0.0,
    end_time: float | None = None,
    mode: str = "track",
    confidence: float = 0.5,
    min_area: int = 100,
    annotate: bool = True,
    progress=None,
    checkpoint=None,
    class_conf: dict[str, float] | None = None,
    chunk_size: int | None = None,
) -> dict:
    """视频 → 抽帧 → 注册进项目 → 自动标注，返回汇总信息。"""
    from . import postprocess as pp

    video_path = Path(video_path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")
    stem = unique_video_name(project, video_path.stem)
    video_dir = project.root / "videos" / stem
    frames_dir = video_dir / "frames"
    # 注意：只写"本次新建"的目录，绝不清空已有帧（重复导入会拿到 xxx（2） 这样的新名字）
    frames_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        checkpoint()
    if progress:
        progress(f"读取视频 {video_path.name}", 0.01)
    try:
        meta = extract_frames(
            video_path,
            frames_dir,
            fps=fps,
            max_frames=max_frames,
            max_size=max_size,
            start_time=start_time,
            end_time=end_time,
            progress=progress,
            checkpoint=checkpoint,
        )
    except BaseException:
        import shutil

        if frames_dir.exists() and not any(frames_dir.glob("*.jpg")):
            shutil.rmtree(video_dir, ignore_errors=True)
        raise
    frames = meta["frames"]
    if not frames:
        raise RuntimeError("没有抽到任何帧（视频为空或时间范围不对）")
    if progress:
        progress(f"抽到 {len(frames)} 帧 → 注册进项目", 0.15)

    entries = project.add_video_frames(stem, frames)
    entry_by_index = {e.frame_index: e for e in project.frames_of_video(stem)}
    if progress:
        progress(f"新增 {len(entries)} 张项目图片（已有 {len(frames) - len(entries)} 张跳过）", 0.16)

    if not annotate or not classes:
        return {
            "video": stem, "probe": {k: meta[k] for k in ("fps", "frames", "width", "height", "duration")},
            "sampled": len(frames), "added": len(entries), "annotated": 0,
            "mode": "none", "seconds": meta.get("extract_seconds", 0),
        }

    t0 = time.perf_counter()
    if mode == "track":
        tracker = get_video_tracker()
        try:
            result = tracker.run(
                frames, classes,
                confidence=confidence, min_area=min_area, progress=progress,
                checkpoint=checkpoint, class_conf=class_conf, chunk_size=chunk_size,
            )
            used_mode = "track"
        except Exception as exc:  # 显存不够/模型异常时自动退回逐帧
            if type(exc).__name__ in ("JobCanceled", "JobCanceledError"):
                raise  # 取消不降级，直接向上抛
            if progress:
                progress(f"跟踪模式失败（{type(exc).__name__}），先释放视频模型再退回逐帧标注")
            free_video_tracker()  # 关键：不释放的话图像模型没显存可用
            result = annotate_per_image(
                frames, classes, confidence=confidence, min_area=min_area,
                progress=progress, checkpoint=checkpoint, class_conf=class_conf,
            )
            used_mode = "frame(fallback)"
        finally:
            free_video_tracker()  # 无论成功/失败/取消都释放，避免显存泄漏
    else:
        result = annotate_per_image(
            frames, classes, confidence=confidence, min_area=min_area,
            progress=progress, checkpoint=checkpoint, class_conf=class_conf,
        )
        used_mode = "frame"

    saved = 0
    total_inst = 0
    for fr in result.frames:
        entry = entry_by_index.get(fr["frame_index"])
        if entry is None:
            continue
        records = []
        # 帧内 id 必须唯一（不同类别的 track_id 会重复），按分数从高到低编号
        ordered = sorted(fr["instances"], key=lambda i: -float(i["score"]))
        for k, inst in enumerate(ordered, start=1):
            rec = {
                "id": k,
                "track_id": int(inst.get("track_id", -1)),
                "kept": True,
                "class_name": inst["class_name"],
                "score": float(inst["score"]),
                "box": [float(v) for v in inst["box"]],
                "area": int(inst["area"]),
                "source": "video",
                "segmentation": pp.encode_mask(inst["mask"]),
            }
            records.append(rec)
        annot = {
            "image_id": entry.id,
            "path": entry.relpath,
            "width": meta["frames"][fr["frame_index"] - 1]["width"],
            "height": meta["frames"][fr["frame_index"] - 1]["height"],
            "classes": classes,
            "status": "done",
            "video": stem,
            "frame_index": fr["frame_index"],
            "ts": meta["frames"][fr["frame_index"] - 1]["time"],
            "params": {"confidence": confidence, "min_area": min_area, "mode": used_mode},
            "instances": records,
            "next_id": len(records) + 1,
        }
        project.save_annotation(entry, annot)
        saved += 1
        total_inst += len(records)
        if progress and saved % 10 == 0:
            progress(f"写入标注 {saved}/{len(result.frames)} 帧", 0.97)

    merged = list(project.classes)
    for c in classes:
        if c not in merged:
            merged.append(c)
    if merged != project.classes:
        project.set_classes(merged)

    return {
        "video": stem,
        "probe": {k: meta[k] for k in ("fps", "frames", "width", "height", "duration")},
        "sampled": len(frames),
        "sample_fps": meta["sample_fps"],
        "added": len(entries),
        "annotated": saved,
        "instances": total_inst,
        "tracks": result.per_class,
        "chunk_errors": result.errors[:10],
        "mode": used_mode,
        "seconds": round(time.perf_counter() - t0, 1),
    }
