"""命令行入口：doctor / annotate / export / serve / stats。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config
from .store import ImageEntry, Project, parse_classes


# --------------------------------------------------------------------- 工具
def _log(msg: str) -> None:
    print(msg, flush=True)


def _fmt_eta(seconds: float) -> str:
    seconds = max(seconds, 0)
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def open_or_create_project(args) -> Project:
    root = Path(args.project)
    if (root / "project.json").exists():
        proj = Project.open(root)
    else:
        classes = parse_classes(args.classes)
        image_dir = Path(args.image_dir) if args.image_dir else root / "images"
        proj = Project.create(root, name=root.name, classes=classes, image_dir=image_dir)
        _log(f"[项目] 已创建 {root}（类别 {classes or '待设置'}）")
    if getattr(args, "classes", None):
        classes = parse_classes(args.classes)
        if classes and classes != proj.classes:
            proj.set_classes(classes)
            _log(f"[项目] 类别更新为 {classes}")
    if getattr(args, "image_dir", None):
        proj.meta["image_dir"] = str(Path(args.image_dir).resolve())
        proj.save_meta()
    added = proj.scan()
    if added:
        _log(f"[项目] 扫描到 {added} 张新图片，共 {len(proj.images)} 张")
    return proj


def parse_class_conf(text: str | None) -> dict[str, float]:
    """解析 "car=0.6,person=0.45" 形式的类别阈值。"""
    out: dict[str, float] = {}
    if not text:
        return out
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.rsplit("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    _log("== 环境检查 ==")
    _log(f"Python      : {sys.version.split()[0]}")
    try:
        import torch

        _log(f"PyTorch     : {torch.__version__} (CUDA {torch.version.cuda})")
        _log(f"CUDA 可用   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            _log(f"GPU         : {p.name}  {p.total_memory / 1024**3:.1f} GB")
    except Exception as exc:  # pragma: no cover
        _log(f"PyTorch     : 不可用 ({exc})")
        return 1
    _log(f"权重        : {config.CHECKPOINT} ({'存在' if config.CHECKPOINT.exists() else '缺失'})")
    _log(f"BPE 词表    : {config.BPE_PATH} ({'存在' if config.BPE_PATH.exists() else '缺失'})")
    if not args.load:
        return 0
    _log("== 加载模型 ==")
    from .engine import SAM3Engine

    engine = SAM3Engine()
    _log(json.dumps(engine.info(), ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------- annotate
def cmd_annotate(args) -> int:
    from PIL import Image

    from .engine import SAM3Engine

    proj = open_or_create_project(args)
    classes = parse_classes(args.classes) or proj.classes
    if not classes:
        _log("错误：没有类别提示词。用 --classes \"person,car\" 指定。")
        return 2

    class_conf = parse_class_conf(getattr(args, "class_conf", None))
    if class_conf:
        merged = {**proj.class_conf, **class_conf}
        proj.set_class_conf(merged)
        _log(f"[阈值] 类别阈值已保存：{merged}")
    engine = SAM3Engine(
        confidence=args.conf,
        resolution=args.resolution,
    )
    _log(f"[模型] 加载完成 {engine.load_seconds:.1f}s | {engine.info().get('gpu')}")

    targets = _select_targets(proj, args)
    if not targets:
        _log("没有需要处理的图片（都已完成？加 --overwrite 重标）。")
        return 0

    _log(f"[任务] 待标注 {len(targets)} 张，类别 {classes}")
    started = time.time()
    ok = 0
    failed = 0
    total_inst = 0
    for idx, entry in enumerate(targets, 1):
        t0 = time.time()
        try:
            path = proj.image_path(entry)
            with Image.open(path) as im:
                image = im.convert("RGB")
                image.load()
            state = engine.set_image(image, key=entry.id)
            result = engine.segment_text(
                state,
                classes,
                confidence=args.conf,
                min_area=args.min_area,
                dedup_iou=args.dedup_iou,
                class_agnostic_dedup=not args.per_class_dedup,
                class_conf=proj.class_conf,
            )
            annot = {
                "image_id": entry.id,
                "path": entry.relpath,
                "width": image.width,
                "height": image.height,
                "classes": classes,
                "status": "done",
                "params": {
                    "confidence": args.conf,
                    "min_area": args.min_area,
                    "dedup_iou": args.dedup_iou,
                    "resolution": args.resolution,
                },
                "instances": [
                    {"id": i + 1, "kept": True, **inst.to_record()}
                    for i, inst in enumerate(result.instances)
                ],
            }
            proj.save_annotation(entry, annot)
            ok += 1
            total_inst += result.count
            dt = time.time() - t0
            done = idx
            eta = (time.time() - started) / done * (len(targets) - done)
            _log(
                f"[{idx}/{len(targets)}] {entry.relpath}  "
                f"{result.count} 个实例  {dt:.2f}s  剩余 {_fmt_eta(eta)}"
            )
        except Exception as exc:  # pragma: no cover
            failed += 1
            _log(f"[{idx}/{len(targets)}] {entry.relpath} 失败: {exc}")
        finally:
            if args.gc_every and idx % args.gc_every == 0:
                engine._drop_state()

    _log(
        f"[完成] 成功 {ok} 张 / 失败 {failed} 张 / 共 {total_inst} 个实例，"
        f"耗时 {_fmt_eta(time.time() - started)}"
    )
    _log(f"[下一步] 人工复核：python -m sam3label.cli serve --project {proj.root}")
    return 0 if failed == 0 else 1


def _select_targets(proj: Project, args) -> list[ImageEntry]:
    entries = proj.images
    if args.filter:
        entries = [e for e in entries if args.filter in e.relpath]
    targets = []
    for e in entries:
        exists = proj.annotation_path(e).exists()
        if args.overwrite or not exists:
            targets.append(e)
    if args.limit:
        targets = targets[: args.limit]
    return targets



# --------------------------------------------------------------------- import
def cmd_import(args) -> int:
    from .store import Project

    proj = open_or_create_project(args)
    copy_to = None
    if args.copy:
        copy_to = Path(args.dest).expanduser() if args.dest else proj.root / "images"
        if not copy_to.is_absolute():
            copy_to = proj.root / copy_to
    res = proj.add_external(args.paths, copy_to=copy_to, recursive=args.recursive)
    _log(
        f"[导入] 新增 {len(res['added'])} 张，跳过重复 {len(res['skipped'])} 张"
        f"（{'复制到 ' + str(copy_to) if copy_to else '引用原路径'}，"
        f"{'含子目录' if args.recursive else '只取该层'}），"
        f"项目共 {len(proj.images)} 张"
    )
    for miss in res["missing"]:
        _log(f"  路径不存在: {miss}")
    for bad in res["bad"]:
        _log(f"  不是图片/文件夹: {bad}")
    return 0



# --------------------------------------------------------------------- video
def cmd_video(args) -> int:
    from .video import run_video_import

    proj = open_or_create_project(args)
    classes = parse_classes(args.classes) or proj.classes

    def progress(msg: str) -> None:
        _log(f"  [{msg}]")

    class_conf = parse_class_conf(getattr(args, "class_conf", None))
    if class_conf:
        proj.set_class_conf({**proj.class_conf, **class_conf})
        _log(f"[阈值] 类别阈值已保存：{proj.class_conf}")
    _log(f"[视频] {args.video} → 项目 {proj.root}")
    try:
        res = run_video_import(
            proj,
            args.video,
            classes,
            fps=args.fps,
            max_frames=args.max_frames,
            max_size=args.max_size,
            start_time=args.start,
            end_time=args.end,
            mode=args.mode,
            confidence=args.conf,
            min_area=args.min_area,
            annotate=not args.no_annotate,
            progress=progress,
            class_conf=proj.class_conf,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:
        _log(f"视频处理失败：{exc}")
        return 1
    _log(json.dumps(res, ensure_ascii=False, indent=2))
    if res.get("annotated"):
        _log(f"[下一步] 复核：python -m sam3label.cli serve --project {proj.root}")
    return 0


# --------------------------------------------------------------------- export
def cmd_export(args) -> int:
    from . import exporters

    proj = Project.open(args.project)
    entries = None
    if args.group:
        entries = proj.entries_of_groups(args.group)
        if not entries:
            _log(f"错误：分组 {args.group} 下没有图片")
            return 2
    elif args.filter:
        entries = [i for i in proj.images if args.filter in i.relpath]
        if not entries:
            _log(f"错误：没有路径包含「{args.filter}」的图片")
            return 2
    _log(f"[导出] 范围 {len(entries) if entries is not None else len(proj.images)} 张")
    out = Path(args.out) if args.out else proj.root / "exports" / f"export-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    formats = parse_classes(args.formats) if args.formats else ["coco", "yolo", "masks", "labelme", "previews"]
    written: dict[str, str] = {}
    if "coco" in formats:
        written["coco"] = str(
            exporters.export_coco(
                proj, out / "instances_coco.json", only_reviewed=args.only_reviewed,
                polygons=args.coco_polygons, entries=entries,
            )
        )
    if "yolo" in formats:
        written["yolo"] = str(
            exporters.export_yolo(proj, out / "yolo_seg", only_reviewed=args.only_reviewed, entries=entries)
        )
    if "masks" in formats:
        written["masks"] = str(
            exporters.export_masks(
                proj, out / "masks", only_reviewed=args.only_reviewed,
                instance_png=args.instance_png, entries=entries,
            )
        )
    if "labelme" in formats:
        written["labelme"] = str(
            exporters.export_labelme(proj, out / "labelme", only_reviewed=args.only_reviewed, entries=entries)
        )
    if "previews" in formats:
        written["previews"] = str(
            exporters.export_previews(proj, out / "previews", only_reviewed=args.only_reviewed, entries=entries)
        )
    _log(json.dumps(written, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------- refilter
def cmd_refilter(args) -> int:
    proj = Project.open(args.project)
    if args.class_conf:
        proj.set_class_conf({**proj.class_conf, **parse_class_conf(args.class_conf)})
        _log(f"[阈值] 已保存：{proj.class_conf}")
    stats = proj.refilter(apply=args.apply)
    total = {"above": 0, "below": 0, "changed": 0, "manual": 0}
    for cls, v in sorted(stats.items()):
        _log(
            f"  {cls:<14} 阈值 {v['threshold']:<5} ≥ {v['above']:<5} 低于 {v['below']:<5} "
            f"将改变 {v['changed']:<5} 人工 {v['manual']}"
        )
        for k in total:
            total[k] += v.get(k, 0)
    _log(json.dumps({"apply": args.apply, **total}, ensure_ascii=False))
    if not args.apply:
        _log("（预览模式，加 --apply 才会写回标注）")
    return 0


# --------------------------------------------------------------------- projects
def cmd_projects(args) -> int:
    """列出 data/projects 下的所有项目。"""
    import json as _json

    root = config.PROJECTS_DIR
    rows = []
    for d in sorted(root.iterdir(), key=lambda x: x.name.lower()) if root.exists() else []:
        if not (d / "project.json").exists():
            continue
        try:
            media = _json.loads((d / "project.json").read_text(encoding="utf-8"))
            n = 0
            p2 = d / "images.json"
            if p2.exists():
                n = len(_json.loads(p2.read_text(encoding="utf-8")).get("images", []))
        except Exception:
            continue
        rows.append((media.get("name") or d.name, n, str(d)))
    if args.json:
        _log(_json.dumps([{"name": a, "images": b, "dir": c} for a, b, c in rows], ensure_ascii=False, indent=2))
    else:
        _log(f"{'项目':<16}{'图片':>6}   目录")
        for name, n, path in rows:
            _log(f"{name:<16}{n:>6}   {path}")
        _log("\n启动某个项目：sam3label serve --project <目录> --port 8099")
    return 0


# --------------------------------------------------------------------- resync
def cmd_resync(args) -> int:
    proj = Project.open(args.project)
    _log(f"[自检] {proj.root}：{len(proj.images)} 张图，按标注文件回填状态…")
    res = proj.resync_manifest()
    _log(json.dumps(res, ensure_ascii=False))
    _log(json.dumps({k: proj.stats()[k] for k in ("images", "done", "pending", "reviewed", "instances")},
                    ensure_ascii=False))
    return 0


# --------------------------------------------------------------------- stats
def cmd_stats(args) -> int:
    proj = Project.open(args.project)
    _log(json.dumps(proj.stats(), ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------- serve
def cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app

    flags = []
    if args.no_preload:
        flags.append("--no-preload")
    if args.no_scan:
        flags.append("--no-scan")
    app = create_app(
        args.project,
        preload=not args.no_preload,
        confidence=args.conf,
        scan=not args.no_scan,
        host=args.host,
        port=args.port,
        serve_flags=flags,
    )
    _log(f"[服务] http://{args.host}:{args.port}  项目 {args.project}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# --------------------------------------------------------------------- 参数
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sam3label", description="SAM3 半自动分割标注系统"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="检查环境、权重、显存")
    d.add_argument("--load", action="store_true", help="真正加载一次模型")
    d.set_defaults(func=cmd_doctor)

    a = sub.add_parser("annotate", help="批量自动标注")
    a.add_argument("--project", required=True, help="项目目录（不存在会自动创建）")
    a.add_argument("--image-dir", default=None, help="原图目录")
    a.add_argument("--classes", default=None, help="类别提示词，逗号分隔")
    a.add_argument("--conf", type=float, default=config.CONFIDENCE)
    a.add_argument("--class-conf", default=None,
                   help='按类别覆盖阈值，如 "car=0.6,person=0.45"（也会写进项目设置）')
    a.add_argument("--resolution", type=int, default=config.RESOLUTION)
    a.add_argument("--min-area", type=int, default=config.MIN_AREA)
    a.add_argument("--dedup-iou", type=float, default=config.DEDUP_IOU)
    a.add_argument(
        "--per-class-dedup",
        action="store_true",
        help="只在同类内去重（默认跨类别去重）",
    )
    a.add_argument("--overwrite", action="store_true", help="重标已有结果的图片")
    a.add_argument("--limit", type=int, default=0, help="最多处理多少张")
    a.add_argument("--filter", default=None, help="只处理路径包含该子串的图片")
    a.add_argument("--gc-every", type=int, default=20, help="每 N 张清理一次显存")
    a.set_defaults(func=cmd_annotate)


    i = sub.add_parser("import", help="把本机文件夹/图片加入项目（默认只引用不复制）")
    i.add_argument("--project", required=True)
    i.add_argument("paths", nargs="+", help="文件夹或图片路径")
    i.add_argument("--copy", action="store_true", help="复制进项目（默认只登记路径）")
    i.add_argument("--dest", default=None, help="--copy 时的目标目录")
    i.add_argument("--recursive", action="store_true", help="连子目录一起加（默认只加该层）")
    i.add_argument("--image-dir", default=None)
    i.add_argument("--classes", default=None)
    i.set_defaults(func=cmd_import)

    e = sub.add_parser("export", help="导出标注")
    e.add_argument("--project", required=True)
    e.add_argument("--out", default=None)
    e.add_argument("--formats", default=None, help="coco,yolo,masks,labelme,previews")
    e.add_argument("--only-reviewed", action="store_true", help="只导出已复核的图")
    e.add_argument("--coco-polygons", action="store_true", help="COCO 用多边形而非 RLE")
    e.add_argument("--instance-png", action="store_true", help="额外导出逐实例 PNG")
    e.add_argument("--group", action="append", default=None,
                   help='只导出某个分组，可重复：--group "video:飞书xxx"')
    e.add_argument("--filter", default=None, help="只导出路径包含该子串的图片")
    e.set_defaults(func=cmd_export)

    rf = sub.add_parser("refilter", help="按类别阈值重筛已有实例")
    rf.add_argument("--project", required=True)
    rf.add_argument("--class-conf", default=None, help='先更新阈值，如 "car=0.6,person=0.45"')
    rf.add_argument("--apply", action="store_true", help="真正写回（默认只预览）")
    rf.set_defaults(func=cmd_refilter)

    pj = sub.add_parser("projects", help="列出所有项目")
    pj.add_argument("--json", action="store_true")
    pj.set_defaults(func=cmd_projects)

    rs = sub.add_parser("resync", help="按标注文件回填清单状态（修复统计不一致）")
    rs.add_argument("--project", required=True)
    rs.set_defaults(func=cmd_resync)

    s = sub.add_parser("stats", help="查看项目统计")
    s.add_argument("--project", required=True)
    s.set_defaults(func=cmd_stats)


    vd = sub.add_parser("video", help="导入视频：抽帧 + 自动标注")
    vd.add_argument("--project", required=True)
    vd.add_argument("--video", required=True, help="视频文件路径（mp4/mov/avi/mkv…）")
    vd.add_argument("--classes", default=None, help="类别提示词，逗号分隔")
    vd.add_argument("--fps", type=float, default=2.0, help="抽帧帧率（0=每帧）")
    vd.add_argument("--max-frames", type=int, default=300)
    vd.add_argument("--max-size", type=int, default=1600, help="帧长边上限（0=不缩放）")
    vd.add_argument("--chunk-size", type=int, default=80,
                    help="长视频分段跟踪的段长（越小越省显存，track id 会在段边界重置）")
    vd.add_argument("--start", type=float, default=0.0, help="起始时间（秒）")
    vd.add_argument("--end", type=float, default=None, help="结束时间（秒）")
    vd.add_argument("--mode", choices=["track", "frame"], default="track",
                    help="track=视频模型跨帧跟踪（推荐）；frame=逐帧独立标注")
    vd.add_argument("--conf", type=float, default=config.CONFIDENCE)
    vd.add_argument("--class-conf", default=None, help='按类别覆盖阈值，如 "car=0.6,person=0.45"')
    vd.add_argument("--min-area", type=int, default=config.MIN_AREA)
    vd.add_argument("--no-annotate", action="store_true", help="只抽帧不标注")
    vd.add_argument("--image-dir", default=None)
    vd.set_defaults(func=cmd_video)

    v = sub.add_parser("serve", help="启动复核 Web 界面")
    v.add_argument("--project", required=True)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--conf", type=float, default=config.CONFIDENCE)
    v.add_argument("--no-preload", action="store_true", help="启动时不加载模型")
    v.add_argument("--no-scan", action="store_true",
                   help="启动时不扫描 image_dir，只显示项目里已有的图片")
    v.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    config.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        _log(f"错误：{exc}")
        return 2
    except KeyError as exc:
        _log(f"错误：{exc}")
        return 2
    except KeyboardInterrupt:
        _log("\n已中断")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
