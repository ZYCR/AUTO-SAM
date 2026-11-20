# 设计说明

## 数据流

```
原图/视频帧 ──scan/import──> project/images.json（清单）
                                  │
      annotate / segment ─────────┴──> annotations/<id>.json（实例 + COCO RLE）
                                  │
            复核界面（FastAPI）<──┴── 掩码叠加 PNG / 实例 id 标签图
                                  │
             export ──────────────> COCO / YOLO-seg / 语义 PNG / LabelMe / 预览
```

## 几个关键取舍

**1. 掩码用 COCO RLE 存盘**
体积小、可无损还原、pycocotools 通吃；前端不解析 RLE，改由服务端渲染成叠加 PNG
（`/media/overlay`）与实例 id 标签图（`/media/labelmap`）。

**2. 命中测试用标签图而不是外框**
服务端把实例 id 编进 RGB（`id = r + g*256 + b*65536`）渲染一张标签图，前端取像素即得实例。
外框重叠、框内空白都不会点错；掩码重叠处按面积从大到小写入，因此点击会选中更小的内层实例。

**3. 按类别阈值 + 批量重筛**
阈值存在 `settings.class_conf`，作用两处：推理时每个类别用各自的 `confidence_threshold`；
已有结果可"批量重筛"（只翻 `kept`，不重跑模型）。人工点过「排除/恢复」的实例带 `manual` 标记，
重筛跳过，避免覆盖人工判断。

**4. 视频分段跟踪**
SAM3 的检测-跟踪关联需要 `N检测 × N航迹 × H×W` 的 IoU 矩阵，长视频 + 多个目标时会占几 GB 显存。
按 `chunk_size` 切段、每段独立 session，单段失败只记录错误并继续；`track` 失败会先释放视频模型
再退回逐帧模式（否则两个模型同时驻留会二次 OOM）。

**5. 缓存与并发**
- `decode_cache`：解码后的掩码按 `(image_id, updated)` 缓存，最多留 2 张图；
- `overlay_cache`：按 `(image_id, updated, highlight, hide, hcls)` 缓存 PNG，最多 16 项；
- 所有 GPU 调用的端点用同一把 `gpu_lock` 串行化；视频任务期间先卸载图像模型；
- 清单 `images.json` 按 mtime 热加载，写标注时按 image id 找"当前清单里的那一条"，
  避免任务线程持有旧对象把状态写丢。

**6. 项目与任务**
一个服务进程服务一个项目；切换项目＝后台重启进程指向新项目（前端轮询 `/api/state` 后刷新）。
任务（批量标注/视频）跑在后台线程里，`pause/resume/cancel` 通过 checkpoint 协作式生效，
只在帧/图片边界检查，正在跑的推理不能打断。
