# AUTO-SAM · 基于 SAM 3 的本地自动分割标注平台

给一句话类别清单（`person, car, glass, floor`），自动把整个文件夹/整段视频里的目标分割标注好，
再在本地网页里人工复核（改类别 / 编辑掩码 / 删除 / 调阈值），最后导出 **COCO / YOLO-seg / 语义 PNG / LabelMe**。

> 推理后端是 Meta 的 **SAM 3（Segment Anything with Concepts）**。
> 本仓库**只包含平台本体代码与 6 张示例图**，不包含模型权重（3.2 GiB）——
> 权重请按 [模型权重](#模型权重下载) 一节自行下载。

---

## 目录

- [能做什么](#能做什么)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [模型权重下载](#模型权重下载)
- [Web 复核界面](#web-复核界面)
- [命令行](#命令行)
- [视频导入与自动标注](#视频导入与自动标注)
- [导出](#导出)
- [数据格式](#数据格式)
- [性能实测](#性能实测)
- [HTTP API](#http-api)
- [常见问题](#常见问题)
- [已知限制](#已知限制)
- [许可与引用](#许可与引用)

## 能做什么

| 能力 | 说明 |
|---|---|
| 文本提示批量分割 | 开放词表，按类别清单自动标注整个目录 |
| 按掩码命中 | 点击以"点在哪个实例的掩码里"为准；重叠处取面积更小者；点背景取消选中 |
| 选中即高亮 | 选中实例的掩码提亮 + 3px 白边，其余压暗 |
| 画笔精修 | 加/减画笔、撤销、保存回写 RLE，不影响其它实例 |
| 按类别阈值 | 每个类别单独设置信度阈值，一键批量重筛（不重跑模型）；人工"排除/恢复"的实例不受影响 |
| 类别颜色可调 | 每类颜色自定义，列表圆点 / 画布方框 / 掩码叠加 / 导出预览统一 |
| 视频导入自动标注 | 抽帧 + 跨帧跟踪（分段跟踪抗显存），帧作为图片进入项目 |
| 按来源分组 | 视频抽帧 🎬 / 文件夹导入 📁 分组显示，可折叠、全选、整组删除 |
| 多项目管理 | 界面里切换项目、从文件夹新建项目 |
| 任务可控 | 批量标注/视频任务可暂停、继续、取消，带真实百分比进度 |
| 多种导出 | 全部 / 仅已复核 / 指定分组 / 勾选的图片 |

## 目录结构

```
AUTO-SAM/
├── sam3label/              # 平台本体（Python 包）
│   ├── config.py           # 路径与默认参数（全部可用环境变量覆盖）
│   ├── engine.py           # SAM3 图像模型封装：set_image / segment_text / 框提示
│   ├── video.py            # 视频：探测、抽帧、分段跨帧跟踪、逐帧降级
│   ├── postprocess.py      # RLE 编解码、掩码 NMS、多边形化、可视化
│   ├── store.py            # 项目/清单/标注 JSON、分组、阈值、颜色、自检
│   ├── exporters.py        # COCO / YOLO-seg / 语义 PNG / LabelMe / 预览
│   ├── server.py           # FastAPI 后端（任务、缓存、项目切换）
│   └── cli.py              # doctor / annotate / video / projects / serve / export / refilter / resync / stats
├── web/                    # 前端（原生 JS + Canvas，无构建步骤）
├── scripts/                # setup_env.sh / download_weights.sh / serve.sh / start_samples.sh / run_demo.sh
├── docs/                   # API.md / ARCHITECTURE.md
├── data/projects/samples/  # 6 张示例图 + 标注（96 实例），开箱可看
├── pyproject.toml
└── LICENSE
```

## 快速开始

需要：Linux + NVIDIA GPU（≥ 8 GB 显存推荐）、Python 3.12、能访问 PyPI 镜像。

```bash
# 1) 建环境（Python 3.12 + torch cu128 + sam3 + web 依赖）
./scripts/setup_env.sh

# 2) 下权重（见下一节；约 3.2 GiB）
./scripts/download_weights.sh

# 3) 自检
.venv/bin/python -m sam3label.cli doctor --load

# 4) 打开示例项目（6 张示例图，启动不扫描、不标注任何其它图片）
./scripts/start_samples.sh        # http://127.0.0.1:8099
```

> 国内网络说明：`download.pytorch.org` 往往只有 1 MB/s 且 `github.com` 的 git 协议可能 TLS 失败，
> 因此脚本默认走 **PyPI/阿里云镜像取代码与 torch、ModelScope 取权重**。这些地址都可在脚本里替换。

## 模型权重下载

**权重不随仓库分发**，请任选一个来源，下载后放到 `weights/sam3.pt`
（约 **3,450,062,241 字节**，可用大小校验完整性）：

| 来源 | 地址 | 是否需要申请 |
|---|---|---|
| ModelScope（推荐，无需审批） | <https://modelscope.cn/models/facebook/sam3> → `sam3.pt` | 否 |
| Hugging Face 官方 | <https://huggingface.co/facebook/sam3> | 是（gated，需申请 + token） |

一键下载（走 ModelScope，含断点续传与大小校验）：

```bash
./scripts/download_weights.sh
```

脚本同时会还原文本分词器需要的 BPE 词表 `.venv`→ `assets/bpe_simple_vocab_16e6.txt.gz`
（内容即官方仓库的 `merges.txt`，48895 行）。

路径可用环境变量覆盖：

```bash
export SAM3_CHECKPOINT=/abs/path/sam3.pt
export SAM3_BPE=/abs/path/bpe_simple_vocab_16e6.txt.gz
export SAM3_WEIGHTS_DIR=/abs/weights
```

## Web 复核界面

```bash
./scripts/start_samples.sh            # 示例项目（干净，只有 6 张图）
./scripts/serve.sh data/projects/xxx  # 任意项目
```

打开 <http://127.0.0.1:8099>。

- **左侧**：缩略图按来源分组（🎬 视频帧 / 📁 文件夹导入），带状态徽标与实例数；
  右上角勾选框 + 顶部选择条可批量移除（`移除` / `移除并删文件`，项目外原图不会被删）；
  列表分页加载（每页 400，滚到底自动加载）。
- **中间**：原图 + 掩码 + 方框；单击选中并弹操作菜单；滚轮缩放、拖空白平移。
- **右侧**：实例按类别分组，可折叠；每类/每实例有显示勾选框；每行 `编辑 / 排除 / 删除`。
- **顶部**：项目切换、类别清单、类别阈值与颜色、添加图片、批量标注、导出。
- **底部**：任务进度条（暂停/继续/取消/关闭）、翻页按钮、缩放控件、按键诊断。

快捷键（只保留常用几个）：

| 键 | 作用 | 键 | 作用 |
|---|---|---|---|
| `d` / `a` | 下一张 / 上一张（也支持 `←/→`、`↑/↓`、`j/k`） | `Delete` | 删除选中实例（`Ctrl+Z` 撤销最近一次） |
| `Enter` | 标记当前图「已复核」 | `g` / `r` | 标记完成 / 已复核 |
| `S` / `B` / `E` | 选择 / 框选补标 / 画笔精修 | `F` | 适应窗口 |

> 如果快捷键无反应：先看工具栏「键 …」诊断行是否显示按键；不显示说明按键被浏览器扩展拦截，
> 用工具栏的 `◀ 上一张 / 下一张 ▶` 按钮或隐身窗口。

## 命令行

```bash
# 环境自检（--load 会真正加载模型）
python -m sam3label.cli doctor --load

# 批量标注（项目不存在会自动创建）
python -m sam3label.cli annotate --project data/projects/mydata \
  --image-dir /path/to/images --classes "person,car,glass" \
  [--conf 0.5] [--min-area 100] [--dedup-iou 0.75] [--class-conf "glass=0.7"] \
  [--overwrite] [--limit N] [--filter 子串]

# 视频导入 + 自动标注
python -m sam3label.cli video --project data/projects/vid \
  --video /path/to/clip.mp4 --classes "person,box" \
  [--fps 2] [--max-frames 300] [--max-size 1600] [--chunk-size 80] \
  [--mode track|frame] [--start 0] [--end 20]

# 复核界面 / 项目列表 / 统计
python -m sam3label.cli serve --project data/projects/mydata [--port 8099] [--no-scan] [--no-preload]
python -m sam3label.cli projects
python -m sam3label.cli stats --project data/projects/mydata

# 导出（可限定分组/路径）
python -m sam3label.cli export --project data/projects/mydata \
  --formats coco,yolo,masks,labelme,previews \
  [--only-reviewed] [--group "video:clip"] [--filter 子串] [--out DIR]

# 按类别阈值重筛已有标注（不需要 GPU）
python -m sam3label.cli refilter --project data/projects/mydata \
  --class-conf "glass=0.75" [--apply]

# 用标注文件回填清单状态（修复统计不一致）
python -m sam3label.cli resync --project data/projects/mydata
```

## 视频导入与自动标注

两种模式：

| 模式 | 做法 | 特点 |
|---|---|---|
| `track`（默认） | SAM3 视频模型：首个采样帧文本提示 → 跨帧跟踪传播 | 同一目标段内共享 `track_id`，时间上连贯 |
| `frame` | 每帧当独立图像跑图像模型 | 显存稳、不会因目标过多 OOM，但实例 id 不跨帧连续 |

要点：

- 视频先按 `--fps` 抽帧（默认 2）存到 `data/projects/<项目>/videos/<视频名>/frames/`，
  帧同时注册进项目（`relpath = <视频名>/000001.jpg`）；
- **重复导入同一个视频会自动编号** `xxx`、`xxx（2）`、`xxx（3）`…，不会覆盖已有分组；
- 长视频按 `--chunk-size`（默认 80 帧）**分段跟踪**：SAM3 的检测-跟踪关联要算
  `N检测 × N航迹 × H×W` 的 IoU 矩阵，目标多时会吃几 GB 显存；分段后单段失败只跳过该段，
  其它段与类别照常出结果；
- 视频模型约需 7 GB 空闲显存；服务端在跑视频任务前会**先释放图像模型**，任务结束再懒加载回来。

## 导出

| 格式 | 内容 | 用途 |
|---|---|---|
| `coco` | `instances_coco.json`，RLE 分割 + bbox + area + score | mmdetection / detectron2 / 通用 |
| `yolo` | `images/ labels/ data.yaml`，归一化多边形 | ultralytics 分割训练 |
| `masks` | `*.png` 语义掩码（像素值 = 类别 id + 1）+ `palette.json` | 语义分割训练 |
| `labelme` | `*.json` 多边形 | LabelMe / X-AnyLabeling 继续精修 |
| `previews` | 叠加掩码与标签的 jpg | 质检 / 汇报 |

导出范围：**全部 / 仅已复核 / 指定分组 / 左侧勾选的图片**。

未标注像素在 COCO/YOLO 里就是背景；语义分割训练时把 `masks/*.png` 里的 `0` 映射成
`ignore_index` 即可（**不需要单独建 `unknown` 类别**，"未知"用忽略表达，而不是当成一个类去学）。

## 数据格式

```
data/projects/<项目>/
├── project.json        # 类别表、图像目录、阈值、每类颜色、排除列表
├── images.json         # 清单：id / relpath / status / n_instances / 视频帧元信息 / 分组
├── annotations/<id>.json
├── videos/<视频名>/frames/     # 视频抽帧
└── exports/                    # 导出结果
```

单个实例：

```json
{"id": 3, "kept": true, "class_name": "car", "score": 0.93,
 "box": [x0, y0, x1, y1], "area": 25189, "source": "text",
 "track_id": 5, "segmentation": {"size": [853, 1280], "counts": "..."}}
```

## 性能实测

RTX 4060 Ti 16 GB / 1280×853 输入：

| 项 | 数值 |
|---|---|
| 模型加载 | 6.6–7.4 s，占 3.2 GB 显存 |
| 单图推理（`set_image` + 10 个类别提示） | ≈ 1.85 s（图像编码 0.6 s + 每类 0.12–0.14 s） |
| 12 MP 大图（4032×2688，3 类） | 1.5 s，显存峰值 5.2 GB |
| 视频跨帧跟踪 | ≈ 0.3–0.6 s/帧，显存峰值 5.5–7.6 GB |
| 高亮叠加层渲染 | 0.16 s（首次）/ 0.04 s（切换选中）/ 1 ms（缓存命中） |

## HTTP API

常用接口（完整列表见 [docs/API.md](docs/API.md)）：

```
GET  /api/state                    项目统计 / 类别 / 阈值 / 颜色 / 引擎信息
GET  /api/images?status=&q=&offset=&limit=
GET  /api/images/{id}              单图标注（掩码走 PNG 接口）
POST /api/images/{id}/segment      按类别重跑
POST /api/images/{id}/prompt       追加一个提示词
POST /api/images/{id}/box_prompt   框选补标
POST /api/images/{id}/instances/{iid}/mask   画笔精修（add/sub 涂抹层）
GET  /media/image|thumb|overlay|labelmap/{id}
POST /api/thresholds /api/colors /api/refilter
POST /api/export                   支持 groups / images 限定范围
POST /api/video                    视频导入 + 自动标注（后台任务）
POST /api/jobs/{id}/pause|resume|cancel
```

## 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: psutil` | 官方 `sam3` 包漏声明依赖：`uv pip install psutil` |
| 找不到权重 | 跑 `./scripts/download_weights.sh`，确认 `weights/sam3.pt` 大小为 3,450,062,241 字节 |
| CUDA OOM（视频） | 关掉占显存的程序（如 UE 仿真）；把「最多帧数 / 长边 / 每段帧数」调小；或改用 `--mode frame` |
| 视频任务失败 | 任务条会显示具体原因（哪一类、哪一段）；分段失败不影响其它段 |
| 快捷键无反应 | 看工具栏「键 …」诊断行；不显示则是扩展拦截（换隐身窗口），或把焦点从搜索框移开 |
| 统计和实际不一致 | `python -m sam3label.cli resync --project <项目>` 按标注文件回填清单状态 |
| 缩略图与图片不一致 | 已按源文件 mtime 自动失效重建；仍异常时删掉 `data/projects/<项目>/.cache/thumbs` |
| 端口占用 | `fuser -k 8099/tcp` 后重启 |

## 已知限制

- 只做"图像/视频帧"标注，不支持在界面上对原始视频做时间轴浏览；
- `track` 模式的分段会在段边界重置 `track_id`；多类别需逐类各跑一次跟踪，耗时随类别数线性增长；
- 暂停/取消只在帧边界、图片边界生效，正在跑的模型推理不能打断（视频模型加载阶段也不响应取消）；
- 泛类别（`glass`、`floor`、`obstacle` 等）容易过检：`obstacle` 是语义角色而非视觉类别，
  建议拆成具体物体词，并对泛类单独提高阈值；
- Web 界面面向单机本地使用，未做鉴权，请只监听 `127.0.0.1`。

## 许可与引用

- **本仓库代码**：MIT License，见 [LICENSE](LICENSE)。
- **SAM 3 模型与权重**：由 Meta 以 *SAM License* 发布，本仓库不再分发；
  下载与商用请遵守 <https://github.com/facebookresearch/sam3> 的许可条款。
- 示例图片来自公开示例照片，仅用于演示标注效果。

```bibtex
@misc{carion2025sam3segmentconcepts,
  title={SAM 3: Segment Anything with Concepts},
  author={Carion, Nicolas and others},
  year={2025}, eprint={2511.16719}, archivePrefix={arXiv}, primaryClass={cs.CV}}
```
