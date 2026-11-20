"""全局路径与默认参数。所有路径都可以用环境变量覆盖，方便换机器部署。"""

from __future__ import annotations

import os
from pathlib import Path

# 减少显存碎片：视频跟踪里 mask_iou 会要大块连续显存
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# 项目根目录（本文件位于 <root>/sam3label/config.py）
ROOT = Path(__file__).resolve().parent.parent

# SAM3 权重与 BPE 词表
WEIGHTS_DIR = Path(os.environ.get("SAM3_WEIGHTS_DIR", ROOT / "weights"))
CHECKPOINT = Path(os.environ.get("SAM3_CHECKPOINT", WEIGHTS_DIR / "sam3.pt"))
BPE_PATH = Path(
    os.environ.get("SAM3_BPE", ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz")
)

# 工作区默认位置
DATA_DIR = Path(os.environ.get("SAM3_DATA_DIR", ROOT / "data"))
PROJECTS_DIR = Path(os.environ.get("SAM3_PROJECTS_DIR", DATA_DIR / "projects"))

DEVICE = os.environ.get("SAM3_DEVICE", "cuda")

# 推理分辨率（官方 1008；显存紧张时可降到 672/504，会损失小目标精度）
RESOLUTION = int(os.environ.get("SAM3_RESOLUTION", "1008"))

# 默认置信度阈值（SAM3 presence*score 的乘积）
CONFIDENCE = float(os.environ.get("SAM3_CONFIDENCE", "0.5"))

# 单张图默认最小实例面积（像素），用于过滤碎片
MIN_AREA = int(os.environ.get("SAM3_MIN_AREA", "100"))

# 跨类别去重（不同提示词命中同一物体时）的掩码 IoU 阈值
DEDUP_IOU = float(os.environ.get("SAM3_DEDUP_IOU", "0.75"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 支持的视频容器（解码走 OpenCV，常见编码都能读）
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".flv", ".wmv"}


def ensure_dirs() -> None:
    for p in (WEIGHTS_DIR, DATA_DIR, PROJECTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
