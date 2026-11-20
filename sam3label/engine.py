"""SAM3 推理引擎封装：一次加载模型，多次复用图像 embedding。

对外只暴露两个动作：
  set_image(pil)                -> 计算并缓存图像特征（约 0.3~1s）
  segment_text(state, class…)   -> 对每个类别提示词跑一次 grounding，取回掩码
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config
from .postprocess import bbox_from_mask


@dataclass
class Instance:
    """一个实例的中间表示（掩码用 bool numpy 保存）。"""

    class_name: str
    score: float
    box: list[float]  # xyxy, 原图像素
    mask: np.ndarray  # HxW bool
    source: str = "text"  # text | box

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def to_record(self) -> dict:
        from .postprocess import encode_mask

        return {
            "class_name": self.class_name,
            "score": float(self.score),
            "box": [float(v) for v in self.box],
            "area": self.area,
            "source": self.source,
            "segmentation": encode_mask(self.mask),
        }


@dataclass
class SegmentResult:
    instances: list[Instance]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.instances)


def _pil_to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


class SAM3Engine:
    """单例式引擎。CUDA 上下文只初始化一次。"""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        bpe_path: str | Path | None = None,
        device: str | None = None,
        resolution: int | None = None,
        confidence: float | None = None,
    ) -> None:
        checkpoint = Path(checkpoint or config.CHECKPOINT)
        bpe_path = Path(bpe_path or config.BPE_PATH)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"找不到权重 {checkpoint}；先运行 scripts/download_weights.sh"
            )
        if not bpe_path.exists():
            raise FileNotFoundError(f"找不到 BPE 词表 {bpe_path}")

        self.device = device or config.DEVICE
        self.resolution = resolution or config.RESOLUTION
        self.confidence = confidence if confidence is not None else config.CONFIDENCE

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        t0 = time.perf_counter()
        self.model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
            enable_segmentation=True,
            # 不做 SAM1 交互式点提示（省 ~1.5GB 显存）；box 提示走几何编码器即可
            enable_inst_interactivity=False,
            compile=False,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=self.resolution,
            device=self.device,
            confidence_threshold=self.confidence,
        )
        self.load_seconds = time.perf_counter() - t0
        self._state: dict | None = None
        self._state_key: str | None = None

    # ------------------------------------------------------------------ 图像
    def set_image(self, image: Image.Image, key: str | None = None) -> dict:
        """计算图像特征。若 key 命中缓存则直接返回。"""
        if key is not None and self._state_key == key and self._state is not None:
            return self._state
        self._drop_state()
        self._state = self.processor.set_image(image)
        self._state_key = key
        return self._state

    def _drop_state(self) -> None:
        if self._state is not None:
            self._state = None
            self._state_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 文本
    def segment_text(
        self,
        state: dict,
        classes: list[str],
        confidence: float | None = None,
        min_area: int = 0,
        dedup_iou: float | None = None,
        class_agnostic_dedup: bool = False,
        class_conf: dict[str, float] | None = None,
    ) -> SegmentResult:
        """对一组类别提示词做 grounding，返回去重后的实例列表。"""
        from .postprocess import filter_instances

        conf = self.confidence if confidence is None else confidence
        self.processor.set_confidence_threshold(conf, state=None)
        timings: dict[str, float] = {}
        collected: list[Instance] = []
        t_all = time.perf_counter()
        for cls in classes:
            # 每个类别可以用自己的阈值（覆盖全局 conf）
            thr = float((class_conf or {}).get(cls, conf))
            self.processor.set_confidence_threshold(thr, state=None)
            t0 = time.perf_counter()
            out = self.processor.set_text_prompt(prompt=cls, state=state)
            timings[f"text:{cls}"] = time.perf_counter() - t0
            masks = out["masks"]
            scores = out["scores"]
            if masks is None or len(masks) == 0:
                continue
            masks_np = masks.detach().cpu().numpy()
            if masks_np.ndim == 4:  # [N,1,H,W]
                masks_np = masks_np[:, 0]
            masks_np = masks_np.astype(bool)
            scores_np = scores.detach().cpu().numpy()
            for m, s in zip(masks_np, scores_np):
                collected.append(
                    Instance(
                        class_name=cls,
                        score=float(s),
                        box=bbox_from_mask(m),
                        mask=m,
                    )
                )
            # 释放全分辨率 logits，避免多类别叠加爆显存
            for k in ("masks_logits", "masks", "boxes", "scores"):
                if k in out:
                    del out[k]
        timings["grounding_total"] = time.perf_counter() - t_all
        kept = filter_instances(
            collected,
            min_area=min_area,
            dedup_iou=config.DEDUP_IOU if dedup_iou is None else dedup_iou,
            class_agnostic=class_agnostic_dedup,
        )
        return SegmentResult(instances=kept, timings=timings)

    # ------------------------------------------------------------------ 框提示
    def segment_with_box(
        self,
        state: dict,
        box_xyxy: list[float],
        image_size: tuple[int, int],
        class_name: str = "object",
        text_hint: str | None = None,
        confidence: float | None = None,
    ) -> SegmentResult:
        """用框提示补标一个漏掉的实例（可附带文本提示）。"""
        from .postprocess import filter_instances

        conf = self.confidence if confidence is None else confidence
        width, height = image_size
        x0, y0, x1, y1 = box_xyxy
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        bw, bh = (x1 - x0) / width, (y1 - y0) / height
        t0 = time.perf_counter()
        self.processor.reset_all_prompts(state)
        self.processor.set_confidence_threshold(conf, state=None)
        if text_hint:
            self.processor.set_text_prompt(prompt=text_hint, state=state)
        out = self.processor.add_geometric_prompt(
            box=[cx, cy, bw, bh], label=True, state=state
        )
        masks = out["masks"]
        scores = out["scores"]
        instances: list[Instance] = []
        if masks is not None and len(masks) > 0:
            masks_np = masks.detach().cpu().numpy()
            if masks_np.ndim == 4:  # [N,1,H,W]
                masks_np = masks_np[:, 0]
            masks_np = masks_np.astype(bool)
            scores_np = scores.detach().cpu().numpy()
            for m, s in zip(masks_np, scores_np):
                instances.append(
                    Instance(
                        class_name=text_hint or class_name,
                        score=float(s),
                        box=bbox_from_mask(m),
                        mask=m,
                        source="box",
                    )
                )
        kept = filter_instances(instances, min_area=0, dedup_iou=1.01)
        return SegmentResult(
            instances=kept, timings={"box_prompt": time.perf_counter() - t0}
        )

    # ------------------------------------------------------------------ 信息
    def info(self) -> dict:
        info = {
            "device": self.device,
            "resolution": self.resolution,
            "confidence": self.confidence,
            "load_seconds": round(self.load_seconds, 2),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": None,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "total_gb": round(props.total_memory / 1024**3, 1),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            }
        return info
