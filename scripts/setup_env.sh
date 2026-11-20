#!/usr/bin/env bash
# 创建独立运行环境（Python 3.12 + torch cu128 + sam3）
# 说明：torch 走阿里云 pytorch-wheels 镜像，其余包走阿里云 PyPI 镜像；
#       download.pytorch.org 在本机只有 ~1MB/s，实测镜像可到 5-8MB/s。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
VENV=${VENV:-$ROOT/.venv}

echo "== 1/3 创建 venv ($VENV) =="
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
fi

echo "== 2/3 安装 torch 2.10.0+cu128 / torchvision 0.25.0+cu128 =="
WHEELS=${WHEELS:-/tmp/sam3-wheels}
mkdir -p "$WHEELS"
for f in "torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
         "torchvision-0.25.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"; do
  out="$WHEELS/${f//%2B/+}"
  if [ ! -f "$out" ]; then
    echo "下载 $f"
    curl -s --noproxy '*' -L -C - -o "$out" "https://mirrors.aliyun.com/pytorch-wheels/cu128/$f"
  fi
done
uv pip install --python "$VENV/bin/python" --default-index https://mirrors.aliyun.com/pypi/simple/ \
  "$WHEELS"/torch-2.10.0+cu128-*.whl "$WHEELS"/torchvision-0.25.0+cu128-*.whl

echo "== 3/3 安装 sam3 与 Web 依赖 =="
uv pip install --python "$VENV/bin/python" --default-index https://mirrors.aliyun.com/pypi/simple/ \
  sam3 fastapi "uvicorn[standard]" opencv-python-headless pillow pyyaml

echo
echo "完成。验证："
echo "  $VENV/bin/python -m sam3label.cli doctor --load"
