#!/usr/bin/env bash
# 启动"干净示例"界面：只有 7 张示例图，启动不扫描、不标注任何其它图片
#   ./scripts/start_samples.sh [端口]
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PORT=${1:-8099}

exec "$PY" -m sam3label.cli serve \
  --project data/projects/samples \
  --port "$PORT" \
  --no-scan
