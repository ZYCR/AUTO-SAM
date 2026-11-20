#!/usr/bin/env bash
# 启动复核界面（已有项目）
#   ./scripts/serve.sh [项目目录] [端口]      # 不传项目默认用干净示例 samples
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PROJ=${1:-data/projects/samples}
PORT=${2:-8099}

exec "$PY" -m sam3label.cli serve --project "$PROJ" --port "$PORT"
