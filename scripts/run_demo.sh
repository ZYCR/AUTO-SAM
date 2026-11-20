#!/usr/bin/env bash
# 重建示例项目（6 张示例图）的标注，然后起复核界面
#   ./scripts/run_demo.sh [端口]
# 注意：会用 10 个示例类别重跑示例图，覆盖原有标注；只想看现成结果用 start_samples.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PORT=${1:-8099}
PROJ=data/projects/samples
CLASSES=${CLASSES:-"car,person,building,tree,cup,bag,phone,tablet,leaf,mountain"}

"$PY" -m sam3label.cli annotate \
  --project "$PROJ" --image-dir "$PROJ/images" \
  --classes "$CLASSES" --conf 0.5 --overwrite

"$PY" -m sam3label.cli stats --project "$PROJ"

echo "== 复核界面: http://127.0.0.1:$PORT =="
exec "$PY" -m sam3label.cli serve --project "$PROJ" --port "$PORT"
