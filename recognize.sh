#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PATH="${1:-}"
THRESHOLD="${2:-0.50}"

if [[ -z "$IMAGE_PATH" ]]; then
  echo "用法: bash recognize.sh 图片路径 [阈值]" >&2
  echo "示例: bash recognize.sh /path/to/photo.jpg" >&2
  exit 2
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "错误: 找不到图片: $IMAGE_PATH" >&2
  exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "face310" ]]; then
  CONDA_SCRIPT=""
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      CONDA_SCRIPT="$candidate"
      break
    fi
  done

  if [[ -z "$CONDA_SCRIPT" ]]; then
    echo "错误: 未找到 Conda；请先激活 face310 环境。" >&2
    exit 2
  fi

  # shellcheck disable=SC1090
  source "$CONDA_SCRIPT"
  conda activate face310
fi

exec python "$PROJECT_ROOT/face_cli.py" recognize \
  --image "$IMAGE_PATH" \
  --threshold "$THRESHOLD"
