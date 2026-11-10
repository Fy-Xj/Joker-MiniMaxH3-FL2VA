#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 输出规格核查（薄封装）
#
# 用法：  bash scripts/verify_output.sh <mp4文件>
# 依赖：  tests/check_spec.py
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
# shellcheck disable=SC1091
source "$HERE/env.sh"

TARGET="${1:-$AFS/fl2va_official.mp4}"

if [ ! -f "$TARGET" ]; then
  echo "!! 文件不存在: $TARGET"
  exit 2
fi

echo "--- ffmpeg 流信息（若可用）---"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -hide_banner -i "$TARGET" 2>&1 | grep -E "Duration|Stream" | sed 's/^/  /'
else
  echo "  （未安装 ffmpeg，跳过）"
fi
echo ""

python3 "$ROOT/tests/check_spec.py" "$TARGET"
RC=$?

echo ""
echo "--- 首末帧抽取（用于人工核对画面内容）---"
if command -v ffmpeg >/dev/null 2>&1; then
  FRAMES="$HOME/frames"; mkdir -p "$FRAMES"
  for t in 0.05 5.00; do
    out="$FRAMES/f_$(echo "$t" | tr '.' '_').png"
    ffmpeg -y -hide_banner -loglevel error -ss "$t" -i "$TARGET" -frames:v 1 "$out" 2>/dev/null \
      && echo "  抽出 $t s -> $out"
  done
else
  echo "  （未安装 ffmpeg，跳过）"
fi

exit $RC
