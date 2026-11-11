#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— FL2VA 正式生成（参考实现路径，已验证 ✅）
#
# 用法：
#   bash scripts/run_fl2va_legacy.sh [输出文件名]
#
# 产物规格（与官方评分脚本示例一致）：
#   768×928 / 124 帧 / 24 FPS / 5.175 s / H.264(High) yuv420p / AAC 32kHz 立体声
#
# 重要：不要传 --height / --width
#   不指定时 pipeline 由首帧图（800×965）推导尺寸 -> 768×928。
#   若手动指定 384×576，则与官方参考输出规格不一致，评分会扣分。
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

OUTNAME="${1:-fl2va_official.mp4}"
OUTDIR="${OUTDIR:-$HOME/out}"
LM="${LM:-$HOME/pkg/InfiniLM}"
SRC_TAR="${SRC_TAR:-$AFS/upload_src.tar.gz}"

# ---- 前置检查 ----
echo "=========== 前置检查 ==========="
[ -f "$FIRST_FRAME" ] || { echo "!! 官方首帧图不存在: $FIRST_FRAME"; exit 1; }
echo "首帧图: $FIRST_FRAME"
python3 -c "
from PIL import Image; import sys
im = Image.open(sys.argv[1]); print('尺寸:', im.size, im.mode)
" "$FIRST_FRAME"
[ -d "$MODEL/diffusers" ] || { echo "!! 模型目录不存在: $MODEL/diffusers"; exit 1; }
ls "$MODEL/diffusers" | head -12
python3 -c "import torch;print('torch', torch.__version__, 'devices', torch.cuda.device_count())" 2>&1 | tail -1
echo "FA_SO_PATH=${FA_SO_PATH:-<none>}"

# ---- 准备源码 ----
if [ ! -d "$LM" ]; then
  mkdir -p "$HOME/pkg"
  tar -xzf "$SRC_TAR" -C "$HOME/pkg" InfiniLM
fi
mkdir -p "$OUTDIR"
cd "$LM" || exit 1

echo ""
echo "=========== 生成 FL2VA（124 帧 / 50 步 / 8 卡 / 参考实现）==========="
date
python3 -u examples/minimax_h3.py fl2va \
  --model-path="$MODEL/diffusers" \
  --text="A dog falls asleep" \
  --first-frame="$FIRST_FRAME" \
  --gpus=0,1,2,3,4,5,6,7 \
  --legacy \
  --legacy-text-gpus=0,1,2,3 \
  --legacy-transformer-gpus=4,5,6,7 \
  --num-frames=124 \
  --steps=50 \
  --seed=42 \
  --output="$OUTDIR/$OUTNAME" 2>&1 | tail -45
RC=$?
date
echo "exit=$RC"

echo ""
echo "=========== 产物核查 ==========="
ls -lh "$OUTDIR" 2>&1
if [ -f "$OUTDIR/$OUTNAME" ]; then
  echo "FINAL_MP4_OK"
  cp "$OUTDIR/$OUTNAME" "$AFS/$OUTNAME" 2>/dev/null && echo "copied_to_AFS: $AFS/$OUTNAME"
  bash "$HERE/verify_output.sh" "$OUTDIR/$OUTNAME"
else
  echo "NO_OUTPUT"
  exit 1
fi
echo ""
du -sh "$AFS" 2>/dev/null
echo "FL2VA 流程结束。"
