#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 官方评分脚本
#
# 用法：
#   bash scripts/run_compare.sh                     # 完整跑（自己生成 reference + infinilm）
#   bash scripts/run_compare.sh --reuse             # 复用已有 reference 视频（省 10 分钟）
#
# 说明：
#   minimax_h3_compare.py 内部对每个 task 跑两次：
#       reference : minimax_h3.py ... --legacy      ← PyTorch 标准答案
#       infinilm  : minimax_h3.py ...（无 --legacy）← InfiniLM 原生
#   然后比较 Size / Length / 平均 SSIM。
#
#   脚本写死的参数：STEPS=50, NUM_FRAMES=124, SEED=42, FL2VA_PROMPT="A dog falls asleep"
#   且不传 --height/--width（尺寸由首帧推导 -> 768×928）
#
# 注意：初赛判定要求本脚本能完整跑通（--tasks=fl2va）。
#       截至本工程提交时，infinilm 侧仍受阻（见技术报告 §8.3）。
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

LM="${LM:-$HOME/pkg/InfiniLM}"
SRC_TAR="${SRC_TAR:-$AFS/upload_src.tar.gz}"
REUSE=0
[ "${1:-}" = "--reuse" ] && REUSE=1

if [ ! -d "$LM" ]; then
  mkdir -p "$HOME/pkg"
  tar -xzf "$SRC_TAR" -C "$HOME/pkg" InfiniLM
fi
cd "$LM" || exit 1

[ -f "$FIRST_FRAME" ] || { echo "!! 缺少官方首帧图: $FIRST_FRAME"; exit 1; }

ARGS=(
  --first-frame="$FIRST_FRAME"
  --tasks=fl2va
  --legacy-gpus=0,1,2,3,4,5,6,7
  --legacy-memory-fraction=0.4
  --infinilm-gpus=0,1,2,3,4,5,6,7
)

if [ "$REUSE" = "1" ] && [ -f "$AFS/fl2va_official.mp4" ]; then
  echo "复用已有 reference 视频: $AFS/fl2va_official.mp4"
  ARGS+=(--reference-fl2va-video="$AFS/fl2va_official.mp4")
fi

echo "=========== 运行官方评分脚本 ==========="
echo "python examples/minimax_h3_compare.py ${ARGS[*]}"
date
python3 -u examples/minimax_h3_compare.py "${ARGS[@]}" 2>&1 | tail -60
RC=$?
date
echo "exit=$RC"

echo ""
echo "=========== 结果目录 ==========="
OUTDIR="$LM/minimax_h3_comparison"
ls -lh "$OUTDIR" 2>&1
for f in "$OUTDIR"/*.mp4; do
  [ -f "$f" ] || continue
  cp "$f" "$AFS/" 2>/dev/null && echo "copied $(basename "$f") -> $AFS/"
done

echo ""
echo "=========== 期望输出格式 ==========="
cat <<'EOF'
[result] MiniMax-H3 video comparison

  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: xx.xxx%
EOF
exit $RC
