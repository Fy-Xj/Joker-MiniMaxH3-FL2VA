#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 编译安装 InfiniLM（Python 扩展）
#
# 用法：  bash scripts/build_infinilm.sh [InfiniLM源码目录]
# 前提：  已成功执行 build_infinicore.sh（$INFINI_ROOT/include 与 lib 必须存在）
#
# 关键点：
#   - 必须已应用 P3 补丁（python.module -> python.library），否则 xmake f 报
#     "unknown rule(python.module)"
#   - _infinilm 链接 $INFINI_ROOT/lib 下的 infinicore_cpp_api / infiniop /
#     infinirt / infiniccl，所以 InfiniCore 必须先 install
#   - 产出的 _infinilm*.so 必须同时放到 $PL/ 和 $PL/infinilm/
#     （infinilm/lib/__init__.py 用的是绝对导入 `import _infinilm`）
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

LM="${1:-$HOME/pkg/InfiniLM}"
SRC_TAR="${SRC_TAR:-$AFS/upload_src.tar.gz}"
JOBS="${JOBS:-32}"

if [ ! -d "$LM" ]; then
  echo "[1/4] 解压 InfiniLM 源码"
  [ -f "$SRC_TAR" ] || { echo "!! 缺少 $SRC_TAR"; exit 1; }
  mkdir -p "$HOME/pkg"
  tar -xzf "$SRC_TAR" -C "$HOME/pkg" InfiniLM
fi
cd "$LM" || exit 1

echo ""
echo "[2/4] 前置检查：InfiniCore 是否已安装"
if [ ! -d "$INFINI_ROOT/include" ] || [ ! -d "$INFINI_ROOT/lib" ]; then
  echo "!! 缺少 $INFINI_ROOT/include 或 lib，请先执行 build_infinicore.sh"
  exit 1
fi
ls "$INFINI_ROOT/lib" | head -6

echo ""
echo "[3/4] 配置并编译 _infinilm（约 7 分钟）"
rm -rf build .xmake 2>/dev/null
"$XMAKE" f -y -m release --ccache=n 2>&1 | tail -6
date
"$XMAKE" build -j "$JOBS" _infinilm 2>&1 | tail -10
date

echo ""
echo "[4/4] 安装并部署 .so"
"$XMAKE" install _infinilm 2>&1 | tail -3

SO=$(find "$LM/build" -name "_infinilm*.so" 2>/dev/null | head -1)
echo "SO=$SO"
if [ -n "$SO" ]; then
  mkdir -p "$PL/infinilm"
  # 若 python 包尚未复制，先复制
  [ -d "$PL/infinilm" ] || cp -r "$LM/python/infinilm" "$PL/"
  cp -r "$LM/python/infinilm/." "$PL/infinilm/" 2>/dev/null
  cp -f "$SO" "$PL/"            && echo "  copied -> $PL/          (供绝对导入 import _infinilm)"
  cp -f "$SO" "$PL/infinilm/"   && echo "  copied -> $PL/infinilm/"
else
  echo "  !! 未找到 _infinilm*.so"
fi

echo ""
echo "---- 自检 ----"
python3 - <<'PY'
for code, name in [
    ("import infinicore", "INFINICORE_OK"),
    ("import infinilm", "INFINILM_OK"),
    ("from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder", "ENCODER_IMPORT_OK"),
    ("from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner", "RUNNER_IMPORT_OK"),
]:
    try:
        exec(code)
        print("  " + name)
    except Exception as e:
        print(f"  {name.replace('_OK','_FAIL')} {type(e).__name__}: {str(e)[:140]}")
PY
echo ""
du -sh "$AFS" 2>/dev/null
echo "InfiniLM 构建流程结束。"
