#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 编译安装 InfiniCore（Hygon gfx936 + InfiniCCL + ATen）
#
# 用法：  bash scripts/build_infinicore.sh [InfiniCore源码目录]
# 默认：  $HOME/pkg/InfiniCore（若不存在则从 $AFS/upload_src.tar.gz 解压）
#
# 关键点：
#   - 必须先应用 patches/apply_all.sh（否则 add_orders 报错，配置直接失败）
#   - --aten=y 必需：原生路径需要 ATen 桥接，否则报 "cast_ requires ATen"
#   - --ccache=n 必需：ccache 与 xmake 2.9.9 高并发归档存在竞态
#   - -j 32 而非 64：-j 64 更容易触发 "cannot remove file ... No such file" 竞态
#   - 绝对不要再执行 `xmake f -y -cv`（无选项会重置全部配置为默认值！）
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

IC="${1:-$HOME/pkg/InfiniCore}"
SRC_TAR="${SRC_TAR:-$AFS/upload_src.tar.gz}"
JOBS="${JOBS:-32}"

if [ ! -d "$IC" ]; then
  echo "[1/4] 解压 InfiniCore 源码"
  [ -f "$SRC_TAR" ] || { echo "!! 缺少 $SRC_TAR"; exit 1; }
  mkdir -p "$HOME/pkg"
  tar -xzf "$SRC_TAR" -C "$HOME/pkg" InfiniCore
fi
cd "$IC" || exit 1

echo ""
echo "[2/4] 准备 --flash-attn 目录（★ 缺了这一步，海光 mha 算子不会被编译，一跑就段错误）"
FAKE="$AFS/fakeflash"
if [ -z "$FA_SO_PATH" ] || [ ! -f "$FA_SO_PATH" ]; then
  echo "  !! 未找到 flash_attn_2_cuda*.so，请先跑 scripts/setup_python.sh（步骤 4）"
  exit 1
fi
mkdir -p "$FAKE/flash_attn/lib"
ln -sf "$FA_SO_PATH" "$FAKE/flash_attn/lib/libflash_attention.so"
export FLASH_ATTN_2_CUDA_SO="$FA_SO_PATH"
echo "  FA_SO_PATH               = $FA_SO_PATH"
echo "  FLASH_ATTN_2_CUDA_SO     = $FLASH_ATTN_2_CUDA_SO"
echo "  $FAKE/flash_attn/lib/libflash_attention.so -> $FA_SO_PATH"

echo ""
echo "[2/4] 配置（唯一一次！）"
echo "      --hygon-dcu=y --hygon-arch=gfx936 --cuda=/opt/dtk/cuda/cuda"
echo "      --ccl=y --aten=y --cudnn=n --ccache=n --flash-attn=$FAKE"
rm -rf build .xmake 2>/dev/null
"$XMAKE" f -y -cv \
  --hygon-dcu=y --hygon-arch=gfx936 \
  --cuda=/opt/dtk/cuda/cuda \
  --ccl=y --aten=y --cudnn=n --ccache=n \
  --flash-attn="$FAKE" 2>&1 | grep -E "^\s+(aten|hygon-dcu|hygon-arch|ccl|cudnn|flash-attn|cuda|mode)\s*=|error|warning: " | head -20

echo ""
echo "      校验：上面输出里必须有 aten = true / hygon-dcu = true /"
echo "            hygon-arch = gfx936 / ccl = true / cudnn = false /"
echo "            ★ flash-attn = <非空路径>   ← 这一项为空则海光 mha 不会被编译，运行必崩"
echo "      ★ 千万不要为了'再看一眼配置'再发一次 xmake f（无选项会重置为默认！）"

echo ""
echo "[3/4] 编译（约 22 分钟，-j $JOBS）"
date
"$XMAKE" build -j "$JOBS" 2>&1 | tail -12
"$XMAKE" build -j "$JOBS" _infinicore 2>&1 | tail -8
date

echo ""
echo "[4/4] 安装"
"$XMAKE" install 2>&1 | tail -4
"$XMAKE" install _infinicore 2>&1 | tail -3

echo ""
echo "---- 产物检查（libinfiniop.so 必须 >20 MB，否则说明 Hygon 算子没编进去）----"
ls -lh "$INFINI_ROOT/lib" 2>&1

echo ""
echo "---- ★ 校验 mha 是否已改调厂商 flash-attn（结果必须 >= 1）----"
CNT=$(strings "$INFINI_ROOT/lib/libinfinicore_cpp_api.so" 2>/dev/null | grep -c "vllm_mha_varlen_fwd")
echo "  libinfinicore_cpp_api.so 中 vllm_mha_varlen_fwd 出现次数 = $CNT"
if [ "${CNT:-0}" -ge 1 ]; then
  echo "  FLASH_MHA_OK"
else
  echo "  !! FLASH_MHA_MISSING —— P5 补丁未生效或未编译，运行会退化为 ATen matmul（SSIM 只有 ~10%）"
fi
echo ""
echo "---- Python 扩展（必须同时存在于包内与 sys.path 顶层）----"
SO=$(find "$IC/build" -name "_infinicore*.so" 2>/dev/null | head -1)
if [ -n "$SO" ]; then
  cp -f "$SO" "$PL/"          && echo "  copied -> $PL/"
  cp -f "$SO" "$PL/infinicore/" && echo "  copied -> $PL/infinicore/"
else
  echo "  !! 未找到 _infinicore*.so"
fi

echo ""
echo "---- 自检 ----"
python3 - <<'PY'
try:
    import infinicore
    print("  INFINICORE_OK", infinicore.__file__)
except Exception as e:
    print("  INFINICORE_FAIL", type(e).__name__, str(e)[:120]); raise SystemExit(0)
try:
    d = infinicore.device("cuda", 0)
    print("  CUDA_ALIAS_OK", d)
except Exception as e:
    print("  CUDA_ALIAS_FAIL", type(e).__name__, str(e)[:120])
try:
    import torch
    t = infinicore.from_torch(torch.randn(4, 8, dtype=torch.bfloat16, device="cuda"))
    print("  FROM_TORCH_OK", t.shape)
except Exception as e:
    print("  FROM_TORCH_FAIL", type(e).__name__, str(e)[:120])
PY
echo ""
du -sh "$AFS" "$INFINI_ROOT" 2>/dev/null
echo "InfiniCore 构建流程结束。"
