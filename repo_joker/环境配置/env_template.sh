#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 环境变量模板
#
# 用法：
#   cp 环境配置/env_template.sh 环境配置/env.local.sh
#   按实际部署修改下方「需修改」段落
#   source 环境配置/env.local.sh
# =============================================================================

# ======================= 需修改：按平台实际情况填写 =======================
export SHARED="/public/home/aclsylqidf/shared"      # 共享只读模型根目录
export FIRST_FRAME="$AFS/Golde33443.png"            # 官方首帧图（800×965）
# ==========================================================================

# ======================= 通常无需修改 =======================
# 个人 AFS 根目录（平台注入）
export AFS="${IFAB_AFS_HOME:-$HOME}"

# 派生路径
export MODEL="$SHARED/MiniMax-H3"
export PL="$AFS/pylibs"                 # Python 依赖
export INFINI_ROOT="$AFS/infini"        # InfiniCore 安装前缀
export XMAKE="$AFS/xmake299"            # xmake 2.9.9

# 产物与临时目录
export OUTDIR="${OUTDIR:-$HOME/out}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# DTK / HIP 环境
if [ -f /opt/dtk/env.sh ]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi

# Python 搜索路径（shim 用于修复 /etc 只读导致的 pip 崩溃）
export PYTHONPATH="$PL:$AFS/shim${PYTHONPATH:+:$PYTHONPATH}"

# 动态库搜索路径
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# FlashAttention 动态库（缺了 SDPA 会 fallback math 路径，需 158 GiB 显存）
if [ -z "${FA_SO_PATH:-}" ] && [ -d "$PL" ]; then
  FA_SO_PATH="$(find "$PL" -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit 2>/dev/null)"
  export FA_SO_PATH
fi

# xmake 需要提权运行（无 root 场景）
export XMAKE_ROOT=y

# 构建并行度（128 核机器建议 32；过高会触发 xmake 归档竞态）
export JOBS="${JOBS:-32}"
# ==========================================================================

mkdir -p "$TMPDIR" "$HF_HOME" "$OUTDIR" 2>/dev/null

echo "[env] SHARED       = $SHARED"
echo "[env] MODEL        = $MODEL"
echo "[env] FIRST_FRAME  = $FIRST_FRAME"
echo "[env] AFS          = $AFS"
echo "[env] PL           = $PL"
echo "[env] INFINI_ROOT  = $INFINI_ROOT"
echo "[env] XMAKE        = $XMAKE"
echo "[env] FA_SO_PATH   = ${FA_SO_PATH:-<not found>}"
echo "[env] HF_HOME      = $HF_HOME"
echo "[env] JOBS         = $JOBS"
