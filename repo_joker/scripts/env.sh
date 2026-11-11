#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 统一环境变量
#
# 用法：  source scripts/env.sh
# 说明：  所有构建/运行脚本都会自行 source 本文件，也可单独 source 用于交互调试。
# =============================================================================

# ---- 目录约定（可按实际部署调整）----
export AFS="${IFAB_AFS_HOME:-$HOME}"          # 个人 AFS 根目录
export PL="$AFS/pylibs"                       # Python 依赖安装目录
export INFINI_ROOT="$AFS/infini"              # InfiniCore 安装前缀
export XMAKE="$AFS/xmake299"                  # xmake 2.9.9 可执行文件
export SHARED="${SHARED:-/public/home/aclsylqidf/shared}"   # 共享只读模型目录
export MODEL="$SHARED/MiniMax-H3"             # MiniMax-H3 权重根目录
export FIRST_FRAME="${FIRST_FRAME:-$AFS/Golde33443.png}"    # 官方首帧图

# ---- DTK 环境 ----
if [ -f /opt/dtk/env.sh ]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi

# ---- Python 路径 ----
# $AFS/shim 提供 sitecustomize.py，修复 /etc 只读导致的 pip 崩溃
export PYTHONPATH="$PL:$AFS/shim${PYTHONPATH:+:$PYTHONPATH}"

# ---- 动态库路径 ----
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- FlashAttention 动态库（SDPA 后端必需，缺了 math 路径需 158 GiB 显存）----
if [ -z "$FA_SO_PATH" ] && [ -d "$PL" ]; then
  FA_SO_PATH="$(find "$PL" -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit 2>/dev/null)"
  export FA_SO_PATH
fi

# ---- 缓存写临时盘，避免占满 4 GiB AFS 配额 ----
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR" "$HF_HOME" 2>/dev/null

# ---- xmake ----
export XMAKE_ROOT=y

# ---- 打印（静默模式由 QUIET=1 控制）----
if [ "${QUIET:-0}" != "1" ]; then
  echo "[env] AFS          = $AFS"
  echo "[env] PL           = $PL"
  echo "[env] INFINI_ROOT  = $INFINI_ROOT"
  echo "[env] XMAKE        = $XMAKE"
  echo "[env] MODEL        = $MODEL"
  echo "[env] FIRST_FRAME  = $FIRST_FRAME"
  echo "[env] FA_SO_PATH   = ${FA_SO_PATH:-<not found>}"
fi
