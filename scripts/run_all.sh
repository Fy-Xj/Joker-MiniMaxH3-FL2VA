#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 一键复现（评委用）
#
#   用法：  bash scripts/run_all.sh
#   前提：  8 卡海光 gfx936 节点 + 官方镜像；仓库已 clone 到本地
#   耗时：  约 45 分钟（依赖 8 + InfiniCore 23 + InfiniLM 7 + compare 10）
#
#   做什么：
#     1) 解压 source/*.tar.gz 里的完整源码（已含全部补丁）
#     2) 安装 Python 依赖（含 torchvision / scikit-image）
#     3) 编译 InfiniCore（自动创建 fakeflash 并传 --flash-attn）
#     4) 编译 InfiniLM
#     5) 跑官方评分脚本 minimax_h3_compare.py
#
#   每一步失败都会立刻停下并打印原因，不会继续往下跑。
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

echo "=========================================================="
echo " Joker · MiniMax-H3 FL2VA · 一键复现"
echo " 仓库目录: $REPO"
echo "=========================================================="

# ---- 0) 环境变量 ----
# shellcheck disable=SC1091
source "$HERE/env.sh"

# ---- 1) 解压源码 ----
echo ""
echo "########## 步骤 1/5  解压完整源码 ##########"
mkdir -p "$HOME/pkg"
for t in InfiniCore InfiniLM; do
  TGZ="$REPO/source/${t}-patched.tar.gz"
  if [ ! -f "$TGZ" ]; then
    echo "!! 缺少 $TGZ"; exit 1
  fi
  if [ -d "$HOME/pkg/$t" ]; then
    echo "  $t 已存在，跳过"
  else
    tar -xzf "$TGZ" -C "$HOME/pkg" && echo "  解压 $t OK"
  fi
done
ls -d "$HOME/pkg/InfiniCore" "$HOME/pkg/InfiniLM"

# ---- 2) 依赖 ----
echo ""
echo "########## 步骤 2/5  Python 依赖 ##########"
bash "$HERE/setup_python.sh" || { echo "!! 依赖安装失败"; exit 1; }

# ---- 3) 编译 InfiniCore ----
echo ""
echo "########## 步骤 3/5  编译 InfiniCore ##########"
bash "$HERE/build_infinicore.sh" "$HOME/pkg/InfiniCore" || { echo "!! InfiniCore 编译失败"; exit 1; }

# ---- 4) 编译 InfiniLM ----
echo ""
echo "########## 步骤 4/5  编译 InfiniLM ##########"
bash "$HERE/build_infinilm.sh" "$HOME/pkg/InfiniLM" || { echo "!! InfiniLM 编译失败"; exit 1; }

# ---- 5) 官方评分脚本 ----
echo ""
echo "########## 步骤 5/5  官方评分脚本（FL2VA）##########"
echo "  约 10 分钟（复用已生成的 reference）"
bash "$HERE/run_compare.sh" || { echo "!! compare 失败"; exit 1; }

echo ""
echo "=========================================================="
echo " 全部完成"
echo "=========================================================="
echo "  期望看到："
echo "    [result] MiniMax-H3 video comparison"
echo "      Size:         PASS (PyTorch 768x928, InfiniLM 768x928)"
echo "      Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)"
echo "      Average SSIM: 61.182%"
echo ""
