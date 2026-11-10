#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— Python 依赖安装
#
# 关键设计（血泪教训，见技术报告 §3）：
#   1. 必须整套从厂商源 pypi.sourcefind.cn 安装，禁止与 PyPI 混装
#      - transformers 5.22.1 要求 tokenizers<=0.23.0，而 PyPI 无该版本
#      - hub 2.4.0 需要 httpx2/httpcore2（不是 httpx/httpcore）
#      - diffusers 0.42 需要 hub.resolve_revision（平台自带 hub 0.36 无此 API）
#   2. 不安装 triton：解压后 >1 GB，会撑爆 4 GiB AFS；且 SDPA 走 .so 不需要它
#   3. 所有 pip 用 --no-deps，避免把 torch(731 MB) 装进 AFS
#   4. torchvision 必须源码编译（官方要求，否则 processor 为 None）
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

VEND="https://pypi.sourcefind.cn/release/dtk/"
TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"
SRC_TAR="${SRC_TAR:-$AFS/upload_src.tar.gz}"
WHL_TAR="${WHL_TAR:-$AFS/upload_wheels.tar.gz}"

echo "=========================================================="
echo " 步骤 1/5  写入 sitecustomize.py（修复 /etc 只读导致 pip 崩溃）"
echo "=========================================================="
mkdir -p "$AFS/shim"
cat > "$AFS/shim/sitecustomize.py" <<'PY'
"""修复 /etc 只读环境下的两个问题：
   1) pip._vendor.distro 读 /etc/debian_version 抛 PermissionError
   2) mimetypes.init() 读 /etc/mime.types 抛 PermissionError
"""
import sys
import types

# ---- 1) 替换 pip._vendor.distro ----
if "pip._vendor.distro" not in sys.modules:
    m = types.ModuleType("pip._vendor.distro")

    class _Distro(object):
        name = "Ubuntu"
        version = "22.04"
        codename = "jammy"
        id = "ubuntu"
        like = "debian"

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, n):
            return None

    m.LinuxDistribution = _Distro
    m.Distro = _Distro
    m.name = lambda *a, **k: "Ubuntu"
    m.version = lambda *a, **k: "22.04"
    m.codename = lambda *a, **k: "jammy"
    m.id = lambda *a, **k: "ubuntu"
    m.like = lambda *a, **k: "debian"
    sys.modules["pip._vendor.distro"] = m

# ---- 2) 手工构造 mimetypes 数据库 ----
try:
    import mimetypes as _m
    _m.inited = True
    if getattr(_m, "_db", None) is None:
        db = _m.MimeTypes()
        for _t, _e in (
            ("application/octet-stream", "whl"),
            ("application/zip", "zip"),
            ("text/plain", "txt"),
            ("application/x-tar", "tar"),
        ):
            db.add_type(_t, _e)
        _m._db = db
except Exception:
    pass
PY
echo "  写入 $AFS/shim/sitecustomize.py"
python3 -c "import mimetypes; print('  mimetypes OK ->', mimetypes.guess_type('a.whl'))"
python3 -m pip --version 2>&1 | head -1

echo ""
echo "=========================================================="
echo " 步骤 2/5  安装基础依赖（厂商源整套，--no-deps）"
echo "=========================================================="
mkdir -p "$PL"
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip install --no-cache-dir --upgrade --target="$PL" \
  --no-deps --no-compile -i "$VEND" \
  diffusers transformers tokenizers huggingface_hub accelerate \
  httpx2 httpcore2 av einops regex imageio safetensors 2>&1 | tail -14

echo ""
echo "=========================================================="
echo " 步骤 3/5  补装 hub 的 TLS 依赖 truststore（PyPI 清华镜像）"
echo "=========================================================="
python3 -m pip install --no-cache-dir --no-deps --no-compile --target="$PL" \
  -i "$TUNA" --trusted-host pypi.tuna.tsinghua.edu.cn \
  truststore wheel 2>&1 | tail -6

echo ""
echo "=========================================================="
echo " 步骤 4/5  安装厂商版 flash-attn（SDPA 后端，必需）"
echo "=========================================================="
python3 -m pip install --no-cache-dir --upgrade --target="$PL" \
  --no-deps --no-compile -i "$VEND" \
  "flash-attn==2.6.1+das.opt1.dtk2604.torch290" 2>&1 | tail -8

echo ""
echo "=========================================================="
echo " 步骤 5/5  源码编译 torchvision 0.24.0（官方要求）"
echo "=========================================================="
if python3 -c "import torchvision" 2>/dev/null; then
  echo "  torchvision 已存在，跳过"
else
  if [ ! -f "$SRC_TAR" ]; then
    echo "  !! 缺少源码包 $SRC_TAR，跳过 torchvision"
  else
    mkdir -p "$HOME/pkg"
    tar -xzf "$SRC_TAR" -C "$HOME/pkg" torchvision-src
    cd "$HOME/pkg/torchvision-src/vision-0.24.0" || exit 1
    export BUILD_VERSION=0.24.0
    export MAX_JOBS=32
    export FORCE_CUDA=1
    export TORCHVISION_USE_NVJPEG=0
    export TORCHVISION_USE_VIDEO_CODEC=0
    export TORCHVISION_USE_FFMPEG=0
    echo "  编译中（约 4～5 分钟）..."
    python3 setup.py bdist_wheel 2>&1 | tail -6
    for w in dist/*.whl; do
      [ -f "$w" ] || continue
      python3 -m pip install --no-index --no-deps --target="$PL" --upgrade "$w" 2>&1 | tail -3
    done
    cd "$HERE" || exit 1
  fi
fi

echo ""
echo "=========================================================="
echo " 验证"
echo "=========================================================="
python3 - <<'PY'
mods = ["torch","diffusers","transformers","tokenizers","huggingface_hub",
        "httpx2","httpcore2","av","einops","torchvision","truststore"]
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "")
        print(f"  OK  {m:16s} {v}")
    except Exception as e:
        print(f"  FAIL {m:16s} {type(e).__name__}: {str(e)[:60]}")

print()
for code, name in [
    ("from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference", "MODULAR_H3_OK"),
    ("from diffusers import MiniMaxH3Transformer3DModel", "H3_MODEL_OK"),
    ("from transformers import Qwen3VLForConditionalGeneration", "QWEN3VL_OK"),
    ("from huggingface_hub import resolve_revision", "RESOLVE_REV_OK"),
    ("import torchvision.transforms.functional", "TV_TRANSFORMS_OK"),
]:
    try:
        exec(code)
        print("  " + name)
    except Exception as e:
        print(f"  {name.replace('_OK','_FAIL')} {type(e).__name__}: {str(e)[:80]}")
PY
echo ""
du -sh "$AFS" 2>/dev/null
echo "依赖安装流程结束。"
