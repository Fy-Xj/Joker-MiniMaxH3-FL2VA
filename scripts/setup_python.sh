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
echo " 步骤 3b  安装 scikit-image + scipy（官方 compare 算 SSIM 必需）"
echo "=========================================================="
echo "  ★ 必须 --no-deps：否则 pip 会把 numpy 升到 2.x，torch.from_numpy 直接报"
echo "    \"Numpy is not available\"，整个模型都跑不了"
python3 -m pip install --no-cache-dir --upgrade --target="$PL" \
  --no-deps --no-compile -i "$TUNA" --trusted-host pypi.tuna.tsinghua.edu.cn \
  scikit-image scipy tifffile lazy_loader networkx packaging 2>&1 | tail -10
echo "  校验："
python3 -c "import numpy; print('    numpy =', numpy.__version__)" 2>&1 | tail -1
python3 -c "import torch, numpy; torch.from_numpy(numpy.zeros(1)); print('    FROM_NUMPY_OK')" 2>&1 | tail -2
python3 -c "from skimage.metrics import structural_similarity; print('    SKIMAGE_OK')" 2>&1 | tail -2

echo ""
echo "=========================================================="
echo " 步骤 4/5  安装厂商版 flash-attn（SDPA 后端，必需）"
echo "=========================================================="
python3 -m pip install --no-cache-dir --upgrade --target="$PL" \
  --no-deps --no-compile -i "$VEND" \
  "flash-attn==2.6.1+das.opt1.dtk2604.torch290" 2>&1 | tail -8

echo ""
echo "=========================================================="
echo " 步骤 5/5  torchvision（Qwen3-VL 的 processor 依赖）"
echo "=========================================================="
if python3 -c "import torchvision; from torchvision.ops import nms; print('  torchvision', torchvision.__version__, 'nms OK')" 2>/dev/null; then
  echo "  ✅ 已可用（含 nms），跳过"
else
  echo "  当前 torchvision 缺失或不完整（processor 会变成 None）"
  echo "  先尝试直接从厂商源安装 wheel："
  python3 -m pip install --no-cache-dir --upgrade --target="$PL" \
    --no-deps --no-compile -i "$VEND" "torchvision==0.24.0" 2>&1 | tail -6
  if python3 -c "import torchvision; from torchvision.ops import nms" 2>/dev/null; then
    echo "  ✅ 厂商源 wheel 安装成功"
  else
    echo "  ⚠️ 厂商源没有可用的 torchvision wheel，需要源码编译"
    TV_SRC_DIR=""
    if [ -f "$SRC_TAR" ]; then
      # 路径 A：平台本地源码包
      mkdir -p "$HOME/pkg"
      tar -xzf "$SRC_TAR" -C "$HOME/pkg" torchvision-src 2>/dev/null && \
        TV_SRC_DIR="$HOME/pkg/torchvision-src/vision-0.24.0"
    fi
    if [ -z "$TV_SRC_DIR" ] || [ ! -d "$TV_SRC_DIR" ]; then
      # 路径 B：从镜像下载（平台可访问 ghfast.top）
      echo "  本地无 torchvision 源码，尝试从镜像下载 v0.24.0 ..."
      mkdir -p "$HOME/pkg"
      TV_URL="${TV_SRC_URL:-https://ghfast.top/https://github.com/pytorch/vision/archive/refs/tags/v0.24.0.tar.gz}"
      if (cd "$HOME/pkg" && curl -fL --retry 2 -o vision.tar.gz "$TV_URL" 2>&1 | tail -3); then
        tar -xzf "$HOME/pkg/vision.tar.gz" -C "$HOME/pkg" 2>/dev/null
        TV_SRC_DIR="$HOME/pkg/vision-0.24.0"
      fi
    fi
    if [ -z "$TV_SRC_DIR" ] || [ ! -d "$TV_SRC_DIR" ]; then
      echo "  !! 无法获得 torchvision 0.24.0 源码（本地包与镜像均失败）"
      echo "     补救：设置 SRC_TAR 指向含 torchvision-src 的压缩包，"
      echo "           或设置 TV_SRC_URL 指向可访问的 v0.24.0 源码地址后重跑本脚本"
      echo "     注意：torchvision 缺失会导致 Qwen3-VL processor 为 None，"
      echo "           即使编译成功也无法运行推理"
    else
      echo "  源码目录: $TV_SRC_DIR"
      cd "$TV_SRC_DIR" || exit 1
      export BUILD_VERSION=0.24.0
      export MAX_JOBS=32
      export FORCE_CUDA=1
      export TORCHVISION_USE_NVJPEG=0
      export TORCHVISION_USE_VIDEO_CODEC=0
      export TORCHVISION_USE_FFMPEG=0
      echo "  源码编译中（约 4～5 分钟）..."
      python3 setup.py bdist_wheel 2>&1 | tail -6
      for w in dist/*.whl; do
        [ -f "$w" ] || continue
        python3 -m pip install --no-index --no-deps --target="$PL" --upgrade "$w" 2>&1 | tail -3
      done
      cd "$HERE" || exit 1
      python3 -c "import torchvision; from torchvision.ops import nms; print('  ✅ 源码编译完成', torchvision.__version__)" 2>&1 | tail -2
    fi
  fi
fi

echo ""
echo "=========================================================="
echo " 安装完成 —— 关键依赖自检"
echo "=========================================================="
python3 - <<'PY'
import importlib, sys
mods = ["diffusers", "transformers", "tokenizers", "huggingface_hub", "accelerate",
        "av", "safetensors", "numpy", "torch", "torchvision", "skimage"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL {m:18s} {type(e).__name__}: {str(e)[:70]}")
try:
    import numpy, torch
    torch.from_numpy(numpy.zeros(1))
    print("  OK   from_numpy 可用")
except Exception as e:
    print(f"  FAIL from_numpy: {e}")
try:
    from torchvision.ops import nms
    print("  OK   torchvision.ops.nms 可用（Qwen3-VL processor 需要）")
except Exception as e:
    print(f"  FAIL torchvision.ops.nms: {type(e).__name__}")
try:
    from skimage.metrics import structural_similarity
    print("  OK   skimage SSIM 可用（官方 compare 需要）")
except Exception as e:
    print(f"  FAIL skimage: {type(e).__name__}")
PY

echo ""
echo "  依赖目录: $PL"
du -sh "$PL" 2>/dev/null
echo ""
echo "  下一步："
echo "    bash scripts/build_infinicore.sh     # 约 23 分钟（自动创建 fakeflash 并传 --flash-attn）"
echo "    bash scripts/build_infinilm.sh       # 约 7 分钟"
echo "    bash scripts/run_compare.sh          # 官方评分脚本（约 10~22 分钟）"
echo ""
echo "  若上面自检有 FAIL，请先把对应的项修好再编译，否则运行时会报错。"

