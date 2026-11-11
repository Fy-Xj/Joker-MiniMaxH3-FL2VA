#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 原生路径诊断脚本（复现失败案例）
#
# 用途：复现技术报告 §8.4~8.5 记录的两组实验，用于后续接手者继续排查。
#
# 用法：
#   bash scripts/diag_native.sh variants     # 四组控制变量（TP × 尺寸）
#   bash scripts/diag_native.sh cast         # dtype 假设验证（int64→int32）
#   bash scripts/diag_native.sh all          # 全部
#
# 全部为「冒烟」配置：124 帧（模型下限）/ 2 步，单个变体约 2~3 分钟。
#
# 已知结果（2026-11-06 实测）：
#   variants: B(TP8/576) 段错误；C、D(TP4) 引擎初始化 OOM
#   cast:     int / int_float 两个变体均段错误
# =============================================================================
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

MODE="${1:-all}"
LM="${LM:-$HOME/pkg/InfiniLM}"
SRC_TAR="${SRC_TAR:-$AFS/upload_src.tar.gz}"
OUT="$HOME/out"; mkdir -p "$OUT"
export PYTHONFAULTHANDLER=1
ulimit -c 0 2>/dev/null

if [ ! -d "$LM" ]; then
  mkdir -p "$HOME/pkg"
  tar -xzf "$SRC_TAR" -C "$HOME/pkg" InfiniLM
fi
cd "$LM" || exit 1

# ---------------------------------------------------------------------------
# 生成带 dtype 修补的派生脚本
# ---------------------------------------------------------------------------
gen_cast() {
python3 - <<'PY'
src="examples/minimax_h3.py"; dst="examples/minimax_h3_cast.py"
s=open(src,encoding="utf-8").read()

old='''            text_encoder = MiniMaxH3ConditionEncoder(
                paths.partition_dir / "text_encoder",
                dtype=DTYPE,
                tp_device_ids=devices,
            )'''
new='''            _te_devs = legacy_text_devices or devices[: max(1, len(devices) // 2)]
            print(f"[model] CAST-HYBRID: torch Qwen3-VL on {_te_devs}, native DiT on {devices}")
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                str(paths.partition_dir),
                subfolder="text_encoder",
                dtype=DTYPE,
                device_map="balanced_low_0",
                max_memory=make_max_memory(_te_devs, legacy_memory_fraction),
                low_cpu_mem_usage=True,
                local_files_only=True,
            )'''
if old not in s:
    raise SystemExit("!! 条件编码器代码块未匹配（上游版本可能已变更）")
s = s.replace(old, new, 1)

anchor = "def resolve_legacy_device_groups("
if anchor not in s:
    raise SystemExit("!! 未找到插入锚点")
idx = s.index(anchor)

inject = '''# ===== 诊断修补：DiT runner 传给 C++ 的整型张量转 int32 =====
import os as _os
import torch as _torch

_CAST_MODE = _os.environ.get("CAST_MODE", "int")
print(f"[cast] mode = {_CAST_MODE}")

_orig_to_ic = getattr(
    MiniMaxH3TransformerRunner._to_infinicore,
    "__func__",
    MiniMaxH3TransformerRunner._to_infinicore,
)


def _patched_to_ic(tensor):
    if _CAST_MODE in ("int", "int_float"):
        if tensor.dtype == _torch.int64:
            tensor = tensor.to(_torch.int32)
    if _CAST_MODE == "int_float":
        if tensor.dtype == _torch.float32:
            tensor = tensor.to(_torch.bfloat16)
    return _orig_to_ic(tensor)


MiniMaxH3TransformerRunner._to_infinicore = staticmethod(_patched_to_ic)
print("[cast] patched MiniMaxH3TransformerRunner._to_infinicore")
# ===== 修补结束 =====


'''
open(dst,"w",encoding="utf-8").write(s[:idx] + inject + s[idx:])
print("patched ->", dst)
PY
  python3 -c "import ast;ast.parse(open('examples/minimax_h3_cast.py',encoding='utf-8').read());print('AST_OK')"
}

# ---------------------------------------------------------------------------
# 统一的运行封装
# ---------------------------------------------------------------------------
run_smoke() {  # $1=script $2=gpus $3=extra $4=outfile $5=label [$6=castmode]
  echo ""
  echo "=============== [$5] gpus=$2 extra=$3 ==============="
  date
  if [ -n "${6:-}" ]; then export CAST_MODE="$6"; fi
  python3 -u "$1" fl2va \
    --model-path="$MODEL/diffusers" \
    --text="A dog falls asleep" \
    --first-frame="$FIRST_FRAME" \
    --gpus="$2" $3 \
    --num-frames=124 \
    --steps=2 \
    --output="$OUT/$4" 2>&1 | tail -22
  echo "[$5] exit=$?"
  date
  if [ -f "$OUT/$4" ]; then
    echo "[$5] PRODUCED $4"
    cp "$OUT/$4" "$AFS/$4" 2>/dev/null && echo "[$5] copied_to_AFS"
  else
    echo "[$5] NO_OUTPUT"
  fi
}

case "$MODE" in
  variants|all)
    echo "########## 前置：生成混合后端脚本（P4 补丁）##########"
    [ -f examples/minimax_h3_hybrid.py ] || \
      bash "$HERE/../patches/apply_all.sh" "$HOME/pkg/InfiniCore" "$LM" >/dev/null 2>&1
    ls -l examples/minimax_h3_hybrid.py 2>&1

    echo ""
    echo "########## B. TP=8 + 384x576（验证是否与尺寸有关）##########"
    run_smoke examples/minimax_h3_hybrid.py "0,1,2,3,4,5,6,7" "--height=384 --width=576" \
      "b_tp8_576.mp4" "B-tp8-576"

    echo ""
    echo "########## C. TP=4 + 默认尺寸（验证是否 TP=8 特有）##########"
    run_smoke examples/minimax_h3_hybrid.py "0,1,2,3" \
      "--legacy-text-gpus=0,1,2,3 --legacy-transformer-gpus=0,1,2,3" \
      "c_tp4_928.mp4" "C-tp4-928"

    echo ""
    echo "########## D. TP=4 + 384x576 ##########"
    run_smoke examples/minimax_h3_hybrid.py "0,1,2,3" \
      "--legacy-text-gpus=0,1,2,3 --legacy-transformer-gpus=0,1,2,3 --height=384 --width=576" \
      "d_tp4_576.mp4" "D-tp4-576"
    ;;
esac

case "$MODE" in
  cast|all)
    echo ""
    echo "########## 生成 dtype 修补脚本 ##########"
    gen_cast

    echo ""
    echo "########## 变体 1：int64 -> int32 ##########"
    run_smoke examples/minimax_h3_cast.py "0,1,2,3,4,5,6,7" "" "cast_int.mp4" "cast-int" "int"

    if [ ! -f "$OUT/cast_int.mp4" ]; then
      echo ""
      echo "########## 变体 2：int64 -> int32 且 float32 -> bf16 ##########"
      run_smoke examples/minimax_h3_cast.py "0,1,2,3,4,5,6,7" "" "cast_int_float.mp4" "cast-int-float" "int_float"
    fi
    ;;
esac

echo ""
echo "########## 汇总 ##########"
ls -lh "$OUT" 2>&1
du -sh "$AFS" 2>/dev/null
echo "########## DONE $(date) ##########"
echo ""
echo "参考结果（2026-11-06 实测）："
echo "  B  -> Segmentation fault (exit 139)，与分辨率无关"
echo "  C  -> cudaMalloc 失败 / RankWorker failed to initialize（引擎初始化 OOM）"
echo "  D  -> 同 C"
echo "  cast-int / cast-int-float -> 均 Segmentation fault (exit 139)"
