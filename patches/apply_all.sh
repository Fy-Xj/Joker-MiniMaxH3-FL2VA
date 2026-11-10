#!/bin/bash
# =============================================================================
# Joker 队 · 嵩山杯 2026 赛题一 —— 仓库补丁一键应用脚本
#
# 用法：
#   bash patches/apply_all.sh <InfiniCore目录> <InfiniLM目录>
#
# 示例：
#   bash patches/apply_all.sh ~/pkg/InfiniCore ~/pkg/InfiniLM
#
# 特性：
#   - 幂等：已应用的补丁自动跳过
#   - 幂等：P2-b / P3 的路径注入只做一次
#   - 每步打印 APPLIED / SKIP / FAIL
# =============================================================================
set -o pipefail

IC="${1:-}"
LM="${2:-}"

if [ -z "$IC" ] || [ -z "$LM" ]; then
  echo "用法: bash $0 <InfiniCore目录> <InfiniLM目录>"
  exit 2
fi

for d in "$IC" "$LM"; do
  [ -d "$d" ] || { echo "!! 目录不存在: $d"; exit 2; }
done

# ---- 解析 torch / python 路径 ----
TI=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include'))" 2>/dev/null)
TL=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))" 2>/dev/null)
PI=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])" 2>/dev/null)

if [ -z "$TI" ] || [ -z "$TL" ] || [ -z "$PI" ]; then
  echo "!! 无法解析 torch / python 路径，请先激活 PyTorch 环境（source /opt/dtk/env.sh）"
  exit 2
fi

echo "torch include : $TI"
echo "torch lib     : $TL/lib"
echo "python include: $PI"
echo ""

OK=0; SKIP=0; FAIL=0
note() { echo "  --> $1"; }

# ---------------------------------------------------------------------------
# P1: InfiniCore/xmake/hygon.lua -- 注释 add_orders
# ---------------------------------------------------------------------------
echo "[P1] InfiniCore/xmake/hygon.lua : 注释 add_orders"
F="$IC/xmake/hygon.lua"
if [ ! -f "$F" ]; then
  echo "  FAIL 文件不存在"; FAIL=$((FAIL+1))
elif grep -q '^[[:space:]]*-- add_orders("cuda.env", "hygon.env")' "$F"; then
  echo "  SKIP 已应用"; SKIP=$((SKIP+1))
else
  sed -i 's|^[[:space:]]*add_orders("cuda.env", "hygon.env")|    -- add_orders("cuda.env", "hygon.env")  -- xmake 2.9.9 无此 API|' "$F"
  if grep -q -- '-- add_orders' "$F"; then
    echo "  APPLIED"; OK=$((OK+1)); note "已注释第 72 行"
  else
    echo "  FAIL 未匹配到 add_orders"; FAIL=$((FAIL+1))
  fi
fi

# ---------------------------------------------------------------------------
# P2-a: InfiniCore/xmake.lua + xmake/cpu.lua -- 注释 add_requires
# ---------------------------------------------------------------------------
echo "[P2-a] InfiniCore : 注释 add_requires (boost / pybind11 / openmp)"
F1="$IC/xmake.lua"; F2="$IC/xmake/cpu.lua"
done_a=0
if [ -f "$F1" ]; then
  sed -i 's|^add_requires("boost"|-- add_requires("boost"|' "$F1"
  sed -i 's|^add_requires("pybind11")|-- add_requires("pybind11")|' "$F1"
  done_a=1
fi
if [ -f "$F2" ]; then
  sed -i 's|^[[:space:]]*add_requires("openmp")|    -- add_requires("openmp")|' "$F2"
  sed -i 's|^[[:space:]]*add_packages("openmp")|    -- add_packages("openmp")|' "$F2"
  done_a=1
fi
if [ "$done_a" = "1" ]; then echo "  APPLIED"; OK=$((OK+1)); else echo "  FAIL 目标文件缺失"; FAIL=$((FAIL+1)); fi

# ---------------------------------------------------------------------------
# P2-b: InfiniCore/xmake.lua -- _infinicore 注入头文件与链接路径
# ---------------------------------------------------------------------------
echo "[P2-b] InfiniCore : _infinicore 注入 pybind11(torch) 路径"
if grep -q 'JOKER_TORCH_INC' "$F1" 2>/dev/null; then
  echo "  SKIP 已应用"; SKIP=$((SKIP+1))
else
  python3 - "$F1" "$TI" "$PI" "$TL" <<'PY'
import sys
path, ti, pi, tl = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lines = open(path, encoding="utf-8").read().split("\n")
start = next((i for i, l in enumerate(lines) if l.strip() == 'target("_infinicore")'), None)
if start is None:
    print("  FAIL 未找到 target(\"_infinicore\")"); sys.exit(1)
for i in range(start, len(lines)):
    if lines[i].strip() == 'add_packages("pybind11")':
        ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines[i] = ind + '-- add_packages("pybind11")'
        lines.insert(i+1, ind + 'add_includedirs("%s", {public = true})  -- JOKER_TORCH_INC' % ti)
        lines.insert(i+2, ind + 'add_includedirs("%s", {public = true})  -- JOKER_PY_INC' % pi)
        lines.insert(i+3, ind + 'add_linkdirs("%s/lib", {public = true})  -- JOKER_TORCH_LIB' % tl)
        open(path, "w", encoding="utf-8").write("\n".join(lines))
        print("  APPLIED  at line", i + 1)
        sys.exit(0)
print("  FAIL _infinicore 内未找到 add_packages(\"pybind11\")")
sys.exit(1)
PY
  if [ $? -eq 0 ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); fi
fi

# ---------------------------------------------------------------------------
# P3: InfiniLM/xmake.lua -- python.module -> python.library
# ---------------------------------------------------------------------------
echo "[P3] InfiniLM/xmake.lua : python.module -> python.library"
F3="$LM/xmake.lua"
if [ ! -f "$F3" ]; then
  echo "  FAIL 文件不存在"; FAIL=$((FAIL+1))
elif grep -q 'add_rules("python.library"' "$F3"; then
  echo "  SKIP 已应用"; SKIP=$((SKIP+1))
else
  sed -i 's|add_rules("python.module"|add_rules("python.library"|' "$F3"
  sed -i 's|^add_requires("pybind11")|-- add_requires("pybind11")|' "$F3"
  if grep -q 'add_rules("python.library"' "$F3"; then
    echo "  APPLIED"; OK=$((OK+1))
  else
    echo "  FAIL 未匹配 python.module"; FAIL=$((FAIL+1))
  fi
fi

# ---------------------------------------------------------------------------
# P3-b: InfiniLM/xmake.lua -- _infinilm 注入头文件
# ---------------------------------------------------------------------------
echo "[P3-b] InfiniLM : _infinilm 注入 pybind11(torch) 路径"
if grep -q 'JOKER_TORCH_INC' "$F3" 2>/dev/null; then
  echo "  SKIP 已应用"; SKIP=$((SKIP+1))
else
  python3 - "$F3" "$TI" "$PI" <<'PY'
import sys
path, ti, pi = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
old = 'add_packages("pybind11")'
if old not in s:
    print("  FAIL 未找到 add_packages(\"pybind11\")"); sys.exit(1)
new = ('-- add_packages("pybind11")\n'
       '    add_includedirs("%s", {public = true})  -- JOKER_TORCH_INC\n'
       '    add_includedirs("%s", {public = true})  -- JOKER_PY_INC' % (ti, pi))
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("  APPLIED")
PY
  if [ $? -eq 0 ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); fi
fi

# ---------------------------------------------------------------------------
# P4: InfiniLM/examples/minimax_h3_hybrid.py -- 混合后端入口（新增文件）
# ---------------------------------------------------------------------------
echo "[P4] InfiniLM : 生成 examples/minimax_h3_hybrid.py（混合后端）"
SRC="$LM/examples/minimax_h3.py"
DST="$LM/examples/minimax_h3_hybrid.py"
if [ ! -f "$SRC" ]; then
  echo "  FAIL 缺少 examples/minimax_h3.py"; FAIL=$((FAIL+1))
elif [ -f "$DST" ]; then
  echo "  SKIP 已存在 $DST"; SKIP=$((SKIP+1))
else
  python3 - "$SRC" "$DST" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
old = '''            text_encoder = MiniMaxH3ConditionEncoder(
                paths.partition_dir / "text_encoder",
                dtype=DTYPE,
                tp_device_ids=devices,
            )'''
new = '''            _te_devs = legacy_text_devices or devices[: max(1, len(devices) // 2)]
            print(f"[model] HYBRID: torch Qwen3-VL on {_te_devs}, native DiT on {devices}")
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
    print("  FAIL 未匹配到条件编码器代码块（上游版本可能已变更）"); sys.exit(1)
open(dst, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("  APPLIED ->", dst)
PY
  if [ $? -eq 0 ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); fi
fi

# ---------------------------------------------------------------------------
echo ""
echo "================ 汇总 ================"
echo "  APPLIED : $OK"
echo "  SKIP    : $SKIP"
echo "  FAIL    : $FAIL"
echo "====================================="
[ "$FAIL" -eq 0 ] || exit 1
echo "全部补丁就绪。下一步："
echo "  bash scripts/build_infinicore.sh"
echo "  bash scripts/build_infinilm.sh"
