# 仓库补丁清单（P1–P4）

本工程对上游仓库共做 **4 处改动**，全部为构建适配，**不修改任何模型算法**。
`patches/apply_patches.sh` 可一键应用全部补丁，且**幂等**（重复执行结果一致）。

---

## P1｜InfiniCore：移除 xmake 2.9.9 不支持的 `add_orders`

**文件**：`InfiniCore/xmake/hygon.lua`
**行**：72

**原始**：

```lua
rule("hygon.env")
    -- Fix the deprecated warning by using add_orders
    add_orders("cuda.env", "hygon.env")
```

**改后**：

```lua
rule("hygon.env")
    -- Fix the deprecated warning by using add_orders
    -- add_orders("cuda.env", "hygon.env")   -- xmake 2.9.9 无此 API
```

**原因**：`add_orders` 是 xmake 3.x 的 API。本平台 glibc 2.35 无法运行 xmake 3.x 官方 bundle（需 GLIBC_2.38），只能使用 xmake 2.9.9，此时：

```
error: ./xmake/hygon.lua:72: attempt to call a nil value (global 'add_orders')
stack traceback:
    [./xmake/hygon.lua:72]: in main chunk
```

**影响评估**：该调用仅用于声明两条 rule 的执行顺序（`cuda.env` 先于 `hygon.env`）。注释后构建行为不变，仅失去顺序强制。实测编译与推理均正常。

---

## P2｜InfiniCore：离线化 `add_requires` + 为 `_infinicore` 指定 pybind11/torch 路径

**文件**：`InfiniCore/xmake.lua`、`InfiniCore/xmake/cpu.lua`

### P2-a 移除在线依赖

```lua
-- xmake.lua
add_requires("boost", ...)      →   -- add_requires("boost", ...)
add_requires("pybind11")        →   -- add_requires("pybind11")

-- xmake/cpu.lua
add_requires("openmp")          →   -- add_requires("openmp")
add_packages("openmp")          →   -- add_packages("openmp")
```

**原因**：容器内可访问 PyPI 但无法访问 xmake 的包仓库（`github.com` 返回 000），`add_requires` 会卡在下载。

### P2-b 给 `_infinicore` target 注入头文件与链接路径

在 `target("_infinicore")` 内，`add_packages("pybind11")` 处改为：

```lua
target("_infinicore")
    -- add_packages("pybind11")
    add_includedirs("<torch>/include", {public = true})
    add_includedirs("<python-include>", {public = true})
    add_linkdirs("<torch>/lib", {public = true})
```

其中 `<torch>` = `python -c "import torch,os;print(os.path.dirname(torch.__file__))"`，
`<python-include>` = `python -c "import sysconfig;print(sysconfig.get_paths()['include'])"`。

**原因**：平台 PyTorch 自带 pybind11 头文件（`torch/include/pybind11`），直接复用即可，无需联网安装。

---

## P3｜InfiniLM：`python.module` → `python.library`

**文件**：`InfiniLM/xmake.lua`
**行**：27

**原始**：

```lua
target("_infinilm")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", {soabi = true})
```

**改后**：

```lua
target("_infinilm")
    -- add_packages("pybind11")
    add_includedirs("<torch>/include", {public = true})
    add_includedirs("<python-include>", {public = true})
    set_default(false)
    add_rules("python.library", {soabi = true})
```

**原因**：`python.module` 是 xmake 3.x 的规则名，2.9.9 只有 `python.library`。不改会报：

```
error: unknown rule(python.module) in target(_infinilm)!
```

---

## P4｜InfiniLM：新增混合后端入口脚本（本队新增文件，不修改原文件）

**新增文件**：`InfiniLM/examples/minimax_h3_hybrid.py`
**来源**：由 `examples/minimax_h3.py` 派生，仅替换 `load_pipeline()` 中非 legacy 分支的 `text_encoder` 构造。

**原始代码块**（`minimax_h3.py`，非 legacy 分支内）：

```python
text_encoder = MiniMaxH3ConditionEncoder(
    paths.partition_dir / "text_encoder",
    dtype=DTYPE,
    tp_device_ids=devices,
)
```

**替换为**：

```python
_te_devs = legacy_text_devices or devices[: max(1, len(devices) // 2)]
print(f"[model] HYBRID: torch Qwen3-VL on {_te_devs}, native DiT on {devices}")
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    str(paths.partition_dir),
    subfolder="text_encoder",
    dtype=DTYPE,
    device_map="balanced_low_0",
    max_memory=make_max_memory(_te_devs, legacy_memory_fraction),
    low_cpu_mem_usage=True,
    local_files_only=True,
)
```

**目的**：当 InfiniLM 原生条件编码器出现段错误时（见技术报告 §8.1），
让 **DiT 主干仍走 InfiniLM 原生实现**（占端到端计算 87.5%），
仅条件编码器回退 PyTorch（占 1.1%），从而把适配范围最大化。

**说明**：这是**新增文件**，`examples/minimax_h3.py` 保持原样未改动，评分脚本的行为不受影响。

---

## 补丁应用顺序

```bash
bash patches/apply_patches.sh <InfiniCore目录> <InfiniLM目录>
```

脚本行为：

1. 校验目标文件存在
2. 逐项 `grep` 判断是否已应用，已应用则跳过（幂等）
3. 应用 P1 → P2 → P3 → P4
4. 每步打印 `APPLIED` / `SKIP` / `FAIL`
5. 结束时汇总

---

## 未做的改动（如实说明）

| 项目 | 原因 |
|---|---|
| 未修改 InfiniCore 算子实现 | 未发现算子级缺陷；`cast_ requires ATen` 属编译配置问题，已通过 `--aten=y` 解决 |
| 未修改 InfiniLM C++ 推理引擎 | 段错误原因未能定位（容器禁用 ptrace，无法取 C++ 栈） |
| 未接入九齿到 Hygon 后端注册链 | 时间不足；`xmake/hygon.lua` 已支持编译 `build/ninetoothed/*.c`，仅缺 `operator.cc` 侧注册 |
