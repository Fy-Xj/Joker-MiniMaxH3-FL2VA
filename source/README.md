# 完整可编译源码（已应用本队补丁）

本目录提供**本队实际使用的完整源码包**，解压即可编译，无需再找上游版本、也无需手动打补丁。
目的是满足"提供可复现的源代码"的要求。

## 文件

| 文件 | 大小 | 内容 |
|---|---|---|
| `InfiniCore-patched.tar.gz` | 1.9 MB | InfiniCore 完整源码（分支 `minimax-h3`），**已应用 P1 / P2 / P5** |
| `InfiniLM-patched.tar.gz` | 0.4 MB | InfiniLM 完整源码（分支 `minimax_h3`），**已应用 P3 / P4** |
| `changes-InfiniCore.patch` | 6.0 KB | 与上游的 unified diff（逐行可审） |
| `changes-InfiniLM.patch` | 1.3 KB | 与上游的 unified diff（逐行可审） |

> 压缩包内顶层目录分别为 `InfiniCore/` 与 `InfiniLM/`。

## 快速使用

```bash
# 1) 解压到 $HOME/pkg（脚本默认目录）
mkdir -p $HOME/pkg
tar -xzf source/InfiniCore-patched.tar.gz -C $HOME/pkg
tar -xzf source/InfiniLM-patched.tar.gz  -C $HOME/pkg

# 2) 依赖
source scripts/env.sh
bash scripts/setup_python.sh        # 期望 SKIMAGE_OK / FROM_NUMPY_OK

# 3) 编译（脚本会自动创建 $AFS/fakeflash 并传 --flash-attn）
bash scripts/build_infinicore.sh    # 期望 INFINICORE_OK / FLASH_MHA_OK
bash scripts/build_infinilm.sh      # 期望 INFINILM_OK / RUNNER_IMPORT_OK

# 4) 官方评分脚本
bash scripts/run_compare.sh         # 期望 Average SSIM ≈ 61%
```

> **注意**：源码已含补丁，**不要再执行** `patches/apply_all.sh`（会因"已应用"而全部 SKIP，不会出错，但没必要）。

## 改了什么（与上游的差异）

### InfiniCore（4 处）

| 文件 | 改动 | 原因 |
|---|---|---|
| `xmake/hygon.lua` | 注释 `add_orders("cuda.env","hygon.env")` | xmake 2.9.9 无此 API，配置阶段直接报错 |
| `xmake.lua` | 注释 `add_requires("boost"/"pybind11")`；文件头注入 torch / python 头文件与库路径；注释 `add_packages("pybind11")` | 平台离线，无法下载第三方包；改用 torch 自带的 pybind11 |
| `xmake/cpu.lua` | 注释 `add_requires("openmp")` / `add_packages("openmp")` | 同上，离线取不到 |
| `src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc` | **改调厂商 `flash::vllm_mha_varlen_fwd`** | **最关键**：原实现是手写 `at::matmul + at::softmax`，把 `[B,H,S,S]` 以 bf16 落盘（S=26448 时每卡约 9.8 GB），logits 仅 8 位尾数，精度不足并随 49 步扩散迭代放大，最终输出为棋盘状噪声。改用与 PyTorch 参考侧相同的 flash 内核后，`Average SSIM` 从 9.923% 提升到 **61.182%** |

`mha` 修改的要点：
1. 全自注意力 → varlen 只有单序列，`cu_seqlens = [0, B*S]`（int32，与 q 同设备）
2. 张量由 `[B,S,H,D]` 重排为 `[B*S, H, D]`
3. 该函数第 4~10 个参数是**非 const 引用**，必须传具名变量（不能传 `std::nullopt`）
4. 用 try/catch 包住，失败时回退 `at::scaled_dot_product_attention`

### InfiniLM（2 处）

| 文件 | 改动 | 原因 |
|---|---|---|
| `xmake.lua` | `add_rules("python.module")` → `("python.library")`；注释 `add_requires("pybind11")` 并注入 torch 头文件路径 | `python.module` 规则在 xmake 2.9.9 中不存在（`unknown rule(python.module)`） |
| `examples/minimax_h3_hybrid.py` | 新增（由 `minimax_h3.py` 派生） | 诊断用混合后端脚本：条件编码器走 PyTorch、DiT 走 InfiniLM，用于定位问题出在哪一侧。**非官方提交路径**，官方评分脚本跑的是原版 `minimax_h3.py` |

## 实测结果（8 × 海光 gfx936）

| 指标 | 数值 |
|---|---|
| `Average SSIM`（官方 `minimax_h3_compare.py`） | **61.182%** |
| Size / Length | PASS / PASS |
| DiT 去噪 49 次调用 | 188.080 s（PyTorch 530.820 s → **2.82×**） |
| 端到端 | 428.370 s（含冷加载 181.4 s，纯计算约 247.0 s） |
| 画面 | 金毛小狗睁眼（0.05 s）→ 打瞌睡（2.5 s）→ 闭眼睡着（5.0 s） |

## 关于上游版本

平台无法直连 `github.com`，上游源码通过镜像 `ghfast.top` 于 2026-11-03 拉取，
**镜像 clone 未保留 `.git` 目录，因此无法记录上游 commit hash**。
本目录的 tar.gz 即为该版本应用补丁后的完整快照，可直接作为复现基准。
`changes-*.patch` 为与之对应的逐行 diff，便于审阅。
