# InfiniCore · MiniMax-H3 海光适配改动（Joker）

本仓库是嵩山杯 2026 赛题一（Joker 队）对 **InfiniCore** 所做的全部修改，
用于满足"报告中需包含所有更改涉及仓库的链接、branch 与 commit"的要求。

## 上游

| 项 | 内容 |
|---|---|
| 上游仓库 | `InfiniTensor/InfiniCore` |
| 分支 | `minimax-h3` |
| 拉取方式 | 平台无法直连 github，使用镜像 `https://ghfast.top/https://github.com/InfiniTensor/InfiniCore.git --branch minimax-h3` |
| 拉取时间 | 2026-11-03 |
| **上游 commit** | 未能记录（镜像 clone 未保留 `.git`，`git log` 不可用）。改动位置均以上游文件内容为准，见下方逐条说明 |

## 改动清单（4 个文件）

### 1. `xmake/hygon.lua` — P1

注释掉 `hygon.env` 规则中的 `add_orders("cuda.env", "hygon.env")`。

原因：xmake 2.9.9 没有 `add_orders` API（报 `attempt to call a nil value (global 'add_orders')`），
而平台 glibc 只有 2.35，用不了需要 GLIBC_2.38 的 xmake 3.x。

### 2. `xmake.lua` — P2

- 注释 `add_requires("boost")`、`add_requires("pybind11")`（离线环境无法下载）
- 在文件头注入 torch / python 的头文件与库路径（`JOKER_GLOBAL_INC` / `JOKER_GLOBAL_PY` / `JOKER_GLOBAL_LIB`）
- 注释所有 `add_packages("pybind11")`，改用 torch 自带的 pybind11

### 3. `xmake/cpu.lua` — P2

注释 `add_requires("openmp")` 与 `add_packages("openmp")`（离线环境取不到）。

### 4. `src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc` — **P5，最关键**

**问题**：该文件名为 `mha_flashattn_hygon.cc`，实现里却没有 flash attention：

```cpp
auto attn_weight = at::matmul(q.permute({0,2,1,3}), k_attn.permute({0,2,3,1})) * scale;
auto attn = at::softmax(attn_weight, -1);
auto result = at::matmul(attn, v_attn.permute({0,2,1,3})).permute({0,2,1,3});
```

它把 `[B,H,S,S]` 的注意力矩阵**以 bf16 真实算出来并落盘**：
S = 26448、TP=8 时每 rank 7 个 head → `7 × 26448² × 2 B ≈ 9.8 GB/卡`；
中间 logits 只有 8 位尾数，softmax 精度不足，再经 49 步扩散迭代累积，
最终输出退化为棋盘状噪声。

**修改**：改为调用厂商 `flash_attn_2_cuda*.so` 提供的 varlen 内核
（`flash::vllm_mha_varlen_fwd`，与 PyTorch 参考侧同一 kernel）：

- 全自注意力 → varlen 只有单序列，`cu_seqlens = [0, B*S]`（int32，与 q 同设备）
- 张量由 `[B,S,H,D]` 重排为 `[B*S, H, D]`
- 该函数第 4~10 个参数是**非 const 引用**，必须传具名变量，不能传 `std::nullopt`
- 用 try/catch 包住，失败时回退 `at::scaled_dot_product_attention`

## 实测效果（8 × 海光 gfx936，官方评分脚本）

| 指标 | 修改前 | 修改后 |
|---|---|---|
| **Average SSIM** | 9.923% | **61.182%** |
| DiT 单次前向 | 8.380 s | **3.812 s** |
| DiT 49 次调用 | 410.605 s | **188.080 s**（对 PyTorch 530.820 s 为 **2.82×**） |
| 输出画面 | 棋盘状噪声 | 金毛小狗睁眼 → 打瞌睡 → 闭眼睡着 |

## 说明

- 本仓库**只包含被修改的文件**，目录结构与上游一致，便于逐文件对照
- 完整的上游源码请取 `InfiniTensor/InfiniCore` 的 `minimax-h3` 分支
- 配套的补丁脚本与完整技术报告在本队主仓库
  `Joker-MiniMaxH3-FL2VA` 的 `patches/` 与 `技术报告.md`
