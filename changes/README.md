# 本队对上游仓库的全部改动

本目录集中存放本队在初赛中**对上游仓库所做的全部修改**，逐文件对应上游路径，便于核验。

| 子目录 | 上游仓库 | 上游分支 | 上游 commit | 改动文件数 |
|---|---|---|---|---|
| `InfiniCore/` | `InfiniTensor/InfiniCore` | `minimax-h3` | 未能记录（镜像 clone 未保留 `.git`，见下） | 4 |
| `InfiniLM/` | `InfiniTensor/InfiniLM` | `minimax_h3` | 未能记录（同上） | 2 |

**九齿（ninetoothed）**：本次**未修改**其仓库。只用九齿 DSL 做了 `timestep_embedding`
的独立基准测试（gfx936 2.92×、gfx906 2.99×，精度无损），但因未接入 Hygon 后端注册链，
**未进入正式推理路径**，故无对应改动仓库。

**关于上游 commit**：平台无法直连 github.com，上游源码通过镜像
`https://ghfast.top/https://github.com/InfiniTensor/...` 拉取（2026-11-03），
镜像 clone 未保留 `.git` 目录，`git log` 不可用，因此无法记录上游 commit。
本目录中的文件均是在该上游版本上修改后的**完整文件**，可直接与上游对照。

## InfiniCore 改动（4 个文件）

| 文件 | 改动 | 作用 |
|---|---|---|
| `xmake/hygon.lua` | 注释 `add_orders("cuda.env", "hygon.env")` | xmake 2.9.9 无此 API |
| `xmake.lua` | 注释 `add_requires(boost/pybind11)`；注入 torch/python 路径；注释 `add_packages("pybind11")` | 离线构建，改用 torch 自带 pybind11 |
| `xmake/cpu.lua` | 注释 `add_requires(openmp)` / `add_packages(openmp)` | 离线环境取不到 |
| `src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc` | **改调 `flash::vllm_mha_varlen_fwd`** | **最关键：SSIM 9.923% → 61.182%** |

## InfiniLM 改动（2 个文件）

| 文件 | 改动 | 作用 |
|---|---|---|
| `xmake.lua` | `python.module` → `python.library`；注入 pybind11 路径 | 上游规则在 xmake 2.9.9 中不存在 |
| `examples/minimax_h3_hybrid.py` | 新增（由 `minimax_h3.py` 派生） | 诊断用混合后端脚本，**非官方提交路径** |

## 实测效果（8 × 海光 gfx936，官方评分脚本）

| 指标 | 改动前 | 改动后 |
|---|---|---|
| Average SSIM | 9.923% | **61.182%** |
| DiT 49 次调用 | 410.605 s | **188.080 s**（对 PyTorch 530.820 s 为 2.82×） |
| 输出画面 | 棋盘状噪声 | 金毛小狗睁眼 → 打瞌睡 → 闭眼睡着 |

详细的逐项说明见 `InfiniCore/README.md`、`InfiniLM/README.md`，
以及本仓库根目录的 `patches/PATCHES.md`（含可一键应用的 `apply_all.sh`）。
