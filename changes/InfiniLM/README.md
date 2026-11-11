# InfiniLM · MiniMax-H3 海光适配改动（Joker）

本仓库是嵩山杯 2026 赛题一（Joker 队）对 **InfiniLM** 所做的全部修改，
用于满足"报告中需包含所有更改涉及仓库的链接、branch 与 commit"的要求。

## 上游

| 项 | 内容 |
|---|---|
| 上游仓库 | `InfiniTensor/InfiniLM` |
| 分支 | `minimax_h3` |
| 拉取方式 | 镜像 `https://ghfast.top/https://github.com/InfiniTensor/InfiniLM.git --branch minimax_h3` |
| 拉取时间 | 2026-11-03 |
| **上游 commit** | 未能记录（镜像 clone 未保留 `.git`）；改动位置均以上游文件内容为准 |

## 改动清单（2 个文件）

### 1. `xmake.lua` — P3 / P3-b

- `add_rules("python.module")` → `add_rules("python.library")`
  原因：上游使用的 `python.module` 规则在当前 xmake 2.9.9 中不存在
  （报 `unknown rule(python.module) in target(_infinilm)`）
- 注释 `add_requires("pybind11")`，改为使用 torch 自带的 pybind11
  （注入 `JOKER_TORCH_INC` / `JOKER_PY_INC` 两个头文件路径）

### 2. `examples/minimax_h3_hybrid.py` — P4（新增文件，不改上游文件）

由 `examples/minimax_h3.py` 派生，把条件编码器换成 PyTorch 实现、DiT 仍走 InfiniLM，
用于做"编码器 / DiT 分别是谁出错"的对照实验。

**说明**：这是诊断用的辅助脚本，**不属于官方提交路径**。官方评分脚本
`minimax_h3_compare.py` 的 InfiniLM 侧跑的是原版 `minimax_h3.py`（不加 `--legacy`）。

## 实测效果

与 InfiniCore 的 P5 补丁配合（mha 改调厂商 flash-attn），8 × 海光 gfx936 上：

- 官方 `minimax_h3_compare.py` 完整运行通过（exit=0）
- `Size: PASS` / `Length: PASS` / **`Average SSIM: 61.182%`**
- DiT 去噪 49 次调用 188.080 s，对 PyTorch 530.820 s 为 **2.82×**

## 说明

- 本仓库只包含被修改的文件，目录结构与上游一致
- 完整上游源码请取 `InfiniTensor/InfiniLM` 的 `minimax_h3` 分支
- 配套补丁脚本与完整技术报告在本队主仓库 `Joker-MiniMaxH3-FL2VA`
