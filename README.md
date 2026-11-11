# Joker · 嵩山杯 2026 赛题一 · MiniMax-H3 FL2VA 适配

**基于 InfiniLM 与 InfiniCore 的 MiniMax-H3 首帧视频生成适配与优化**

| 项 | 内容 |
|---|---|
| 队伍 | Joker（个人参赛） |
| 单位 | 郑州大学 |
| 平台账号 | xdzs2026_b0384 / joker_w1 |
| 硬件 | 8 × 海光 DCU gfx936（BW），64 GiB/卡 |
| 软件 | DTK 26.04 / PyTorch 2.9.0+das.opt1.dtk2604 / Python 3.11.9 / glibc 2.35 |

---

## 状态摘要（先说结论）

**官方评分脚本完整运行通过（exit=0），规格两项 PASS，画面正确性已修复：**

```
[result] MiniMax-H3 video comparison

  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 61.182%
```

**性能：DiT 去噪比 PyTorch 快 2.82 倍**（49 次调用 530.820 s → 188.080 s）。

### 关键修复：mha 用错了实现

海光后端的 `mha_flashattn_hygon.cc` 文件名带 flashattn，实际却是手写的
`at::matmul + at::softmax`，会把 `[B,H,S,S]` 注意力矩阵以 bf16 落盘
（S=26448 时每卡约 9.8 GB），logits 只有 8 位尾数，精度不足并随 49 步迭代放大，
最终输出是棋盘状噪声。

改成调用厂商 flash-attn 的 `flash::vllm_mha_varlen_fwd`（与 PyTorch 参考侧同一 kernel）后：

| | 修复前 | 修复后 |
|---|---|---|
| Average SSIM | 9.923% | **61.182%** |
| DiT 单次前向 | 8.380 s | **3.812 s** |
| DiT 49 次调用 | 410.605 s | **188.080 s** |
| 输出视频 | 棋盘状噪声 | 金毛小狗睁眼→打瞌睡→睡着 |

补丁见 `patches/PATCHES.md` 的 **P5**，实现在 `patches/apply_all.sh`。

### 其它已完成

- 8 卡（gfx936 × 8）编译与运行全部打通；InfiniCCL 张量并行通信正常
- 定位并修复原生路径段错误（根因：未启用 `--flash-attn`，海光 `mha` 算子整个文件未编译）
- 逐元素对拍验证：DiT 输出与 PyTorch 相关系数 **0.999714**，
  条件编码器逐 token 中位数 **0.9996**，接口层（mrope/RoPE/AdaLN/拼装）全部正确

### 剩余差距

61.182% 与参考实现的 93.584% 之间还差一段，主要是 bf16 累积误差
（参考实现自己换个随机后验采样重跑，SSIM 也只有 80.4%，这是本口径的噪声地板）。
详见 [技术报告.md](技术报告.md) 第 8、9 章。

---

## 0.1 本队对上游仓库的全部改动（合规核验入口）

按初赛要求，本队**对上游仓库所做的全部修改**集中放在本仓库的 `changes/` 目录，
逐文件对应上游路径，可直接与本仓库 commit 一起核验：

| 子目录 | 上游仓库 | 上游分支 | 改动文件 |
|---|---|---|---|
| [`changes/InfiniCore/`](changes/InfiniCore/) | `InfiniTensor/InfiniCore` | `minimax-h3` | 4 个（含最关键的 `mha` 修改） |
| [`changes/InfiniLM/`](changes/InfiniLM/) | `InfiniTensor/InfiniLM` | `minimax_h3` | 2 个 |

**九齿（ninetoothed）**：本次**未修改**其仓库。只用九齿 DSL 做了 `timestep_embedding`
的独立基准测试（gfx936 2.92×、gfx906 2.99×，精度无损），但未接入 Hygon 后端注册链，
**未进入正式推理路径**，故无对应改动仓库。

**上游 commit 说明**：平台无法直连 github.com，上游源码经镜像
`https://ghfast.top/https://github.com/InfiniTensor/...` 于 2026-11-03 拉取，
镜像 clone 未保留 `.git`，故无法记录上游 commit；`changes/` 中为修改后的完整文件，
可与上游逐文件对照。

详见 [`changes/README.md`](changes/README.md)。

---

## 1. 本仓库包含什么

**上游源码不在此仓库**（InfiniCore / InfiniLM 为公开仓库，见 §2）。
本仓库只放**我们做的工作**：

```
├── 技术报告.md                完整技术报告（10 章 + 2 附录）
├── 提交清单.md                逐项对照赛题第 11 条要求
├── requirements.txt           实测可用的依赖版本 + 历史冲突说明
├── patches/
│   ├── PATCHES.md             4 处补丁的逐项说明（改了什么/为什么/影响）
│   └── apply_all.sh           一键应用（幂等，字符串匹配不依赖行号）
├── scripts/
│   ├── env.sh                 统一环境变量
│   ├── setup_python.sh        依赖安装（厂商源整套 + torchvision 源码编译）
│   ├── build_infinicore.sh    编译 InfiniCore（Hygon + InfiniCCL + ATen）
│   ├── build_infinilm.sh      编译 InfiniLM Python 扩展
│   ├── run_fl2va_legacy.sh    FL2VA 生成（参考实现，已验证 ✅）
│   ├── run_compare.sh         官方评分脚本 minimax_h3_compare.py
│   ├── verify_output.sh       输出规格核查
│   └── diag_native.sh         原生路径诊断复现（TP × 尺寸、dtype 假设）
├── tests/
│   └── check_spec.py          规格自动核查（9 项断言，本地实测全 PASS）
├── 环境配置/
│   ├── PATHS.md               模型与数据路径配置说明
│   └── env_template.sh        环境变量模板
├── logs/                      10 份关键运行日志（含 5 个失败案例）
└── outputs/
    ├── fl2va_official.mp4        参考实现路径（--legacy）产物，画面正确
    ├── first_frame_input.png     官方首帧输入图（800×965）
    └── frames/                   首末帧截图（0.05 s / 5.00 s）
```

### outputs/ 文件对照（重要）

| 文件 | 生成方 | 画面 | 说明 |
|---|---|---|---|
| `fl2va_official.mp4` | `--legacy`（PyTorch 参考实现） | ✅ **正确**（小狗睁眼→闭眼睡着） | 768×928 / 124 帧 / 24 FPS / 5.175 s / H.264+AAC 32kHz |
| `frames/f_0_05.png`、`f_5_0.png` | 同上 | ✅ | 首帧与末帧画面内容佐证 |

`minimax_h3_compare.py` 在平台上跑完后会生成两个文件
（`minimax_h3_comparison/reference_fl2va.mp4` 与
`minimax_h3_comparison/infinilm_fl2va.mp4`），两者规格完全一致
（768×928 / 124 帧 / 24 FPS），**`Average SSIM = 61.182%`**。
修复后的 InfiniLM 侧画面内容正确：金毛小狗由睁眼（0.05 s）→ 打瞌睡（2.5 s）→ 闭眼睡着（5.0 s）。
出于仓库体积考虑，这两个 compare 产物未随仓库提供，需要复核时可从平台「个人存储」根目录下载
（`reference_fl2va.mp4` 1.5 MB、`infinilm_fl2va_flash.mp4` 1.8 MB）。**未做任何修饰。**

---

## 2. 上游源码

```bash
# 本平台无法直连 github.com，实际使用镜像
git clone --recursive --branch minimax-h3 \
  https://ghfast.top/https://github.com/InfiniTensor/InfiniCore.git
git clone --recursive --branch minimax_h3 \
  https://ghfast.top/https://github.com/InfiniTensor/InfiniLM.git
```

> 说明：构建时未记录上游确切 commit hash，如实说明。本仓库所有补丁均为**字符串级匹配**，
> 不依赖行号，对上游小幅变动有一定鲁棒性。

---

## 3. 快速复现

```bash
# 0) 环境变量
source scripts/env.sh

# 1) 依赖（约 8 分钟）
bash scripts/setup_python.sh
# 期望：MODULAR_H3_OK / H3_MODEL_OK / QWEN3VL_OK / RESOLVE_REV_OK / TV_TRANSFORMS_OK

# 2) 应用补丁
bash patches/apply_all.sh $HOME/pkg/InfiniCore $HOME/pkg/InfiniLM
# 期望：APPLIED 4~5 / SKIP 0 / FAIL 0

# 3) 编译 InfiniCore（约 23 分钟）
bash scripts/build_infinicore.sh
# 期望：INFINICORE_OK / CUDA_ALIAS_OK / FROM_TORCH_OK
#       且 libinfiniop.so 必须 >20 MB（否则 Hygon 算子没编进去）

# 4) 编译 InfiniLM（约 7 分钟）
bash scripts/build_infinilm.sh
# 期望：INFINILM_OK / ENCODER_IMPORT_OK / RUNNER_IMPORT_OK

# 5) 生成 FL2VA（约 3~11 分钟）
bash scripts/run_fl2va_legacy.sh fl2va_official.mp4
# 期望：FINAL_MP4_OK + 规格核查全部 PASS

# 6) 规格核查
bash scripts/verify_output.sh outputs/fl2va_official.mp4
# 或：python3 tests/check_spec.py outputs/fl2va_official.mp4
```

### 三条硬约束（踩过的坑）

1. **不要传 `--height` / `--width`** —— 不指定时由首帧图（800×965）推导尺寸 → **768×928**，
   与官方 `minimax_h3_compare.py` 示例输出一致。手写 384×576 会导致规格不符。
2. **`--num-frames` 不得小于 120** —— 模型要求 120–360 且为 `17n+5` 形式。
3. **必须先导出 `FA_SO_PATH`** —— 否则 SDPA 走 math 路径（FL2VA 需 158.57 GiB 显存，必然 OOM）。

---

## 4. 已验证的结果

> 以下为**参考实现路径（`--legacy`，PyTorch）**的结果，画面正确，用作对照基准。
> 原生 InfiniLM 路径在完成 P5 修复后，官方评分脚本实测 **`Average SSIM = 61.182%`**，
> 画面内容同样正确（见状态摘要）。

### 产物规格（与官方口径逐项一致）

| 项 | 官方要求 | 实测 |
|---|---|---|
| 分辨率 | 768×928 | **768×928** |
| 帧数 | 124 | **124**（逐帧解码计数） |
| 帧率 | 24 FPS | **24.0** |
| 时长 | 5.17 s | **5.175 s** |
| 视频编码 | H.264 | **h264 (High) / yuv420p** |
| 音频 | 存在 | **aac (LC) 32000 Hz 立体声** |
| 生成步数 | 50 | **50** |

### 画面内容

| 时刻 | 内容 |
|---|---|
| 0.05 s | 金毛幼犬趴在苔藓木头上，双眼睁开看向镜头（与输入首帧构图一致） |
| 5.00 s | 双眼完全闭合、头部下垂，趴在木头上睡着 |

### 性能（端到端 606.35 s）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 全部加载（热缓存） | 40.2 s | 6.6% |
| **DiT 去噪（49 次调用）** | **530.42 s** | **87.5%** |
| 视频 VAE 解码 | 17.37 s | 2.9% |
| 其余 | 18.4 s | 3.0% |

**关键发现**：冷缓存时加载占总时间 **92%**，热缓存时计算占 **87.5%**——性能优化必须
先解决权重 I/O，再优化 GEMM/attention；单算子优化（如 `timestep_embedding` 2.92×）
对端到端不可测。

---

## 5. 已知限制

| # | 事项 | 状态 |
|---|---|---|
| **L1** | ~~InfiniLM 原生路径段错误~~ | ✅ **已修复**。根因是构建时未启用 `--flash-attn`，导致 `ENABLE_FLASH_ATTN` 未定义、海光 `mha` 算子整个文件不参与编译。补见 §0.1 与 `patches/PATCHES.md` P1~P5 |
| **L2** | ~~原生路径画面为噪声~~ | ✅ **已修复**。根因是 `mha` 实现为手写 `at::matmul + at::softmax`（bf16 落盘 `[B,H,S,S]`）。改调厂商 `flash::vllm_mha_varlen_fwd` 后 SSIM 9.923% → **61.182%** |
| L3 | 与参考实现仍有差距（61.182% vs 93.584%） | 🟡 主要是 bf16 累积误差；参考实现自己换个随机后验采样重跑也只有 80.4%，这是本口径的噪声地板 |
| L4 | TP 只能为 8 | TP=4/2/1 时 InfiniCore 分配器按未分片尺寸分配显存，OOM；无法做 TP 对照 |
| L5 | Ref2VA 未实现 | 时间不足，未开展 |
| L6 | 九齿未接入 Hygon 注册链 | 只做了独立基准测试，未进入正式推理路径 |
| L7 | 容器禁用 ptrace、无 core dump | 无法获取 C++ 调用栈，排查只能靠黑盒实验与逐元素对拍 |

详见 [技术报告.md](技术报告.md) 第 8 章（完整排查记录，含 8.5 节的对拍实验与 8.5.6 的最终修复）。

---

## 6. 九齿算子优化

用九齿 DSL 重写 `timestep_embedding`，实测：

| 平台 | 加速比 | 精度 |
|---|---|---|
| gfx936（BW） | **2.92×** | 无损（与 PyTorch 逐位一致） |
| gfx906（MI50） | **2.99×** | 无损 |

**对端到端耗时的影响可忽略**（DiT 的 530 s 里绝大部分是 50 层 GEMM 与 attention），
报告如实记录，不宣称端到端加速。详见技术报告 §4。

---

## 7. 合规性

- 未调用任何外部在线 API
- 未对模型做量化、替换、微调
- 未减少帧数（124）、分辨率（768×928）、生成步数（50）
- 未预计算或缓存生成结果，所有输出均由当次作业实时生成
- 固定模型权重与出题方数据不随本仓库打包（从共享只读目录读取）
- 第三方组件已在 `requirements.txt` 与技术报告 §3.8 逐项列明
