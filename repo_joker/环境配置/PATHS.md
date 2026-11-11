# 模型与数据路径配置

本文件说明本工程涉及的全部路径，对应赛题第 11 条「模型和数据路径配置」。

---

## 1. 路径总表

| 类别 | 环境变量 | 实际路径 | 权限 | 说明 |
|---|---|---|---|---|
| 共享模型根 | `$SHARED` | `/public/home/aclsylqidf/shared` | **只读** | 平台提供，不占个人额度 |
| H3 权重根 | `$MODEL` | `$SHARED/MiniMax-H3` | 只读 | 331 GB |
| FL2VA 分区 | `$MODEL/diffusers` | `.../MiniMax-H3/diffusers` | 只读 | 135 GB，FL2VA 用这个 |
| Ref2VA 分区 | `$MODEL/Ref2VA` | `.../MiniMax-H3/Ref2VA` | 只读 | 124 GB，**本工程未使用** |
| 官方首帧图 | `$FIRST_FRAME` | `$AFS/Golde33443.png` | 读写 | 800×965 PNG，1.36 MB |
| 个人 AFS | `$AFS` | `$IFAB_AFS_HOME` | 读写 | 4 GiB 配额 |
| Python 依赖 | `$PL` | `$AFS/pylibs` | 读写 | 约 2.3 GiB |
| InfiniCore 安装 | `$INFINI_ROOT` | `$AFS/infini` | 读写 | 38 MB（lib + include） |
| pip 修复模块 | — | `$AFS/shim` | 读写 | 9 KB，`sitecustomize.py` |
| xmake | `$XMAKE` | `$AFS/xmake299` | 可执行 | 2.4 MB |
| 推理缓存 | `$HF_HOME` | `$HOME/.cache/huggingface` | 临时 | **不写 AFS**，作业结束即清 |
| 输出目录 | `$OUTDIR` | `$HOME/out` | 临时 | 产物需复制到 AFS 才持久 |

---

## 2. `--model-path` 的语义

`examples/minimax_h3.py` 通过 `resolve_model_paths()` 解析，**两种写法都可用**：

| 写法 | 解析结果 |
|---|---|
| `--model-path=$SHARED/MiniMax-H3` | `partition_dir` 自动按 task 选择（fl2va→diffusers，ref2va→Ref2VA） |
| `--model-path=$SHARED/MiniMax-H3/diffusers` | 直接使用该分区 |

本工程统一使用第二种，明确无歧义。

---

## 3. `diffusers` 分区内部结构（实测）

```
MiniMax-H3/diffusers/
├── model_index.json               # ModularPipeline 入口
├── modular_model_index.json
├── audio_scheduler/
├── audio_vae/                     # 音频 VAE（输出 AAC 音轨的来源）
├── processor/                     # Qwen3-VL processor（需要 torchvision）
├── scheduler/
├── text_encoder/                  # Qwen3-VL-32B（14 分片 + config.json）
├── tokenizer/
├── transformer/                   # H3 DiT（14 分片）
└── vae/                           # 视频 VAE
```

**注意**：`processor/` 依赖 torchvision 的 `torchvision::nms` 算子。
若 torchvision 未正确编译，会得到
`AttributeError: 'NoneType' object has no attribute 'image_processor'`。

---

## 4. 缓存路径的刻意选择

| 变量 | 取值 | 原因 |
|---|---|---|
| `HF_HOME` | `$HOME/.cache/huggingface` | **指向临时盘**。平台文档建议放 AFS，但本工程 AFS 仅剩 1.5 GiB，放缓存会撑爆配额。作业结束临时盘清空，无副作用。 |
| `TMPDIR` | `$HOME/tmp` | 同上；torchvision 编译中间产物可达数百 MB |
| `HF_HUB_OFFLINE` | 未设置 | 权重全在本地，实际不触发联网；若需完全离线可设为 `1` |

---

## 5. 空间占用实测

```
$AFS 总计           2.5 GiB / 4.0 GiB
├── pylibs/         2.3 GiB   （diffusers + transformers + flash-attn 的 .so(2.0G) + torchvision）
├── infini/          38 MiB   （lib: 4 个 .so + include）
├── xmake299         2.4 MiB
├── shim              9 KiB
├── Golde33443.png   1.3 MiB
└── (产物) fl2va_*.mp4  1.5 MiB
```

**关键约束**：`flash_attn_2_cuda*.so` 单文件 **2.0 GiB**，是 AFS 占用的主要来源。
因此**不能安装 triton**（解压再增 1 GiB 会超限）。

---

## 6. 部署到新环境时需修改的项

修改 `scripts/env.sh`（或复制 `环境配置/env_template.sh`）中的：

```bash
SHARED=...        # 换成本平台的共享模型目录
MODEL=$SHARED/MiniMax-H3
FIRST_FRAME=...   # 官方首帧图位置
```

其余（`AFS`、`PL`、`INFINI_ROOT`、`XMAKE`）由 `$IFAB_AFS_HOME` 自动推导，通常无需改动。
