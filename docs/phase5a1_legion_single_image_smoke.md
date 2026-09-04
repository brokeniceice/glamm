# Phase 5A-1 — LEGION Single-Image Smoke Test

日期：2026-08-31（UTC）  
阶段结论：**S-A**  
范围：仅验证官方公开 intermediate `legion_LE` checkpoint 能否完成一次 localization + explanation 推理；不是 LEGION vs R1 性能实验。

## 1. 结论

固定到官方 LEGION commit `d21535dd45f6fea509337a83095966f0b86ac924` 和 HuggingFace revision `f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa` 后，公开的 intermediate LE checkpoint 在一张非 benchmark 普通照片上完成了**唯一一次** forward：

| 成功条件 | 结果 | 证据 |
|---|---|---|
| model load 成功 | YES | 两个 checkpoint shard 均加载完成 |
| explanation 非空 | YES | 官方 `explanation.json` 已生成，清洗后 475 个字符 |
| generation 至少一个 `[SEG]` | YES | token ID `32004`，共 **2** 个 |
| SAM 产生 `mask.png` | YES | 两张 per-`[SEG]` mask 经官方 `logit > 0` + union 写出 |
| mask 与原图同尺寸 | YES | 均为 `3840×2560`（W×H） |

因此本阶段分类为：

```text
S-A
```

这只证明“当前公开 checkpoint 能跑”。它仍是官方发布的 **intermediate LE checkpoint**，不是论文 final checkpoint；本结果不支持任何性能、质量或 LEGION vs R1 优劣结论。

## 2. 固定身份

### 2.1 LEGION source

| Field | Value |
|---|---|
| Official repository | `https://github.com/opendatalab/LEGION.git` |
| Local checkout | `/home/yz/groundingLMM_official/external/LEGION_official` |
| Commit | `d21535dd45f6fea509337a83095966f0b86ac924` |
| Branch | `main` |
| Core code modified | NO |
| Worktree after run | clean，`## main...origin/main` |

官方 repo 中包含 tracked `.pyc` 文件。本次运行产生的 untracked Python cache 已清理；误删的 tracked cache 已从同一 HEAD 精确恢复，最终 `git status --porcelain` 为空，源码文件从未修改。

### 2.2 Public intermediate LE checkpoint

| Field | Value |
|---|---|
| HuggingFace repo | `khr0516/legion_LE` |
| Fixed revision | `f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa` |
| Local directory | `/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/legion_LE_f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa` |
| Index | 968 keys；16,752,883,392 tensor bytes |
| Shard 1 | 9,976,691,902 bytes；SHA-256 `07080b5ac1840f0fb5c2a569796058b42e0699277bac6ba3a77aa8793dc6daeb` |
| Shard 2 | 6,776,538,784 bytes；SHA-256 `b743f1cce5bccbe613f94c108dfb5d327c18096161fdf7303b1497fd2efe7fb7` |
| tokenizer.model | 499,723 bytes；SHA-256 `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` |

两个大分片使用固定 revision 的 resolve URL 断点续传。最终字节数和三项官方 LFS SHA-256 全部一致。

### 2.3 CLIP

| Field | Value |
|---|---|
| Model | `openai/clip-vit-large-patch14-336` |
| Fixed revision | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| Local directory | `/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| `pytorch_model.bin` SHA-256 | `c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801` |
| Vision config | hidden size 1024；image size 336 |

本机原 HuggingFace cache 已存在该 exact revision，因此将其实体复制到指定 `/data/.../checkpoints` 目录并校验 SHA-256；推理以本地目录离线加载。Transformers 报告 full CLIP checkpoint 中 text tower 权重未被 `CLIPVisionModel` 使用，这是官方 vision-only 加载方式的预期提示，不是缺权重。

## 3. 环境与 dependency deviations

独立环境：`/home/yz/miniconda3/envs/legion`，未修改 R1 的 `glamm_official` 环境。

| Component | Version |
|---|---|
| Python | 3.10.21 |
| PyTorch | 1.13.1+cu117 |
| torchvision | 0.14.1+cu117 |
| Transformers | 4.28.0 |
| PEFT | 0.5.0 |
| DeepSpeed | 0.12.5 |
| FlashAttention | 2.3.6 |
| MMCV-full | 1.4.7，CUDA ops enabled |
| CUDA build toolkit | 11.7 |

对 `requirements/default.txt` 的 156 个 `==` pin 做 installed metadata 比对：**156/156 exact，0 missing，0 version mismatch**。

额外/安装层 deviations：

1. 官方依赖必须先安装 PyTorch，再以 `--no-build-isolation` 构建 `flash-attn==2.3.6`；否则 build isolation 看不到 Torch。
2. 未固定的 setuptools 使用 `68.2.2`，以保留旧源码构建所需的 `pkg_resources`。
3. 按官方 README 从 `mmcv v1.4.7` 源码构建，MMCV source commit 为 `e319369068c8fe813086ecddf1070ac510860592`，设置 `MMCV_WITH_OPS=1`、`TORCH_CUDA_ARCH_LIST=8.6`。
4. 为 MMCV CUDA ops 添加 NVIDIA `cuda-11.7.1` 开发头文件/库；这不改变官方 Python pins。
5. `pip check` 只报告官方固定的 `ninja==1.11.1.1` 为 “not supported on this platform”；同一 ninja 实际成功完成 FlashAttention/MMCV 构建。该 checker 警告保留在日志中。

完整安装日志与环境快照位于：

```text
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/dependency_install.log
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/environment/pip_freeze.txt
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/environment/conda_explicit.txt
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/environment/runtime_versions.txt
```

## 4. Launcher workaround

官方 `scripts/loc_exp/infer.py` 未修改。外部 wrapper 为：

```text
/home/yz/groundingLMM_official/scripts/phase5a1_legion_smoke_wrapper.py
```

它只做以下启动与观察工作：

1. 将官方 repo 加入 `PYTHONPATH` / `sys.path`；
2. 设置单进程 distributed 环境：`RANK=0`、`WORLD_SIZE=1`、`LOCAL_RANK=0`、`MASTER_ADDR=127.0.0.1`、`MASTER_PORT=29517`；
3. 显式传 `--local_rank 0`，解决官方代码无条件构造 `DistributedSampler` 的问题；
4. 将 checkpoint、image、output 三个占位路径替换为本次固定绝对路径；
5. 将官方请求的 CLIP repo ID 旁路解析到固定 revision 本地目录；
6. 不改变返回值地观察 `evaluate()` 的 generated IDs / mask shapes，以及 tokenizer decode 文本和 PyTorch peak memory。

传给官方 `infer.py` 的参数只有：

```text
--hf_model_path <fixed local checkpoint>
--image_dir <single-image directory>
--output_dir <run output directory>
--local_rank 0
```

未传入或修改 `image_size`、`model_max_length`、conversation type、prompt、threshold、beam 或 generation 参数；官方固定 prompt、`max_tokens_new=512`、`num_beams=1`、mask `logit > 0` 和 multi-mask union 均保持原样。

## 5. Input

| Field | Value |
|---|---|
| Image | `matt_mcnulty_nyc_ordinary_photo.jpg` |
| Source | 系统文件 `/usr/share/backgrounds/matt-mcnulty-nyc-2nd-ave.jpg` |
| Package | `ubuntu-wallpapers-focal` |
| Local run copy | `/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/input/matt_mcnulty_nyc_ordinary_photo.jpg` |
| Size | 3840×2560（W×H） |
| SHA-256 | `7231c2921ac2db970ee859f21264f0a79545d3a891392068621ab0eb99ce64f9` |

这是普通 NYC 场景照片，不来自 official1000、SynthScars、LOKI、RichHF、R1 数据或任何本项目 benchmark/封存集。输入目录中只有这一张图片，未抽样、未换图。

## 6. Single forward result

### 6.1 GPU 与运行

| Field | Value |
|---|---|
| Physical GPU | index 1，NVIDIA RTX A6000 |
| UUID | `GPU-69f825a5-8566-423b-f02b-f5454cf6d425` |
| Driver | 570.133.07 |
| Total / free before run | 49,140 / 48,546 MiB |
| PyTorch peak allocated | 18,552,989,696 bytes（17.279 GiB） |
| PyTorch peak reserved | 19,379,781,632 bytes（18.049 GiB） |
| Wrapper elapsed | 36.629 s |
| Official dataloader inference progress | 23.54 s for 1/1 |
| Successful forwards | **1** |

物理 GPU 0 当时被其他进程占用，因此选择空闲 GPU 1；没有在其他 GPU 上重复 forward。

### 6.2 Explanation 与 `[SEG]`

官方保存的 cleaned explanation 为：

> Upon examining the image. I have found: A dimly lit subway platform features a distorted and unrecognizable text on the right side of the column, while the left side of the column is missing a portion of its structure.To elaborate, I have found the following artifacts. Text on the right side of the column :The text on the right side of the column is distorted and unrecognizable. The left side of the column :The left side of the column is missing a portion of its structure.

解释非空。旁路记录的 raw decode 中存在 **2 个 `[SEG]`**；generated sequence shape 为 `[1, 279]`，每个 `[SEG]` 对应一张 SAM mask，返回 tensor shape 为 `[2, 2560, 3840]`。

本阶段不评价上述 explanation 对真实图片是否正确。普通图片上模型仍声明存在 artifacts 只是本次公开 intermediate checkpoint 的单样本输出，不能扩展为模型质量结论。

### 6.3 Mask

| Field | Value |
|---|---|
| Output | `/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/output/matt_mcnulty_nyc_ordinary_photo.jpg/mask.png` |
| Shape | `[2560, 3840]`（H×W） |
| Mode / values | grayscale `L`；`{0, 255}` |
| Foreground pixels | 50,345 / 9,830,400 |
| Foreground ratio | `0.0051213582`（0.5121%） |
| SHA-256 | `38eac20512e125dff026fd62eb77c4aaa440e2a98421a5c5df48240e2a702513` |

mask 非空且尺寸与原图严格一致。这里只记录 smoke 所要求的 foreground ratio；未计算 IoU、F1 或任何 benchmark metric。

## 7. Evidence and storage

大文件、缓存、构建源与运行输出统一位于当前项目 `checkpoints` 链接所对应的 `/data` 目录：

```text
/home/yz/groundingLMM_official/checkpoints
  -> /data/yz/myLISA_storage/checkpoints

/data/yz/myLISA_storage/checkpoints/phase5a1_legion/
```

阶段目录在完成时约 28 GiB，包含 16.75 GB LEGION tensor bytes、约 1.6 GiB CLIP、Conda/PIP 构建缓存、MMCV source 和全部日志。关键证据：

```text
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/checkpoint_download.log
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/checkpoint_sha256.txt
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/clip_sha256.txt
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/inference.log
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/observation.json
/data/yz/myLISA_storage/checkpoints/phase5a1_legion/run/output_validation.json
```

项目内的轻量机器记录位于 `artifacts/phase5a1_legion/`：`manifest.json`、`observation.json`、`output_validation.json` 和 checksum 文件。

## 8. Phase boundary

本阶段未训练、fine-tune、merge、调参或修改 LEGION 核心代码；未运行 R1、official1000、SynthScars、LOKI、RichHF；未访问封存集；未计算 IoU/F1；没有因输出质量重采样。

Phase 5A-1 到此停止。下一阶段若获正式授权，公开 intermediate LEGION 与 R1 的共同测试集比较只能作为参考性/诊断性 Phase 5A-2；论文主要 baseline 仍应来自后续按共同训练协议重训官方 LEGION 的 Phase 5A-3。
