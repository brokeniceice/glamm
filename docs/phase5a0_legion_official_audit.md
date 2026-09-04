# Phase 5A-0 LEGION Official Audit

审计日期：2026-08-31（UTC）  
阶段：Phase 5A-0 — LEGION Official Repository Audit & Reproduction Setup  
范围：repository + checkpoint + inference-path audit；未训练、未 benchmark、未 forward、未访问任何封存集。

| Question | Answer | Evidence |
|---|---|---|
| 官方 repo 是否可用 | YES（源码可获取；launcher 非开箱即用） | 官方仓库已 clone，固定到 clean commit `d21535d`；静态语法通过 |
| 官方 final LE weights 是否仍存在 | 公开：NO；作者侧是否仍可恢复：UNKNOWN | README 明确称找不到 original final weights，只发布 intermediate LE weights |
| 当前公开 intermediate LE weights 是否存在 | YES | `khr0516/legion_LE@f8c28b3`，2 个 `.bin` 分片，共 16,752,883,392 tensor bytes |
| checkpoint 是否支持 localization | CONDITIONALLY | 完整 SAM、LLM、`text_hidden_fcs` 均在权重索引中；需依赖、CLIP、GPU 和正确 launcher |
| checkpoint 是否支持 explanation | CONDITIONALLY | 完整 LLM/tokenizer/`[SEG]` 和官方文本输出链；本阶段未 forward |
| 是否需要重新训练 | NO（跑公开 intermediate LE）；YES/UNKNOWN（若要求 exact paper-final） | 公开权重可用于 LE；exact final 未公开且作者称已无法定位 |
| 是否需要 merge | NO（公开 checkpoint）；训练产物才需要 | 公共目录已是 sharded HF full-model，0 LoRA/adapter keys |
| 是否需要 GLaMM base | 推理 NO；训练/重做 merge YES | 公共分片含完整 LLM；merge step 2 默认输入 GLaMM-GranD-Pretrained |
| 是否需要 SAM weight | 推理 NO；训练/重做 merge YES | 公共分片含 594 个 SAM keys；训练/merge 参数要求 `sam_vit_h_4b8939.pth` |
| inference script 是否完整 | 调用链 YES；stock launcher NO | `infer.py` 完整产出 text+mask；`infer.sh` 缺 PYTHONPATH 和分布式启动条件 |
| 能否对任意 folder 图片推理 | CONDITIONALLY（仅 flat、全部条目均为可读图片） | `GCGEvalDDP` 直接 `os.listdir`，无扩展名/文件类型过滤 |
| 能否直接导出 mask | YES | 原图分辨率二值 `mask.png`；阈值为 logit `> 0`，多 `[SEG]` mask 取 union |
| 是否需要改代码才能跑 | 核心模型 NO；官方 shell 需 wrapper/启动参数修正 | 不建议 patch；Phase 5A-1 可用外部 wrapper 提供 PYTHONPATH 与单进程 DDP 环境 |

> `LE = Localization + Explanation`。本报告中的“可复现”仅指当前公开 intermediate LE checkpoint 的官方推理路径，不代表论文表格 exact final model 或论文指标可复现。

## 1. Executive Summary

**最终分类：Case A（带明确限制）。**

`khr0516/legion_LE` 不是 LoRA adapter，也不是只含部分模块的 delta；它是已经完成官方 step-2 merge、可由 `LegionForCausalLM.from_pretrained()` 加载的完整 HF localization+explanation 模型目录。权重索引覆盖 LLaMA 语言模型、LM head、MM projector、region encoder、SAM ViT-H image encoder、SAM prompt encoder、SAM mask decoder 和 `text_hidden_fcs`。运行时仍需单独取得 `openai/clip-vit-large-patch14-336`，因为官方 merge 明确排除了 `vision_tower` 权重。

与此同时，官方 README 明确将这份权重称为 **intermediate weights**，并称 original final weights 因存储变更无法定位。因此：

- 公开 intermediate checkpoint 可以有条件地执行 localization 和 explanation：`SUPPORTED BY STATIC EVIDENCE`；
- 它是论文 localization/explanation 表格 exact final checkpoint：`NO`；
- exact final LE checkpoint 的公开可用性：`NO`；
- exact final detection checkpoint 的公开可用性：`NO`，且当前 LE 权重没有 classification `prediction_head`；
- 当前公开权重实际 forward 是否成功、输出质量如何：本阶段 `UNTESTED`，等待 Phase 5A-1 授权。

官方 `infer.py` 的模型与输出链完整，但 `infer.sh` 不是可直接复现入口：它没有设置 `PYTHONPATH`，没有建立推理代码无条件使用的分布式进程组，并保留三处 `/path/to/...` 占位符。这些是启动层 blocker，不应被误写成“checkpoint 缺失”或“必须重训”。

## 2. Repository Identity

| Field | Value |
|---|---|
| Local path | `/home/yz/groundingLMM_official/external/LEGION_official` |
| Remote | `https://github.com/opendatalab/LEGION.git`（fetch/push） |
| Branch | `main` |
| Full commit | `d21535dd45f6fea509337a83095966f0b86ac924` |
| Latest commit | `d21535d upload weights for artifact loacalization and explanation` |
| Commit date | `2025-10-22T16:27:42+08:00` |
| Worktree | clean，`## main...origin/main` |
| Tags | none |

所有后续 LEGION 静态结论必须绑定上述 full commit。机器记录见 `artifacts/phase5a0_legion/repo_commit.txt`。

官方 repo 本身没有被 patch。主项目仍为 `main`，此前全部未提交改动均保留；本阶段只新增 `external/LEGION_official/`、本报告、审计 artifacts，并创建独立 conda 环境。

## 3. Repository Structure

三层目录树见 `artifacts/phase5a0_legion/repo_tree.txt`。关键目录作用如下：

| Path | Role |
|---|---|
| `README.md` | 官方安装、数据、LE/Detection/Controller 入口与权重状态声明 |
| `scripts/loc_exp/` | localization+explanation 训练与 folder inference |
| `scripts/cls/` | 独立 binary detection head 的训练与评估 |
| `scripts/merge_weights/` | DeepSpeed ZeRO consolidation 与 PEFT LoRA merge/HF export |
| `scripts/refine/` | controller 的 regeneration/inpainting；本阶段不执行 |
| `model/Legion.py` | `LegionForCausalLM`、SAM 文本提示分割链与独立 `LegionForCls` |
| `model/llava/` | LLaVA/LLaMA、多模态图像 token、CLIP tower、region feature 注入 |
| `model/SAM/` | vendored SAM ViT-H / prompt encoder / mask decoder |
| `model/layers.py` | 多层 CLIP region feature fusion 与 ROI query 模块 |
| `dataset/` | GLaMM mixed datasets 与 SynthScars/LEGION GCG loader |
| `eval/` | 图像预处理、mask 工具、flat-folder dataset、distributed init |
| `mmdet/` | vendored MMDetection，用于 region feature/ROI 组件 |
| `requirements/` | defender/default 与 controller/regenerator 两套依赖 |
| `tools/` | 特殊 token、常量和训练/推理辅助函数 |

仓库历史提供了直接的 GLaMM 继承证据：早期 `model/GLaMM.py` 在 commit `9ea46d5` 被 98% similarity rename 为 `model/Legion.py`；当次定位模型修改主要是类名/初始化函数重命名，localization 的 `[SEG] hidden state -> text_hidden_fcs -> SAM` 核心结构保持不变。artifact specialization 主要来自 LEGION/SynthScars 训练数据、固定 artifact prompt 和训练得到的参数，而不是额外引入一个新的 artifact-specific localization module。

## 4. Official Weight Status

官方 README 最新声明（`README.md` 的 Latest News）区分了三类权重：

1. **Original final weights**：作者称因 storage changes 无法定位；没有公开下载入口。
2. **Public intermediate LE weights**：`https://huggingface.co/khr0516/legion_LE`，作者称可做 artifact localization and explanation。
3. **Detection weights**：作者写明 detection task 需用户自行完成；当前 LE 权重不含 detection head。

核心回答：

| ID | Question | Answer | Confidence |
|---|---|---|---|
| Q1 | 论文最终 localization 表格的 exact final checkpoint 是否公开存在？ | **NO** | 官方 README 明确发布的是 intermediate 而非 final |
| Q2 | 论文最终 explanation 表格的 exact final checkpoint 是否公开存在？ | **NO** | LE 共享模型；没有另一份 final explanation checkpoint |
| Q3 | `khr0516/legion_LE` 是否可执行 localization？ | **CONDITIONALLY** | 代码+keys 支持；尚未实际 forward |
| Q4 | `khr0516/legion_LE` 是否可执行 explanation？ | **CONDITIONALLY** | 代码+tokenizer 支持；尚未实际 forward |
| Q5 | 当前公开权重是否就是论文表格对应 final model？ | **NO** | 与作者“intermediate weights”表述直接冲突 |

“final weights 不存在”应保守解释为：**没有可验证的官方公开 exact-final artifact**。作者机器/备份中是否还存在可恢复副本无法由公开材料证明，因此只能标 `UNKNOWN`，不能扩大为物理意义上的绝对不存在。

## 5. HuggingFace legion_LE Audit

### 5.1 Repository identity

| Field | Value |
|---|---|
| Repo | `khr0516/legion_LE` |
| Pinned revision | `f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa` |
| Revision title | `upload weights for artifact localization and explanation` |
| Revision time | `2025-10-22T07:42:11Z` |
| Previous revision | `ab87c945dbdb2cf0496340eb67f1926900808047` |
| Model card/README | absent |
| Declared library/pipeline | absent |
| Tags | `pytorch`, `llava` |

完整文件与 LFS SHA-256 见 `artifacts/phase5a0_legion/hf_file_inventory.txt`。主 revision 共 10 个文件：配置/分词器文件齐全、2 个 `.bin` 分片、1 个 index；没有 `.safetensors`、`.pt`、`.pth`、adapter config 或单独 projector 文件。

### 5.2 Checkpoint key evidence

`pytorch_model.bin.index.json` 声明 968 keys、16,752,883,392 tensor bytes。按模块前缀统计：

| Key family | Count | Meaning |
|---|---:|---|
| LLM/embed/norm/lm_head | 323 | 完整 32-layer LLaMA-class language model + output head |
| SAM image encoder | 457 | 完整 ViT-H grounding encoder |
| SAM prompt encoder | 17 | text prompt 到 SAM sparse/dense prompts |
| SAM mask decoder | 120 | 完整 mask decoder |
| `text_hidden_fcs` | 4 | 4096→4096→256 的 `[SEG]` hidden projection |
| MM projector | 4 | CLIP global features 到 LLM hidden space |
| Region encoder | 43 | 多层 region/ROI feature module |
| CLIP vision tower | 0 | 运行时从外部 `openai/clip-vit-large-patch14-336` 初始化 |
| Detection prediction head | 0 | 当前权重不支持独立 detection classifier |
| LoRA/adapter keys | 0 | 不是 adapter-only checkpoint |

因此 checkpoint 的**文件形式**属于 A：完整 merged localization+explanation model；但其**训练选择身份**仍是 intermediate，不是 paper-final。这两个维度不能混为一谈。

### 5.3 Runtime bases

- `GLaMM-GranD-Pretrained`：公开 merged checkpoint 推理不需要；重新执行 merge step 2 才需要。
- `sam_vit_h_4b8939.pth`：公开 merged checkpoint 推理不需要；训练/重新 merge 才需要。
- `openai/clip-vit-large-patch14-336`：推理需要，且不在公开分片中。
- tokenizer：已包含；`[SEG]` ID 为 `32004`，另含 `<p>`, `</p>`, `<bbox>`, `<point>`, `<im_start>`, `<im_end>`。

## 6. Localization / Explanation Inference Path

### 6.1 `infer.sh`

`scripts/loc_exp/infer.sh` 只有一条命令：

```text
python scripts/loc_exp/infer.py \
  --hf_model_path /path/to/legion/ckpt \
  --image_dir /path/to/image \
  --output_dir /path/to/save
```

- Python entry：`scripts/loc_exp/infer.py`
- model/image/output：全部为 hard-coded placeholders；Python 内 `image_dir` 默认值本来是 `./inference_data`，但 shell 覆盖为 placeholder。
- GPU：shell 未指定；Python 使用 `.cuda()`，因此落在当前默认 CUDA device。
- precision：固定 BF16；注释仅提示可手工改 FP32。
- distributed launch：无 `torchrun`/launch/env 设置。
- `PYTHONPATH`：未设置；从 repo root 直接执行会报 `No module named 'eval'`。

### 6.2 Model loading

`scripts/loc_exp/infer.py:113-140`：

1. `AutoTokenizer.from_pretrained(hf_model_path)`；
2. 从 tokenizer 取得 `[SEG]` token ID；
3. `LegionForCausalLM.from_pretrained(hf_model_path, torch_dtype=bfloat16)`；
4. `initialize_vision_modules(config)` 建立外部 CLIP tower；
5. model 与 CLIP tower 转 BF16/CUDA；
6. `CLIPImageProcessor.from_pretrained(model.config.vision_tower)`。

模型类为 `model/Legion.py::LegionForCausalLM`，继承 `model/llava/model/language_model/llava_llama.py::LlavaLlamaForCausalLM`，其底层是 Transformers `LlamaForCausalLM`。HF config 为 32 layers、hidden 4096、32 heads，对应 GLaMM 使用的 LLaMA-class 7B backbone。视觉系统是双路径：

- global/LLM vision：CLIP ViT-L/14-336；
- grounding vision：SAM ViT-H（1024 输入）。

### 6.3 Input and prompt

- 输入单位：一个 flat folder；DataLoader batch 默认 1。
- 格式：代码没有 allowlist；所有 `os.listdir()` 条目都进入 `cv2.imread()`。因此只能保守承诺 OpenCV 能解码且目录中没有子目录/非图片的文件。
- prompt：固定为 artifact detailed analysis，覆盖 physical / structural / distortion artifacts，并要求 interleaved segmentation masks。
- localization 与 explanation：一次 LLM generation 同时产生自然语言与 `[SEG]` token；每个 `[SEG]` 触发一张 mask。

### 6.4 Output

- `pred_masks`：SAM postprocess 后的 **logits**，每个 `[SEG]` 一张，分辨率恢复为原图 `H×W`。
- threshold：`scripts/loc_exp/infer.py:162-164`，严格 `logit > 0`。
- multi-mask aggregation：所有 `[SEG]` binary mask 做 `torch.any` union。
- mask file：`output_dir/<input filename>/mask.png`，8-bit `{0,255}`、原图分辨率。
- tensor/NumPy：不保存。
- explanation：`output_dir/<input filename>/explanation.json`；内容只是 `{image_path: cleaned_caption}`。
- structured per-region JSON：没有。
- `[SEG]`：generation 内存在并驱动 mask，保存文本前被删除。
- `<p>...</p>` phrase：解析到 `phrases`，但之后未保存也未用于输出关联。

完整调用图见 `artifacts/phase5a0_legion/inference_callgraph.txt`。

## 7. Model Architecture Trace

### 7.1 Exact segmentation path

```text
scripts/loc_exp/infer.py::inference
  image (BGR -> RGB)
    |-- CLIPImageProcessor.preprocess
    |     -> model/llava/.../clip_encoder.py::CLIPVisionTower
    |     -> model/llava/llava_with_region_arch.py::encode_images
    |     -> LegionModel.mm_projector
    |     -> LlavaLlamaForCausalLM.generate
    |     -> text tokens + generation hidden states
    |
    `-- ResizeLongestSide(1024)
          -> eval/utils.py::grounding_image_ecoder_preprocess
          -> LegionForCausalLM.get_grounding_encoder_embs
          -> SAM ImageEncoderViT

generated [SEG] token positions
  -> model/Legion.py::LegionForCausalLM.evaluate
  -> _process_hidden_states
  -> LegionModel.text_hidden_fcs[0]
       Linear(4096,4096) -> ReLU -> Linear(4096,256)
  -> _generate_and_postprocess_masks
  -> SAM PromptEncoder(text_embeds=256-D prompts)
  -> SAM MaskDecoder(multimask_output=False)
  -> low_res mask logits
  -> SAM.postprocess_masks(resized input size, original image size)
  -> per-[SEG] original-resolution logits
  -> infer.py: logits > 0 -> union -> mask.png
```

### 7.2 Relationship with GLaMM

- 官方 README 明确称 built upon GLaMM。
- 代码历史显示 `model/GLaMM.py` 98% rename 成 `model/Legion.py`；核心 LE 架构未被替换。
- 仍使用 `[SEG]`（不是 tokenizer 中的 `<SEG>`）；ID 从公开 tokenizer 动态解析。
- `text hidden state -> MLP projection -> SAM prompt encoder/mask decoder` 与 GLaMM 路线一致。
- localization 没有新增独立 artifact-specific encoder/decoder；artifact specialization 来自训练与 prompt。
- `region_encoder`/MM projector 属于继承的 GLaMM/LLaVA region path；当前 folder inference 传 `bboxes=None`，不使用 region boxes。
- localization 与 explanation 共用同一个 `LegionForCausalLM` 和一次 generation。
- detection 是另一个 `LegionForCls` class：从 CLIP penultimate hidden state 的 CLS token 接独立两层 `prediction_head`，不是 LE generation 的自动判定结果。
- 公开 `legion_LE` index 没有 `prediction_head`，因此 public LE model 不具备 detection head。

Issue #10 的用户报告也与此结构一致：公开 LE 权重会把输入送入 artifact prompt/mask 路径，但没有 detection gate 去先判断 real/fake。该报告是用户现象，不是 maintainer-confirmed 结论。

## 8. Merge Pipeline

### Step 1

脚本：`scripts/merge_weights/step1.sh`

- 输入：DeepSpeed checkpoint directory，硬编码 tag `global_step703`，以及 checkpoint 自带 `zero_to_fp32.py`。
- 操作：调用 `zero_to_fp32.py <checkpoint_dir> <checkpoint_dir>/global_step703/pytorch_model.bin --tag global_step703`。
- 输出：合并后的单一 FP32 `pytorch_model.bin`。

### Step 2

脚本：`scripts/merge_weights/step2.sh` → `merge_lora_weights.py`

- 输入：
  - `MBZUAI/GLaMM-GranD-Pretrained` 或本地等价 base；
  - step-1 `pytorch_model.bin`；
  - SAM ViT-H checkpoint；
  - output directory。
- 操作：
  1. 建立 tokenizer/special tokens；
  2. 从 GLaMM base 构造 `LegionForCausalLM`；
  3. 初始化 CLIP/SAM；
  4. 按 LoRA r=8、alpha=16、target `q_proj,v_proj` 包装模型；
  5. 给 state-dict keys 增加 `base_model.model.` 并 `strict=True` load；
  6. `merge_and_unload()`；
  7. 排除所有 `vision_tower` keys；
  8. `save_pretrained()` 模型与 tokenizer。
- 输出：可由 HF `from_pretrained()` 加载的 merged directory。

### Public checkpoint merge status

| Check | Answer | Basis |
|---|---|---|
| 是否完成 step 1 | UNKNOWN（使用者无需再做） | 当前公开目录不保留原 DeepSpeed/tag provenance，无法证明是否逐字执行 hard-coded step 1 |
| 是否完成 step 2 | YES | full HF shards + tokenizer；0 LoRA keys；SAM/LLM/projectors 已内嵌；CLIP 被排除 |
| 当前使用是否还需 merge | NO | 直接指向 `hf_model_path` 即是官方推理预期结构 |

另有一个静态入口缺口：`merge_lora_weights.py` 使用 `from train import ...`，但当前 repo root 没有 `train.py`，真正文件是 `scripts/loc_exp/train.py`；`step2.sh` 只导出 `PYTHONPATH=./`。因此若未来重新 merge，必须先用不修改核心实现的 import-path wrapper 或由人工批准修复。当前 public checkpoint 已 merge，不受此缺口影响。

## 9. Dependency Audit

完整关键版本与资产角色见 `artifacts/phase5a0_legion/dependency_inventory.txt`。

| Component | Official pin / requirement |
|---|---|
| Python | README 3.10；已创建隔离 `legion` env，Python 3.10.21 |
| torch | `1.13.1+cu117` |
| torchvision | `0.14.1+cu117` |
| transformers | `4.28.0` |
| peft | `0.5.0` |
| deepspeed | `0.12.5` |
| flash-attn | `2.3.6` |
| mmcv | source build tag `v1.4.7`, `MMCV_WITH_OPS=1` |
| mmdet | `2.22.0`，vendored in repository |
| bitsandbytes | not listed |
| opencv-python | `4.8.0.74` |
| kornia | `0.7.3` |
| CUDA | torch wheels target 11.7；driver/system CUDA 未进一步 pin |

隔离状态：

- 新环境：`/home/yz/miniconda3/envs/legion`；只安装最小 Python 3.10 基础环境。
- 未在新环境安装 torch、mmcv、flash-attn 或 requirements。
- 未升级/降级 `glamm_official` 中任何包。
- 现有 R1 环境保持不变。

静态检查：shell `bash -n` 和关键 Python 文件 `py_compile` 均通过。`infer.py --help` 的 import-path/依赖失败已记录，未通过安装或 patch 掩盖。

## 10. GitHub Issue Findings

审计入口：`https://github.com/opendatalab/LEGION/issues`。与权重/推理直接相关的公开证据：

| Issue | Explicit evidence | Maintainer resolution visible at audit time |
|---|---|---|
| [#5 Request for LEGION Model Weights](https://github.com/opendatalab/LEGION/issues/5) | 2025-04 用户请求实验权重 | 无；open |
| [#6 Issues with Model Output and Request for Pretrained Weights](https://github.com/opendatalab/LEGION/issues/6) | 用户自训出现全黑 mask、重复/无意义文本 | 无；open；不能外推到当前公开权重 |
| [#9 Pretrained weights release](https://github.com/opendatalab/LEGION/issues/9) | 2025-10 仍有人询问 release plan | 无；open；之后 README/HF 发布 intermediate LE |
| [#10 Detection vs Localization&Explanation](https://github.com/opendatalab/LEGION/issues/10) | 用户称已用公开权重推理，但所有图都被定位为 artifact，并询问 detection head 关系 | 无；open；至少构成第三方“能启动公开权重”的弱证据 |
| [#11 Has anyone been able to reproduce the results?](https://github.com/opendatalab/LEGION/issues/11) | 用户称作者预训练权重与自训结果不佳，并请求 best-performance weights | 无；open；支持“公开 intermediate ≠ best/final”但不是定量验证 |

没有找到 maintainer 对 intermediate 与 paper-final 数值差异、checkpoint 缺文件、官方 inference bug 或成功复现论文表格的明确补充说明。Issues 只能作为辅助证据；权重身份的主证据仍是官方 README、HF revision 和 state-dict index。

## 11. Reproduction Blockers

### Blockers for first public-checkpoint smoke inference

1. **Weights 未下载**：本阶段只审计 metadata/index，遵守不执行正式推理；下一阶段需人工授权下载约 16.75 GB shards。
2. **Heavy dependencies 未安装**：独立 `legion` env 已建，但 requirements/mmcv/flash-attn 尚未装。
3. **External CLIP required**：`openai/clip-vit-large-patch14-336` 不在 public shards。
4. **Stock shell import path**：缺 `PYTHONPATH=./`。
5. **Stock DDP setup**：`DistributedSampler(rank=-1)` 与 plain-python launcher 不兼容；需外部 wrapper 提供一致的单进程 distributed env/CLI rank，或另行批准最小修复。
6. **Folder hygiene**：必须是 flat image-only folder；没有格式过滤。
7. **Resource requirement**：模型约 16.75 GB tensor data，加 CLIP、generation KV/hidden states、SAM；需要后续单独做显存 preflight，不能在本阶段假设任意 GPU 可运行。

### Blockers for exact paper-table reproduction

1. exact final LE weights 未公开，公开的是 intermediate；
2. public Issues 没有 best checkpoint 或 table checkpoint 的可验证映射；
3. 当前阶段禁止 benchmark，亦没有建立论文表格与公开 intermediate 之间的指标等价关系。

这些 blocker 不等价于“必须重训才能做任何 localization”。对 public intermediate 的第一次 smoke inference，静态证据不要求重训。

## 12. Minimal Requirements for First Inference

Phase 5A-1 若获正式授权，最小前置项应为：

1. 继续固定 LEGION source `d21535dd...` 与 HF revision `f8c28b3...`；
2. 在独立 `legion` env 内按官方版本安装，禁止修改 R1 环境；
3. 安装/验证 mmcv v1.4.7 ops、torch 1.13.1+cu117、transformers 4.28.0、PEFT 0.5.0 等；
4. 下载并 hash-verify 2 个 public `.bin` shards 与 tokenizer/config；
5. 单独准备/缓存 `openai/clip-vit-large-patch14-336`；
6. 使用外部 launcher wrapper 设置 repo-root `PYTHONPATH` 与一致的单进程 distributed rank，不修改官方 core code；
7. 使用 1 张非封存、非 benchmark 的普通测试图片；
8. 只验证 model load、1 次 forward、`mask.png`/`explanation.json` 生成和尺寸，不计算指标；
9. 保留 stdout/stderr、GPU/依赖 inventory、文件 SHA-256 和输出 manifest；
10. 完成后 STOP，再请求是否进入任何正式数据评估。

## 13. Final Reproducibility Classification

**Classification: Case A — public intermediate LE checkpoint is a complete merged model suitable for a conditional first smoke inference.**

严格限定：

- Case A 判定针对 **当前公开 intermediate checkpoint 的可加载形态与 LE 调用链**；
- 不代表论文 localization/explanation 表格 exact final checkpoint 可复现；
- 不代表公开 checkpoint 的质量与 paper-final 相同；
- 不包括 detection；public index 无 `prediction_head`；
- 不代表 stock `infer.sh` 可以原样运行，必须先解决 launcher 层条件；
- 本阶段没有 forward，因此运行成功仍是 `UNTESTED`。

不归入 Case B：公开 checkpoint 已 merge，不需要再跑官方 merge。  
不归入 Case C：公开 checkpoint 没有缺 GLaMM LLM 或 SAM 参数；只把 CLIP 作为标准外部 tower。  
不归入 Case D：没有证据表明 public intermediate 做 localization 必须重训；只有 exact paper-final 无法由当前公开权重替代。

## 14. Next-Step Recommendation

建议下一阶段只做 `Phase 5A-1 — LEGION Public Intermediate LE Single-Image Smoke Test`，并设置硬门：

- 先审计磁盘/显存/driver 与 dependency resolution；
- 获得单独下载与依赖安装授权；
- 下载固定 HF revision 并校验 LFS SHA-256；
- 不 merge、不训练、不访问 SynthScars/LOKI/RichHF-18K/official1000；
- 用非封存单图验证 public checkpoint 的真实 load/forward/output；
- 不比较 R1，不计算 mIoU/F1，不改 evaluator；
- 如发现 checkpoint load mismatch、无 `[SEG]`、空 mask 或 launcher 协议偏差，立即 STOP 并报告，不静默 patch。

Phase 5A-0 到此完成并停止；未进入 Phase 5A-1。

## Evidence Files

- `artifacts/phase5a0_legion/repo_commit.txt`
- `artifacts/phase5a0_legion/repo_tree.txt`
- `artifacts/phase5a0_legion/hf_file_inventory.txt`
- `artifacts/phase5a0_legion/dependency_inventory.txt`
- `artifacts/phase5a0_legion/inference_callgraph.txt`
- `artifacts/phase5a0_legion/checkpoint_analysis.txt`

Primary public sources:

- Official repository: https://github.com/opendatalab/LEGION
- Official README at pinned commit: https://github.com/opendatalab/LEGION/blob/d21535dd45f6fea509337a83095966f0b86ac924/README.md
- Public checkpoint at pinned revision: https://huggingface.co/khr0516/legion_LE/tree/f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa
- HF model metadata API: https://huggingface.co/api/models/khr0516/legion_LE?blobs=true
- HF revision history API: https://huggingface.co/api/models/khr0516/legion_LE/commits/main
