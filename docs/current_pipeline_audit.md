# GLaMM AIGC 当前流水线审计（Phase 0）

审计日期：2026-08-09  
审计基线：`7bef9a2 Add image forensics training support`，并包含当前工作区中已经冻结的数据 manifest 产物  
审计范围：tokenizer、conversation、labels、collate、GLaMM forward/generate、classification、segmentation、NPR/SRM、SynthScars 与真实图数据冻结流程  
本轮行为：只读审计；未修改模型、dataset 或训练逻辑，也未启动训练。

## 1. 结论摘要

最关键结论如下。

1. **当前 `[CLS]` 是 generated token，不是 fixed forensic query。** 训练时它由 collate 插在 `ASSISTANT:` 后，属于 assistant target；推理时 `evaluate()` 强制它成为第一个新生成 token。
2. `[CLS]` 的训练 label **不是 `-100`**。本地 tokenizer/collate 实测其 `input_id=32007`、`label=32007`，因此 causal LM CE 必须预测它。
3. 分类头训练时读取 teacher forcing 序列中 `[CLS]` 位置的最后层 hidden state；本地 FullScope 配置为 `[N, L, 4096] -> [N, 4096] -> Linear(4096,2) -> [N,2]`。
4. 当前取证 manifest 使用的是大写 `[REAL]` / `[FAKE]`，而计划文本使用 `[Real]` / `[Fake]`。两者都没有被注册为 special token；在本地 FullScope tokenizer 中会分别拆成 4/4/3/4 个 subword。
5. **SynthScars/Real adapter 与冻结 split 尚未接入 `train.py`。** 目前没有取证 conversation grammar，没有 Real/Fake LM 标签监督，也没有实际的 unified Real/Fake GLaMM batch。
6. 因为 Real 数据尚未进入训练入口，不能说“Real segmentation loss 已按设计 skip”；准确表述是：**当前根本没有 Real 训练样本到达该 loss**。模型 loss 内部也没有按 authenticity 显式 gate segmentation。
7. GLaMM mask 分支实际使用的是 **`[SEG]` 前一个位置**的 causal hidden state，而不是 `[SEG]` token 自身位置的 hidden state；随后经过 `4096 -> 4096 -> 256` 投影并送入 SAM prompt/mask decoder。
8. 当前 NPR+SRM 是与 GLaMM 完全解耦的独立图像级分类器。默认输入 `[B,3,224,224]`，NPR/SRM 各自得到 `[B,512,28,28]`，池化为 `[B,512]`，相加后输出 `[B,1]` fake logit。它没有保留可直接供 GLaMM 融合的空间 feature API。
9. 冻结的内部数据池在排除 18 个与官方 SynthScars test 近重复的 train identity 后为 **11,046 Real + 11,046 Fake**；Human/Animal/Object/Scene 数量逐类完全一致，且内部 train/val/test 未发现 sample ID、文件名、Real SHA256 或 pHash 连通组跨 split 重叠。
10. Real/Fake 当前只有 manifest 层，不存在共同的 GLaMM preprocessing class，因此 **resize/crop/augmentation/normalization 的一致性尚未实现，也无法通过现有训练代码确认**。

Phase 0 的核心问题已经回答，因此可以进行人工确认；按照研发计划，本审计完成后停止，不进入 Phase 1。

## 2. 对 15 个审计问题的逐项回答

### 2.1 `[CLS]` 如何注册？

- 常量定义：`tools/utils.py:19`，`DEFAULT_CLS_TOKEN = "[CLS]"`。
- 训练入口：`train.py:129-158`。加载 tokenizer 后无条件调用 `tokenizer.add_tokens([DEFAULT_CLS_TOKEN], special_tokens=True)`，再用无额外 special token 的编码结果取得 `cls_token_idx`。
- demo：`app.py:41-55` 使用同一逻辑。
- GCG/region/referring-seg inference 入口也各自添加 `[CLS]`。
- `train.py:138-150` 只在 `not args.pretrained` 时添加 `<im_start>`、`<im_end>`、`<bbox>`、`<point>`、`[SEG]`、`<p>`、`</p>`；`[CLS]` 则无论 `pretrained` 与否都尝试添加。
- `train.py:236-237` 在模型准备阶段执行 `resize_token_embeddings(len(tokenizer))`。

本地 `checkpoints/GLaMM-FullScope` tokenizer 只读实测：加载时词表长度为 32,007，已有 `[SEG]=32004`；添加 `[CLS]` 后词表长度为 32,008，`[CLS]=32007`。

### 2.2 dataset 中 `[CLS]` 是 input token 还是 assistant target token？

原始 caption/region/segmentation dataset 不生成 `[CLS]`。`dataset/dataset.py:176-180` 在 `inference=False` 时把每个 conversation 中：

```text
<sep>ASSISTANT: 
```

替换成：

```text
<sep>ASSISTANT: [CLS] 
```

因此它位于 assistant answer 的第一个位置，是 **assistant target token**。它虽然自然也存在于完整 teacher-forcing `input_ids` 中，但不是用户输入/prefix 中固定存在的 query。

该替换应用于当前所有非 inference 的 caption、region、segmentation/GCG conversation，而不只取证任务。

### 2.3 `[CLS]` 对应 label 是否为 `-100`？

不是。

`dataset/dataset.py:227-253` 只 mask system/user/instruction 区域；assistant answer 保持为监督 label。本地对真实 `custom_collate_fn` 的轻量实测得到：

```text
... ASSISTANT: [CLS] [FAKE] malformed hand [SEG].</s>

position  token_id  label
47        32007     32007   # [CLS]
```

因此 `[CLS]` 的 label 不是 `IGNORE_INDEX=-100`。

### 2.4 训练时 `[CLS]` 是否需要 LM CE 预测？

是。

`model/llava/model/language_model/llava_llama.py:109-120` 使用标准 causal shift：`logits[..., :-1]` 预测 `labels[..., 1:]`。由于 `[CLS]` label 未 mask，它必须由 `[CLS]` 前一位置（即 assistant prefix 结尾）的 logits 预测，并进入 `output.loss`。`model/GLaMM.py:350-385` 再将其包含在 `ce_loss = output.loss * ce_loss_weight` 中。

### 2.5 推理时 classification head 的 hidden state 从哪里获得？

当前意图是从**生成出来的 `[CLS]` token 位置**获得：

1. `model/GLaMM.py:387-395` 的 `evaluate()` 默认设置 `force_cls_token=True`。
2. 它通过 `forced_decoder_ids=[[input_ids.shape[1], cls_token_idx]]` 强制第一个新 token 为 `[CLS]`。
3. 生成完成后，`model/GLaMM.py:408-409` 只在 prompt 之后搜索 `[CLS]`。
4. `_extract_cls_logits()`（`model/GLaMM.py:228-239`）从最后 hidden state 取该位置的向量，再送入 `classification_head`。

训练路径与推理路径的差异是：训练读取 teacher-forced `[CLS]` 位置，推理依赖 generate 先产生/强制 `[CLS]`，再试图读取 generation hidden state。

这里存在一个高优先级静态风险：Transformers 的 `generate(..., output_hidden_states=True)` 通常返回“按 generation step 嵌套的各层 hidden states”，而 `_get_last_hidden_state()` 只做一层 `[-1]`。`evaluate()` 随后直接访问 `.shape`。当前代码没有单元测试证明这一结构在项目锁定的 Transformers 版本中正确；进入 Phase 1 前必须用真实 checkpoint 做 generate smoke test。

### 2.6 当前 `[Real]/[Fake]` 如何生成和监督？

**当前没有进入 GLaMM 生成或监督。**

- `dataset/forensics/real_images.py:15-17` 定义 `REAL_VERDICT_TOKEN = "[REAL]"`。
- `dataset/forensics/synthscars.py:24-25` 定义 `FAKE_VERDICT_TOKEN = "[FAKE]"`。
- adapter 只把它们写入 manifest dict 的 `verdict_token` 字段；没有构造 GLaMM conversation，也没有注册到 `Hybrid*Dataset` registry。
- `train.py:55-75,288-338` 只支持现有 caption/region/segmentation dataset 名称。
- 全仓唯一能够向模型提供 `cls_label` 的通道是 11-field sample（`dataset/dataset.py:148-167`），但当前没有 dataset 将取证 adapter dict 转为这种训练 tuple。

本地 tokenizer 实测：

| 字符串 | subword 数 | token 片段 |
|---|---:|---|
| `[REAL]` | 4 | `▁[`, `RE`, `AL`, `]` |
| `[FAKE]` | 4 | `▁[`, `FA`, `KE`, `]` |
| `[Real]` | 3 | `▁[`, `Real`, `]` |
| `[Fake]` | 4 | `▁[`, `F`, `ake`, `]` |

因此当前 manifest 中的 verdict 并不是单一 label token，且大小写 grammar 尚未统一。

### 2.7 `[SEG]` 在 Real/Fake 中如何出现？

- 当前 Fake adapter 提供 `refs`、polygon 和 union mask，但不生成含 `[SEG]` 的 conversation。
- 当前 Real adapter 提供空 `refs` 与原尺寸全零 union mask，也不生成含 `[SEG]` 的 conversation。
- 所以在**当前取证数据**中，Real/Fake 都没有实际进入 LM target 的 `[SEG]`。
- 现有非取证 segmentation/GCG dataset 会在 answer 中显式构造 `[SEG]`，例如 `dataset/gcg_datasets/GranDf_gcg_ds.py:102-118`；这些 target 在 collate 时会变为 `[CLS] ... [SEG] ...`。

### 2.8 Real 当前是否计算 segmentation loss？

没有，但原因不是 conditional supervision 已实现，而是 Real 数据没有接入训练。

还需注意：

- `dataset/forensics/real_images.py:148-152` 当前把 Real 写成 `has_mask_label=True`，并在 decode 时返回原图大小的全零 mask。
- `model/GLaMM.py:350-385` 没有读取 authenticity 或 `has_mask_label`，也没有 Real/Fake gate；它只依据是否产生了 pred mask 以及 mask tensor 是否非空来累积 BCE/Dice。

因此未来如果直接把当前 Real zero mask 与 `[SEG]` 一起接入，它会被当作有效 segmentation supervision；这与计划“Real 不计算 segmentation loss”冲突。

### 2.9 `[SEG]` 如何得到 mask？

当前完整路径为：

```text
conversation answer 中出现 [SEG]
  -> input_ids 中定位 [SEG]
  -> _create_seg_token_mask(..., select_preceding=True)
  -> 选择 [SEG] 前一位置的 LLM hidden state [M,4096]
  -> text_hidden_fcs: Linear(4096,4096) + ReLU + Linear(4096,256)
  -> text embedding [M,256]
  -> SAM prompt_encoder(text_embeds=[M,1,256])
  -> sparse/dense prompt embeddings
  -> SAM mask_decoder(image embedding + positional encoding + prompts)
  -> low-resolution mask
  -> SAM postprocess_masks(resized input size, original size)
  -> 每张图的 pred mask [M,H,W]
```

关键代码：

- mask 对齐：`model/GLaMM.py:193-222`
- text projection：`model/GLaMM.py:74-80,310-325`
- SAM prompt/mask decoder：`model/GLaMM.py:327-344`
- mask loss：`model/GLaMM.py:350-385`
- multimodal image token 展开与 image patch label masking：`model/llava/llava_with_region_arch.py:83-246`

`select_preceding=True` 与 causal LM 的预测语义一致：前一位置的 hidden state 用来预测 `[SEG]`。但它意味着文档中若写“`[SEG]` hidden state”并不精确。

### 2.10 classification head 输入 shape 是什么？

本地 FullScope config：`hidden_size=4096`。

```text
last_hidden_state                 [N, L_expanded, 4096]
cls_token_mask                    [N, L_expanded]
cls_hidden_states                 [N, 4096]
classification_head Linear        4096 -> 2
cls_logits                        [N, 2]
```

其中 `N` 是 conversation 数，不一定等于图像数。一张图可以对应多个 conversation；`model/GLaMM.py:241-255` 按 `offset` 对有效 conversation logits 取均值，得到 image logits `[B,2]`。classification CE 则在 conversation 级 logits 上计算，并将 image label repeat 到各 conversation（`model/GLaMM.py:257-273`）。

当前所有标准 GLaMM dataset 都不给 `cls_labels`，所以 classification head 虽然被设为 trainable（`train.py:269-275`），`cls_loss` 实际恒为可微的 0。

### 2.11 NPR/SRM 输入、输出、checkpoint、冻结状态

#### 输入与预处理

`npr_expert/transforms.py:29-43`：

```text
PIL RGB
 -> Resize(256,256)
 -> train: RandomCrop(224) / eval: CenterCrop(224)
 -> train only: RandomHorizontalFlip
 -> ToTensor [0,1]
 -> ImageNet mean/std normalize
 -> [B,3,224,224]
```

#### 实测 shape

对 `OfficialNPRSRM().eval()` 输入 `[2,3,224,224]` 的 CPU hook 实测：

| 节点 | shape |
|---|---|
| input | `[2,3,224,224]` |
| NPR conv1 | `[2,64,112,112]` |
| NPR maxpool | `[2,64,56,56]` |
| NPR layer1 | `[2,256,56,56]` |
| NPR layer2 | `[2,512,28,28]` |
| NPR avgpool | `[2,512,1,1]` |
| SRM stem | `[2,512,28,28]` |
| SRM pool | `[2,512,1,1]` |
| fused global feature | `[2,512]` |
| `fc1` / output | `[2,1]` |

实现位于 `npr_expert/official_npr_srm.py:47-115,178-240`。NPR 使用 `image - nearest-down/up(image)` 残差；SRM 使用固定 5x5 高通核生成 9 通道残差。两个全局特征在 `features + srm_gate * srm_features` 后立即融合。

这意味着当前接口不能分别返回 `g_npr`、`g_srm`，也不再返回旧版 `[B,512,16,16]` 空间特征；V1 late fusion 以后需要新增 feature-return API，但本轮未修改。

#### checkpoint

当前本地可用的主要 checkpoint：

- `checkpoints_stage1/official_npr_srm_20260802_062826/best.pth`：`official_npr_srm_v1`，epoch 28，best val AP 0.98889569。
- `checkpoints_stage1/official_npr_srm_focal_warmstart_20260802_113842/best.pth`：`official_npr_srm_focal_v2`，epoch 8，best val AP 0.99195453；以第一项为 warm start，并使用缓存/冻结 FOCAL feature。
- 另有 no-FOCAL warm-start control、非 warm-start FOCAL 和 pure frozen-FOCAL-linear 实验。当前 GLaMM config 没有指定其中任何一个为正式融合 checkpoint。
- FOCAL 作者权重默认路径：`/data/yz/myLISA_storage/checkpoints/FOCAL/FOCAL_ViT_weights.pth`（`npr_expert/train.py:23`）。

#### 冻结状态

- `NPRExpert` / `OfficialNPRSRM` 默认是可训练的。
- 只有显式调用 `.freeze()` 或 `from_checkpoint(..., freeze=True)` 才冻结，见 `npr_expert/model.py:36-59`、`npr_expert/official_npr_srm.py:243-251`。
- FOCAL extractor 在构造时永久 `requires_grad_(False)` 且强制 eval，见 `npr_expert/focal_extractor.py:26-48,71-92`。
- NPR/SRM 当前完全没有挂到 GLaMM module tree，因此在 GLaMM 训练中谈不上 frozen/trainable；它们根本不参与 forward。

### 2.12 当前采样逻辑

数据冻结流程如下：

```text
SynthScars 官方 annotation-centric JSON
 -> 按 image stem 分组，同 identity 多 annotation/多文件 variant 合并
 -> 每图保留所有 refs，生成一张 union evidence mask
 -> 11,064 个 train image identity

真实源原始 pinned manifests
 -> OpenImages 8,000 + PASS 6,000 + COCO 4,000 + FFHQ 2,500 + iNaturalist 1,000
 -> SHA256 精确去重排除 3 张，剩 21,497
 -> PASS 6,000 张全部用 GPT 内容分类；46 ambiguous 不参与匹配
 -> 使用冻结的 SynthScars CLIP content prediction 作为四类目标数
 -> 每类按 sha256(seed:category:sample_id) 稳定排序截取
 -> 选出 11,064 Real，与 11,064 Fake 四类逐类相等
 -> 排除 18 个与官方 SynthScars test 近重复的 train Fake
 -> 同时排除同类别 18 个 Real，保留平衡
 -> 内部池 11,046 Real + 11,046 Fake
 -> 按类别分别做 8:1:1；pHash 连通组整体进入同一 split
```

关键代码：

- SynthScars 分组与 mask：`dataset/forensics/synthscars.py:161-290`
- Real source mapping：`dataset/forensics/real_images.py:65-153,180-261`
- PASS label merge 与 deterministic matching：`scripts/data/build_content_matched_real.py:84-165`
- pHash group-aware split：`scripts/data/split_unified_forensics.py:100-199`

### 2.13 Real/Fake 与四类实际数量和比例

#### 冻结内部池（排除官方 test leakage 后）

| domain | total | Human | Animal | Object | Scene |
|---|---:|---:|---:|---:|---:|
| Fake / SynthScars | 11,046 | 5,815 (52.643%) | 1,525 (13.806%) | 1,939 (17.554%) | 1,767 (15.997%) |
| Real total | 11,046 | 5,815 (52.643%) | 1,525 (13.806%) | 1,939 (17.554%) | 1,767 (15.997%) |

#### Real source 分布

| source | total (% Real) | Human | Animal | Object | Scene |
|---|---:|---:|---:|---:|---:|
| OpenImagesV7 | 4,267 (38.629%) | 2,626 | 477 | 650 | 514 |
| COCO2017 | 2,263 (20.487%) | 1,510 | 211 | 323 | 219 |
| PASS | 2,397 (21.700%) | 44 | 353 | 966 | 1,034 |
| FFHQ | 1,635 (14.802%) | 1,635 | 0 | 0 | 0 |
| iNaturalist | 484 (4.382%) | 0 | 484 | 0 | 0 |

#### 内部 split

| split | Real | Fake | total |
|---|---:|---:|---:|
| train | 8,836 | 8,836 | 17,672 |
| val | 1,106 | 1,106 | 2,212 |
| test | 1,104 | 1,104 | 2,208 |

各 split 内 Real/Fake 的四类计数也逐类完全相等。独立评估资源另有官方 SynthScars test 1,000 Fake，以及去除 2 个不可解码 TIFF 后的 RAISE held-out 998 Real。

统计来源：

- `outputs/data_audits/content_matched_real_v1/summary.json`
- `outputs/data_audits/unified_forensics_split_v1/summary.json`
- 本轮对 `train/val/test_{real,fake}.jsonl` 的只读重新计数。

### 2.14 resize/crop/augmentation/normalization 是否完全一致？

**当前答案：尚未实现共同链，不能确认一致。**

取证 adapter 目前只负责：

- SynthScars：PIL 读取尺寸、polygon rasterization、返回 numpy mask。
- Real：PIL 读取尺寸、返回原尺寸 zero mask。

它们都没有创建 GLaMM 所需的 `global_enc_image` 和 `grounding_enc_image`。冻结 manifest 也未接入 `Hybrid*Dataset`。

现有 GLaMM segmentation dataset 的标准双路处理可作为未来统一 wrapper 的基准：

```text
同一 RGB image
  -> CLIPImageProcessor.preprocess -> global encoder tensor
  -> ResizeLongestSide(image_size=1024)
     -> subtract [123.675,116.28,103.53]
     -> divide [58.395,57.12,57.375]
     -> right/bottom pad to 1024x1024
     -> grounding encoder tensor
```

例见 `dataset/gcg_datasets/GranDf_gcg_ds.py:96-100,139-157`。这些路径没有通用 image augmentation；dataset 层的随机性主要来自样本、问题、类别/ref 选择。

NPR/SRM 使用另一条 `Resize(256)->Crop(224)->ImageNetNormalize` 链，训练时还带 random crop/flip。这条 expert preprocessing 不能与 GLaMM preprocessing 混为一谈，但同一 expert transform 对 Real/Fake 必须保持一致。

### 2.15 是否改变训练逻辑？

没有。本轮只新增此审计文件并执行只读统计/轻量 CPU shape 与 tokenizer/collate 验证。

## 3. 当前 token flow

### 3.1 训练

```text
DEFAULT_CLS_TOKEN = "[CLS]"
  -> tokenizer.add_tokens(..., special_tokens=True)
  -> resize_token_embeddings

dataset 原始 conversation
  USER: <image> question
  ASSISTANT: answer [SEG]
  -> custom_collate_fn 对所有 non-inference conversation 插入 [CLS]
  USER: <im_start><image><im_end> question
  ASSISTANT: [CLS] answer [SEG]
  -> tokenizer_image_token (<image> -> IMAGE_TOKEN_INDEX=-200)
  -> input_ids
  -> labels clone
  -> system/user/instruction labels = -100
  -> assistant labels保留；[CLS] label保留
  -> multimodal prepare: image token 展开为 576 patch embeddings，patch labels=-100
  -> Llama teacher forcing
       LM head: 预测 [CLS] 及完整 assistant answer
       CLS head: 读取 [CLS] 位置 h_cls
       SEG branch: 读取每个 [SEG] 前一位置 h_seg_predictor
```

取证 verdict 当前停在 manifest：

```text
manifest.verdict_token = [REAL]/[FAKE]
  -X-> conversation
  -X-> tokenizer registration
  -X-> input_ids/labels
  -X-> LM supervision
```

### 3.2 推理

```text
USER prompt + empty ASSISTANT prefix
  -> model.evaluate(force_cls_token=True)
  -> generate 强制第一个新 token 为 [CLS]
  -> 后续文本自由生成
  -> 从 prompt 后生成序列搜索 [CLS]
  -> h_generated_cls -> Linear(4096,2)
  -> 从生成序列搜索 [SEG]
  -> 每个 [SEG] 前一生成位置 hidden -> SAM mask
```

当前 LM 没有被本仓库的 unified data 教成先生成 `[REAL]/[FAKE]`。

## 4. 当前 dataset flow

### 4.1 实际 `train.py` flow

```text
HybridCapDataset / HybridRegDataset / HybridSegDataset
  -> 各 dataset 随机抽样
  -> 10-field legacy tuple（无 cls_label）
  -> custom_collate_fn
  -> [CLS] 插入、tokenize、label mask、offset
  -> GLaMM
```

`HybridDatasetBase.__getitem__()` 使用 `np.random.choice` 按配置权重选择子 dataset，但无论收到哪个 `idx` 都调用 `selected_dataset[0]`；子 dataset 内部通常再次随机采样（`dataset/dataset.py:71-78`）。

### 4.2 已冻结但未消费的取证 flow

```text
SynthScarsAdapter / UnifiedRealImageAdapter
  -> dict manifest records
  -> content match + split scripts
  -> outputs/data_audits/unified_forensics_split_v1/*.jsonl
  -> [缺失] unified GLaMM dataset wrapper
  -> [缺失] common image preprocessing
  -> [缺失] Real/Fake conversation grammar
  -> [缺失] 11-field training tuple / cls_label
```

## 5. 当前 loss flow

```text
L_total = L_text + L_mask + L_cls

L_text = ce_loss_weight * causal LM CE
  当前包含 [CLS] 与现有 assistant answer/[SEG] 的预测
  当前不包含取证 [REAL]/[FAKE]，因为取证 conversation 未接入

L_mask = bce_loss_weight * BCEWithLogits + dice_loss_weight * Dice
  仅按 pred mask / gt mask 数量循环
  无 authenticity gate
  无 has_mask_label gate

L_cls = cls_loss_weight * CrossEntropy(cls_logits, cls_labels)
  只监督 cls_label >= 0 且确实找到 [CLS] 的 conversation
  当前标准 dataset 的 cls_labels=None，因此为 0

L_consistency = 不存在
L_artifact = 不存在
```

默认权重来自 `train.py:92-95`：`ce=1.0`、`dice=0.5`、`bce=2.0`、`cls=1.0`。

## 6. 数据重复与 source shortcut 审计

对冻结内部 split 的只读复核：

- train/val/test 两两间 sample ID 重复：Real 0，Fake 0。
- train/val/test 两两间 `image_name` 重复：Real 0，Fake 0。
- Real `content_sha256` 跨 split 重复：0。
- pHash connected component 跨 split：Real 0，Fake 0。
- Fake 有 14 个多成员 pHash group，均整体落在单一 split。
- 初始 pHash 审计发现 23 个候选 pair，其中 real candidate 内 3、SynthScars 内 14、RAISE 内 6；2 个 RAISE TIFF 不可解码。
- 官方 SynthScars test 对 train fake 找到 18 个阈值内近重复 identity；冻结内部池已排除这 18 Fake 及 18 个类别匹配 Real，修复后 cross leakage 为 0。

仍然存在明显的 dataset-source shortcut 风险：

- Fake 全部来自 SynthScars；Real 来自五个不同真实源。
- Real 原始源统计中有大量 JPEG、2,500 PNG（FFHQ）和 held-out TIFF；Fake 的尺寸/格式分布与此不同。
- FFHQ 只贡献 Human，iNaturalist 只贡献 Animal；source 与 content category 有强相关。
- Real 固定短解释、Fake 长解释且带 localization，会形成输出 grammar shortcut。
- 内容匹配只能控制 Human/Animal/Object/Scene，不能控制 compression、resolution、aspect ratio、watermark、sensor/noise 和生成器风格。

因此 source statistics/source classifier 实验仍然必要，但不属于本轮模型修改。

## 7. 潜在 bug、label leakage 与阻塞项

按优先级列出。

### P0：进入 Phase 1 前必须处理/验证

1. **Unified data 未接入。** 当前模型中的 classification head 与取证 adapter 是两条断开的代码路径。
2. **verdict 大小写与 token 粒度不统一。** manifest 为 `[REAL]/[FAKE]`，计划为 `[Real]/[Fake]`，且均非 special token。
3. **Real mask flag 与训练目标冲突。** Real manifest 写 `has_mask_label=True` 并返回 zero mask，而模型没有 conditional seg gate。
4. **无共同 Real/Fake preprocessing。** 目前不能满足“相同 resize/crop/augmentation/normalization”。
5. **generate hidden-state 结构未验证。** `evaluate()` 对 generation hidden state 的解包方式可能与 Transformers 返回结构不兼容；必须真实 checkpoint smoke test。
6. **当前 classification head 未被任何实际 dataset 监督。** 标准 dataset 都返回 10 fields，`cls_labels=None`。

### P1：高风险实现问题

1. `custom_collate_fn` 用 batch 第一个元素决定是否 stack/丢弃 `grounding_enc_images`、`bboxes`、`label_list` 等（`dataset/dataset.py:206-217`）。若未来 mixed Real/Fake sample 的这些字段一个为 `None`、一个为 tensor，结果依赖 batch 顺序。
2. `masks=None` 在 collate 中被改成空 Python list，但返回处又以 `masks_list[0] is None` 判断；因此“无 mask”语义不一致。
3. image expansion 在 `model/GLaMM.py:203-204` 和 collate truncation 中硬编码 `575`，只对当前 24x24 CLIP patch 数成立；更换 vision tower/feature layout 会静默错位。
4. `_compute_loss_components()` 在 gt mask 多于 pred mask 时直接截断 gt（`model/GLaMM.py:361-367`），可能把 missing `[SEG]` 变成未报告的监督丢失。
5. `_inference_path()` 将各样本的 `hidden_states` 直接 append 后 `torch.cat`（`model/GLaMM.py:275-290`）；训练模式和 eval 模式下 `hidden_states` 类型不同，缺乏覆盖测试。
6. 当前 `[CLS]` 被插入所有 caption/region/segmentation target；即使没有 classification label，LM 仍被训练为每个回答先生成 `[CLS]`。这改变了原始 GLaMM grammar，并可能影响旧 checkpoint 的生成兼容性。

### P2：研究有效性与 shortcut 风险

1. Generated CLS 的分类表示来自 target/generated token 生命周期，训练与部署的数据条件不同。
2. Real/Fake 输出长度、固定句式、是否有 `[SEG]` 都可能成为 grammar shortcut。
3. 仅按内容类别平衡不能消除 source/format/compression shortcut。
4. NPR/SRM 训练器默认会评估 LOKI，历史日志显示结果已被观察；后续论文实验必须记录此前已看过 LOKI，且不得继续用其调参/early stopping。
5. SynthScars manifest 当前没有 `artifact_type` 字段，Physics/Distortion/Structure 也没有从官方 annotation schema 映射出来。

## 8. Generated CLS 与 Fixed Forensic Query 的代码级比较

本节只比较，不提前选择。

| 维度 | Generated CLS（当前） | Fixed Forensic Query（待实现） |
|---|---|---|
| token 所在位置 | assistant target 首 token | assistant prefix/model input 中固定 token |
| `[CLS]/[FORENSIC]` label | 当前为 token id，参与 LM CE | 必须为 `-100`，不参与 LM CE |
| 训练表示 | teacher forcing 下 `[CLS]` 位置 hidden | 固定 query 位置 hidden |
| 推理表示 | 必须先生成/强制 `[CLS]`，再解析 generate hidden state | forward prompt 时即可取得，不依赖先生成 token |
| 与文本 verdict 的时序 | h_cls 在 `[REAL]/[FAKE]` 之前，看不到未来 verdict token | 同样可放在 verdict 前，作为稳定 query |
| 现有改动量 | 已存在；主要需要修 generate 与取证数据接入 | 需要新增 token strategy、prefix 构造、label mask、hidden index、generate prefix 处理 |
| 训练/推理一致性 | teacher forcing vs autoregressive hidden-state plumbing 不同 | query 始终固定，通常更容易保持一致 |
| 对旧 GLaMM grammar 的影响 | LM 被要求所有回答先生成 `[CLS]` | 可只在取证配置/任务中加入固定 query |
| 主要风险 | generation failure/强制 token、hidden 对齐、额外 LM 目标 | prefix 边界 mask 错误、query 重复插入、save/load token 配置 |
| 预计代码成本 | 低到中：接通数据并补 generate tests | 中：collate/grammar/model/evaluate/config/tests 多点改造 |

### Generated CLS 最小改造面

- 保留 `custom_collate_fn` 插入逻辑，但限制到取证策略/任务。
- 接通 unified dataset 的 11-field tuple 与 `[REAL]/[FAKE] explanation [SEG]` grammar。
- 明确并测试 `[CLS]` label 参与 CE。
- 修复/验证 `evaluate()` 的 generation hidden-state 对齐。
- 增加 tokenizer save/load 与强制首 token generate test。

### Fixed Forensic Query 最小改造面

- 新增独立 `[FORENSIC]`（或明确复用 `[CLS]`）special token 与 config token id。
- 在 assistant prefix/model input 边界固定插入，避免成为待预测的第一个 target。
- `_process_conversation()` 对 query label 显式设 `-100`，并用 token-level index 而不是脆弱字符串替换验证。
- classification head 从固定 query 的最后层 hidden state读取。
- `evaluate()` 直接在 prompt/prefix forward 中取得该表示，再生成 `[REAL]/[FAKE] + explanation + optional [SEG]`。
- 增加两种 token strategy 的 input_ids/labels、forward、generate、tokenizer save/load tests。

两种方案都需要先解决共同问题：unified dataset wrapper、verdict token 规范、Real conditional segmentation、共同 preprocessing、mixed collate 语义与 evaluation 输出。

## 9. Phase 0 验收判断

验收问题：当前 `[CLS]` 是生成 token 还是固定输入 token？

> **答案：当前是 Generated CLS。** 训练时它是 assistant target，label 不为 `-100`，参与 LM CE；推理时必须先由 generate 强制/产生，再从其 generation hidden state送入 classification head。它不是 fixed forensic query。

本轮到此停止。按照 `docs/CODEX_NEXT_PLAN_GLAMM_AIGC.md`，下一步应先由人工确认此审计，再决定是否执行 Phase 1/第二条指令；本轮不继续实现。
