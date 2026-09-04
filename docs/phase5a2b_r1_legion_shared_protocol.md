# Phase 5A-2B — Freeze R1 vs LEGION Shared Localization Protocol

## 0. 冻结结论

Phase 5A-2 的公开 intermediate LEGION reference comparison 只使用一个主测试集：**official SynthScars test / official1000，Fake-only，N=1,000，original RGB**。

冻结条件为：

```text
R1:       existing G0 (canonical image-only free generation; no redesign/retraining)
LEGION:   official L-FREE (official prompt → free generation → [SEG] → SAM)
R1 Phrase:     N/A
R1 TF-PHRASE:  N/A
```

共同、可直接执行的 ordered sample manifest 已存在：

```text
outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl
SHA256: fff3c3839e54d831955f8055501882f27a408c885e0524c2d7bf5d3edbe8c700
rows: 1,000; unique sample_id: 1,000
ordered sample-ID SHA256 (UTF-8, newline after every ID):
34b136b24365aedf4878a86b3f12856a974991d3615f1c603213792c99a2e114
```

该 manifest 的每行均为 `forensics_domain=fake`、`class_label=1`、`dataset_split=test`，有非空 `refs` 和 mask label，且 1,000 个绝对 `image_path` 当前均存在。未来 Phase 5A-2 只可按此文件的**原顺序**读取图像；不能重新抽样、重排、增删、改用 corruption 或混入其他数据集。

## 1. R1 现有正式 localization 测试盘点

| Population / split | R1 条件 | 实际 N | target | 在本次 LEGION public comparison 中 |
|---|---|---:|---|---|
| DEV internal validation Fake | G0 / Phrase / TF | 1,106 | SynthScars official polygon-derived per-image union | 不使用：开发/selector 集 |
| Internal test Fake | G0 / Phrase / TF | 1,104 | 同上 | 不使用：封存集，且不是 Phase 5A 既定共同集 |
| Internal test Fake corruption | G0, JPEG70/JPEG80/Gaussian5/Gaussian10 | 1,104/condition | 同上 | 不使用：robustness 条件，不是 original shared protocol |
| official SynthScars test / official1000 Fake | G0 / Phrase / TF | 1,000 | 同上 | **使用 original G0 only** |
| official1000 corruption | G0, JPEG70/JPEG80/Gaussian5/Gaussian10 | 1,000/condition | 同上 | 不使用：不是 original shared protocol |
| LOKI | 仅 G1 | 229 | filled bbox per-image union | 不使用：没有 R1 G0，且 target 类型不同 |
| AIGI-test | 仅 G1 | 861 | released pixel-mask union | 不使用：没有 R1 G0，且不是 SynthScars target |

证据：当前 P1/R1 baseline 索引将 official1000 G0 明确列为 SynthScars official final localization，N=1,000；同时规定 G0 是 deployable primary，Phrase/TF 是 language/oracle diagnostics（`docs/p1_reusable_evaluation_baselines.md:24-26, 134-154, 175-211, 242-272`）。Phase 5A-0 的既定路线也明确此项目后续拥有共同 official1000，而不是要自动扩展至其他 benchmark。

因此本协议不访问或计划任何未列入主集的数据集；internal test、LOKI、AIGI 和所有 corruption 均不在下阶段执行范围。

## 2. 精确集合与历史 R1 G0 的核验

比较了以下只读 machine-readable records：

1. shared manifest：`outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl`；
2. historical P1 G0 prediction：`outputs/phase3a_phrase_grounding/evaluation/official1000/G0/predictions.jsonl`，SHA256 `709d74dd322ca491969f2b295797ef3f845bfd52b429cd9c652e0fc57b4b10ad`；
3. R1 G0 scored prediction record：`outputs/p1_r1_reusable_matrix/jobs/official_g0.json`，SHA256 `f6b790f3d1a8d09055071ae2c0dc195178d3aac3d7a6cfd4fd2f430de3885d03`。

核验结果：

| Check | Result |
|---|---|
| manifest rows / unique IDs | `1000 / 1000` |
| P1 G0 rows / unique IDs | `1000 / 1000` |
| R1 G0 records / unique IDs | `1000 / 1000` |
| manifest = P1 ordered ID list | exact true |
| manifest = R1 ordered ID list | exact true |
| 三者集合相等 | exact true |
| 1,000 image paths available | `1000 / 1000` |
| R1 checkpoint | `outputs/phase4hd/r1/selected_checkpoint.pt`，SHA256 `9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5` |

这冻结的是 image identity，而不是只冻结数量；LEGION 的下一阶段输入清单就是上述 manifest 的 `sample_id`、`image_path` 和 `refs`，按原行序。

## 3. 共同 evaluator v1

### 3.1 GT target construction

对每张 official1000 图像，在其 original image resolution `(H,W)`：

```text
GT = OR over every ref in row.refs
     [ OR over every polygon in ref.polygons ]
```

每个 polygon 先按 `ref.source_image_size → (H,W)` 分别缩放 x/y，再用 COCO polygon rasterization 生成 binary mask；所有 polygon/ref 按 bool OR 合并。GT 必须非空。

固定代码语义：

- `dataset/forensics/synthscars.py:147-158,277-290`：polygon rasterization 与按目标尺寸缩放；
- `dataset/forensics/unified.py:136-146`：对全部 `refs` 取 per-image union；
- 禁止用 LEGION training validation 的 per-reference `zip(gt,pred)`、LOKI bbox 或 RichHF target 替代。

### 3.2 Prediction normalization and alignment

所有模型输出首先必须是 original-resolution logit masks `[K,H,W]` 或一个 binary union `[H,W]`。共同语义为：

```text
assert predicted H,W == GT H,W == original image H,W
per-mask binary_i = (logit_i > 0)
prediction = OR_i(binary_i)
```

这与在 union 前取 `amax(logits) > 0` 等价。不得在 GT 尺寸之外做 metric，不得对 final binary mask 做额外 resize、形态学后处理、连通域筛选或 threshold tuning。

- LEGION L-FREE：官方 `model.evaluate` 后的 SAM `postprocess_masks` 已恢复 original size；按 `scripts/loc_exp/infer.py:162-180` 固定 `>0` 后 `torch.any(..., dim=0)`。
- R1 G0：现有 code 在 original geometry 上 `inverse_sam_logits`，并用 `>0` metric（`tools/phase4c_b.py:107-121`）；历史 G0 evaluator 的多-mask 规则也为 logits `amax` 后 `>0`（`eval/forensics.py:64-81`）。

### 3.3 Empty/no-SEG/failure policy

对本 Fake-only、有非空 GT 的 1,000 张图，以下情况一律生成 `(H,W)` 全零 prediction，且不重试、不换 prompt、不调整 generation：

- 没有生成 `[SEG]`；
- `[SEG]` 存在但模型没有返回 mask；
- 返回 zero masks；
- 返回 mask 但无法验证 original-resolution shape。

最后一类在下一阶段应记录为 `INVALID_SHAPE` 并按空预测计分，而不是静默 resize；它是 protocol failure，不是可调参信号。全零预测的 `TP=0, FP=0, FN=|GT|`，所以 per-image FG IoU/F1 均为 0。

历史 R1 G0 已符合该 policy：1,000 张中 968 张生成且仅生成一个 `[SEG]`/mask，32 张没有 `[SEG]`，都已经以空预测、IoU/F1=0 计入。R1 G0 不需要多-mask 合并的特殊分支，因为其有效轨迹恰有一个 `[SEG]`；单 mask union 与共同规则完全一致。

### 3.4 Per-image metrics and dataset aggregation

每张图计算：

```text
TP = |prediction ∩ GT|
FP = |prediction \ GT|
FN = |GT \ prediction|
FG IoU = TP / (TP + FP + FN)
FG F1  = 2TP / (2TP + FP + FN)
```

冻结报告项目：

1. **primary**：mean per-image FG IoU；
2. median per-image FG IoU；
3. mean per-image FG F1；
4. global FG IoU：`sum(TP) / sum(TP+FP+FN)`；
5. global FG F1：`2sum(TP) / (2sum(TP)+sum(FP)+sum(FN))`；
6. `N`、valid `[SEG]` count、no-`[SEG]`/empty/invalid-shape counts；
7. paired R1–LEGION per-image FG IoU/F1 delta、bootstrap CI、W/T/L、Wilcoxon（只在两者都完成此 exact manifest 后计算）。

不报告或不比较：README paper mIoU/F1、fg/bg mIoU、category breakdown、per-reference IoU、LOKI bbox metrics。原因是它们不都能由已冻结的 R1 G0 per-sample record 和 shared target 无歧义重建，或 target/aggregation 不相同。

固定 metric implementation 对应 `eval/forensics.py:64-106` 与 `tools/phase4c_b.py:113-121`。所有比例在分母为零时的定义虽保留原实现，但本 manifest 的 GT 非空，实际不会触发该边界。

## 4. 固定的模型侧协议

### R1 G0

- checkpoint：上述 selected R1 checkpoint，禁止重训、选模或改模型；
- prompt：历史 `unified_forensics_v1`，SHA256 `32f0856f718fb3e19f34b552892636fb6c2613adbae4afb8826dea89f670654d`；
- generation mode：`unified_fake_generate`；
- 图像外的 GT input：无 authenticity、phrase 或 explanation；
- generation/metric threshold：mask logit `>0`；
- classification gate：关闭；
- historical P1 trajectory 为 exact replay；R1 仅在该冻结 trajectory 的 SAM decoding 路径产生 R1 mask。

### LEGION public intermediate L-FREE

- source commit：`d21535dd45f6fea509337a83095966f0b86ac924`；
- checkpoint：`khr0516/legion_LE` revision `f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa`；
- 固定官方 L-FREE prompt（`scripts/loc_exp/infer.py:144-145`）；
- free generation，官方 `max_tokens_new=512`、`num_beams=1`；
- generated `[SEG]` hidden state → official SAM → original resolution；
- each mask logit `>0`，multiple masks → pixelwise union；
- 不提供 GT authenticity、GT phrase、GT explanation 或 GT mask 给模型；
- 不构造 Phrase 或 TF。

R1 与 LEGION 的 user prompt 文本不相同。这是保留各自 canonical/official deployment protocol 的条件级比较，而不是强制替换 prompt 的新实验；prompt 文本差异必须随结果披露。

## 5. 历史 R1 G0 是否可直接复用

**结论：可以直接复用，不需要重跑 R1 evaluation 或 inference。**

判定依据：

| Frozen field | Historical R1 official_g0 status | Result |
|---|---|---|
| image ID set and order | 与 shared manifest exact equal | pass |
| GT construction | official SynthScars all-reference polygon union | pass |
| original-resolution evaluation | R1 inverse SAM geometry → original `(H,W)` | pass |
| threshold | logit `>0` | pass |
| multi-mask union | valid R1 trajectories each have exactly one `[SEG]`; singleton equals union | pass |
| no-[SEG] / empty policy | 32/1,000 as zero prediction, IoU/F1=0 | pass |
| per-image and global aggregation | stored TP/FP/FN plus FG IoU/F1; same formulas | pass |
| prompt/input condition | canonical free-generation G0, no GT context | pass |

可复用的 R1 artifact 是 `outputs/p1_r1_reusable_matrix/jobs/official_g0.json`：它保存 1,000 条按 manifest 顺序的 per-sample `sample_id, foreground_iou, foreground_f1, tp, fp, fn, valid_q_seg` 及其冻结汇总指标。它**没有**保存 R1 的 raw logits/binary mask；这不阻止本次严格数值/paired comparison，但若未来另行授权要求 R1–LEGION spatial panel、mask-level审计或不同 metric，则必须仅为那个新增目标重跑 R1 mask decoding，不能倒推现有 masks。

当前可复用 R1 G0 baseline（仅作后续 paired reference，不与 README 数字混用）：N=1,000，mean FG IoU `0.286588`，median FG IoU `0.232139`，mean FG F1 `0.393974`，global FG IoU `0.267042`，global FG F1 `0.421521`，`valid_q_seg=968/1000`。

## 6. 下一阶段执行清单

| Dataset | Shared samples | R1 G0 prediction available | LEGION needed | Metrics |
|---|---|---|---|---|
| official SynthScars test / official1000, original RGB | `test_combined.jsonl`; N=1,000; ordered-ID SHA256 `34b136…e114` | Yes — `jobs/official_g0.json`, per-sample scoring records and frozen summary; raw masks not retained | Run official L-FREE exactly once per manifest row; save per-image explanation, SEG count, original-size union binary mask and TP/FP/FN record | mean/median FG IoU, mean FG F1, global FG IoU/F1, validity counts; paired statistics |

Phase 5A-2B 完成后，下一阶段可直接读取的唯一 manifest 为：

```text
/home/yz/groundingLMM_official/outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl
```

启动前必须先复核其 file SHA256、row count、unique IDs、ordered-ID SHA256、所有 `image_path` availability，以及 R1 artifact/LEGION source/checkpoint revisions。任一不一致即 STOP，不改 sample set 或协议补救。

## 7. 停止点

本阶段仅完成静态协议冻结和已有 artifact 核验：未启动 LEGION 批量 inference、未访问 internal test/LOKI/AIGI、未跑 R1、未运行 Phrase/TF、未训练或修改模型。

## 8. Protocol amendment v2 — authorized robustness extension

用户随后正式授权在本协议上执行两个实验：original shared comparison，以及同一 official1000 的 localization robustness comparison。原 §0 的 “original RGB only” 限制据此仅保留为主结果定义，不再排除下列**预先存在且已用于 R1**的四个 deterministic corruption 条件：`jpeg70`、`jpeg80`、`gaussian5`、`gaussian10`。

每个 condition 都必须使用 §0 的同一 1,000 条 ordered manifest、同一 GT、同一 full-N empty/no-`[SEG]` failure policy 与同一 evaluator。corruption 在模型预处理前、RGB `uint8` 像素上施加：JPEG 使用 quality 70/80、subsampling 2、`optimize=False`；Gaussian 使用 sigma 5/10，seed 为 `sha256("3407:<sample_id>:<condition>")` 前八字节。R1 historical corruption RGB SHA256 必须逐图匹配。

每个 condition 同时报告：

1. full-N=1,000 end-to-end primary metrics（无 `[SEG]`/空 mask 计空预测）；
2. model-specific `[SEG]` coverage 与 failure counts；
3. 仅当两个模型都完成后，从固定 `both_valid_seg`（双方均至少一个 `[SEG]` 且 mask original-size valid）子集得到的 conditional mask-quality diagnostic。该子集由模型输出决定，**不能作为 primary、不能替代 full-N paired conclusion，也不能跨条件直接比较其 N**。

R1 Phrase/TF-PHRASE 继续为 LEGION comparison 的 `N/A`；没有构造 known-fake 或 teacher-forced LEGION 分支。

## 9. Protocol amendment v3 — authorized LOKI localization supplement

用户随后明确授权在 official1000 original + robustness matrix 完成后自动接续 LOKI localization。该补充严格限制为现有 LEGION-compatible LOKI localization manifest 的 229 张 fully-synthetic images：

```text
manifest: datasets/LOKI/legion_localization/manifest.jsonl
rows / unique IDs: 229 / 229
manifest SHA256: c9a29854717867419c2386be75bf3cf4c0366f43230e62b653594ca908628bc8
ordered-ID SHA256: 0b9b182f3b1b2d9b747858250360360018d6f56e4176e39bf20488837e609503
GT: union of filled regional xywh bounding boxes at original 512x512 resolution
```

LEGION 仍使用官方 L-FREE：官方 prompt、image-only free generation、`[SEG]`→SAM、logit `>0`、multiple masks union、no-`[SEG]`/empty prediction 计全零。不得把 LOKI region descriptions、global description、bbox 或 GT authenticity 输入 LEGION。

现有 R1 LOKI artifact `outputs/p1_r1_reusable_matrix/jobs/loki_g1.json` 只有 **G1**，即 known-Fake assistant prefix 条件；没有冻结的 R1 LOKI G0。故本补充的模型间数字必须明确标记为：

```text
CROSS-PROTOCOL DIAGNOSTIC:
R1 G1 vs LEGION official L-FREE
```

它不是严格同输入条件的主 baseline，不能与 official1000 的 R1 G0 vs LEGION L-FREE 主结果合并，也不能声称复现 LEGION paper LOKI 数字。报告 full-N=229 mean/median FG IoU、mean FG F1、global FG IoU/F1、coverage/failure counts及 paired descriptive statistics；双方有效 `[SEG]` 子集仍只作条件性诊断。
