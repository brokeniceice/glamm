# Phase 1B：Unified GLaMM Baseline Training Preflight

## 结论

Phase 1B 已完成代码审计、loss/token/gradient/leakage 测试、固定 32 Real + 32 Fake 受控过拟合、Phase 1A evaluator 回归与最终评估，以及完整训练配置准备。没有启动完整长训练，也没有接入 NPR、SRM、forensic fusion、consistency loss、artifact-type head 或 Known-Fake auxiliary training。

受控子集能够明显记忆：classification head accuracy、LM verdict accuracy 和 CLS-LM agreement 均从初始化附近提升到 1.0；最后 32 step 的平均 total loss 从最初 32 step 的 4.7489 降至 0.4421；TF-FullContext Mean IoU 从 0.0580 提升到 0.4210。另一方面，GT-Fake Generate 的 `[SEG]` trigger rate 仅从 0.3125 提升到 0.34375，存在明确的 generation/grounding activation gap。因此当前训练链路可以学习，但 autoregressive explanation / `[SEG]` 激活仍是正式长训练前需要人工接受或另立阶段处理的风险。本阶段按要求只记录诊断，不改变 grammar 或模型结构。

## 1. 当前 baseline architecture

固定 baseline 是：GLaMM + fixed `[CLS]` forensic query + 二分类 classification head。`[CLS]` 位于 assistant prefix 后、verdict 前；其 label 为 `-100`，不由 LM CE 预测，最后层 hidden state 送入 classification head。LM 从该位置的 next-token logits 中只截取 `[REAL]`/`[FAKE]` 两个 token 做 verdict probability。

训练 grammar 保持不变：

```text
Fake: [CLS] [FAKE] <artifact explanation> [SEG]
Real: [CLS] [REAL] No identifiable synthetic artifact evidence is detected.
```

`[REAL]`、`[FAKE]`、Fake explanation 和 Fake `[SEG]` 参与 LM CE；Fake `[SEG]` 前一位置的 causal predictor hidden 驱动原 GLaMM mask decoder。Real 的 `seg_valid=False`，没有有效 GT mask。

禁用项在两个 Phase 1B 配置中均被显式固定为 false/none：NPR、SRM、consistency loss、forensic fusion、artifact-type head、extra evidence query、Known-Fake auxiliary training。frozen split 和 token 顺序未改动。

## 2. Trainable / frozen modules

完整机器可读审计见 `outputs/phase1b_preflight/trainable_parameters.json` 和 `outputs/phase1b_overfit/trainable_parameters.json`。

| 模块 | Total | Trainable | Frozen | LR |
|---|---:|---:|---:|---:|
| LLM base | 6,476,271,616 | 0 | 6,476,271,616 | - |
| LoRA | 4,194,304 | 4,194,304 | 0 | 3e-4（warmup 审计时 1.5e-5） |
| Vision tower | 303,507,456 | 0 | 303,507,456 | - |
| mm_projector | 20,979,712 | 0 | 20,979,712 | - |
| Grounding encoder（不含单列 mask decoder） | 637,032,268 | 0 | 637,032,268 | - |
| SAM mask decoder | 4,058,340 | 4,058,340 | 0 | 3e-4 |
| text_hidden_fcs | 17,830,144 | 17,830,144 | 0 | 3e-4 |
| classification head | 8,194 | 8,194 | 0 | 3e-4 |
| token embeddings | 131,112,960 | 131,112,960 | 0 | 3e-4 |
| lm_head | 131,112,960 | 131,112,960 | 0 | 3e-4 |
| 原 GLaMM region encoder | 299,142,912 | 299,142,912 | 0 | 3e-4 |

总参数 8,025,250,866；trainable 587,459,814；frozen 7,437,791,052。region encoder 是原 GLaMM 训练策略保留的 trainable 模块，本阶段没有新增 bbox 输入，也没有擅自改变其冻结策略。所有可训练组使用同一 base LR 3e-4；审计发生在 20-step linear warmup 的第一个 LR 点，因此 JSON 同时记录 base LR 与 current-at-audit LR。

## 3. Loss formula 与实际权重

实际 loss 为：

```text
L_total = 1.0 * L_text
        + 1.0 * L_cls
        + 2.0 * L_mask_bce
        + 0.5 * L_mask_dice
```

consistency weight 固定为 0。上述权重与现有 `train.py` 默认值一致，没有因 overfit 结果调整。Real-only 实测：text 8.8125、cls 1.15419、BCE 0、Dice 0。Fake-only 实测：text 3.32813、cls 0.01176、加权 BCE 2.12979、加权 Dice 0.42197。

## 4. Mixed batch mask normalization

mask loss 遍历每个样本，仅将 `seg_valid=True` 的 pred/GT mask 加入累计值，最终除以有效 mask 数 `num_masks`，不除以 Real+Fake batch size。Real 不进入 mask denominator。

真实 mixed forward 的预测受 BF16 batch shape/kernel 影响，与 Fake 单独 forward 有 3.77% 的数值差异；这不是 denominator 差异。将同一 pred mask 和同一 GT 分别送入真实 `_compute_loss_components` 后，mixed 与 Fake-only mask loss 的绝对差为 **0.0**。对应 unit test 已锁定该行为。

## 5. fixed `[CLS]` future-label leakage

在同一 batch 中使用相同 image、user prompt、assistant prefix 与 `[CLS]`，只改变 `[CLS]` 后的 GT 为 Real target 或 Fake target。eval mode 的 causal forward 得到：

```text
max_abs(h_cls_real_target - h_cls_fake_target) = 0.0
```

结论：fixed `[CLS]` hidden 看不到 future verdict、explanation 或 `[SEG]`，P0 leakage 验收通过。

## 6. Special token 与 gradient 验证

| Token | ID | 单 token | embedding trainable |
|---|---:|---|---|
| `[CLS]` | 32007 | 是 | 是 |
| `[REAL]` | 32008 | 是 | 是 |
| `[FAKE]` | 32009 | 是 | 是 |
| `[SEG]` | 32004 | 是 | 是 |

隔离 backward 的 embedding row grad norm：Real-only `[CLS]=199.261`、`[REAL]=19.597`；Fake-only `[CLS]=25.347`、`[FAKE]=16.090`。第一训练 step 的 `[CLS]=102.046`、`[REAL]=0.917`、`[FAKE]=21.540`。因此 `[CLS]` 虽为 ignored label，仍通过 classification loss 获得梯度；两个 verdict token 均可学习。tokenizer 与 model/tokenizer save-load roundtrip 测试通过。

## 7. Answer / token length statistics

统计 frozen train split 的全部 8,836 个 Fake：

| 项目 | min | mean | median | std | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fake full sequence | 105 | 203.07 | 182 | 68.80 | 289 | 339 | 448.65 | 1075 |
| Fake explanation | 33 | 131.07 | 110 | 68.80 | 217 | 267 | 376.65 | 1003 |
| `[SEG]` position | 103 | 201.07 | 180 | 68.80 | 287 | 337 | 446.65 | 1073 |
| Fake answer | 36 | 134.07 | 113 | 68.80 | 220 | 270 | 379.65 | 1006 |
| Real answer | 12 | 12 | 12 | 0 | 12 | 12 | 12 | 12 |

Real/Fake answer 长度分布差异很大，可能给 LM verdict / generation 提供 grammar shortcut；因 `[CLS]` 位于 target 前，该差异不构成 classification head 的 future-label shortcut。本阶段不修改 grammar。

## 8. `[SEG]` truncation statistics 与 P0 修复

在 multimodal image expansion 预留后，raw token cutoff 为 961。朴素尾截断会影响 1/8,836 个 Fake，并删除其中的 `[SEG]`。现改为训练期 SEG-preserving truncation：超预算且 `[SEG]` 位于 cutoff 后时，裁剪 explanation/prefix 可用区间并保留以 `[SEG]` 为结尾的监督尾部。

修复后：受截断 Fake 数 1；朴素策略会丢 `[SEG]` 数 1；实际策略丢 `[SEG]` 数 **0**。任何无法保留所需 `[SEG]`/pred mask 的 Fake 仍触发明确 warning/counter，不会静默跳过或截断 GT。

## 9. Overfit subset 与实际曝光

随机种子固定为 3407。子集从 frozen train manifest 确定性选择 32 Real + 32 Fake；Real/Fake 各自含 human、animal、object、scene 每类 8 张，合计每类 16 张。静态来源为 COCO2017 3、FFHQ 4、OpenImagesV7 11、PASS 8、iNaturalist 6、SynthScars 32。

训练 320 optimizer steps，batch size 2、gradient accumulation 1，共完整重复子集 10 次。每个 batch 实测均为 1 Real + 1 Fake：

- domain exposure：Real 320，Fake 320；
- content exposure：human/animal/object/scene 各 160；
- source exposure：COCO2017 30、FFHQ 40、OpenImagesV7 110、PASS 80、iNaturalist 60、SynthScars 320。

每 32 optimizer steps 恰好完成一次 64-sample pass；每个 pass 的来源曝光固定为 COCO2017 3、FFHQ 4、OpenImagesV7 11、PASS 8、iNaturalist 6、SynthScars 32，四个 content type 各 16。十个 pass 均一致。因此 controlled DataLoader 没有破坏预期 balance。该 run 为单进程，不声称验证多 GPU DistributedSampler；正式配置没有被执行。

## 10. Gradient audit

第一训练 step 的关键模块 grad norm：classification head 47.635、LoRA 25.215、token embeddings 112.416、text_hidden_fcs 9.076、SAM mask decoder 34.913。

loss isolation 中，Real-only mask decoder grad 为 0，classification head grad 为 98.132；Fake-only mask decoder grad 为 33.990，text_hidden_fcs 为 9.267。当前路径中 mask decoder 只从 mask loss 获得该监督，因此结果同时证明 Real 不更新 mask path、Fake 有有效 mask gradient。

## 11. Controlled overfit 训练与 loss 趋势

配置见 `configs/phase1b_overfit.yaml`。320 steps、BF16、AdamW、LR 3e-4、20-step linear warmup+decay、gradient clip 1.0；没有写死 GPU 数量。没有 NaN、Inf、长期恒定或 loss 爆炸。

| 32-step 窗口均值 | Total | Text/CE | CLS | BCE | Dice | CLS Acc | LM Acc | Agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steps 1-32 | 4.7489 | 2.5974 | 0.71139 | 0.96669 | 0.47336 | 0.6094 | 0.5625 | 0.5469 |
| steps 289-320 | 0.4421 | 0.04516 | 0.000030 | 0.15221 | 0.24473 | 1.0000 | 1.0000 | 1.0000 |

分类、verdict、agreement 均达到可记忆水平；text、cls、mask 两项 loss 均明显下降。第一版 64-step probe 只用于发现学习不足，随后在不改 loss 权重的前提下从头运行预先封顶的 320-step 受控实验；最终文档和 artifact 只以 320-step run 为准。

## 12. Phase 1A evaluator：初始化 vs overfit

初始与最终均在同一 64-sample subset、同一 400-token generation budget 上重跑。阈值和 denominator 沿用 Phase 1A protocol。

| 指标 | 初始化 | 320-step 最终 |
|---|---:|---:|
| Classification accuracy | 0.4844 | 1.0000 |
| LM verdict accuracy | 0.5000 | 1.0000 |
| CLS-LM agreement | 0.9844 | 1.0000 |
| TF-FullContext Mean IoU | 0.0580 | 0.4210 |
| TF-FullContext Mean pixel-F1 | 0.0905 | 0.5478 |
| GT-Fake Generate Mean IoU | 0.00781 | 0.04398 |
| GT-Fake Generate Mean pixel-F1 | 0.01470 | 0.07002 |
| GT-Fake Generate SEG trigger | 0.3125 | 0.34375 |
| Joint Mean IoU | 0.04973 | 0.24532 |
| Joint Mean pixel-F1 | 0.07689 | 0.32320 |
| Joint SEG trigger | 0.8750 | 0.7500 |
| Joint classification pass | 0.96875 | 1.0000 |

TF-FullContext 最终 Mean IoU 比 GT-Fake Generate 高 0.37704，约为后者 9.57 倍。mask decoder 在 teacher-forced full context 下已明显学起来，但 GT-Fake 自回归路径仍只在 11/32 个 Fake 上触发 `[SEG]`，故主要瓶颈是 autoregressive explanation / `[SEG]` activation，而不是 mask supervision 完全失效。

Joint 最终 IoU 比 GT-Fake 高 0.20134，且 classification pass 已是 1.0；因此二者差距**不主要来自 classification error**，更可能来自两种 protocol 的 prompt/context 与生成激活路径差异。按阶段约束未据此改训练逻辑。

## 13. 发现的 bug 与修复

1. `prepare_model_for_training` 曾未返回包装后的 PEFT model，调用方可能继续持有错误对象；现显式 return 并由 `train.py` 接收。
2. 朴素 truncation 可让最长 Fake 丢失 `[SEG]`；已加入 SEG-preserving training truncation 和回归测试，修复后为 0。
3. Phase 1B runner 最初只移动顶层 tensor，遗漏 list 内 mask tensor；已改成逐元素移动。主训练 `dict_to_cuda` 原本已支持 list tensor，此项是 preflight runner 集成修复。
4. autoregressive cache 路径对 `past_key_values=None` 和缺失/空 attention mask 不稳；已增强 GLaMM forward 与 multimodal helper 的 cache 兼容，generation test 通过。
5. 初次 generation probe 使用 64-token cap，小于 overfit subset 的最长 answer（399 token），不可作最终比较；最终统一改为 400，并重跑初始与最终评估。64-token结果已被覆盖/废弃。

未发现 mixed mask denominator 被 Real batch size 稀释；新增等价性测试用于防止回归。BF16 下不同 batch shape 的预测差异不被误判为 denominator bug。

## 14. Full baseline config

完整训练配置已准备在 `configs/phase1b_unified_baseline.yaml`，明确固定：

- `token_strategy: fixed_cls_query`；
- unified frozen manifests；
- BF16；LoRA r=8、alpha=16、dropout=0.05、targets `q_proj,v_proj`；
- per-device batch 2、gradient accumulation 10、seed 3407；
- AdamW 3e-4、betas 0.9/0.95、weight decay 0、clip 1.0；
- linear warmup scheduler；
- text/cls/BCE/Dice 权重 1/1/2/0.5；consistency 0；
- 所有禁用 forensic feature 显式为 false/none。

该文件仅完成准备，未启动正式训练。GPU 数量未写死。

## 15. Checkpoint selection audit

当前 `train.py` 并不存在适用于 unified forensic baseline 的统一 best metric：普通 validation 路径将 `ce_loss` meter 的均值称为 validation loss 并据其最小值选 best，不能覆盖 classification 与 conditional localization；`mask_validation` 路径则按 gIoU 选 best，不能覆盖 Real/Fake detection 与 generation trigger。因此不能直接把现有 best 逻辑当作本 baseline 的合理 selection policy。

人工决策选项：

- Option A，validation total loss：和训练目标一致、实现简单，但不同 loss 尺度/权重会影响排序，且不直接反映 generation activation。
- Option B，validation classification F1：稳定且易解释，但完全忽略 explanation/localization。
- Option C，预注册 unified validation score：可组合 classification F1、TF-FullContext localization 与 `[SEG]` trigger，但需人工确定组成、归一化与权重，避免事后调参。

建议无条件保存 last，并在人工选定 metric 后保存 best。不得使用 LOKI performance early-stop 或选 checkpoint；`configs/phase1b_unified_baseline.yaml` 将 selection 状态标为 `awaiting_human_decision`。

## 16. 修改文件与实验 artifacts

Phase 1B 直接新增/修改：

- `configs/phase1b_overfit.yaml`
- `configs/phase1b_unified_baseline.yaml`
- `scripts/phase1b_preflight.py`
- `train.py`
- `dataset/dataset.py`
- `dataset/forensics/unified.py`
- `model/GLaMM.py`
- `model/llava/llava_with_region_arch.py`
- `tests/test_unified_forensics_pipeline.py`
- 本文档

Phase 1A evaluator 文件未改变其指标定义；训练后通过原 evaluator 接口运行 Detection、GT-Fake Generate、TF-FullContext 和 Joint。`outputs/phase1b_overfit/` 已包含 config、git commit、subset JSONL、data audit、逐 step train log/metrics、parameter audit、initial/final evaluation records、tokenizer、console/test logs 和约 1.1 GiB 的 trainable-state final checkpoint。没有保存大量中间 checkpoint。

## 17. 完整单元测试结果

执行：

```bash
/home/yz/miniconda3/envs/glamm_official/bin/python -m unittest discover -v tests
```

结果：**Ran 53 tests，OK**。其中 Unified pipeline 14 项、Phase 1A forensic evaluation protocol 14 项全部通过；Detection、GT-Fake Generate、TF-FullContext、TF-MinimalContext、Joint、causal SEG hidden alignment、token save/load、Real/Fake dataset、mixed collate、single forward、Real mask skip、Fake gradient、generation、model roundtrip、future-label leakage、mask denominator 和 late-SEG truncation 均覆盖。预期 warning 仅来自专门验证 pred/GT mask count mismatch 会显式报警的测试，以及 tiny tokenizer 的人为 max-length 提示。

## 18. 必答诊断问题

1. **fixed `[CLS]` 是否不存在 future-label leakage？** 是；同 batch A/B 的 hidden max absolute difference 为 0.0。
2. **`[CLS]` embedding 是否通过 classification loss 获得 gradient？** 是；Real/Fake 隔离 backward 与首 step 均为正。
3. **`[REAL]`/`[FAKE]` 是否能稳定学习？** 是；两行 embedding 均有梯度，subset LM verdict accuracy 最终为 1.0。
4. **Real 是否完全不参与 mask loss？** 是；BCE/Dice 与 mask decoder gradient 均为 0。
5. **mixed batch 是否不会稀释 Fake mask loss？** 是；同 pred/GT 的绝对差为 0.0。
6. **Fake mask decoder 是否获得有效 gradient？** 是；隔离 grad norm 33.990，首 step 34.913。
7. **是否存在 `[SEG]` 被 sequence truncation 删除？** 修复前朴素策略有 1 个；当前策略为 0。
8. **64-sample subset 是否能够明显 overfit？** 是；classification、LM verdict、agreement 到 1.0，所有主要 loss 下行，TF localization 明显提升。
9. **TF-FullContext localization 是否能够被学起来？** 是；Mean IoU 0.0580 -> 0.4210，F1 0.0905 -> 0.5478。
10. **GT-Fake Generate 是否能够稳定生成 `[SEG]`？** 否；最终 trigger 仅 0.34375，仍有 21/32 trigger failure。
11. **TF-FullContext 与 GT-Fake Generate gap 有多大？** 最终 Mean IoU 绝对差 0.37704，TF 约为 GT-Fake 的 9.57 倍。
12. **Joint 与 GT-Fake Generate gap 是否主要来自 classification？** 否；最终 classification pass=1.0，差距主要在 protocol context/generation activation path。
13. **实际 DataLoader 是否保持 frozen dataset balance？** 是（本 controlled subset）；每 batch 1:1，总曝光 Real/Fake=320/320，四类各 160。
14. **Phase 1A 全部 regression tests 是否仍通过？** 是；完整测试共 53 项全部通过。

## 19. 停止点

Phase 1B 到此停止。当前结果不构成论文结论，也没有用 LOKI 调参或选择 checkpoint。下一步必须等待人工检查，尤其是是否接受当前 GT-Fake `[SEG]` activation gap，以及正式训练采用哪一种 checkpoint-selection metric。
