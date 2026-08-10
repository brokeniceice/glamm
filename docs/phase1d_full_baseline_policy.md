# Phase 1D：Full Unified Baseline Training Policy Finalization

## 1. 结论与边界

Phase 1D 已完成 evaluation、optimizer、text-loss normalization 与 checkpoint selection 的冻结。正式选择：embedding LR=`3e-4`，lm_head LR=`3e-4`，text loss=`token_mean`，region encoder frozen，primary checkpoint selector=`val_total_loss/min`。已生成 `configs/phase2a_unified_baseline_full.yaml`，但没有启动 Phase 2A 正式训练。

本阶段只运行固定 64-sample、seed 3407、各 320 optimizer steps 的受控消融及 one-batch dry preflight；没有 NPR、SRM、forensic fusion、consistency loss、L_seg_token、artifact head、Known-Fake auxiliary 或 prompt diversity，没有修改 grammar/frozen split，也没有用 test/LOKI/RAISE 选超参。

## 2. Phase 1C 最终结论

Phase 1C 已证明 canonical Unified distribution 内 G0/G1 generation-localization 接近 TF；terminal EOS 是历史 partial prefix 的主要 bug；mask branch 可在 8 Fake 上过拟合到 TF Mean IoU 0.95865；region encoder hook=0、grad=0；embedding 与 lm_head 未 tied；Real/Fake 虽为 1:1，token-mean LM denominator 中 Real/Fake 为 8.22%/91.78%（Fake/Real=11.17）。当前无证据支持增加 L_seg_token 或新结构。

## 3. Joint canonicalization 与最终 protocol

旧 Joint 同时改变 prompt 与 scoring gate，无法把 generation shift 和 classification error 分开。现已固定：

- G0：canonical Unified user prompt + assistant prefix `[CLS]`，自由生成 verdict/explanation/`[SEG]`，classification 不参与 localization scoring。
- G1（默认 GT-authenticity-conditioned localization）：同一 canonical prompt + structural prefix `[CLS] [FAKE]`，prefix 无 terminal EOS，自由续写 explanation/`[SEG]`。
- Joint：生成输入与 G0 完全相同，唯一差别是 scoring gate；`cls_pred=Real` 时 localization=0。
- TF-FullContext：GT verdict + GT explanation 的 teacher-forced diagnostic upper bound。
- legacy/corrected old-prompt G2：仅保留复现，不是默认 protocol。

canonical user content：

```text
The <image> provides an overview of the picture.
Determine whether this image is authentic and explain the forensic evidence.
```

`canonical_prompt_id=unified_forensics_v1`，`canonical_prompt_sha256=32f0856f718fb3e19f34b552892636fb6c2613adbae4afb8826dea89f670654d`。训练 sample metadata 与 evaluation JSONL 均持久化 fingerprint；evaluation 额外保存 raw prompt、assistant prefix 和 generated token IDs。

## 4. G0/Joint equivalence

新增 `test_joint_equals_g0_when_all_fake_classifications_pass`。在 3e-4 controlled checkpoint 的 32 Fake smoke 中，classification pass=1.0；每个样本的 prompt hash、raw prompt、generated token IDs、SEG position、intersection/union 均完全相等。因此：

| Mode | Trigger | Mean IoU | Global IoU | Mean F1 |
|---|---:|---:|---:|---:|
| G0 | 0.96875 | 0.45596 | 0.60018 | 0.57069 |
| Joint | 0.96875 | 0.45596 | 0.60018 | 0.57069 |

另有 gate regression：同一有效 mask 在 `cls_pred=Real` 时 G0 正常计分、Joint 强制为 0。

## 5. Region encoder freeze

freeze 只修改 `requires_grad` 与 optimizer membership，不改 forward code。真实 FullScope one-batch audit：

| 项目 | 参数数 |
|---|---:|
| Total | 8,025,250,866 |
| Trainable before freeze | 587,459,814 |
| Region encoder removed | 299,142,912 |
| Trainable after freeze | 288,316,902 |
| Trainable region after | 0 |

同一 checkpoint、同一 mixed batch、eval mode 的 freeze 前后 max absolute difference：classification logits=0、full LM logits=0、Real/Fake verdict logits=0、pred masks=0、total/text/cls/BCE/Dice loss 全部为 0。region encoder 不在 optimizer。

## 6. Optimizer groups

最终组互斥、无重复参数：

| Group | Params | LR |
|---|---:|---:|
| LoRA | 4,194,304 | 3e-4 |
| classification_head | 8,194 | 3e-4 |
| text_hidden_fcs | 17,830,144 | 3e-4 |
| SAM mask_decoder | 4,058,340 | 3e-4 |
| embeddings | 131,112,960 | 3e-4 |
| lm_head | 131,112,960 | 3e-4 |

Mixed backward 的组 gradient norm 分别为 22.7533、62.9449、3.8876、13.4078、101.7789、9.2857，均大于 0。row-only gradient masking helper 保留但默认关闭。

## 7. Embedding/lm_head LR ablation

三个 run 均从 `checkpoints/GLaMM-FullScope` 重新初始化，固定相同 64 samples、顺序、seed、grammar、token-mean、region freeze、core LR 与 320 steps。1e-4 只因 3e-5 LM verdict 在 Real 上明显不足而按预注册 gate 补跑。

| Embed/LM LR | CLS acc | LM acc | G0 trigger | G0 Mean IoU | G1 trigger | G1 Mean IoU | TF Mean IoU | Repetition |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3e-4 | 0.98438 | 0.95312 | 0.96875 | 0.45596 | 1.00000 | 0.45750 | 0.51867 | 0.03125 |
| 3e-5 | 1.00000 | 0.84375 | 0.90625 | 0.15982 | 0.90625 | 0.16539 | 0.31295 | 0.12500 |
| 1e-4 | 1.00000 | 0.89062 | 0.84375 | 0.19468 | 0.84375 | 0.21342 | 0.32939 | 0.15625 |

低 LR 虽能学习 classification，却没有保持 LM verdict、explanation/SEG activation 与 localization，因此不满足“保持学习能力前提下最小 drift”的选择条件。正式 embedding/lm_head LR 均选 3e-4。

## 8. Vocabulary drift

下表为 row-wise mean L2 delta（完整 JSON 同时包含 median/p95/max/relative norm）：

| LR | Embed new | Embed old | LM new | LM old |
|---:|---:|---:|---:|---:|
| 3e-4 | 0.180692 | 0.00284731 | 0.338987 | 0.281752 |
| 1e-4 | 0.067511 | 0.00069532 | 0.133042 | 0.138606 |
| 3e-5 | 0.014102 | 0.00011394 | 0.029129 | 0.024656 |

3e-4 相对 3e-5 的 old-vocab mean drift：embedding 24.99×，lm_head 11.43×；new-token mean drift：12.81×/11.64×。这是明确代价，但低 LR 的 G0 Mean IoU 下降 0.29614（3e-5）或 0.26128（1e-4），且 repetition/trigger 退化，因此本轮 controlled evidence 支持 3e-4。

## 9. Text-loss normalization ablation

LR 固定为已选 3e-4；两个 run 都从同一 raw base 开始、固定 64 samples/seed/320 steps，唯一变量是 normalization。token-mean run 直接复用严格相同的 LR 3e-4 run，artifact hard-linked 到 `text_loss_ablation/token_mean`。

| Normalization | CLS acc | LM acc | G0 trigger | G0 Mean IoU | G1 Mean IoU | TF Mean IoU | Exact explanation | Prefix overlap | Length ratio | Repetition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token_mean | 0.98438 | 0.95312 | 0.96875 | 0.45596 | 0.45750 | 0.51867 | 0.78125 | 0.86797 | 0.96381 | 0.03125 |
| per_sample_mean | 1.00000 | 0.98438 | 0.87500 | 0.25744 | 0.25774 | 0.37532 | 0.62500 | 0.69732 | 1.08107 | 0.09375 |

per-sample 改善 Detection/Real balance，但明显损害 Fake explanation exact/prefix、SEG trigger、G0/G1/TF localization 并增加 repetition，故最终选择 token-mean。Real grammar 不做无信息扩写。

最终定义：对所有非 `IGNORE_INDEX` causal assistant target token 汇总 CE 后求 mean。Real/Fake diagnostic CE 只用于日志，不改变 total objective。

## 10. Validation total loss 与辅助指标

旧 validation 将 CE meter 错称 val loss 的路径已修复。primary value 现在来自完整 `loss` meter，并有 regression 检查：

```text
val_total_loss = L_text + L_cls + weighted_fake_mask_BCE + weighted_fake_mask_Dice
               = L_text + L_cls + 2.0 * raw_BCE + 0.5 * raw_Dice
```

Real 保持不计算 mask loss。validation 同时写出 classification Accuracy/Precision/Recall/F1/AUC、LM verdict Accuracy/F1/Fake Recall、CLS-LM agreement，以及 TF-FullContext Mean/Global IoU 与 Mean/Global F1；这些均不参与 primary selection。G0/G1 可按固定低频率运行。

## 11. Checkpoint policy

固定 total budget=5,000 optimizer steps（10 epochs × 500 steps）。每 epoch 始终刷新 `ckpt_model_last_epoch`；当且仅当 `val_total_loss` 更低时另存 `ckpt_model_best`。primary selector=`best_val_total_loss`、mode=`min`。test、external SynthScars test、RAISE 与 LOKI 明确禁止参与 early stopping、hyperparameter 或 checkpoint selection。

## 12. Final full config

`configs/phase2a_unified_baseline_full.yaml` 固定 GLaMM/fixed_cls_query、canonical prompt/grammar/fingerprint、trainable modules、六 optimizer groups、token-mean、Fake-only mask weights、5,000-step budget、last+best total-loss checkpoint policy以及所有禁止 feature=false。配置状态为 `preregistered_not_started`。

## 13. Final one-batch preflight

Real-only、Fake-only、Mixed 各执行一轮 forward；Real/Fake 各执行 backward，Mixed 另执行全 optimizer group gradient audit：

| Case | Total | Text | CLS | weighted BCE | weighted Dice |
|---|---:|---:|---:|---:|---:|
| Real | 11.15396 | 8.37500 | 2.77896 | 0 | 0 |
| Fake | 3.92994 | 3.17188 | 0.04467 | 0.21425 | 0.49914 |
| Mixed | 5.97937 | 3.82812 | 1.41444 | 0.23770 | 0.49910 |

Real mask decoder grad=0；Fake mask decoder grad>0；Real/Fake classification 与对应 token rows 均有 gradient；对同一预测的 Fake-only 与 Mixed mask loss absolute difference=0，证明 denominator 不被 Real 稀释。

Mixed tensor shapes：global `[2,3,336,336]`、grounding `[2,3,1024,1024]`、input/labels/attention `[2,154]`、offset `[3]`、cls_labels `[2]`、seg_valid `[2]`。

## 14. Regression tests

完整执行：

```text
/home/yz/miniconda3/envs/glamm_official/bin/python -m unittest discover -v tests
Ran 74 tests
OK
```

覆盖 Joint/G0 canonical equality与 gate、G1 continuation、region freeze/output equivalence/optimizer exclusion、group uniqueness/LR、drift、token/per-sample math、Real/Fake CE、true val total、CE masquerade prevention、total-loss selector、last+best save、fingerprint，以及 Phase 0.5/1A/1B/1C 全部既有回归。

## 15. 修改文件

- `dataset/forensics/unified.py`：canonical prompt常量与训练 fingerprint metadata。
- `dataset/dataset.py`：collate保留 fingerprint。
- `eval/forensics_eval.py`：G0/Joint canonical mapping与 evaluation fingerprint/raw prompt/prefix。
- `eval/forensics.py`：record propagation、stored detection aggregation。
- `model/llava/model/language_model/llava_llama.py`：暴露 multimodal-expanded labels。
- `model/GLaMM.py`：expanded-label normalization、Real/Fake CE、validation TF masks。
- `train.py`：region freeze、互斥 optimizer groups、完整 val total、辅助 validation metrics、last+best policy。
- `configs/phase1d_lr_3e4.yaml`、`phase1d_lr_3e5.yaml`、`phase1d_lr_1e4.yaml`、`phase1d_text_per_sample.yaml`：受控消融。
- `configs/phase2a_unified_baseline_full.yaml`：最终冻结配置。
- `scripts/phase1d_training_policy.py`：bounded ablation、drift与五协议评估。
- `scripts/phase1d_final_preflight.py`：真实 FullScope freeze equivalence和 one-batch audit。
- `tests/test_phase1d_training_policy.py`：Phase 1D regression。
- `docs/phase1a_localization_evaluation_protocol.md`、`docs/phase1c_seg_activation_diagnosis.md`：仅追加最终 protocol 状态。

## 16. Artifacts

`outputs/phase1d_training_policy/` 包含：`joint_protocol_fix/`、三个 LR runs、两个 text-loss runs、`parameter_audit/`、`final_preflight/` 和 `test_log.txt`。每个 bounded run 保存 config、git HEAD `7bef9a2c94f9add141adc9ed0659c45577da555c`、320-step metrics JSONL、五协议 evaluation JSON、vocabulary drift、optimizer audit、trainable-state checkpoint与 tokenizer。

## 17. 十六项验收回答

1. Joint/G0 是否完全相同 canonical prompt：是。
2. classification pass=1 时 Joint localization 是否等于 G0：是，逐样本与 aggregate 均相等。
3. G1 是否仍为默认 GT-conditioned localization：是。
4. region encoder 是否安全 freeze：是，0 trainable且不进 optimizer。
5. freeze 是否改变当前 output：否，所有 audited max abs=0。
6. 正式 embedding LR：`3e-4`。
7. 正式 lm_head LR：`3e-4`。
8. 3e-4 vs 3e-5 old-vocab drift：embedding 24.99×，lm_head 11.43×；低 LR 同时严重损害 localization。
9. 最终 text loss：token-mean。
10. 原因：per-sample 虽改善 Detection，却显著降低 explanation、trigger、G0/G1/TF并增加 repetition。
11. validation total 是否含全部 loss：是，text+cls+weighted conditional masks。
12. primary selector：minimum `val_total_loss`。
13. 是否使用 test/LOKI 选模型：完全没有。
14. final trainable parameters：288,316,902。
15. 是否生成 full config：是。
16. 是否具备 Phase 2A 条件：代码、协议、配置、真实 one-batch preflight 和回归均通过，具备启动条件；仍按要求等待人工检查。

## 18. 停止点

Phase 1D 到此停止。未运行正式 Full Unified Baseline，未接入任何禁止模块或 loss。下一步仅在人工确认后启动 Phase 2A。
