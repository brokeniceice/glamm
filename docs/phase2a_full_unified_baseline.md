# Phase 2A — Full Unified GLaMM Baseline

## 结论

Phase 2A 完成了第一个严格可复现的 Unified GLaMM baseline。训练达到精确的 5000 optimizer steps（100,000 sample exposures），完成 10 次 validation，并只依据 `min val_total_loss` 选择 step 2500 / epoch 5 为 best checkpoint。未使用 internal test、LOKI、RAISE 或 external SynthScars 做训练决策，未加入 NPR、SRM、forensic fusion 或 consistency loss。

代码已真实验证双 GPU DeepSpeed forward/backward/optimizer/scheduler、rank sampler sharding、参数同步、checkpoint/resume 与 world-size-invariant loss normalization。根据训练期间的人工资源调整，正式轨迹的 step 1–1000 使用 GPU0+GPU1，之后在保持 effective global batch、LR、loss、数据顺序语义和 optimizer-step budget 不变的前提下转换为单 GPU 续训，并最终在 physical GPU1 完成 step 5000。

## 固定实验定义

- 配置：`configs/phase2a_unified_baseline_full.yaml`
- 模型：GLaMM + fixed `[CLS]` + classification head
- Fake target：`[CLS] [FAKE] <artifact explanation> [SEG]`
- Real target：`[CLS] [REAL] No identifiable synthetic artifact evidence is detected.`
- `region_encoder=frozen`
- BF16；LoRA / classification head / text hidden FCs / SAM mask decoder / embeddings / LM head LR 均为 `3e-4`
- `L_total = L_text + L_cls + 2.0 L_mask_BCE + 0.5 L_mask_Dice`
- text normalization：global `token_mean`
- mask normalization：global `seg_valid` Fake count；Real mask loss 为 0
- NPR=false，SRM=false，forensic_fusion=none，consistency=0，seg-token auxiliary=0
- frozen train/val/test：17,672 / 2,212 / 2,208；test 为 1,104 Real + 1,104 Fake
- canonical prompt SHA256：`32f0856f718fb3e19f34b552892636fb6c2613adbae4afb8826dea89f670654d`

## 硬件与 distributed 配置

预注册的两张目标卡均为 NVIDIA RTX A6000，`50,908,823,552` bytes（nvidia-smi 标称 49,140 MiB，通常称 48 GB）。软件环境为 PyTorch 1.13.1+cu117、CUDA 11.7、DeepSpeed 0.12.5。

双卡正式启动命令对应：

```bash
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 \
  scripts/phase2a_distributed_train.py \
  --mode train --gradient-accumulation-steps 5 --workers 8
```

DeepSpeed 使用 ZeRO stage 2、BF16、gradient clipping 1.0、overlap communication、reduce-scatter，以及 linear warmup/decay（100 warmup steps）。双卡设置为 `micro_batch=2 × world_size=2 × GAS=5 = EGB 20`。经人工要求切换到单卡后的设置为 `micro_batch=10 × world_size=1 × GAS=2 = EGB 20`，因此总曝光仍严格是 `5000 × 20 = 100,000`，即 `100000 / 17672 = 5.658669081` effective dataset passes。

## Sampler audit

`BalancedDistributedForensicsSampler` 的完整 epoch 审计结果：rank0 与 rank1 各 8,836 个样本，跨 rank overlap 为 0，两 rank union 严格等于 17,672 条 frozen train manifest。同 seed + epoch 可复现，不同 epoch 顺序发生变化。正式 runtime 另保存前 20 optimizer steps 的每个 rank 的 `sample_id/image_path/GT/content/source`；各 optimizer window 保持 10 Real + 10 Fake 的全局平衡。

证据：`distributed_sampler_audit.json`、`distributed_sampler_runtime.json`。

## Distributed loss normalization 与数值 parity

训练 runner 不使用“各 rank local mean 后再等权平均”。每个 gradient-accumulation window 先统计全局有效 text token、classification sample 和 `seg_valid` mask 数，再用全局分母构造各组件梯度。因此：

- text CE = 所有 rank 的有效 target-token CE sum / global valid-token count；
- classification CE = global sample sum / global sample count；
- BCE/Dice = Fake valid-mask sum / global valid-mask count；
- Real-only distributed mask loss 精确为 0；mixed Real/Fake 不受 rank Fake 比例影响。

相同 20-sample global set 的 1GPU vs 2GPU smoke：text loss mean relative difference `0.000889`，mask BCE `0.013139`，mask Dice `0.001051`；全部关键参数在两个 rank 的 max-abs difference 为 0。BF16 下的总体小差异处于预设容差内，测试判定通过。

## Throughput、DataLoader 与显存

| 项目 | 实测值 |
|---|---:|
| 1GPU smoke samples/s | 1.278813 |
| 2GPU smoke samples/s | 1.861273 |
| 双卡 speedup | 1.455469× |
| 2GPU worst-case samples/s | 1.191935 |
| 2GPU worst-case rank0 peak | 24,096,249,856 bytes |
| 2GPU worst-case rank1 peak | 24,096,249,856 bytes |

DataLoader benchmark 的 2/4/8 workers 分别为 19.45/52.79/57.09 samples/s，最终选择每 rank 8 workers；数据准备速度显著高于约 1.86 samples/s 的训练吞吐，因此没有主要 DataLoader bottleneck。启用了 pinned memory、persistent workers、prefetch 和 non-blocking transfer。

单 GPU micro-batch 10 的一次从 step 3500 replay 在 step 3994 遇到 allocator OOM：47.41 GiB capacity 中 31.57 GiB allocated、33.56 GiB reserved、仅 20 MiB free，随后申请 1.55 GiB 失败。该次 replay 的未持久化 step 3501–3994 全部丢弃；从 durable step 3500 在 physical GPU1 原配置恢复，未修改 LR/loss/batch/sample exposure，稳定完成至 step 5000。此事件及处理记录于 `gpu_memory_stress.json` 和 `training_completion_audit.json`。

## 训练曲线

以下为 canonical、去除失败 replay 后严格 step 1–5000 的 epoch mean：

| Epoch | Total | Text | CLS | Mask BCE | Mask Dice |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.141680 | 1.250561 | 0.227222 | 0.279038 | 0.384859 |
| 2 | 1.559814 | 0.927787 | 0.095899 | 0.208264 | 0.327863 |
| 3 | 1.386743 | 0.830696 | 0.062694 | 0.186185 | 0.307168 |
| 4 | 1.263247 | 0.758795 | 0.045308 | 0.168685 | 0.290459 |
| 5 | 1.170016 | 0.703078 | 0.031617 | 0.160234 | 0.275087 |
| 6 | 1.100921 | 0.655033 | 0.023253 | 0.157857 | 0.264779 |
| 7 | 1.017220 | 0.599413 | 0.019933 | 0.144741 | 0.253134 |
| 8 | 0.943970 | 0.557777 | 0.009141 | 0.137179 | 0.239873 |
| 9 | 0.889237 | 0.523621 | 0.004654 | 0.132530 | 0.228432 |
| 10 | 0.842027 | 0.493882 | 0.003333 | 0.124559 | 0.220252 |

五个 loss component 均平滑下降，无 NaN/Inf、gradient overflow、rank desync 或 loss-normalization 异常。canonical `metrics.jsonl` 恰有 5,000 行，SHA256 为 `9c8a841cc8ce8a5cc116c46381a509834d0e43db2e08246199954a5bc2bc662f`。

## Validation 曲线与 checkpoint 选择

| Epoch/step | val total | CLS acc | LM acc | CLS–LM agree | TF mean IoU | TF global IoU |
|---|---:|---:|---:|---:|---:|---:|
| 1/500 | 1.682129 | .957052 | .960217 | — | .274378 | — |
| 2/1000 | 1.571300 | .976040 | .975588 | — | .289796 | — |
| 3/1500 | 1.535476 | .976944 | .978300 | — | .302482 | — |
| 4/2000 | 1.545545 | .977848 | .978300 | — | .300983 | — |
| **5/2500** | **1.529054** | **.983273** | **.984177** | **.999096** | **.325395** | **.372487** |
| 6/3000 | 1.538196 | .985533 | .984629 | — | .326400 | — |
| 7/3500 | 1.563020 | .980561 | .979204 | — | .331866 | — |
| 8/4000 | 1.564325 | .985081 | .986438 | — | .340121 | — |
| 9/4500 | 1.591470 | .982369 | .984177 | — | .348887 | — |
| 10/5000 | 1.585654 | .985986 | .985986 | .997288 | .347640 | .380940 |

Primary selector 是唯一预注册的 `min val_total_loss`，因此 best 为 epoch 5 / step 2500，`val_total_loss=1.5290540713317924`。last 为 epoch 10 / step 5000。后期 TF localization 继续改善而 total validation loss 略回升，不改变 checkpoint selection。

Best：`checkpoints/phase2a_unified_baseline/single/best`  
Last：`checkpoints/phase2a_unified_baseline/single/last`

二者通过仓库 `checkpoints` 到 `/data` 的软链接保存，不占用主目录的模型权重空间。

## Frozen internal test（best checkpoint only）

<!-- PHASE2A_TEST_RESULTS_START -->
### Detection

| Head | Accuracy | Precision | Recall/Fake Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Classification | 0.972373 | 0.971093 | 0.973732 | 0.972411 | 0.996282 |
| LM verdict | 0.968750 | 0.967480 | 0.970109 | 0.968792 | 0.996339 |

CLS–LM agreement：**0.994565**。

### Localization overall

| Mode | Mean IoU | Global IoU | Mean Pixel-F1 | Global Pixel-F1 | SEG trigger |
|---|---:|---:|---:|---:|---:|
| G0 | 0.139568 | 0.133436 | 0.200415 | 0.235454 | 0.986413 |
| G1 | 0.137260 | 0.132652 | 0.196853 | 0.234232 | 0.992754 |
| tf_full_context | 0.344065 | 0.388522 | 0.454585 | 0.559619 | N/A (teacher-forced) |
| joint | 0.139568 | 0.133461 | 0.200415 | 0.235492 | 0.986413 |

### Localization by content type

**G0**

| Content | N | Mean IoU | Global IoU | Mean Pixel-F1 | Global Pixel-F1 |
|---|---:|---:|---:|---:|---:|
| Human | 581 | 0.157627 | 0.149347 | 0.226348 | 0.259882 |
| Animal | 152 | 0.128748 | 0.091970 | 0.180902 | 0.168447 |
| Object | 194 | 0.127590 | 0.117880 | 0.184037 | 0.210900 |
| Scene | 177 | 0.102710 | 0.143766 | 0.149996 | 0.251391 |

**G1**

| Content | N | Mean IoU | Global IoU | Mean Pixel-F1 | Global Pixel-F1 |
|---|---:|---:|---:|---:|---:|
| Human | 581 | 0.154008 | 0.148925 | 0.221171 | 0.259243 |
| Animal | 152 | 0.126019 | 0.096479 | 0.177614 | 0.175980 |
| Object | 194 | 0.128589 | 0.108479 | 0.183758 | 0.195726 |
| Scene | 177 | 0.101444 | 0.146311 | 0.147902 | 0.255274 |

**tf_full_context**

| Content | N | Mean IoU | Global IoU | Mean Pixel-F1 | Global Pixel-F1 |
|---|---:|---:|---:|---:|---:|
| Human | 581 | 0.354592 | 0.396659 | 0.473613 | 0.568011 |
| Animal | 152 | 0.348868 | 0.378270 | 0.452837 | 0.548906 |
| Object | 194 | 0.373629 | 0.437842 | 0.479710 | 0.609027 |
| Scene | 177 | 0.272986 | 0.330873 | 0.366086 | 0.497227 |

**joint**

| Content | N | Mean IoU | Global IoU | Mean Pixel-F1 | Global Pixel-F1 |
|---|---:|---:|---:|---:|---:|
| Human | 581 | 0.157627 | 0.149347 | 0.226348 | 0.259882 |
| Animal | 152 | 0.128748 | 0.091970 | 0.180902 | 0.168447 |
| Object | 194 | 0.127590 | 0.117880 | 0.184037 | 0.210900 |
| Scene | 177 | 0.102710 | 0.143889 | 0.149996 | 0.251579 |

G0/Joint 对 1104 个 Fake 的 sample IDs 与生成轨迹 equivalence：**passed**。
<!-- PHASE2A_TEST_RESULTS_END -->

Detection 在全部 2,208 样本上报告 classification head、LM verdict 和 agreement。G0/G1/TF-FullContext/Joint 在全部 1,104 个有有效 evidence mask 的 GT Fake 上报告 Mean IoU、Global IoU、Mean/Global Pixel-F1、SEG trigger，并按 Human/Animal/Object/Scene 分组。G0 与 Joint 共享完全相同的 canonical free-generation trajectory；Joint 仅额外施加 classification-head Fake gate，并有逐样本 token/prompt/SEG-position equivalence audit。

## Explanation outputs 与 language drift

每个 GT Fake 的 G0/G1 prediction 均保存 `generated_explanation`、`gt_explanation`、generated/GT token length、`seg_triggered`、top-level `repetition_flag`、详细 repetition n-gram statistics、stop reason 和完整 generated token IDs。完整结果位于 `test/G0|G1/predictions.jsonl`，并在 `predictions/explanations_G0.jsonl` 提供统一入口。

<!-- PHASE2A_LANGUAGE_RESULTS_START -->
- G0: SEG trigger `0.986413`；repetition `0.019928`；mean generated tokens `110.657`；EOS-before-SEG `0.000000`；max-token-before-SEG `0.000000`。
- G1: SEG trigger `0.992754`；repetition `0.026268`；mean generated tokens `111.323`；EOS-before-SEG `0.000000`；max-token-before-SEG `0.000000`。
- joint: SEG trigger `0.986413`；repetition `0.019928`；mean generated tokens `110.657`；EOS-before-SEG `0.000000`；max-token-before-SEG `0.000000`。
<!-- PHASE2A_LANGUAGE_RESULTS_END -->

## Runtime warnings

1. 单 GPU 第一次 step-3500 replay 的 allocator OOM 已如上记录；没有改变实验定义，失败 replay 未进入 canonical metrics。
2. Hugging Face 初始化会尝试网络 metadata HEAD；最终 evaluation 显式使用本地 cache offline 模式，避免网络不可达造成等待，不改变权重。
3. CLIP vision checkpoint 打印“text weights unused”提示，这是只加载 CLIP vision tower 的预期提示。
4. 全量回归中的 Fake pred/GT count mismatch warning 来自专门验证“不得静默截断”的负向测试，属于预期行为。

## Artifacts

- `run_metadata.json`、`effective_batch_audit.json`
- `distributed_sampler_audit.json`、`distributed_sampler_runtime.json`
- `distributed_loss_parity.json`
- `gpu_memory_stress.json`、`throughput_benchmark.json`、`throughput_benchmark_workers.json`
- `training_completion_audit.json`、`validation_curve.json`
- `metrics.jsonl`（canonical 1–5000）及各 resume segment provenance
- `validation/epoch_01` … `epoch_10`
- `test/detection|G0|G1|tf_full_context|joint`
- `predictions/`
- `checkpoints/best|last`（仅软链接到 `/data` 权重）

## 修改文件

- `dataset/forensics/distributed.py`：balanced rank sharding sampler
- `tools/distributed_loss.py`：global numerator/denominator normalization
- `scripts/phase2a_distributed_train.py`：DeepSpeed train/validation/checkpoint/resume runner
- `scripts/phase2a_preflight.py`：metadata、sampler、loader 与 smoke gates
- `scripts/convert_zero2_world2_to_world1.py`：经审计的 ZeRO-2 DP world-size conversion
- `scripts/run_phase2a_detached.sh`：脱离终端的续训与显存监控
- `scripts/phase2a_final_evaluate.py`：best-only、逐样本可恢复 final evaluation
- `scripts/phase2a_finalize_artifacts.py`：metrics provenance merge 与 final artifact audit
- `eval/forensics.py`、`eval/forensics_eval.py`：完整 explanation 字段和 canonical generation evaluation
- `tests/test_phase2a_distributed_training.py`：Phase 2A distributed regression
- `configs/phase2a_unified_baseline_full.yaml`：实际 continuation 配置及不变实验定义

## Final tests

`CUDA_VISIBLE_DEVICES='' python -m pytest -q tests`：**83 passed, 5 warnings**。覆盖 world-size=2 sampler 无重叠/可复现、global text/mask/classification parity、EGB/sample exposure equality、parameter synchronization、canonical prompt、rank0-only checkpoint、resume、best-val selection、Real-only/mixed mask semantics、explanation audit fields，以及 Phase 1A–1D 全部既有 regression。

## 是否可进入下一阶段

<!-- PHASE2A_READINESS_START -->
**可以进入下一阶段 forensic feature enhancement。** Phase 2A 的训练、validation、best-only frozen internal test、完整 predictions、artifact audit 与 regression 均已完成；该 best checkpoint 可作为后续 NPR/SRM/fusion 的正式对照组。
<!-- PHASE2A_READINESS_END -->

Phase 2A 没有运行 external evaluation，也没有实现或启动 NPR、SRM、forensic fusion、consistency loss。后续增强必须继续使用相同 frozen split、protocol、metric，并以本阶段 best checkpoint 作为正式对照组。
