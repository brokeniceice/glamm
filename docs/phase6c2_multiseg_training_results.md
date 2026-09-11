# Phase 6C.2 — Native MultiSEG Controlled Training

仅使用 internal TRAIN 与 internal validation；未访问 internal test 或任何 external/OOD benchmark。L0/L1严格复用，L2从P1初始化，L3以selected L2为base且只导入L1的utility/rectifier。

## 共同 image-level union 口径

| Arm | mean IoU | median IoU | mean F1 | global IoU | global F1 |
|---|---:|---:|---:|---:|---:|
| L0_P1_single | 0.148233 | 0.037966 | 0.210697 | 0.146524 | 0.255597 |
| L1_R1_single | 0.195477 | 0.085664 | 0.270091 | 0.200328 | 0.333789 |
| L2_P1_multi | 0.138912 | 0.023798 | 0.196517 | 0.128005 | 0.226958 |
| L3_R1_multi | 0.148025 | 0.030010 | 0.207133 | 0.136086 | 0.239570 |

## 效应与 interaction

| Contrast | Δmean IoU | Δmedian IoU | Δmean F1 | Δglobal IoU | Δglobal F1 |
|---|---:|---:|---:|---:|---:|
| L2_minus_L0 | -0.009322 | -0.014168 | -0.014180 | -0.018519 | -0.028638 |
| L3_minus_L1 | -0.047452 | -0.055654 | -0.062958 | -0.064242 | -0.094219 |
| L1_minus_L0 | +0.047244 | +0.047698 | +0.059394 | +0.053804 | +0.078192 |
| L3_minus_L2 | +0.009113 | +0.006212 | +0.010616 | +0.008081 | +0.012612 |
| interaction | -0.038131 | -0.041486 | -0.048779 | -0.045723 | -0.065581 |

## 合同与审计

- multiSEG target：append style；ordered phrase-[SEG]-mask pairs；无 union-mask training supervision。
- loss：GLaMM global per-mask mean；双卡按跨 rank 总 mask 数归一化。
- L3：T=sum(K)，一套共享 R1 参数；无 K 套参数复制。
- selector：internal-validation canonical G0 mean IoU，平局取更早 epoch。
- pair-level 与 K bucket 仅为 multiSEG diagnostic，不用于 interaction。

`P1_MULTI_TRAINING_VALID = YES`

`R1_MULTI_TRAINING_VALID = YES`

`MULTISEG_EFFECT_ON_P1 = mean_IoU -0.009322`

`MULTISEG_EFFECT_ON_R1 = mean_IoU -0.047452`

`R1_INTERACTION = mean_IoU -0.038131`

**STOP：未运行任何 external localization benchmark。
