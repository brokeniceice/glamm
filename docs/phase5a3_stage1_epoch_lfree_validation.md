# Phase 5A-3 — Stage-1 Epoch L-FREE Validation Diagnostic

本阶段只在冻结 internal-validation Fake 1,106 张 unique images 上比较现有 Epoch 1/2/3；没有继续训练。

固定协议：官方 image-only L-FREE、free generation、`[SEG]→SAM`、mask logit `>0`、multiple masks union；无 `[SEG]`/空 mask 在 full-N 中计零。GT 是每图全部官方 refs polygon 的原分辨率 pixel union。

| Epoch | TF val total loss | N | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid SEG+mask |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.065668 | 1106 | 0.116230 | 0.016106 | 0.169521 | 0.122668 | 0.218530 | 1104/1106 |
| 2 | 1.026487 | 1106 | 0.128368 | 0.016153 | 0.183977 | 0.124166 | 0.220904 | 1099/1106 |
| 3 | 1.018036 | 1106 | 0.126342 | 0.015383 | 0.182273 | 0.127892 | 0.226781 | 1104/1106 |

## Paired changes

- epoch2_minus_epoch1: mean FG IoU delta `+0.012137`, 95% CI `[+0.002583, +0.022111]`, W/T/L `445/305/356`, Wilcoxon p=`0.00047351358`.
- epoch3_minus_epoch2: mean FG IoU delta `-0.002025`, 95% CI `[-0.010221, +0.005880]`, W/T/L `361/351/394`, Wilcoxon p=`0.55954052`.
- epoch3_minus_epoch1: mean FG IoU delta `+0.010112`, 95% CI `[+0.001128, +0.019030]`, W/T/L `431/320/355`, Wilcoxon p=`0.0011346513`.

本结果仅用于判断是否值得授权延长 Stage-1 训练；本阶段不会自动继续训练或选择新 checkpoint。

Machine result: `/home/yz/groundingLMM_official/outputs/phase5a3_stage1_epoch_lfree_validation/results.json`
