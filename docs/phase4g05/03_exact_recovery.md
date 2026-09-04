# P1 exact recovery

## 原 proposed graph 的结构矛盾

`z_L 256 → downsample 64 → evidential fusion → upsample 256` 对 impulse、checkerboard、sharp edge 等高频输入不可逆，因此不能用任何 tolerance 声称 exact recovery。低分辨率 reliability field可以参与 active fusion，但 fallback 不能穿过该 down/up path。

## Hardened identity path

PCERF 在 original-image normalized 256 grid上完成 active late fusion。对每个 sample/pixel，只有 canonical G0 valid、forensic present、非 vacuous、非 off且在 CLIP crop support 内时才运行 fusion。其他位置由 DS neutral-element dispatch 直接选择原 `z_L`：

```text
active = valid_G0 & forensic_present & !vacuous & !off & clip_support
output = where(active, DSmP(ECoLaF(E_L,E_F)), z_L)
```

它不是 learned residual，也不把 `z_L` 先 resize 后再恢复；fallback branch保留输入 dtype 与 bit pattern。

## Synthetic results

constant、impulse、checkerboard、sharp edge、sparse mask、random 与 high-frequency pattern 全部用 `torch.equal`、零容差检查：

```yaml
P1_LANGUAGE_ONLY_EXACT_RECOVERY: PASS
FORENSIC_VACUOUS_EXACT_RECOVERY: PASS
FORENSIC_OFF_EXACT_RECOVERY: PASS
```

完整逐 pattern 结果见 `outputs/phase4g05/exact_recovery.json`。测试无 optimizer update、无 validation data。

## 语义边界

这里的 exact 指 canonical 256×256 `z_L` tensor identity。正式 evaluator仍必须保留 canonical invalid-G0 score 0 policy；exact fallback 不能“救活”没有合法 `[SEG]` 的样本。

