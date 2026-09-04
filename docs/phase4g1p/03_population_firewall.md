# Conditional-utility population firewall

## Source pool

只允许原 G1-C `TRAIN-FIT + TRAIN-CAL`，共7,511 image IDs；原 G1-C TRAIN-AUDIT 1,325 IDs全部排除。development validation、internal test、official1000持续封存。

用 seed material `phase4g1p-conditional-utility-v1` 在 valid/invalid-G0 strata 内按 `SHA256(seed\0image_id)` rank，再固定70/15/15：

| Fold | n | valid | invalid | FG prevalence@256 | region count |
|---|---:|---:|---:|---:|---:|
| UTILITY-FIT | 5,258 | 5,171 | 87 | 0.0422230 | 11,495 |
| UTILITY-CAL | 1,127 | 1,108 | 19 | 0.0423791 | 2,315 |
| UTILITY-AUDIT | 1,126 | 1,108 | 18 | 0.0449929 | 2,412 |

三组 image-ID disjoint、unique membership、exact 7,511 coverage均通过；与旧 G1-C TRAIN-AUDIT overlap=0。invalid-G0保持 invalid，不构造伪 language source；future合法 utility fitting/audit仅消费 valid rows，并单独报告 invalid policy。

Split manifest已在任何 fitting 前冻结于 `outputs/phase4g1p/utility_split_manifest.json`。本阶段没有读取 UTILITY-CAL/AUDIT 的 utility result，也没有根据结果重划 split。
