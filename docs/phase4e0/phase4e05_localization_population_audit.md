# Phase 4E-0.5 — Localization Training Population 审计

## 冻结数据事实

来源为 `outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl`，SHA256 `3a5cd7040ee9ae5b7c8e6cf6cc5f225441d95849984c9ce6ed9abfa07f9f63bc`。

| Split | Total | Fake | Real | `seg_valid=True` | 合法 polygon-derived mask |
|---|---:|---:|---:|---:|---:|
| train | 17,672 | 8,836 | 8,836 | 8,836 Fake | 8,836 Fake |
| validation | 2,212 | 1,106 | 1,106 | 1,106 Fake | 1,106 Fake |

manifest 中 Real 的 `has_mask_label=true` 不能解释为合法 empty-mask supervision：所有 Real 均 `refs=[]`；`UnifiedForensicsDataset.__getitem__` 明确只在 `forensics_domain==fake` 时令 `seg_valid=True` 并构造 `_fake_union_mask`，Real 返回 `masks=None`。项目历史上 Phase 3D.2 只允许 Fake 进入 spatial loss；Phase 3E/4A 虽以 balanced Real/Fake population 训练 language/classification/global objective，但 dense mask loss仍只作用于 Fake。

因此 TF-FDG Stage T/S 是纯 localization decoder training，只使用 train Fake；不得把 Real 补成 empty mask。

```text
LOCALIZATION_TRAIN_POPULATION:
frozen internal train Fake rows only

N_FAKE:
8836

N_REAL:
8836 in the combined classification manifest; 0 in localization optimizer population

REAL_USED_FOR_MASK_TRAINING:
NO

RATIONALE:
Real has no refs/polygons, dataset seg_valid is false, and no historical protocol defines Real as an authoritative empty localization target.
```

## 修正预算

effective batch 固定为 8；每个 full epoch 保留最后一个 4-image partial batch，不重复填充样本。

| Stage | Epochs | Steps/epoch | Total optimizer steps | Fake exposures |
|---|---:|---:|---:|---:|
| Teacher TF | 5 | `ceil(8836/8)=1105` | 5,525 | 44,180 |
| Student G0 | 10 | 1,105 | 11,050 | 88,360 |

epoch 是主预算语义；不为恢复旧 proposal 的 11k/22k step 数而重复 exposure。机器可读证据见 `outputs/phase4e05_tf_fdg_hardening/population_audit.json`。

