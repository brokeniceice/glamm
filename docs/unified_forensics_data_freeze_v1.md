# 统一 AIGC 取证数据冻结记录 v1

## 1. 阶段状态

数据准备阶段已于 2026-08-05 冻结。本阶段没有启动模型训练，没有加载 NPR，也没有修改或删除任何原始数据文件。冻结入口为：

`outputs/data_audits/unified_forensics_split_v1/summary.json`

所有正式划分 manifest 位于：

`outputs/data_audits/unified_forensics_split_v1/`

## 2. 标签策略

- SynthScars fake：使用 image-grouped adapter；同一视觉身份的全部有效 refs/polygons 合并成一个 union evidence mask。
- OpenImagesV7、COCO2017、FFHQ、iNaturalist：使用来源 manifest 的既有内容标签。
- PASS：6,000 张全部使用 `gpt-5.6-luna` 内容标签；Human 70、Animal 742、Object 2,101、Scene 3,041、Ambiguous 46。
- PASS 的 46 张 Ambiguous 保留在全量标签审计中，但不进入内容匹配池。
- 内部 real/fake 内容分布严格匹配；不做重复采样。

## 3. 官方 test 泄漏修复

补充 pHash 审计发现 SynthScars 官方 test 与原 train 各有 18 个同 UUID 视觉身份重合：16 对 pHash 距离为 0，2 对距离为 2。

处理原则：

- 官方 test 1,000 张保持原样；
- 从内部 fake 池排除对应 18 个 train 样本；
- 按内容类别从 real 池确定性排除 18 张以保持平衡；
- 不删除原图，不改写官方 annotation 或上游 manifest；
- 修复后内部池与官方 test 的已知跨集合 pHash 泄漏数为 0。

排除详情：

`outputs/data_audits/unified_forensics_split_v1/official_test_overlap_exclusions.json`

原始泄漏审计与 18 对可视化：

`outputs/data_audits/official_synth_test_leakage_v1/`

## 4. 固定 8:1:1 划分

排除官方 test 重合后，内部池为 11,046 Real + 11,046 Fake。按内容类别分层，并将 pHash 近重复连通组锁在同一 split：

| split | Real | Fake | Combined |
|---|---:|---:|---:|
| train | 8,836 | 8,836 | 17,672 |
| val | 1,106 | 1,106 | 2,212 |
| test | 1,104 | 1,104 | 2,208 |

类别配额在每个 split 内 real/fake 完全一致：

| split | Human/侧 | Animal/侧 | Object/侧 | Scene/侧 |
|---|---:|---:|---:|---:|
| train | 4,652 | 1,220 | 1,551 | 1,413 |
| val | 582 | 153 | 194 | 177 |
| test | 581 | 152 | 194 | 177 |

固定 seed：`unified-forensics-split-v1`。

## 5. 独立测试资源

- `official_synthscars_test.jsonl`：官方 SynthScars test，1,000 张 fake，完全不进入内部 train/val/test。
- `raise_heldout_real_clean.jsonl`：RAISE-1k 中可完整解码的 998 张 real。
- 两张损坏 TIFF `r0515a051t.TIF`、`r0bf7f938t.TIF` 不进入 clean held-out；原文件保留。

内部 `test_combined.jsonl` 是内容匹配且类别平衡的 10% 测试集；官方 SynthScars test 与 RAISE held-out 分别作为额外 fake/real benchmark 报告，不能用于调参或 best checkpoint 选择。

## 6. 正式 manifest

- `train_real.jsonl` / `train_fake.jsonl` / `train_combined.jsonl`
- `val_real.jsonl` / `val_fake.jsonl` / `val_combined.jsonl`
- `test_real.jsonl` / `test_fake.jsonl` / `test_combined.jsonl`
- `official_synthscars_test.jsonl`
- `raise_heldout_real_clean.jsonl`

逐文件 SHA256 固定在 `summary.json` 的 `output_manifest_sha256` 字段中。DataLoader 不得重新扫描来源目录或临时随机划分。

## 7. 阶段验收

- Real/Fake 逐 split 数量相等：通过。
- Real/Fake 逐 split 四类内容配额相等：通过。
- pHash 近重复组跨内部 split：0。
- 官方 test 已知跨内部池重复：修复后 0。
- SynthScars 官方 test 缺图/损坏：0。
- RAISE held-out 已知损坏：2，已显式排除并记录。
- 相关 adapter、标签、匹配与划分测试：通过。

本数据阶段到此结束。后续若进入模型阶段，应从结构化统一推理协议和小规模数据闭环开始，不应重新下载、重采样或改变本冻结版本。
