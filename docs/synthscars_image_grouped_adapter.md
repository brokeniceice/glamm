# SynthScars image-grouped adapter 与标注审计

## 1. 第一版训练语义

第一版统一取证模型对每张图只生成一个 `[SEG]`：

```text
[CLS] [FAKE] <artifact explanation> [SEG]
```

该 `[SEG]` 监督同一图片所有有效 SynthScars ref polygon mask 的布尔并集（union evidence mask），不是逐 ref mask，也不是整张生成图的“伪造像素真值”。adapter 同时保留每个 ref 的 phrase、explanation、polygon、独立 mask 和来源 annotation ID，供追溯与后续 phrase-mask 阶段使用。

## 2. 实现与产物

- adapter：`dataset/forensics/synthscars.py`
- 审计工具：`scripts/data/prepare_synthscars.py`
- 单元测试：`tests/test_synthscars_adapter.py`
- 审计产物：`outputs/data_audits/synthscars_image_grouped_v1/`

复现命令：

```bash
conda run -n glamm_official python scripts/data/prepare_synthscars.py \
  --root datasets/SynthScars \
  --output-dir outputs/data_audits/synthscars_image_grouped_v1 \
  --visualizations-per-split 50
```

manifest 不保存栅格 mask，以免重复占用存储；它保存原始 polygon 和完整追溯信息。adapter 读取样本时用 `pycocotools` 栅格化逐 ref mask，再用布尔 OR 生成严格 `{0,1}` 的 `uint8` union mask。

## 3. 固定数据统计

| split | annotation | 唯一文件名 | 唯一视觉身份 | 多文件变体身份 | 多 annotation 身份 | refs | polygons |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 11,236 | 11,182 | 11,064 | 118 | 171 | 23,935 | 23,935 |
| test | 1,000 | 1,000 | 1,000 | 0 | 0 | 2,674 | 2,674 |

annotation SHA256：

- train：`94100c0b381f4b24745d75e5cba65872686ccf9e37af8e9fd669349e3509cd27`
- test：`3bb55380f9ad75d76c26ad9a18035296be4ab2a7255b5e2fa11b131a2417e7e4`

完整 mask 解码与图像验证结果：

- 缺失图像：0；
- 损坏图像：0；
- 空 union mask：0；
- 真正越界 polygon：0；
- train 空 ref mask：1；
- train 空 ref explanation：3；
- test 空 ref mask/explanation：0。

train union mask 面积占比：

| min | p01 | median | mean | p99 | max |
|---:|---:|---:|---:|---:|---:|
| 0.0000286 | 0.000347 | 0.016279 | 0.042611 | 0.391307 | 0.900231 |

## 4. 异常说明

### 4.1 退化 polygon

train 的 annotation `317`、ref `317:3` 使用三个共线点，shoelace 几何面积为 0，栅格化后为空。该图片还有其他有效 refs，因此最终 union mask 非空。

adapter 不修改原始 polygon，也不伪造区域；返回 `ref_mask_valid` 和 `invalid_ref_ids` 显式标记该 ref。第一版 union mask 只会由实际非空像素组成。阶段 7 做逐 ref 训练前应排除或人工修复这个 ref。

### 4.2 空 ref explanation

train 中三个 ref 的 `explanation` 为空：

- `2983:5`；
- `4850:0`；
- `6318:0`。

三者所属 annotation 的顶层 caption 均非空，所以第一版整图 explanation 仍可使用。adapter 原样保留空 ref explanation，并由审计报告列出；阶段 7 前需单独处理。

### 4.3 边界浮点误差

train 298 个、test 28 个 polygon 的最大坐标比图片宽或高多约 `2.8e-14` 至 `9.1e-13` 像素。这是坐标缩放产生的浮点舍入误差，不是真正越界。审计以 `1e-6` 像素为容差，将其记录为 `boundary_roundoff_polygons`，不裁剪或改写原始标注。

## 5. 重复视觉身份结论

初始文件名审计发现 54 张图片具有重复 annotation。后续全库 pHash 审计又发现 118 个相同 stem 的 `.png/.jpg` 文件对：它们宽高比全部完全一致，统一缩放后的像素 MAE 中位数为 1.37/255、最大为 2.38/255，确认是同一画面的尺寸/编码变体，而不是独立图像。

按 stem 统一视觉身份后，共有 171 个身份包含多条 annotation：170 个含两条、1 个含三条。其 annotation 对统计为：

- 相同 caption：0 对；
- 完全相同 ref：0 对；
- annotation union mask IoU：最小 0、median 0.34001、最大 0.94569；
- mask 完全不重合：19 对。

因此不能随意保留其中一条，也不应把编码/尺寸变体当成独立训练图片。当前处理为：以无后缀 stem 作为视觉身份；选择像素数最大的变体作为 canonical 图像，同尺寸时优先 PNG；将其他变体的 polygon 按宽高比例缩放到 canonical 坐标；聚合 annotation IDs、captions 和 refs 后生成单个 union mask。原始文件名、原始 polygon、来源尺寸和 annotation ID 均保留。

官方 test split 始终保持独立，不参与训练或重复合并。

## 6. 验证

当前单元测试覆盖：

- 同图重复 annotation 聚合；
- annotation ID 保留；
- ref mask 与 union mask 的严格 OR 关系；
- union mask 的 `uint8` 二值性；
- manifest 不嵌入栅格 mask；
- 空 ref explanation 保留；
- 非法 polygon 拒绝。

另生成 train/test 各 50 张确定性 union-mask 红色覆盖图，索引分别记录在 `train_visualizations.json` 和 `test_visualizations.json`。
