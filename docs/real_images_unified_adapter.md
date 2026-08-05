# 多来源真实图统一 adapter 与候选池审计

## 1. 第一版真实样本契约

所有真实图统一使用：

```text
[CLS] [REAL] No reliable synthetic artifact is found. [SEG]
```

其中 `[SEG]` 始终存在，并以原图尺寸的严格 `uint8` 全零 mask 监督。真实样本不包含 refs：

```text
class_label = 0
verdict_token = [REAL]
explanation = No reliable synthetic artifact is found.
refs = []
ref_masks = []
union_evidence_mask = zeros(height, width)
```

实现位于 `dataset/forensics/real_images.py`。该 adapter 只读取各来源已经固定的 `selected.jsonl`，不会重新扫描目录或重新随机选择图片。

## 2. 来源角色

| 来源 | 原始数量 | 统一角色 |
|---|---:|---|
| Open Images V7 | 8,000 | candidate_train |
| PASS | 6,000 | candidate_train |
| COCO 2017 | 4,000 | candidate_train |
| FFHQ | 2,500 | candidate_train |
| iNaturalist | 1,000 | candidate_train |
| RAISE-1k | 1,000 | heldout_test |

RAISE-1k 不进入训练候选池。原始候选 real 共 21,500 张；统一 SHA256 审计排除 3 个 iNaturalist 精确重复后，下游候选 manifest 为 21,497 张。

## 3. 固定产物

- adapter：`dataset/forensics/real_images.py`
- 生成与审计脚本：`scripts/data/prepare_real_images.py`
- 单元测试：`tests/test_real_images_adapter.py`
- 产物目录：`outputs/data_audits/real_images_unified_v1/`

关键文件：

```text
candidate_train_all_manifest.jsonl    # 21,500，精确去重前
candidate_train_manifest.jsonl        # 21,497，精确去重后
heldout_test_manifest.jsonl            # 1,000，全部 RAISE-1k
exact_duplicate_exclusions.json        # 3 条确定性排除记录
audit.json
summary.json
thumbnail_index.json
thumbnails/                             # 六来源各 10 张抽检图
```

复现命令：

```bash
conda run -n glamm_official python scripts/data/prepare_real_images.py \
  --datasets-root datasets \
  --output-dir outputs/data_audits/real_images_unified_v1 \
  --thumbnails-per-source 10
```

## 4. 数据版本

各来源 `selected.jsonl` SHA256：

```text
OpenImagesV7  537d0429fd220d6f0c511555f4a9dbd91f5bdb82c6b36f0a315707d859dcbbe0
PASS          2d0ebbcba34f93051e35659e404c158ea0417173bbe558ef8f2ecde26747849c
COCO2017      47d827b4e52d77cc1bf82e9cdb471a220ad5372763fc9cf35baeaa469abcfeee
FFHQ          262bf519f20fbc87029317d945d0f04398610eb095ee238258c12bedb7bc7562
iNaturalist   bd4b207a00418d54cc9eb6b52ad39acdca279f2b7f119024a4fecad210ce3e98
RAISE-1k      724537f5d34415c2fd46b0bc0c80bb198e81fc20d619176f0f79d9e9d5c06fa1
```

除 FFHQ 外，adapter 复用已验收的逐文件 SHA256。FFHQ 原 manifest 只有官方 MD5，因此生成工具为 2,500 张 FFHQ 图补算 SHA256，最终 22,500 张都具有统一的 `content_sha256`，可做跨来源精确重复检查。

## 5. 全量 adapter 审计

- manifest 映射样本：22,500；
- 缺失图片：0；
- 无法读取图像头：0；
- 文件大小与来源 manifest/checksum 不一致：0；
- sample ID 重复：0；
- 精确 SHA256 重复组：3；
- 跨来源精确重复组：0。

三组重复均来自 iNaturalist，每组两条 observation/photo 记录指向完全相同的图像内容。保留 source manifest 中先出现的样本，排除另外三条；不删除或修改原始图片。

图像尺寸范围：

- width：102–7,360，median 768；
- height：56–4,928，median 681。

格式与颜色模式：

| 项目 | 数量 |
|---|---:|
| JPEG | 19,000 |
| PNG | 2,500 |
| TIFF | 1,000 |
| RGB | 22,373 |
| 灰度 L | 124 |
| CMYK | 3 |

124 张灰度图分布在 Open Images、PASS、COCO 和 iNaturalist；3 张 CMYK 均来自 Open Images。后续统一图像流水线必须对 real/fake 一律显式转 RGB，并使用相同 resize、crop、重编码与归一化，禁止颜色模式、后缀、EXIF 和原始尺寸成为标签捷径。

## 6. 当前粗内容分布

精确去重前 candidate-train 粗标签：

| 类别 | 数量 | 来源说明 |
|---|---:|---|
| Human | 8,740 | Open Images 4,000；COCO 2,240；FFHQ 2,500 |
| Object | 2,180 | Open Images 1,500；COCO 680 |
| Animal | 2,440 | Open Images 1,000；COCO 440；iNaturalist 1,000 |
| Scene | 2,140 | Open Images 1,500；COCO 640 |
| Unknown | 6,000 | 全部 PASS |

这些只是来源 manifest 的粗标签，不是最终内容匹配结果。尤其 FFHQ 只能代表对齐人脸近景，不能代替全部 Human；PASS 必须完成视觉内容分类与人工抽检后才能参与按类别配额选择。

RAISE held-out 的关键词启发式分布为 Scene 859、Human 141，仅用于分层报告，不参与训练配额。

## 7. 尚未完成的过滤

当前 `candidate_train_manifest.jsonl` 只是精确去重后的候选池，不是最终 1:1 训练清单。正式选择与 SynthScars 11,064 个训练视觉身份匹配的 real 之前仍需：

1. PASS Human/Object/Animal/Scene 内容分类和人工抽检；
2. real 来源内部及跨来源 pHash/视觉嵌入近重复审计；
3. real 与 SynthScars 的跨类近重复检查；
4. 根据 SynthScars 内容分布确定来源上限和类别配额；
5. 固定 source-stratified train/validation manifest。

在这些步骤完成前不得把 21,497 张候选池直接交给训练 DataLoader。
