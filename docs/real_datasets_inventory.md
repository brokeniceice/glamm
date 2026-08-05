# 统一 AIGC 取证项目：真实图像数据资产记录

## 1. 文档范围与当前状态

- 记录日期：2026-08-03（UTC）。
- 工作区：`/home/yz/groundingLMM_official`。
- 逻辑数据根目录：`/home/yz/groundingLMM_official/datasets`。
- 实际数据根目录：`/data/yz/myLISA_storage/AIGC`。
- `datasets` 是指向上述大容量存储目录的符号链接。
- 当前数据盘：约 15 TB，总体已用约 7.5 TB，可用约 6.3 TB。
- 本文只记录本轮取得的真实图像来源；训练中的 fake 仍严格限定为 SynthScars。
- 尚未启动任何训练。

当前共有六个真实来源、22,500 张落盘图像：

- 21,500 张构成原始候选训练真实池；统一 SHA256 审计排除 3 个 iNaturalist 精确重复后，下游候选池为 21,497 张；
- RAISE-1k 的 1,000 张只作为相机来源真实图 held-out 测试集，不进入第一轮训练；
- “候选训练池”不等于“已经可以直接训练”：仍需跨源近重复检查、PASS 内容分类、与 SynthScars 的语义匹配和统一预处理审计。

## 2. 总览

| 数据集 | 角色 | 图像数 | 图像字节 | 数据集目录总字节 | 主要格式 | 当前验收 |
|---|---|---:|---:|---:|---|---|
| Open Images V7 | 候选训练 | 8,000 | 2,553,399,262 | 2,605,305,679 | JPEG | 通过 |
| PASS v3.0 | 候选训练 | 6,000 | 747,247,647 | 1,061,366,846 | JPEG | 通过 |
| COCO 2017 | 候选训练，限制占比 | 4,000 | 645,347,376 | 1,736,663,642 | JPEG | 通过 |
| FFHQ | 候选训练，人物近景补充 | 2,500 | 3,426,908,344 | 3,699,467,496 | PNG | 通过 |
| iNaturalist | 候选训练，动物补充 | 1,000 | 462,937,955 | 465,162,926 | JPEG | 通过 |
| RAISE-1k | 仅 held-out real | 1,000 | 22,865,869,444 | 22,866,866,900 | TIFF | 通过 |
| **合计** |  | **22,500** | **30,701,710,028** | **32,434,833,489** |  |  |

换算后，图像本体约 28.59 GiB；连同标注、CSV、JSON、URL 清单和压缩包约 30.21 GiB。文件系统块占用由 `du` 显示约 31 GB。

所有已验收数据集都满足：

1. manifest 条数与预期数量一致；
2. 图像文件数量与 manifest 一致；
3. 没有缺失文件或 manifest 外多余图像；
4. 所有图像可由 Pillow 完整解码；
5. 没有残留 `.part` 文件；
6. `download_failures.jsonl` 为空；
7. 除 FFHQ 外均生成逐文件 SHA256；FFHQ 在 `selected.jsonl` 内保存官方逐图 MD5，并已逐张验证。

## 3. 统一目录约定

每个数据集尽量使用以下结构：

```text
datasets/<DATASET>/
├── images/                         # 实际参与后续审计或加载的图像
├── metadata/                       # 官方类别、许可、URL 或 CSV/JSON 元数据
├── manifests/
│   ├── selected.jsonl              # 固定后的样本清单和溯源信息
│   ├── selection_summary.json      # 选择规则与数量摘要
│   ├── checksums_sha256.jsonl      # 逐文件 SHA256；FFHQ 使用官方 MD5
│   └── download_failures.jsonl     # 下载失败记录，当前均为空
└── archives/                       # 需要保留的官方压缩包（若有）
```

manifest 是后续 adapter 的数据版本入口。训练代码不应重新扫描整个来源目录并随机取图，以免样本集合发生漂移。

## 4. Open Images V7

### 4.1 来源与版本

- 官方版本：Open Images V7，2022 年发布。
- 官方下载说明：<https://storage.googleapis.com/openimages/web/download_v7.html>。
- 使用来源 split：`validation`，候选池共 41,620 张。
- 采用 validation 是为了用很小的官方元数据完成定向下载，避免获取约 561 GB 的完整检测子集。
- 本项目不把 Open Images 当作下游 benchmark，因此该 split 名称只表示原始来源，不表示本项目验证集。

### 4.2 固定选择

- 固定随机种子：`20260803`。
- 总数：8,000。
- Human：4,000。
- Object：1,500。
- Animal：1,000。
- Scene：1,500。
- 候选数量：Human 7,586、Object 14,347、Animal 7,673、Scene 5,106。
- 分类依据：Open Images 人工验证的 image-level labels；优先级为 Human、Animal、Scene、Object，避免单图被重复分配。

### 4.3 许可与溯源

- 当前 8,000 张的逐图许可均为 CC BY 2.0。
- `selected.jsonl` 保存 ImageID、作者、原始 Flickr 页面、原始 URL、许可 URL、标签和下载 URL。
- 图像从 Open Images 官方 S3 对象存储按 ImageID 下载。

### 4.4 路径与校验

- 根目录：`datasets/OpenImagesV7`。
- 图像：`datasets/OpenImagesV7/images`。
- 固定清单：`datasets/OpenImagesV7/manifests/selected.jsonl`。
- 逐图 SHA256：`datasets/OpenImagesV7/manifests/checksums_sha256.jsonl`，8,000 行。
- `selected.jsonl` SHA256：`537d0429fd220d6f0c511555f4a9dbd91f5bdb82c6b36f0a315707d859dcbbe0`。
- checksum manifest SHA256：`c30f105ff2a1ddf8fcc0fc3563345e855ce3af7e7075fa1187fa704ba7181094`。

官方元数据文件 SHA256：

```text
84a4373a0efb7fd6d93fe19b0e7ceb6c1b855c233d13b9b78a9a33655c9fdce3  oidv7-class-descriptions.csv
92ddbdfceb3626e044df5e89100b24f6c22a79c1888a4bddd00a6f231d86d56a  oidv7-val-annotations-human-imagelabels.csv
ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17  validation-images-with-rotation.csv
```

### 4.5 已知风险

- 类别来自标签启发式，不等于与 SynthScars 已完成逐图语义匹配。
- 必须保留作者和许可信息，发布派生清单时满足署名要求。
- JPEG、分辨率和来源元数据不能直接成为 real 标签捷径；训练 loader 必须对所有来源应用相同图像流水线。

## 5. PASS v3.0

### 5.1 来源与版本

- 数据集：PASS（Pictures without humAns for Self-Supervision）v3.0。
- 官方页面：<https://www.robots.ox.ac.uk/~vgg/data/pass/>。
- DOI：`10.5281/zenodo.6615455`。
- 官方完整数据约 1,439,589 张；本项目没有下载 179.6 GB 的完整 tar 包。
- 本项目读取官方 `pass_urls.txt` 和 `pass_metadata.csv`，定向下载固定子集。

### 5.2 固定选择

- 固定随机种子：`20260803`。
- URL/元数据可配对候选数：1,439,588。
- 去重后固定下载：6,000 张唯一图像。
- PASS 是无标签数据集，当前 6,000 张尚未划分 Object/Animal/Scene；必须在进入训练前完成视觉内容分类和 SynthScars 内容匹配。
- PASS 官方设计排除可识别人物，适合补充非人物真实来源。

### 5.3 许可与年代

- 数据集整体为 CC BY 4.0；当前 6,000 条元数据均标记为 `Attribution License`。
- manifest 保存拍摄日期、作者昵称、许可、地理字段、官方图像标识和原始下载 URL。
- 当前样本主要来自 2006–2013 年的 Flickr/YFCC 图像，早于现代文生图普及，适合作为低 AIGC 污染风险的真实来源。

### 5.4 路径与校验

- 根目录：`datasets/PASS`。
- 图像：`datasets/PASS/images`。
- 固定清单：`datasets/PASS/manifests/selected.jsonl`。
- 逐图 SHA256：`datasets/PASS/manifests/checksums_sha256.jsonl`，6,000 行。
- `selected.jsonl` SHA256：`2d0ebbcba34f93051e35659e404c158ea0417173bbe558ef8f2ecde26747849c`。
- checksum manifest SHA256：`1a50e07202598033b245e65b561de2fc72d469264e973288efb0ef48f97ad85b`。

元数据 SHA256：

```text
8b6fde80b48326bda9da0a7c48f92146a58a847f76a1dc40603b7e4f73f5e798  pass_metadata.csv
cc692c3e7094b7e51e218c8bc2e351acdbb419e861976be10472f16ba872566b  pass_urls.txt
```

官方 Zenodo 给出的 `pass_metadata.csv` MD5 为 `0b033707ea49365a5ffdd14615825511`，本地验证一致。

### 5.5 下载过程中的重要发现

- PASS CSV 的 `hash` 是 Multimedia Commons 图像标识，不是文件内容 MD5。
- 官方 URL 清单的选择必须按标识去重，不能仅按 6,000 条记录计数。
- 最终版本已按 6,000 个唯一标识重建，数量、解码和 SHA256 均重新验收。

## 6. COCO 2017

### 6.1 来源与版本

- 官方页面：<https://cocodataset.org/#download>。
- 使用 split：`train2017`。
- 只下载官方 train/val annotations 包，再从 COCO 官方图像服务器按 ID 定向下载 4,000 张；没有下载完整 18 GB train2017 图像 ZIP。

### 6.2 固定选择

- 固定随机种子：`20260803`。
- Human：2,240。
- Object：680。
- Animal：440。
- Scene：640。
- 总数：4,000。
- Human 和 Animal 使用 COCO instance categories；Scene 使用 caption 场景词启发式；剩余无人物/动物图进入 Object。
- `selected.jsonl` 保存 COCO image ID、文件名、尺寸、实例类别、五条 caption、Flickr/COCO URL 和逐图许可信息。

### 6.3 许可分布

当前 4,000 张按 COCO 元数据记录：

- Attribution-NonCommercial-NoDerivs：1,119。
- Attribution-NonCommercial-ShareAlike：1,073。
- Attribution：688。
- Attribution-NonCommercial：563。
- Attribution-ShareAlike：354。
- Attribution-NoDerivs：186。
- No known copyright restrictions：17。

这是研究数据资产记录，不构成法律意见。任何再分发或公开模型/派生数据前都必须重新审查逐图许可，尤其是 NC、ND 和 SA 条款。

### 6.4 路径与校验

- 根目录：`datasets/COCO2017`。
- 图像：`datasets/COCO2017/images`。
- 标注：`datasets/COCO2017/annotations`。
- 固定清单：`datasets/COCO2017/manifests/selected.jsonl`。
- 逐图 SHA256：`datasets/COCO2017/manifests/checksums_sha256.jsonl`，4,000 行。
- `selected.jsonl` SHA256：`47d827b4e52d77cc1bf82e9cdb471a220ad5372763fc9cf35baeaa469abcfeee`。
- checksum manifest SHA256：`561d2f76c3676f26a0e38dd0088ee4c0a50d18a6fff0bdca3e04bd0560a44a6a`。
- 官方 annotations ZIP MD5：`f4bbac642086de4f52a3fdda2de5fa2c`。
- 本地 annotations ZIP SHA256：`113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`。

### 6.5 已知风险

- GLaMM 原始训练大量使用 COCO，COCO 图可能成为“模型熟悉度 = real”的捷径。
- COCO 在真实训练池中必须限制占比，并按来源单独报告 real recall。
- COCO 不应作为本项目的主要真实 held-out 测试来源。
- Scene 分类只是 caption 启发式，训练前仍需图像级复核或视觉嵌入再分类。

## 7. FFHQ

### 7.1 来源与版本

- 官方仓库：<https://github.com/NVlabs/ffhq-dataset>。
- 官方数据：70,000 张 1024×1024 对齐人脸 PNG。
- 本项目只从官方指定的前 60,000 张 training split 中固定选择 2,500 张。
- 固定随机种子：`20260803`。
- 未使用官方后 10,000 张 validation split。

### 7.2 选择与用途

- 用途：补充 SynthScars 中大量人脸、头部、眼睛、嘴、牙齿等人物近景内容。
- 2,500 张均为 1024×1024 RGB PNG。
- FFHQ 的对齐与居中裁剪会形成强风格特征，因此不能占人物真实样本的主体。
- FFHQ 官方明确说明该数据集不用于开发或改进人脸识别技术；本项目只用于合成图真实性判断研究。

### 7.3 许可分布

- Attribution：1,272。
- Attribution-NonCommercial：992。
- Public Domain Mark：177。
- CC0：48。
- United States Government Work：11。
- 数据集元数据和工具本身采用 CC BY-NC-SA 4.0；逐图许可和作者保存在 manifest 中。

### 7.4 路径与校验

- 根目录：`datasets/FFHQ`。
- 图像：`datasets/FFHQ/images`。
- 官方元数据：`datasets/FFHQ/metadata/ffhq-dataset-v2.json`。
- 固定清单：`datasets/FFHQ/manifests/selected.jsonl`。
- `selected.jsonl` 保存 2,500 个互异的官方 `file_md5`，所有本地图像均逐张验证大小、MD5 和 PNG 解码。
- `selected.jsonl` SHA256：`262bf519f20fbc87029317d945d0f04398610eb095ee238258c12bedb7bc7562`。
- 官方 metadata MD5：`425ae20f06a4da1d4dc0f46d40ba5fd6`，本地一致。
- 本地 metadata SHA256：`1be5c2d1a78196d45a9600558c04b46dffefa217b7c664d5be38ed28a4d9a0ee`。
- 官方下载脚本 SHA256：`07313017632ee48e90166d94db93839ad6a5e4c0ba10cbe0327bdad2063579f9`。

### 7.5 下载兼容性记录

- NVIDIA 原始下载脚本不能解析 Google Drive 当前的病毒扫描确认页。
- 本项目仍从同一官方 Google Drive 文件 ID 获取 metadata 和逐图文件；没有改用第三方图像镜像。
- 每张图均用官方 metadata 中的文件大小和 MD5 验证，因此下载兼容处理没有改变数据内容。

## 8. iNaturalist

### 8.1 来源与筛选约束

- 官方 API：<https://api.inaturalist.org/v1/observations>。
- 只选择 `Research Grade`、带照片的观察。
- 观察日期上限：2021-12-31。
- 允许的逐图许可：CC0、CC BY、CC BY-NC。
- 使用官方 iNaturalist Open Data S3 的 `large` 图像版本。
- 通过日期上限降低现代 AIGC 图混入风险。

### 8.2 固定清单

- Mammal：350。
- Bird：300。
- Reptile：100。
- Amphibian：100。
- Ray-finned fish：100。
- Insect：50。
- 总数：1,000。
- 清单创建后默认复用，不在重跑下载脚本时重新请求随机样本。
- 当前采用 API 按 ID 固定分页；最初的 `order_by=random` 因代理缓存会重复返回同一页，已弃用。

### 8.3 许可分布

- CC BY-NC：968。
- CC BY：17。
- CC0：15。
- manifest 保存 observation ID、photo ID、观察日期、taxon、观察者、署名、许可、观察页面和下载 URL。

### 8.4 路径与校验

- 根目录：`datasets/iNaturalist`。
- 图像：`datasets/iNaturalist/images`。
- 固定清单：`datasets/iNaturalist/manifests/selected.jsonl`。
- 逐图 SHA256：`datasets/iNaturalist/manifests/checksums_sha256.jsonl`，1,000 行。
- `selected.jsonl` SHA256：`bd4b207a00418d54cc9eb6b52ad39acdca279f2b7f119024a4fecad210ce3e98`。
- checksum manifest SHA256：`f02dfc6a1aea82f0e6d9649dfc9779f7bcbf58d58c530760c27c016011db24ac`。

### 8.5 已知风险

- iNaturalist 偏向野生动物、物种观察和居中主体，与 SynthScars 中宠物或幻想动物的内容分布不完全相同。
- 绝大多数图像为 CC BY-NC；项目若改变为商业用途必须重新筛选。
- 进入训练前仍需与 Open Images、PASS 和 COCO 的动物子集做 pHash/视觉嵌入去重。

## 9. RAISE-1k

### 9.1 来源与原始下载形式

- 官方页面：<https://loki.disi.unitn.it/RAISE/download.html>。
- 许可范围：非商业研究和教育用途，发表时需引用 RAISE 论文。
- 用户从官方页面取得的 `RAISE_1k.csv.zip` 不是图像包，而是 1,000 张图像的 CSV 下载索引和相机元数据。
- CSV 每行同时提供 Nikon `.NEF` RAW 和官方 `.TIF` URL。
- 本项目没有安装 RAW 解码依赖，也没有下载 NEF；选择下载可由当前环境直接读取的官方 TIFF。

### 9.2 角色边界

- RAISE-1k 只用于相机来源真实图 held-out 测试。
- 不进入第一轮真实训练池。
- 原因：若用高分辨率相机 TIFF 作为训练 real，而 fake 主要是 512×512 PNG，会产生严重的传感器、分辨率和编码格式捷径。
- 后续评估时仍需通过与其他来源相同的 loader 预处理输入模型，但保留原始 TIFF 作为可追溯资产。

### 9.3 数据统计

- 图像数：1,000。
- 图像本体：22,865,869,444 字节，约 21.30 GiB。
- 全部为 RGB TIFF。
- Nikon D7000：729。
- Nikon D90：263。
- Nikon D40：8。
- 分辨率：
  - 4928×3264：571；
  - 4288×2848：177；
  - 3264×4928：158；
  - 2848×4288：86；
  - 3008×2000：6；
  - 2000×3008：2。
- CSV 关键词是多标签：Outdoor 842、Landscape 300、Buildings 292、Indoor 158、People 141、Nature 131、Objects 128。

### 9.4 路径与校验

- 原 CSV ZIP：`datasets/RAISE_1k.csv.zip`。
- 数据根目录：`datasets/RAISE-1k`。
- 解压 CSV：`datasets/RAISE-1k/metadata/RAISE_1k.csv`。
- TIFF：`datasets/RAISE-1k/images`。
- 固定清单：`datasets/RAISE-1k/manifests/selected.jsonl`。
- 逐图 SHA256：`datasets/RAISE-1k/manifests/checksums_sha256.jsonl`，1,000 行。
- `selected.jsonl` 同时保留每张图的 NEF URL，便于未来按相同 ID 补取 RAW。
- `selected.jsonl` SHA256：`724537f5d34415c2fd46b0bc0c80bb198e81fc20d619176f0f79d9e9d5c06fa1`。
- checksum manifest SHA256：`169871b264e9dac6131ae81fae4339713b25153815fe2af2b9ad95a5d9c3b964`。
- 原 CSV ZIP SHA256：`2bf21449ed458502c09407cd9260b5d91fe27e7bc9e98f33bd108bbd8e40f8dc`。
- 解压 CSV SHA256：`b388d84a73b17f3c1790a55e28777c6b9bed2806b2a86b9fc05ed28b863cc8c7`。

### 9.5 adapter 注意事项

- 文件后缀为大写 `.TIF`；文件扫描器必须显式支持 `.TIF/.TIFF/.tif/.tiff`。
- 图像尺寸很大，不应在训练 loop 中反复全分辨率解码后才做无缓存缩放。
- 若创建模型输入缓存，缓存只能是派生资产；不能替代或删除原 TIFF。
- RAISE 原图不能与 fake 使用不同的规范化逻辑或单独保留 EXIF 特征给分类头。

## 10. 可复现下载脚本

本轮新增脚本：

```text
scripts/data/download_openimages_v7_subset.py
scripts/data/download_pass_subset.py
scripts/data/download_coco2017_subset.py
scripts/data/download_ffhq_subset.py
scripts/data/download_inaturalist_subset.py
scripts/data/download_raise1k_tiff.py
```

这些脚本只负责固定选择、下载、解码验证和生成 manifest/checksum；没有启动训练，也没有加载 NPR。

重跑注意：

- 已存在且通过验证的图像会跳过；
- 下载使用临时 `.part` 文件，成功后才原子替换为正式文件；
- 不应在缺少 manifest 的情况下直接更换随机种子；
- iNaturalist 默认复用现有 manifest，只有显式 `--refresh-selection` 才重新请求；
- FFHQ manifest 内的官方逐图 MD5 是内容身份的一部分；
- RAISE 默认只下载 TIFF，不下载 NEF。

## 11. 进入训练前仍必须完成的工作

### 11.1 内容匹配

- 按 SynthScars 的 Human/Object/Animal/Scene 分布构造最终 real 采样池。
- PASS 尚无内容标签，需要视觉模型分类后人工抽查。
- Open Images 与 COCO 的粗分类需要用图像嵌入和抽样审计验证。
- FFHQ 只能补充人脸近景，不能代表全部 Human 内容。

### 11.2 去重与泄漏审计

- 每个来源内部做 SHA256、pHash 和视觉嵌入近重复检查。
- 六个 real 来源之间做跨源近重复检查。
- real 与 SynthScars 做跨类近重复检查。
- COCO 需要额外考虑 GLaMM 原训练数据暴露，不用于主要 held-out 指标。

### 11.3 来源捷径控制

- real/fake 使用相同的解码、颜色转换、resize、crop、重编码和增强逻辑。
- 不允许根据 `.jpg/.png/.TIF` 后缀、原始尺寸、EXIF 是否存在或数据集路径推断标签。
- 正式训练前应决定是否生成统一派生缓存，并保存缓存构造参数和版本哈希。
- 必须按来源分别报告 real recall/false-positive rate。

### 11.4 许可与发布

- 保留逐图作者、原始页面和许可信息。
- 对 CC BY 数据准备署名清单。
- 对 NC/ND/SA 图像，在发布模型、派生数据或公开样例前单独做许可复核。
- FFHQ 不用于人脸识别；RAISE 仅限非商业研究和教育用途。

### 11.5 划分原则

- SynthScars 官方 test 不参与训练。
- RAISE-1k 全部保持 held-out。
- 最终 real train/val/test 不能仅做全池随机拆分，应保留 source-held-out 或至少 source-stratified 评估。
- 任何划分文件都应进入 manifest 并固定哈希，不能在 DataLoader 启动时临时随机生成。

## 12. 与 SynthScars fake 数据的关系

- fake 训练样本仍只来自 SynthScars。
- SynthScars train 有 11,236 条 annotation、11,182 个唯一文件名；合并 118 对同 stem 的 PNG/JPG 尺寸/编码变体后，共 11,064 个唯一训练视觉身份。
- 原始候选训练 real 有 21,500 张；精确去重后的下游候选池有 21,497 张，约为 SynthScars 唯一 train fake 的 1.92 倍。
- 训练 epoch 不应直接使用全部 real 造成类别失衡；计划是在候选池中按内容和来源平衡选择 11,064 张 real，与 fake 视觉身份做 1:1 基线。
- real 输出遵循 `[CLS] [REAL] No reliable synthetic artifact is found. [SEG]`，并监督全零 mask。
- RAISE-1k 不计入上述 21,500 张候选训练 real。

## 13. 当前结论

数据下载、文件级完整性、统一 real adapter、统一 SHA256 和精确重复审计已经完成，详见 `docs/real_images_unified_adapter.md`。下一步继续完成内容匹配与近重复过滤，而不是直接启动训练：

1. 为 PASS 生成内容标签并抽检；
2. 跨源 pHash/嵌入近重复审计；
3. 依据 SynthScars 内容分布建立最终 real train/val manifest；
4. 设计统一解码与格式无关预处理；
5. 仅在这些版本固定后进入小规模平衡过拟合测试。
