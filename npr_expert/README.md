# 独立 NPR+SRM+FOCAL 专家网络

本目录保存从旧项目迁出的图像真假检测专家网络。它不接入 GLaMM、CLIP、tokenizer、
投影层或交叉注意力。可选 FOCAL 分支仅复用 SAM `ImageEncoderViT` 结构并加载作者的
ViT-L 取证权重，不使用 GLaMM 的 SAM ViT-H 权重、prompt encoder 或 mask decoder。

## 当前结构

- NPR 分支严格采用官方 NPR 的残差生成方式和分类主干。
- NPR 主干在第二个残差阶段后直接进行 `1×1` 全局平均池化。
- SRM 分支使用三组固定高通滤波核、一个可学习卷积分支和一个可学习门控系数。
- NPR 与 SRM 的全局特征先相加；可选的冻结 FOCAL ViT-L 分支输出
  `256×64×64` 空间特征，经 mean+max pooling 得到 512 维特征后通过门控融合。
- 融合后的 512 维特征通过 `Linear(512, 1)` 输出真假分类 logit。
- 正 logit 表示伪造图像，训练损失应使用 `BCEWithLogitsLoss`。
- 模型不再生成或返回 `[B, 512, 16, 16]` 空间特征。

其中，SRM 是本项目在官方 NPR 基础上保留的专家增强分支；`16×16` 空间池化和
空间特征返回接口属于旧 LISA 融合适配，已经移除。

## 基本调用

```python
from PIL import Image

from npr_expert import NPRExpert, build_npr_transform

model = NPRExpert()
transform = build_npr_transform()
image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
output = model(image)

output.logits         # [B, 1]，正值倾向伪造类别
output.probabilities  # [B, 2]，依次为真实、伪造概率
output.predictions    # [B]，0 表示真实，1 表示伪造
```

模型默认处于可训练状态。只有明确需要冻结专家网络时，才调用 `model.freeze()`。

## 权重约定

后续重新训练得到的轻量权重保存在项目本地的 `checkpoints_stage1/` 目录。加载器支持
直接保存的 `state_dict`，也支持以 `{"model": state_dict}` 形式保存的训练检查点：

```python
model = NPRExpert.from_checkpoint("checkpoints_stage1/model_epoch_best.pth")
```

旧 LISA 阶段的分类器、旧多分类头以及旧实验权重不再兼容，也不会继续保留。

## 训练

训练入口为 `python -m npr_expert.train`。默认路径已经对应当前工作区的 `datasets`
软链接，训练结果写入本地 `checkpoints_stage1`：

```bash
source /home/yz/miniconda3/etc/profile.d/conda.sh
conda activate glamm_official

python -m npr_expert.train \
  --device cuda:0 \
  --batch-size 64 \
  --epochs 32 \
  --focal-weights /data/yz/myLISA_storage/checkpoints/FOCAL/FOCAL_ViT_weights.pth
```

训练入口默认启用并冻结 FOCAL。其 1024 输入保留在 CPU，由 extractor 按
`--focal-micro-batch-size` 分块送入 GPU；`--no-focal` 可复现原 NPR+SRM 结构。
冻结的 FOCAL 参数不会重复写入 `best.pth` 和 `last.pth`。

若要测量冻结 FOCAL 全局特征本身的二分类能力，可使用纯 FOCAL 模式。该模式完全
跳过 NPR 和 SRM，仅训练 `LayerNorm(512)` 与 `Linear(512, 1)`；FOCAL ViT-L
仍保持冻结。配合预计算缓存可避免每次训练重新提取特征：

```bash
python -m npr_expert.train \
  --device cuda:0 \
  --batch-size 64 \
  --epochs 32 \
  --focal-only \
  --focal-cache /data/yz/myLISA_storage/checkpoints/FOCAL/focal_vit_l_mean_max_all.pt
```

纯 FOCAL 实验不能使用 NPR+SRM 的 `--init-checkpoint`，以免把续训收益误认为
FOCAL 特征收益。

正式运行前应使用 `nvidia-smi` 选择空闲显存充足的 GPU。每次运行会建立带时间戳的
子目录，并生成：

- `training_config.json`：本次完整参数；
- `history.jsonl`：逐 epoch 的训练和验证指标；
- `last.pth`：每个 epoch 更新，可用于恢复训练；
- `best.pth`：验证集 AP 最优权重；
- `evaluation_before_training.json`：随机初始化模型的训练前测试结果；
- `evaluation.json`：训练前、最佳权重训练后以及 AIGI 按来源评估的汇总结果。

默认会在训练前评估一次 AIGI test、DeepfakeJudge、FakeBench 和 LOKI；训练结束加载
`best.pth` 后，再在同一批数据集上评估一次。训练后还会默认执行 AIGI test 按生成来源
评估。可以通过 `--no-evaluate-test`、`--no-evaluate-external` 或
`--no-eval-by-source` 分别关闭。恢复当前结构的训练断点时，训练器会继续使用该断点
所在的原运行目录，并复用已经保存的训练前基线：

```bash
python -m npr_expert.train \
  --device cuda:0 \
  --epochs 32 \
  --resume checkpoints_stage1/某次运行/last.pth
```

`--max-train-samples` 和 `--max-val-samples` 只用于小规模调试，正式训练应保持默认值 0。
