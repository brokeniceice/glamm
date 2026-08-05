# NPR+SRM 结构说明

旧项目实验记录位于只读文件：
`/home/yz/myLISA_old/stage1_upgrade_record.md`。

旧实验最终选择“官方 NPR + SRM”作为后续专家网络改造的起点。因此，本工作区保留：

1. 官方 NPR 的邻近像素关系残差输入；
2. 截止到 ResNet 第二个残差阶段的分类主干；
3. 三组固定 SRM 高通滤波核；
4. 可学习的 SRM 卷积分支和残差门控；
5. 单 logit 的真假二分类形式。

本次迁移明确移除了旧 LISA 融合所需的 `16×16` 空间特征接口，并将 NPR 和 SRM
分支都恢复为全局平均池化后的图像级分类特征。Patch-MIL、patch pooling、旧多分类头、
CLIP 特征对齐和交叉注意力均不属于当前实现。

旧实验权重基于带有 LISA 适配接口的历史代码训练，本工作区不再沿用。接下来应以当前
结构重新训练，并把新权重保存在本地 `checkpoints_stage1/` 目录。
# 官方 FOCAL 零训练图像级评估

环境新增依赖：`torch-kmeans==0.2.0`，与 FOCAL 官方 README 指定版本一致；安装时使用
`pip install --no-deps torch-kmeans==0.2.0`，没有升级 torch 或其他已有依赖。

`python -m npr_expert.eval_focal_zeroshot` 不训练模型。它严格加载作者 ViT-L 权重，使用
OpenCV 拉伸到 `1024×1024`、除以 255、逐像素 L2 normalize、cosine KMeans(K=2)，
并按官方规则把较小簇作为预测篡改掩码。二分类分数包括预测面积、簇中心分离度以及二者
乘积；阈值只在 AIGI val 上校准，然后固定用于 AIGI test 和 LOKI。
