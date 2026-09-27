#!/usr/bin/env python3
"""Publish completed NPR classification results without changing localization tables."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/npr_official_retrain"
REPORT = ROOT / "docs/final_evaluation_report.md"
DETAIL = ROOT / "docs/npr_official_retrain_results.md"
DATASETS = (
    ("validation", "Internal validation", 2212),
    ("internal_test", "Internal2208", 2208),
    ("aigi_holmes", "AIGI-Holmes TestSet", 99999),
    ("genimage", "GenImage held-out", 100000),
    ("loki", "LOKI classification", 2217),
    ("raise998", "RAISE998", 998),
)


def value(x):
    return "-" if x is None else f"{x:.6f}"


def result_row(metrics):
    return (f"| NPR-official-retrained | {metrics['n']} | {value(metrics['accuracy'])} | "
            f"{value(metrics['f1'] if metrics['fake'] else None)} | "
            f"{value(metrics.get('roc_auc'))} | "
            f"{value(metrics['fake_recall'] if metrics['fake'] else None)} | "
            f"{value(metrics['tnr'])} | {value(metrics['fpr'])} |")


def main():
    final = json.loads((OUT / "results.json").read_text())
    if final.get("status") != "COMPLETE":
        raise RuntimeError("NPR aggregate result is not complete")
    protocol = json.loads((OUT / "protocol.json").read_text())
    expected = protocol["counts"]
    for key, _, n in DATASETS:
        item = json.loads((OUT / "evaluation" / key / "results.json").read_text())
        with (OUT / "evaluation" / key / "predictions.jsonl").open() as stream:
            ids = [json.loads(line)["sample_id"] for line in stream if line.strip()]
        if item.get("status") != "COMPLETE" or item["n"] != n or len(ids) != n:
            raise RuntimeError(f"Incomplete dataset: {key}")
        if len(set(ids)) != n or expected[key] != n:
            raise RuntimeError(f"Duplicate IDs or protocol drift: {key}")
        if item["checkpoint_sha256"] != final["checkpoint"]["sha256"]:
            raise RuntimeError(f"Checkpoint mismatch: {key}")

    result = final["results"]
    lines = [
        "# NPR 官方架构：本项目数据重训练与分类评测",
        "",
        "状态：**COMPLETE**。使用官方 NPR 源码的随机初始化分类网络，"
        "在本项目内部训练清单上单独重训 50 轮；固定第 50 轮权重后完成内部测试和四个外部 OOD 数据集。",
        "",
        "## 来源与协议",
        "",
        "- 官方论文：[Tan et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html)；"
        "[官方代码仓库](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)，固定 commit `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`。",
        "- 直接导入官方 `networks.resnet.resnet50(pretrained=False, num_classes=1)`："
        "nearest-neighbor 下/上采样差分作为输入，截断 ResNet50 的 layer1/layer2 后全局池化和单 logit head。"
        "训练从随机权重开始，未载入作者发布的 NPR 权重、ImageNet 权重或项目历史 NPR+SRM 联合权重。",
        "- 内部训练 17,672 张（Real/Fake 各 8,836）；内部验证 2,212 张；"
        "训练 batch 32、Adam 学习率 `2e-4`、BCEWithLogitsLoss、50 轮，按官方代码每 10 轮衰减 0.9。"
        "双卡 DDP 每卡 batch 16、同步 BatchNorm，保持全局 batch 32；此执行方式与官方单卡脚本有浮点/批统计差异。",
        "- 训练图像变换：Resize 到 256×256、RandomCrop 224、随机水平翻转、ImageNet 归一化；"
        "验证和测试：Resize 到 256×256，不裁剪、不翻转、相同归一化。"
        "这些是官方数据代码中的有效变换；项目 adapter 只负责按既定 manifest 读取和记录。",
        "- 仅第 50 轮 checkpoint 用于结果；Fake 为正类，官方 `sigmoid > 0.5` 决策。"
        "内部测试和 OOD 没有参与选择 checkpoint、阈值或超参数。"
        "官方训练脚本每轮调用其测试集；本实验为保持测试隔离，省略该训练中测试调用。",
        "- 执行在第 6 轮结束的完整 checkpoint 处恢复为独立后台进程；模型和 Adam 状态接续，"
        "从第 7 轮起随机裁剪/翻转序列与不中断训练可能不同。50 轮预算、数据清单及选模规则未变。",
        "- [冻结协议与 manifest SHA256](../outputs/npr_official_retrain/protocol.json)、"
        "[checkpoint 身份](../outputs/npr_official_retrain/checkpoint_identity.json)、"
        "[逐图结果目录](../outputs/npr_official_retrain/evaluation/)、"
        "[总结果](../outputs/npr_official_retrain/results.json)。",
        "",
        "## 分类结果",
        "",
        "| 数据集 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label, n in DATASETS:
        m = result[key]
        lines.append(f"| {label} | {n} | {value(m['accuracy'])} | "
                     f"{value(m['f1'] if m['fake'] else None)} | {value(m.get('roc_auc'))} | "
                     f"{value(m['fake_recall'] if m['fake'] else None)} | "
                     f"{value(m['tnr'])} | {value(m['fpr'])} |")
    lines.extend(["", "RAISE998 只有 Real；F1、ROC-AUC、Fake recall 不适用。", "",
                  "### GenImage 分生成器", "",
                  "| 生成器 | N | Accuracy | ROC-AUC | Fake recall | FPR |",
                  "|---|---:|---:|---:|---:|---:|"])
    gen = json.loads((OUT / "evaluation/genimage/results.json").read_text())["per_generator"]
    for key, m in sorted(gen.items()):
        lines.append(f"| {key} | {m['n']} | {value(m['accuracy'])} | "
                     f"{value(m.get('roc_auc'))} | {value(m['fake_recall'])} | {value(m['fpr'])} |")
    lines.extend(["", "## 解释边界", "",
                  "- 这是**单独重训练的 NPR 检测器**；项目历史 NPR+SRM 的共享分类头只能作特征/分支诊断，不能替代此基线。",
                  "- AIGI-Holmes TestSet 与原内部训练清单已有 779 个文件 SHA256 完全重叠；"
                  "NPR 使用全部原训练图，这 779 个重叠会影响该集的独立 OOD 解释。评测保留完整 99,999 张以与其他模型同清单对比。",
                  "- NPR 仅作真假分类，不输出定位 mask。其随机初始化、CNN 取证输入和 256 像素测试变换"
                  "与 CLIP/RINE/C1 不同，跨模型性能差异不是单因素消融。",
                  ""])
    DETAIL.write_text("\n".join(lines))

    report = REPORT.read_text()
    if "| NPR-official-retrained |" in report:
        raise RuntimeError("Report already contains NPR rows; refusing duplicate insertion")
    report = report.replace("分类展示八个模型", "分类展示九个模型", 1)
    report = report.replace(
        "- **RINE-336-adapted** 指既有 `RINE-on-C1` 独立分类器：",
        "- **NPR-official-retrained** 指独立 NPR 基线：直接使用固定官方 NPR ResNet 源码，"
        "在本项目 17,672 张内部训练图上随机初始化、训练 50 轮，固定最后一轮权重；仅有分类输出。\n"
        "- **RINE-336-adapted** 指既有 `RINE-on-C1` 独立分类器：",
        1,
    )
    report = report.replace("C1-raw、公开 LE 第二阶段和两个 RINE 行只用于分类比较。",
                            "C1-raw、NPR、公开 LE 第二阶段和两个 RINE 行只用于分类比较。", 1)
    labels = ("Internal2208（", "AIGI-Holmes TestSet（", "GenImage held-out（",
              "LOKI 分类（", "RAISE998（")
    keys = ("internal_test", "aigi_holmes", "genimage", "loki", "raise998")
    for label, key in zip(labels, keys):
        start = report.index("### " + label)
        next_section = report.find("\n### ", start + 4)
        if next_section < 0:
            next_section = report.find("\n## ", start + 4)
        segment = report[start:next_section]
        anchor = next(line for line in segment.splitlines() if line.startswith("| RINE-336-adapted |"))
        replacement = anchor + "\n" + result_row(result[key])
        report = report[:start] + segment.replace(anchor, replacement, 1) + report[next_section:]
    report = report.replace(
        "- 新增分类结果来源：[公开 LE + aligned Stage-2 汇总]",
        "- NPR 分类结果来源：[NPR 总结果](../outputs/npr_official_retrain/results.json)与"
        "[NPR 重训练记录](npr_official_retrain_results.md)。\n"
        "- 新增分类结果来源：[公开 LE + aligned Stage-2 汇总]",
        1,
    )
    REPORT.write_text(report)
    print(json.dumps({"status": "PUBLISHED", "detail": str(DETAIL), "report": str(REPORT)}))


if __name__ == "__main__":
    main()
