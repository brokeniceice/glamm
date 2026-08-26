# 图像取证融合文献审计

## 方法对照

| 工作 | semantic/forensic 组合方式 | 是否显式 reliability/conflict adaptive fusion | 对本项目结论 |
|---|---|---|---|
| TruFor, CVPR 2023 | RGB + Noiseprint++，cross-modal feature calibration；另输出 reliability map | **部分处理**：预测可靠性，但不是 language–forensic conflict routing | reliability map 有直接取证先例 |
| FakeShield, ICLR 2025 | domain-tag detection + text-guided SAM localization | **NOT ADDRESSED** | 有 language-guided localization，但无按两源质量动态信任机制 |
| ForgeryGPT, arXiv 2024/后续修订 | FL-Expert 先产 mask，再将 image/mask/text token送入 LLM | **NOT ADDRESSED** | 强化 forensic expert 与解释对齐，不处理两个 grounding source 冲突 |
| SIDA, CVPR 2025 | DET/SEG token 交互，SAM-style decoder | **NOT ADDRESSED** | detection knowledge帮助 SEG，但无 calibrated forensic routing |
| Propose-and-Rectify, arXiv 2025 | MLLM proposal；multi-scale forensic rectification；forensic cue 注入 SAM image embedding | **NOT ADDRESSED** | 与 Phase 4F 路径最相似，但原稿未提供 reliability/conflict gate |
| Omni-IML, ICLR 2026 | sample-specific vision vs vision+frequency modal gate；dynamic decoder filters | **处理 sample-adaptive forensic modality**，但不含 language expert | 证明 frequency/forensic 是双刃剑；其 decoder replacement 不兼容 P1 保护 |

## 关键缺口

现有 explainable IFDL 大多解决“如何把 forensic cue 给 MLLM/SAM”或“如何生成解释”，没有回答：当 MLLM language grounding 与 forensic spatial evidence 冲突时，如何用部署时可得且经校准的信号，在像素/样本层面动态调节两者。

因此本项目的潜在方法空间不是“首次加入 forensic feature”，而是：

> 在已经证明 forensic causal utilization 的 P1 grounding 系统中，显式建模 language expert 与 forensic expert 的可靠性、冲突和 Pareto controllability。

## 不可过度迁移的点

- FakeShield/Propose-and-Rectify 的整体性能不能证明其 fusion 可靠性。
- TruFor reliability map 针对自身 localization error，不等于 LLM prompt reliability。
- Omni-IML gate 服务于 image type/frequency utility，且依赖新 encoder/decoder；只可支持 sample-adaptive forensic usage 的必要性。
- 任何 GT mask、domain label 或 oracle phrase 都不能成为部署 gate 输入。

## 取证侧可复用资产

Phase 4C-A selected forensic adapter 已有 `F_forensic [B,256,24,24]` 和 `dense_head logits [B,1,24,24]`，且相对 projection-only mean FG IoU 提升 0.033160，CI 全正。未来 Primary 可把它作为冻结 forensic expert，从 P1 重新开始；不需要从 Phase 4F rectifier warm-start。

