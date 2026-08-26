# Phase 4E-1 — 正式执行前 G0 hidden 可用性阻塞审计

## 结论

Phase 4E-1 尚未发生任何 formal optimizer update。Stage S 的冻结协议存在一个在 Phase 4E-0.5 未被显式定义的边界条件：canonical P1 G0 并非对每张 Fake 图像都生成 `[SEG]`，因而部分样本不存在协议要求的 `[SEG]` 前一 causal hidden `h_G0`。

| Population | 总数 | 合法 `h_G0` | 无 `[SEG]` / 无 `h_G0` |
|---|---:|---:|---:|
| train Fake | 8,836 | 8,690 | 146 |
| validation Fake | 1,106 | 1,078 | 28 |

train 的 146 个无效项全部满足 `seg_count=0` 且 `usable_seg_predictor_position=None`。validation 数据来自先前 fresh canonical P1 batch=1 G0 cache；其 28 个无效项同样没有合法 `q_seg`，因此也不可能回溯出冻结定义中的 raw 4096D `h_G0`。

## 冲突

冻结协议同时规定：

1. Stage S 使用全部 8,836 train Fake、每 epoch 1,105 optimizer steps；
2. student input 必须是 canonical autonomous `h_G0`；
3. `h_G0` 必须是 `[SEG]` 前一 causal state；
4. student 不得接收 TF token、authoritative phrase 或 GT mask 作为输入；
5. 不得在运行中修改 architecture 或 objective。

因此，以下任何自动处理都会新增未注册实验规则：

- 用零向量或 learned sentinel 替代 `h_G0`；
- 使用最后一个 generated token hidden；
- 向 autonomous trajectory 补写 `[SEG]` 后提取 hidden；
- 用 `h_TF` 替代；
- 从 population 中删除 146 个样本；
- 保留样本但跳过其 mask/KD loss。

## 建议的最小协议修正

建议显式冻结为 **valid-G0 conditional optimization**：

- epoch iterator 和 sample order 仍覆盖 8,836 Fake，仍为 1,105 steps/epoch；
- 仅 8,690 个存在 canonical `h_G0` 的样本贡献 Stage S mask/KD loss；
- 146 个无 `[SEG]` 样本不构造 surrogate hidden，不贡献 Stage S loss；
- 报告 nominal exposures 与 effective valid-G0 supervised exposures 两套计数；
- selector 与最终 canonical G0 evaluation 仍覆盖全部 1,106 validation Fake，无 `[SEG]` 的 28 项按 canonical deployable failure 记为 zero localization，不从指标中删除；
- Phrase-only/TF-full 继续按各自合法 trajectory 覆盖全部 1,106，仅作 oracle analysis。

该规则不引入伪造 hidden，也不以 oracle 信息修复 autonomous failure，但它会把 Stage S 的 estimand 明确改为“conditional on canonical SEG emission”，必须由用户明确批准，不能由执行者默认为冻结协议的一部分。

## 当前停止边界

本次属于 mandatory pre-execution audit 发现的 protocol/implementation blocker。Stage T、Stage S、validation selector、qualification、ablation 均未启动；internal test 与 official1000 未访问。

## 后续授权记录（2026-08-25）

用户已明确批准上述 **valid-G0 conditional optimization**，并进一步冻结：batch loss 仅按 `n_valid_g0` 归一化；all-invalid batch 不 backward、不 optimizer step、不推进 scheduler；所有 Stage-S arms 共用相同 8,690/1,078 valid ID sets；正式 validation headline 始终保留全部 1,106 样本及其中 28 个 canonical `[SEG]` failure。

因此本文件前述“当前停止边界”保留为发现问题时的历史事实，不再表示当前执行被阻塞。恢复执行后的机器可读协议见 `outputs/phase4e1_tf_fdg_full_method/preflight/approved_protocol_amendment.json`。
