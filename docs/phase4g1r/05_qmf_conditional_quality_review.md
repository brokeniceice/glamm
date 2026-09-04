# QMF conditional-quality review

Primary source：[Provable Dynamic Fusion for Low-Quality Multimodal Data, ICML 2023](https://proceedings.mlr.press/v202/zhang23ar.html)。

QMF 的 theorem 在二分类 decision-level late fusion 下分解 generalization bound，其中含 `Cov(weight_m, loss_m)`；当 dynamic weight 的均值条件与 static comparator 对齐、且 weight 与 unimodal loss 的相关不为正时，dynamic fusion 可得到更紧的 bound。论文以 uncertainty estimator 实现 quality weight，但它显式把“uncertainty 与 source loss 正相关”作为前提，而不是无条件真理。

## 对 G1-C 的解释

- Forensic QMF relation PASS：当前 effective weight 与 forensic NLL 的负相关存在，说明“质量相关动态权重”方向有部分证据。
- Forensic uncertainty-error FAIL：实现该权重的 estimator 不能继续只来自 forensic intrinsic uncertainty。
- Language QMF INCONCLUSIVE：不能声称两源都已达到 QMF 条件。

## 可迁移为 future validity criterion

Primary 不是复刻 QMF energy uncertainty，而是学习 `U_F(L,F)`。未来必须同时检查：

\[
\operatorname{corr}(U_F,\ell_F-\ell_L)<0
\]

或等价地 `corr(U_F, loss_L-loss_F)>0`，并继续检查最终 forensic effective weight 与 calibrated source NLL 的负相关。相关性必须用预冻结 target/unit、image-cluster bootstrap 与 matched/corruption populations；不能用单个 p-value，也不能事后改 weight 定义。

QMF 的理论设定是 sample-level classification，迁移到 spatial localization 是 ADAPTED evidence。它支持 validity condition，不证明某个 pixel loss、BCE/Dice target 或 utility architecture 必然正确。
