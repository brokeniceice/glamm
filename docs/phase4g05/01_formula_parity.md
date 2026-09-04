# Phase 4G-0.5 公式 parity

## 结论

`FORMULA_PARITY = PASS`。

项目实现分别对齐了 [TMC 官方实现](https://github.com/Han-Zongbo/TMC/tree/a3272b8746861c76a3461943b5eee51df5b5a8fe) 与 [ECoLaF 官方实现](https://github.com/deregnaucourtlucas/ECoLaF/tree/543cdf6c2cfe390f6d3e39ef01fbc66705781f08)。10 类确定性输入上的 ECoLaF discounted mass、conflict、discount、fused mass 与 DSmP probability 均与官方代码逐元素完全一致，最大绝对误差为 0。TMC two-view DS 输出也逐元素一致；vacuous identity 的 float32 最大误差为 `1.9073486328125e-6`。

机器可读结果见 `outputs/phase4g05/formula_parity.json`。

## 必须纠正的接口混写

ECoLaF 官方单模态网络输出的是 `K+1` 个 logits，经 softmax 后直接解释为 `K` 个 singleton mass 与一个 `Ω` mass；它不是 Dirichlet evidential head。TMC 才定义：

```text
e_k >= 0
alpha_k = e_k + 1
S = sum_k alpha_k
b_k = e_k / S
u = K / S
```

PCERF 因而冻结为：TMC-style head 产生 `(b,u)`，将 `[b_0,b_1,u]` 作为 ECoLaF-compatible mass；ECoLaF 官方 kernel 负责 dense conflict discount、Dempster fusion 与 DSmP。两篇论文的职责不再混写。

## ECoLaF parity 边界

对 `M` 个 expert，输入为 `[B,K+1,M,H,W]`。官方 kernel 固定：

- Jousselme-derived pairwise distance；
- conflict scale `1-(2K+1)/(K+1)^2`；二分类时为 `4/9`；
- `lambda=2`，discount 为 `(1-c_m^2)^(1/2)`；
- singleton mass 乘 discount，损失的 mass 全部回收到 `Ω`；
- scalable Dempster 在 log space 计算，内部 `epsilon=1e-10`；
- DSmP `epsilon=1e-4`；
- fusion 强制 float32；输入 BF16 先转 float32；
- ECoLaF kernel 本身不含 validation-selected clamp、temperature 或 gamma。

测试覆盖 uniform、one-sided、conflict、单侧 vacuous、双侧 vacuous、高置信 agreement、高置信 conflict、random tensor 与 extreme logits。全部 finite。

## Empty/vacuous 与 full conflict

ECoLaF released kernel假定固定数量 modality；它会把 vacuous source 也纳入 pairwise conflict，因此不能单独保证项目所需的 P1 logit identity。PCERF 不改写 ECoLaF 公式，而是在 kernel 外冻结 source-availability dispatch：empty/vacuous/off forensic source 是 DS neutral element，直接返回原 `z_L`。有两个 active source 时才进入官方 kernel。

高置信完全相反的 evidence 通过 ECoLaF log-space scalable fusion 保持 finite；不采用 TMC 原始 `1-C` 分母作为 PCERF dense production kernel。

## Annealing 与 regularization

TMC source loss 的公式冻结为 expected cross-entropy 加 incorrect-class evidence 的 KL-to-uniform：

```text
lambda_t = min(1, global_step / annealing_step)
alpha_tilde = (alpha - 1) * (1-y) + 1
L_source = sum_k y_k [digamma(S)-digamma(alpha_k)]
           + lambda_t * KL(Dir(alpha_tilde) || Dir(1))
```

`annealing_step` 只允许由未来人工授权的 G1-C 总预算机械定义为前 10% optimizer steps，不得按 validation IoU 选择。Phase 4G-0.5 没有 optimizer update，也没有冻结 final-training loss 权重；后者属于未来正式训练协议，不影响 inference formula parity。

QMF 在本阶段仅冻结为 reliability validity criterion：future `weight_m` 应与对应 source loss 负相关；其 [官方实现](https://github.com/QingyangZhang/QMF/tree/fe6c4c6ef7cb23f0a89594ee413d485f1854268b) 的 energy weight/ranking loss不是 PCERF 的 DS fusion kernel。

