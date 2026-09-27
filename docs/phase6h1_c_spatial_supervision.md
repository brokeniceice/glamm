# Phase 6H.1 — Correction Spatial Supervision

状态：**STOP**。Phase6H.1 的 A0/A1 匹配训练、selector 与因果比较只使用 internal validation；后续用户单独授权 A1 Official1000 固定评测，见 F 节。未访问 internal test 或 OOD，也未进入 Phase6H.2。

## A. 复现完整性

- 首次 A0/A1 作业随交互会话结束而中断在 epoch 2，未用于任何指标判定；其 epoch 1 检查点与日志保存在 `outputs/phase6h/interrupted_20260924T0257Z/`。当前 A0/A1 从相同初始状态完整重跑，由独立的 systemd 用户服务运行。
- C1 SHA256：`85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff`；A0/A1 Utility 初始 state SHA256 相同：`f151996d37824c8a02f255159f85ebcf91c8c710b4123bee1fc159cf8bae92e8`；Rectifier 初始 state SHA256 相同：`e74affeae1a965deff7826235de09f45665c6817f6f97731397a7f6ea19868b8`。
- A1 相对 A0 只增加 `aux_head.bias, aux_head.weight`，合计 **257** 个参数。原 Utility/Rectifier 可训练参数分别为 371,803/329,985；CLIP、Adapter、C1、SAM 及来源 evidence heads 冻结。
- 原始 Phase6E.2 recipe：seed 3407；10 epochs；batch 8；AdamW lr/weight decay 1e-4；clip 1.0；无 scheduler/gradient accumulation；相同 8836 train Fake 和 1106 DEV Fake、相同样本顺序/cross/shuffle；按 final-mask DEV G0 mean IoU 选 epoch。
- A1 λ=0.1 的前三个 batch `λL_aux/L_original` 为 `0.142207, 0.118308, 0.124586`，未触发一次缩放修正。
- 数据、初始化和逐 epoch 更新数/样本顺序证据见 `outputs/phase6h/initialization_hashes.json`、`outputs/phase6h/A0_reproduction_audit.json` 与各臂 `config_snapshot/protocol.json`。

## B. A0 对 Phase6E.2 的复现

原始 selector 为 epoch **8**、mean FG IoU **0.2027321013**；A0 选 epoch **8**，mean FG IoU **0.203351**，差 **0.000619**。最大逐 epoch IoU 差 **0.003299**，最大逐 epoch original loss 差 **0.001799**；样本顺序、有效曝光数和 optimizer updates 全部逐 epoch 相等：**True**。

`0.202488` 是后续 Phase6F.1/F.3 冻结回放的次级参考值，**不是**原始 6E.2 epoch-8 selector。

| epoch | 6E.2 IoU | A0 IoU | A0−6E.2 | 6E.2 loss | A0 loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.175595 | 0.175987 | 0.000391 | 1.305964 | 1.306402 |
| 2 | 0.182020 | 0.185320 | 0.003299 | 1.290651 | 1.291310 |
| 3 | 0.188608 | 0.188944 | 0.000336 | 1.282317 | 1.283635 |
| 4 | 0.188379 | 0.188784 | 0.000405 | 1.275421 | 1.276849 |
| 5 | 0.194781 | 0.191820 | -0.002960 | 1.271535 | 1.271201 |
| 6 | 0.187686 | 0.190714 | 0.003028 | 1.267802 | 1.268915 |
| 7 | 0.191712 | 0.190396 | -0.001316 | 1.264002 | 1.264270 |
| 8 | 0.202732 | 0.203351 | 0.000619 | 1.259805 | 1.260399 |
| 9 | 0.193503 | 0.192503 | -0.001000 | 1.256754 | 1.258554 |
| 10 | 0.197817 | 0.196744 | -0.001072 | 1.252887 | 1.254104 |

## C. A1 的 C 空间辅助指标

| 指标 | A1 auxiliary mask |
|---|---:|
| mean FG IoU | 0.164076 |
| mean FG F1 | 0.239626 |
| global FG IoU | 0.222366 |
| global FG F1 | 0.363829 |

A0 按预注册要求没有辅助头；因此 A1 的 auxiliary mask 仅证明该受监督 C 可被其 1×1 head 解码，**不能单凭此值量化 C 相对 A0 的可解码性增量**。不把 auxiliary 指标用作 checkpoint selector。

## D. 最终 Utility + SAM 掩码

| 指标 | A0 | A1 | A1−A0 |
|---|---:|---:|---:|
| mean FG IoU | 0.203351 | 0.201945 | -0.001407 |
| mean FG F1 | 0.284868 | 0.283551 | -0.001317 |
| global FG IoU | 0.230169 | 0.224800 | -0.005369 |
| global FG F1 | 0.374208 | 0.367080 | -0.007127 |

1106 张严格配对（1090 有效单 `[SEG]`，16 张无 `[SEG]` 按零计；SEG trigger rate **0.985533**）。IoU paired median Δ **0.000000**，bootstrap 95% CI **[-0.004073, 0.001313]**，胜/平/负 **420/220/466**；F1 paired mean Δ **-0.001317**，CI **[-0.004728, 0.002116]**。global 指标是聚合像素指标，没有样本级 CI。

## E. 预注册判定

主条件 `Δ mean FG IoU ≥ +0.010`：**False**；paired bootstrap CI 下界 >0：**False**；global 反方向 trade-off：**False**。最终判定：**STOP**。

如果辅助掩码表现较好而最终掩码未达门槛，结果只能说明该监督在当前原始 Utility/SAM 路径下未稳定转化为最终收益；不能单凭 A1 auxiliary 指标把瓶颈唯一定位到 Utility 或 SAM。Phase6H.2 需用户另行决定。

## F. 后续单独授权的 A1 Official1000

用户在 A1 selected checkpoint 冻结后单独授权了 Official1000 canonical G0 测试。A1 mean FG IoU 0.316889、global FG IoU 0.333648；历史 Phase6E.2 参考分别为 0.319867/0.343538。严格配对 mean IoU 差 -0.002978，bootstrap 95% CI [-0.006306,0.000408]。A0 没有单独跑 Official1000；这项历史参考差值不能替代内部 A1−A0 因果比较，也不改变本阶段 STOP 判定。详见 `docs/phase6h1_a1_official1000.md`。
