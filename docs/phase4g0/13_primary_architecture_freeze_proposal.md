# Primary Architecture Freeze Proposal：PCERF

状态：**PROPOSED, NOT IMPLEMENTED, NOT TRAINED**。需人工审阅后才可冻结为 Phase 4G-1 协议。

## 冻结组成

1. Language expert：canonical P1 原生输出，checkpoint SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`。
2. Forensic expert：Phase 4C-A selected forensic adapter，feature `F24 [B,256,24,24]` 与原 dense head logits。
3. 两 expert backbone 全冻结，从 P1 + pretrained forensic source fresh initialization；不加载 Phase 4F rectifier。
4. Trainable only：language evidential head、forensic evidential head、ECoLaF-style adaptive discount/calibration。
5. Fusion granularity：pixel/spatial 为主，同时聚合 sample reliability 供审计，标为 hybrid。
6. Output：原 P1 SAM decoder prediction 与 forensic prediction 的 evidential late fusion；不建新 mask decoder。

## 必须精确冻结的 shape 与接口

| Tensor | Shape | 来源 | 梯度 |
|---|---|---|---|
| `h_G0` | `[B,4096]` | 最后一个合法 `[SEG]` 的 causal predictor，即 `[SEG]` 前一 token hidden | frozen |
| `q_seg` | `[B,256]` | P1 `text_hidden_fcs` | frozen |
| `S64` | `[B,256,64,64]` | P1 SAM image encoder/cache | frozen |
| `z_L` | `[B,1,256,256]` | P1 prompt+mask decoder low-res logits | frozen |
| `F24` | `[B,256,24,24]` | forensic adapter | frozen |
| `z_F24` | `[B,1,24,24]` | forensic dense head | frozen |
| `E_L` | `[B,2,64,64]` | language evidence head | trainable |
| `E_F` | `[B,2,24,24]` | forensic evidence head | trainable |
| `m_fused` | `[B,2,64,64]` | conflict-discount + DS combine | trainable w.r.t heads |

## Exact recovery invariants

- forensic source off/vacuous → output等于 P1 logits（数值 tolerance 与 dtype 预注册）。
- reliability heads off + language-only → P1 metrics exact。
- cross-image/shuffle 不得改变 language expert。
- CLIP center-crop 外 forensic support mass 必须为 vacuous，不得用 zero token假装可靠证据。
- invalid G0 不补 hidden、不用 TF hidden、不改 formal 0 policy。

## Reliability loss 与校准

不冻结一个未经核对的自创 evidential loss。Phase 4G-1 实现前先对 TMC/ECoLaF 官方公式做 unit parity，明确 evidence regularization、annealing、conflict normalization 与 empty/vacuous case。校准只用预注册 train-calibration fold；validation 不参与信号选择或 architecture tuning。

## Gate-collapse / dominance 防护

- constant reliability 会在 source-specific NLL/Brier/ECE 与 weight–loss correlation 中暴露；
- 记录 pixel gate/discount 直方图、sample aggregation 与 0/1 饱和比例；
- reliability permutation 必须改变决策并降低 matched performance，否则 fusion router 判无效；
- fused、language-evidence、forensic-evidence loss 分开记录并做 gradient cosine；
- 不默认添加 entropy、load-balance、dropout、OGM-GE 全套正则。

## Freeze gate

只有以下全部通过才允许 full training：ECoLaF公式 parity、P1 exact recovery、geometry/support、gradient isolation、reliability calibration、permutation sensitivity、invalid-G0 policy、sealed test audit。任何一项失败即停止，不退化为 simple sigmoid gate。

