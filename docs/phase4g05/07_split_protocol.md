# Reliability split protocol

## Frozen split

canonical internal train Fake共 8836 个 image ID。用字符串 `phase4g05-reliability-v1\0<sample_id>` 的 SHA256 rank，在 valid-G0与invalid-G0两层内分别按 70/15/15 切分，再合并为：

| Fold | n | valid / invalid G0 | FG prevalence | regions | IDs SHA256 |
|---|---:|---:|---:|---:|---|
| TRAIN-FIT | 6185 | 6083 / 102 | 0.0422054 | 13300 | `c2f4e5f7...997c0` |
| TRAIN-CAL | 1326 | 1304 / 22 | 0.0447898 | 2922 | `9e992127...1bfe7` |
| TRAIN-AUDIT | 1325 | 1303 / 22 | 0.0431677 | 2951 | `b9997704...b3fac` |

前景比例来自 official annotation-derived all-ref union，在 original-image normalized 256×256 grid rasterize。三组 image ID互斥、membership唯一且精确覆盖 8836 个样本。

## 职责

- TRAIN-FIT：未来仅在新授权下拟合 E_L/E_F/fusion preflight parameters。
- TRAIN-CAL：只拟合预注册 calibration parameters；不得更新 feature/evidence heads。
- TRAIN-AUDIT：只读 reliability validity与 frozen expert diagnosis。

不得用同一 image训练 head又宣称 reliability有效。development validation、internal test与official1000均不属于此 split，也未被访问。

`RELIABILITY_SPLIT_FROZEN = YES`。完整 IDs 与哈希见 `outputs/phase4g05/split_manifest.json`。

