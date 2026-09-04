# Reliability permutation 报告

`PERMUTATION_SENSITIVITY = FAIL`。seed 3407 的 preregistered reliability-only permutation 在极低 committed-mass/tie 像素未能保持 source argmax bit-exact，因此 causal-control prerequisite 失败；正式 sensitivity 数值未提交。TRAIN-AUDIT 已消费唯一一次读取，未改写 permutation、未更换 seed、未重跑。失败关闭记录见 `outputs/phase4g1/g1c/permutation_metrics.json`。
