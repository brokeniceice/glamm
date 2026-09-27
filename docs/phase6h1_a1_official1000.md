# Phase6H.1 A1 — Official1000

用户在 A1 完成后授权对冻结的 A1 selected checkpoint 做一次 Official1000 canonical G0 测试。使用 Phase6E.2 原始 Official1000 推理路径；不按官方结果选择 epoch、调整 λ 或重训。

A1 选中 epoch **8**；checkpoint SHA256 `763ac335b44024ac90f7341fbf20b3d7b2362bfe1e01a74608f4fa0ec00c5365`。1000 张 Fake 的 manifest 顺序与历史记录严格相等。

| 指标 | A1 | 历史 6E.2 参考 | 差值 |
|---|---:|---:|---:|
| mean FG IoU | 0.316889 | 0.319867 | -0.002978 |
| mean FG F1 | 0.437495 | 0.440385 | -0.002890 |
| global FG IoU | 0.333648 | 0.343538 | -0.009890 |
| global FG F1 | 0.500354 | 0.511393 | -0.011039 |
| SEG trigger rate | 0.992000 | 0.992000 | +0.000000 |

严格配对 1000 张：IoU 平均差 -0.002978，bootstrap 95% CI [-0.006306, +0.000408]，胜/平/负 477/23/500；F1 平均差 -0.002890，95% CI [-0.006683, +0.000883]。两个均值差的区间均跨 0；global 指标下降。

A0 完整复现已通过，Phase6H.1 内部严格配对 A1−A0 mean IoU -0.001407，95% CI [-0.004073, +0.001313]，预注册判定 **STOP**。

历史 6E.2 官方行只作背景参考；A0 未单独跑 Official1000，不能把 A1 与历史 6E.2 的官方差值直接当作 A1−A0 的因果效应。

完整逐样本记录、冻结协议及配对统计见 `outputs/phase6h/A1_official1000/`。没有访问 internal test，也没有进入 Phase6H.2。
