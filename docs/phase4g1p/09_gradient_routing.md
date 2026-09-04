# Gradient routing and complexity

`REFINEMENT_INCLUDED=NO`。Future trainable blocks只有：language context、forensic context、CMX rectification、local exchange、utility head。Frozen：G1-C source evidential heads、temperatures、P1/SAM、CLIP、Phase4C-A/dense source及official ECoLaF kernel。

一次 synthetic backward、零 optimizer steps：

| Block | params | grad norm | finite | every parameter nonzero |
|---|---:|---:|---|---|
| language context | 116,352 | 2.48701 | yes | yes |
| forensic context | 63,488 | 0.926178 | yes | yes |
| CMX rectification | 6,682 | 0.100800 | yes | yes |
| local exchange | 33,536 | 0.285215 | yes | yes |
| utility head | 151,745 | 3.27305 | yes | yes |

总 trainable parameters=371,803；frozen G1-C source-head parameters=178,820。Frozen grads均 none/zero，source state hash前后相同，`GRADIENT_ISOLATION=PASS`。

Synthetic B=1 complexity estimate：traced trainable conv/linear 2.84359 GFLOPs，frozen source heads 0.887032 GFLOPs，双向 local attention score/value 0.0513802 GFLOPs，总计约3.78200 GFLOPs；插值、normalization、elementwise/ECoLaF未计入。core attention+comparison FP32约10.23 MiB；测得增量 eval-forward CUDA allocator peak约169.56 MiB。未来 training memory会更高，不据此修改 architecture。

详细 hash/route见 `outputs/phase4g1p/gradient_routing.json` 与 `architecture_manifest.json`。保持 `TRAINING_BALANCE_AUDIT_REQUIRED=YES`、`INTERVENTION_REQUIRED=UNRESOLVED`；不自动加入 OGM-GE。
