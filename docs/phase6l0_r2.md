# Phase6L0 — C2-native R2 Localization

Status: **COMPLETE STOP AFTER INTERNAL DEV**.

## Architecture and provenance

R2 uses a new EvidenceConsolidator, geometry-aware local deformable bridge, global q_seg/r_prime conditioner, and SAM residual adapter. It loads no R1 Rectifier or Utility state. The bridge aligns CLIP evidence to the padded S64 lattice. Support is an input feature; deltaS is not multiplied by support. The final residual projection starts at zero, so epoch0 is C2-G0 exactly.

C2 checkpoint SHA256: `4a67e6a87c453d554fa5bd6cf93329ae54c1c2853f0925dc7a63397eef27e8ce`. Frozen SAM state SHA256: `d380db8eec7bae45bc5f2cd9523cc64f6b551c04881c8307bf16025e616b99f6`. Selected R2 checkpoint SHA256: `19ec58057b4a8283c6dc666a0ce901c0ce97f86fee8b03e63fcebfd26861b3c5`.

Tensor contracts: raw CLIP `[B,1024,24,24]`; A `[B,8,24,24]`; E `[B,512,24,24]`; r_prime `[B,4096]`; q_seg `[B,256]`; S64/F64/R64/deltaS `[B,256,64,64]`; z_L `[B,1,256,256]`; final low-res SAM logits `[B,1,256,256]`.

R2 trainable parameters: 4,078,496. Frozen C2 model: 8,025,250,866 parameters (from the formal C2 model loader audit); the lightweight frozen C2 SAM prompt/mask runtime contains 4,064,816 of those parameters. C2, RINE, CLIP, LLM, projectors and SAM are frozen during R2 training.

## Population and training

TRAIN 8836 total; C2-valid 8682; invalid 154. DEV 1106 total; C2-valid 1084; invalid 22. Invalid DEV images are retained and scored as zero.

Ten joint epochs, AdamW lr 1e-4, weight decay 1e-4, batch 8, seed 3407, no scheduler or accumulation; 11050 optimizer updates. The only loss is 2×BCEWithLogits + 0.5×soft Dice on the frozen C2 SAM final mask. Epoch 10 was selected by DEV Mean FG IoU, with earlier epoch as tie break.

| Epoch | Train loss | DEV Mean IoU | Mean F1 | Global IoU | Global F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.672757 | 0.177103 | 0.248768 | 0.172159 | 0.293746 |
| 2 | 0.660697 | 0.156163 | 0.218215 | 0.144782 | 0.252943 |
| 3 | 0.652666 | 0.180685 | 0.256712 | 0.206816 | 0.342747 |
| 4 | 0.648770 | 0.194358 | 0.273086 | 0.202895 | 0.337344 |
| 5 | 0.641142 | 0.196952 | 0.277492 | 0.212204 | 0.350113 |
| 6 | 0.636812 | 0.193147 | 0.272380 | 0.193263 | 0.323923 |
| 7 | 0.629536 | 0.195416 | 0.272308 | 0.200259 | 0.333693 |
| 8 | 0.625313 | 0.195460 | 0.271512 | 0.203885 | 0.338711 |
| 9 | 0.619381 | 0.188883 | 0.263050 | 0.187606 | 0.315939 |
| 10 | 0.614698 | 0.200544 | 0.277964 | 0.204791 | 0.339962 |

## Selected versus epoch0 C2-G0

| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2-G0 epoch0 | 0.155826 | 0.219360 | 0.149181 | 0.259631 | 4069254 | 8870089 | 14337873 | 445317406 |
| C2 + R2 epoch10 | 0.200544 | 0.277964 | 0.204791 | 0.339962 | 4690823 | 4498245 | 13716304 | 449689250 |

Paired selected-minus-epoch0 statistics on the same 1,106 DEV IDs (2,000 bootstrap resamples, seed 3407):

| Metric | Mean delta | Median delta | 95% CI | W/T/L | Wilcoxon p |
| --- | ---: | ---: | --- | --- | ---: |
| FG IoU | 0.044718 | 0.002606 | [0.035488, 0.053808] | 584/221/301 | 2.706e-27 |
| FG F1 | 0.058605 | 0.004974 | [0.047402, 0.069586] | 584/221/301 | 8.804e-28 |

Paired bootstrap global IoU delta: 0.055610, 95% CI [0.038518, 0.072236]. Global F1 delta: 0.080331, 95% CI [0.056444, 0.103312].

## Mechanism diagnostics

Diagnostics are recorded for each epoch in `outputs/phase6l0_r2/diagnostics_epoch*.json`: residual ratio and support behavior, bridge offsets and attention, invalid sampling, and gradients in `training_history.csv`. These were not selectors.

Selected epoch per-image residual ratio mean/median/P95/max: 4.296278/4.255336/4.887380/5.938258; cosine(S64,S_adapt): 0.295965; mean |deltaS| inside/outside support: 0.269221 / 0.029701. Offset mean/median/P95/max (averaged per-image statistics): 3.528083/3.880274/5.443751/5.576445; boundary fraction 0.025065; attention entropy 1.148528; maximum mass 0.207122; invalid sampled-point fraction 0.171512; all-invalid query fraction 0.171512.

Selected epoch gradient means: Evidence 6.499484, Bridge 13.235108, GlobalConditioner 16.627367, blocks 1/2/3 5.174523/4.887329/3.600880, W_out 9.265412; clipping frequency 0.923077. Finite loss and parameters were enforced; frozen SAM and C2 checkpoint hashes were checked after each epoch.

## Conclusion

The internal DEV paired result supports conversion of C2-derived forensic-semantic spatial evidence into SAM-compatible residual adaptation through the jointly trained R2 branch. This does not identify A/E or bridge attention as causal pixel attribution.

## Stop boundary

No internal test, Official1000, localization OOD, R2.1, or architecture search was run. The conclusion is limited to canonical internal DEV.
