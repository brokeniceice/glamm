# Phase 6F.3 — Full-FOV Forensic Evidence Implementation & Frozen Replay

## Protocol

Frozen replay on the same internal-validation Fake population as Phase6F.1 (`n=1106`). No training, optimizer, checkpoint modification/selection, threshold tuning, internal test, Official1000, or OOD access occurred. C1 canonical-G0 queries were reused from the frozen C1 cache; A0 and A1 therefore share identical generation, SEG trigger, sample order, GT, SAM inverse geometry, and mask threshold. The only difference is forensic acquisition/coverage.

Implementation invariants: **PASS**. C1 SHA256: `85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff`. Selected new-R1 SHA256: `c067eb2240af9d5fd61fc5dc367338ed585fcc6a5d5d0f982df0fa886302e603`.

## Overall

| Arm | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 center crop | 0.202488 | 0.104586 | 0.283836 | 0.222581 | 0.364117 |
| A1 full FOV | 0.202649 | 0.104586 | 0.284282 | 0.223402 | 0.365214 |

- A1-A0 IoU: `{'mean_delta': 0.00016109642846279024, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.0015866320912797374, 0.0019996380694669836], 'wins': 67, 'ties': 976, 'losses': 63, 'wilcoxon_statistic': 4054.0, 'wilcoxon_pvalue': 0.6363053897595374}`
- A1-A0 F1: `{'mean_delta': 0.00044663355069450583, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.0018321004401911736, 0.0028705021770375153], 'wins': 67, 'ties': 976, 'losses': 63, 'wilcoxon_statistic': 4014.0, 'wilcoxon_pvalue': 0.5715172879318493}`

## Phase6F.1 exact-crop coverage strata

| Coverage | n | A0 mean IoU | A1 mean IoU | delta IoU | bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| 1.0 | 1018 | 0.214599 | 0.213428 | -0.001171 | [-0.0024037523570372994, -6.72556665931556e-05] |
| [0.9,1.0) | 16 | 0.154201 | 0.131349 | -0.022852 | [-0.0786972776064751, 0.015458296894449042] |
| [0.5,0.9) | 34 | 0.068942 | 0.084246 | 0.015304 | [0.000830175732311338, 0.03134226807316667] |
| (0,0.5) | 23 | 0.025037 | 0.076555 | 0.051518 | [0.006781531814698782, 0.10727760870926306] |
| 0 | 15 | 0.006882 | 0.008930 | 0.002049 | [-0.0010009316257445766, 0.006780266209649738] |

## Aspect and tile-count strata

```json
{
  "aspect_ratio": {
    "aspect_ratio_lt_1.2": {
      "n": 896,
      "A0": {
        "n": 896,
        "mean_fg_iou": 0.22919942787102834,
        "median_fg_iou": 0.14221574011384228,
        "mean_fg_f1": 0.31918638648741027,
        "global_fg_iou": 0.25026315088782153,
        "global_fg_f1": 0.40033676224098536,
        "tp": 6071821,
        "fp": 7619765,
        "fn": 10570160
      },
      "A1": {
        "n": 896,
        "mean_fg_iou": 0.22919942787102834,
        "median_fg_iou": 0.14221574011384228,
        "mean_fg_f1": 0.31918638648741027,
        "global_fg_iou": 0.25026006670089024,
        "global_fg_f1": 0.40033281613362437,
        "tp": 6071821,
        "fp": 7620064,
        "fn": 10570160
      },
      "paired": {
        "n": 896,
        "iou": {
          "mean_delta": 0.0,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            0.0,
            0.0
          ],
          "wins": 0,
          "ties": 896,
          "losses": 0,
          "wilcoxon_statistic": 0.0,
          "wilcoxon_pvalue": 1.0
        },
        "f1": {
          "mean_delta": 0.0,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            0.0,
            0.0
          ],
          "wins": 0,
          "ties": 896,
          "losses": 0,
          "wilcoxon_statistic": 0.0,
          "wilcoxon_pvalue": 1.0
        }
      }
    },
    "aspect_ratio_ge_1.2": {
      "n": 210,
      "A0": {
        "n": 210,
        "mean_fg_iou": 0.08851914171101954,
        "median_fg_iou": 0.0038297135300840906,
        "mean_fg_f1": 0.13300659437260068,
        "global_fg_iou": 0.05500734427626754,
        "global_fg_f1": 0.10427859971724168,
        "tp": 220463,
        "fp": 2242737,
        "fn": 1544683
      },
      "A1": {
        "n": 210,
        "mean_fg_iou": 0.0893675829009236,
        "median_fg_iou": 0.0140303311493028,
        "mean_fg_f1": 0.1353588644062584,
        "global_fg_iou": 0.06593286034642747,
        "global_fg_f1": 0.12370921809277806,
        "tp": 272846,
        "fp": 2373094,
        "fn": 1492300
      },
      "paired": {
        "n": 210,
        "iou": {
          "mean_delta": 0.000848441189904028,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            -0.008623847178443275,
            0.010351155012841777
          ],
          "wins": 67,
          "ties": 80,
          "losses": 63,
          "wilcoxon_statistic": 4054.0,
          "wilcoxon_pvalue": 0.6363053897595374
        },
        "f1": {
          "mean_delta": 0.002352270033657731,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            -0.010057338621366598,
            0.014665342073490283
          ],
          "wins": 67,
          "ties": 80,
          "losses": 63,
          "wilcoxon_statistic": 4014.0,
          "wilcoxon_pvalue": 0.5715172879318493
        }
      }
    }
  },
  "tile_count": {
    "N=1": {
      "n": 894,
      "A0": {
        "n": 894,
        "mean_fg_iou": 0.2297121782689501,
        "median_fg_iou": 0.143418406497355,
        "mean_fg_f1": 0.31990044999185635,
        "global_fg_iou": 0.2504000663134349,
        "global_fg_f1": 0.40051192103930633,
        "tp": 6071821,
        "fp": 7615692,
        "fn": 10560967
      },
      "A1": {
        "n": 894,
        "mean_fg_iou": 0.2297121782689501,
        "median_fg_iou": 0.143418406497355,
        "mean_fg_f1": 0.31990044999185635,
        "global_fg_iou": 0.2504000663134349,
        "global_fg_f1": 0.40051192103930633,
        "tp": 6071821,
        "fp": 7615692,
        "fn": 10560967
      },
      "paired": {
        "n": 894,
        "iou": {
          "mean_delta": 0.0,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            0.0,
            0.0
          ],
          "wins": 0,
          "ties": 894,
          "losses": 0,
          "wilcoxon_statistic": 0.0,
          "wilcoxon_pvalue": 1.0
        },
        "f1": {
          "mean_delta": 0.0,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            0.0,
            0.0
          ],
          "wins": 0,
          "ties": 894,
          "losses": 0,
          "wilcoxon_statistic": 0.0,
          "wilcoxon_pvalue": 1.0
        }
      }
    },
    "N=2": {
      "n": 207,
      "A0": {
        "n": 207,
        "mean_fg_iou": 0.08858134398475344,
        "median_fg_iou": 0.0034918076819769003,
        "mean_fg_f1": 0.13286517192911534,
        "global_fg_iou": 0.0548793395294715,
        "global_fg_f1": 0.10404856266111043,
        "tp": 218550,
        "fp": 2235915,
        "fn": 1527908
      },
      "A1": {
        "n": 207,
        "mean_fg_iou": 0.08940468490527602,
        "median_fg_iou": 0.012304622547389425,
        "mean_fg_f1": 0.13521046105387652,
        "global_fg_iou": 0.06584945048962415,
        "global_fg_f1": 0.12356238577479135,
        "tp": 269726,
        "fp": 2349643,
        "fn": 1476732
      },
      "paired": {
        "n": 207,
        "iou": {
          "mean_delta": 0.0008233409205225617,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            -0.009089172885552527,
            0.01054109421537102
          ],
          "wins": 66,
          "ties": 80,
          "losses": 61,
          "wilcoxon_statistic": 3877.0,
          "wilcoxon_pvalue": 0.6527419845860184
        },
        "f1": {
          "mean_delta": 0.002345289124761203,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            -0.010384671035727358,
            0.015046816764547204
          ],
          "wins": 66,
          "ties": 80,
          "losses": 61,
          "wilcoxon_statistic": 3837.0,
          "wilcoxon_pvalue": 0.5849260053773004
        }
      }
    },
    "N=3": {
      "n": 5,
      "A0": {
        "n": 5,
        "mean_fg_iou": 0.05053631089402827,
        "median_fg_iou": 0.015193561814902178,
        "mean_fg_f1": 0.08565884578385305,
        "global_fg_iou": 0.049334639983494945,
        "global_fg_f1": 0.09403032760696994,
        "tp": 1913,
        "fp": 10895,
        "fn": 25968
      },
      "A1": {
        "n": 5,
        "mean_fg_iou": 0.0520845267603634,
        "median_fg_iou": 0.014091060152874708,
        "mean_fg_f1": 0.08735921743236397,
        "global_fg_iou": 0.056010340370529946,
        "global_fg_f1": 0.10607915136678907,
        "tp": 3120,
        "fp": 27823,
        "fn": 24761
      },
      "paired": {
        "n": 5,
        "iou": {
          "mean_delta": 0.001548215866335133,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            -0.001319645593834965,
            0.006398581056455355
          ],
          "wins": 1,
          "ties": 2,
          "losses": 2,
          "wilcoxon_statistic": 3.0,
          "wilcoxon_pvalue": 1.0
        },
        "f1": {
          "mean_delta": 0.0017003716485109142,
          "median_delta": 0.0,
          "bootstrap_95_ci": [
            -0.0025510893038145467,
            0.008470738222643426
          ],
          "wins": 1,
          "ties": 2,
          "losses": 2,
          "wilcoxon_statistic": 3.0,
          "wilcoxon_pvalue": 1.0
        }
      }
    },
    "N>=4": {
      "n": 0,
      "A0": null,
      "A1": null,
      "paired": null
    }
  }
}
```

## Generation invariance

- C1 trajectory source: frozen Phase6E.2 C1 canonical-G0 cache.
- A0/A1 valid-SEG flags equal: `True`.
- A0/A1 q_seg tensors shared: `True`.
- SEG trigger rate: `0.985533` for both arms.

## Decision

```text
FULL_FOV_SIGNAL_PRESENT_BUT_R1_RECALIBRATION_NEEDED
```

Zero-shot failure, if observed, does not negate Phase6F.1's coverage-bottleneck finding: selected Rectifier and Utility weights were learned under the single-center-crop evidence distribution. No retraining is started by this phase.
