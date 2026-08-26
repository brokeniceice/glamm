# Phase 4F 主结果

| Arm | Selected epoch | G0 mean FG IoU | Phrase | TF |
|---|---:|---:|---:|---:|
| P1 | — | 0.148233 | 0.245798 | 0.342928 |
| FORENSIC-RECT | 9 | 0.171614 | 0.193573 | 0.205262 |
| CLIP-RECT | 5 | 0.157983 | 0.198459 | 0.221938 |

FORENSIC-RECT vs P1 G0 paired result：`{"n": 1106, "mean_difference": 0.02338084271831424, "median_difference": 0.0, "bootstrap_95_ci": [0.014357350493002881, 0.03257244809831621], "wins": 477, "ties": 269, "losses": 360, "wilcoxon_statistic": 141228.0, "wilcoxon_pvalue": 1.0762386976881593e-06}`。
