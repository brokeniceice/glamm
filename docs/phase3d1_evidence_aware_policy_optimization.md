# Phase 3D.1 — Evidence-Aware Policy Optimization

## 实验边界

核心比较严格限定为 P1-FROZEN、P3D1-R3、P3D1-Q2；没有加入 RSFT。R3/Q2 均从同一 P1 SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326` 初始化，K=8、2000 optimizer steps、数据、采样、优化器、学习率、scheduler、可训练模块与 checkpoint 间隔一致，唯一研究变量是冻结 reward。

## Selector

R3 选择 step 750，Q2 选择 step 2000。两者只使用 internal validation Fake mean FG IoU，tie-break 为 FG F1；training reward、test、official1000 均未参与。

## 核心结果

| Arm | mean FG IoU | mean FG F1 | R_phrase_sem | CLS accuracy |
|---|---:|---:|---:|---:|
| P1-FROZEN | 0.148233 | 0.210697 | 0.638102 | 0.986438 |
| P3D1-R3 | 0.153907 | 0.217226 | 0.640366 | 0.985081 |
| P3D1-Q2 | 0.151541 | 0.215564 | 0.646014 | 0.984629 |

Q2−R3 paired bootstrap：FG IoU Δ=-0.002366, 95% CI=[-0.007687407987051258, 0.002902199515104328]；FG F1 Δ=-0.001662, 95% CI=[-0.008134803617325248, 0.0047555003383229775]；R_phrase_sem Δ=+0.005648, 95% CI=[-0.0009990815584417253, 0.012337703254080493]；classification accuracy Δ=-0.000452。

K=8 stochastic evaluation：R3 mean R_ground_rel=0.311288，Q2=0.312762；semantic phrase-mask harmonic consistency R3=0.181929，Q2=0.183000。

## 结论与停止门

主门：**GATE_REWARD_FORMULATION_NOT_PRIMARY_BOTTLENECK**。该结论只回答 Q2 与 R3 哪个更适合作为当前 forensic MLLM policy-optimization signal，不把 selector 内的 validation 差异外推为未知 test/真实图像泛化结论。

Phase 3D.1 到此停止；未自动启动 FEPN、NPR、FOCAL、架构修改、额外 reward tuning 或 RSFT。
