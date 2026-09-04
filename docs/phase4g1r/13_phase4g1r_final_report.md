# Phase 4G-1R final report

本阶段完成 G1-C failure-driven redesign、定向文献审计、conditional-utility tensor graph、监督选项、candidate comparison、causal control 与 future preflight proposal。没有训练、probe、校准、模型实现或 held-out evaluation。

## Mandatory statements

`G1C_INTRINSIC_RELIABILITY_HYPOTHESIS: NOT_SUPPORTED`

`SPATIAL_CONDITIONAL_UTILITY_NEEDED: YES`

`CROSS_SOURCE_INTERACTION_NEEDED: YES`

`PCERF_EVIDENTIAL_REFINEMENT_DIRECTION: OPEN`

`ORIGINAL_RELIABILITY_AWARE_PCERF: NOT_SUPPORTED`

`AHBFR_STATUS: SECONDARY`

`PRIMARY_CANDIDATE: CSCU-LF`

`UTILITY_INTERVENTION_IDENTIFIABLE: YES`

`P1_PATH_PRESERVED: YES`

`ORIGINAL_SAM_DECODER_PRESERVED: YES`

`FORENSIC_GEOMETRY_PRESERVED: YES`

`NEXT_PREFLIGHT_JUSTIFIED: YES`

`FORMAL_TRAINING_JUSTIFIED: NO`

## Scientific interpretation

Forensic intrinsic uncertainty作为 localization reliability 不受支持；language intrinsic reliability 与 cross-image response仍不确定；forensic effective-weight quality relation得到部分正证据；reliability causal contribution没有被有效检验，绝不能写成 false。

CSCU-LF 将核心变量改为 `U_F(L,F)`，利用 CMX-style joint spatial/channel interaction 与 selective context exchange感知 matched/shuffle/cross-image compatibility；source prediction materialize 后，`U_F` 只控制明确标记的 ECoLaF-style adapted discount，因此 future utility intervention 可识别。P1 path、original SAM decoder、forensic geometry/vacuous support 与 exact fallback全部保留。

Phase 4G-1R 到此停止。不得自动实现或训练 CSCU-LF，不得启动 G1-F、AHBFR、Teacher/KD，不得访问 development validation、internal test 或 official1000；等待人工审阅。
