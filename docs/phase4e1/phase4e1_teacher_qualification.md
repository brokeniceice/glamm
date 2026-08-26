# Phase 4E-1 Teacher qualification

- TEACHER_MASK_CAPABILITY：**FAILED**
- TEACHER_IMAGE_SPECIFIC_USE：**TRUE**
- TEACHER_SPATIAL_SPECIFIC_USE：**TRUE**
- Teacher vs P1 TF mean IoU delta：-0.076852
- 实际 KD enable matrix：`{"relation": false, "attention": false, "feature": false, "logit": false}`

Teacher capability gate 失败后，冻结协议禁止启动 Full Stage S；qualification 未参与 teacher 重选。
