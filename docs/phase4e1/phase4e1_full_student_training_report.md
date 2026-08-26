# Phase 4E-1 Full Student 训练报告

Full Stage S **未启动**。原因是 Full Teacher 的 validation Fake canonical TF mean FG IoU 相对 P1 TF 的下降超过冻结门槛，触发 `TEACHER_MASK_CAPABILITY=FAILED`。因此不存在 Full Student exposures、optimizer updates 或 checkpoint；这属于协议规定的停止路径，不是运行故障。
