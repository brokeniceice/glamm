# Phase6I I-JEPA 附加证据：internal DEV 结果

状态：**COMPLETE STOP**。本轮未访问 Official1000、internal test 或外部 OOD。

| 模型 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 | 1106 | 0.206367 | 0.287694 | 0.226734 | 0.369654 |
| A1 | 1106 | 0.208087 | 0.288531 | 0.221507 | 0.362678 |

A1−A0 mean FG IoU = **+0.001720**，逐图 bootstrap 95% CI `[-0.004042, +0.007541]`；胜/平/负 = 419/223/464。Global FG IoU 差 = **-0.005227**。

预注册判定：**STOP: 内部证据不足，不升级**。门槛为 mean IoU 差 ≥ +0.010、配对 CI 下界 > 0、global IoU 差 ≥ −0.005，三者须同时满足。

实验结构与数据边界见 [协议](phase6i_ijepa_r1_internal_protocol.md)；小面积掩码（GT ≤ 5%）与低裁剪覆盖（< 80%）仅作预定次级分组，数值见 JSON，不参与主门槛。逐图结果和哈希见 [`internal_dev_paired.json`](../outputs/phase6i_ijepa_r1/internal_dev_paired.json)。

两臂三阶段训练均完成。原后台总控在最终回放入口因 Python 导入路径报错而标记失败；此后使用同一 selected checkpoint 在卡 1、卡 2 分别完成 1106 张逐图回放，并在不改变训练、选模和评测口径的情况下完成配对。原失败日志与重放日志均保留在输出目录；`pipeline_status.json` 记录了这一恢复过程。
