# Phase 4E-1 Runtime Recovery Log

## 2026-08-25：首次 supervisor 会话回收

- 事件性质：Codex exec 会话随任务回合结束被回收；不是 NaN、gradient explosion、catastrophic collapse 或 model implementation failure。
- 最后完整、可验证的恢复点：Full Stage T teacher epoch 3，global step 3,315。
- 被中断位置：epoch 4 的训练循环完成后，validation 运行到至少 400/1,106；epoch 4 checkpoint 与完整 metrics 尚未写入。
- 恢复规则：只加载完整 epoch 3 checkpoint；以冻结 seed、sample order、optimizer/scheduler state 重新执行整个 epoch 4。丢弃进程内未保存的 epoch 4 状态和不完整 validation，不把它作为 selector candidate。
- scientific boundary：最终正式 exposure/step 报告以完整 checkpoint lineage 为准；另行披露一次被丢弃的 interrupted runtime attempt，不将其计入正式训练曲线。
- process hardening：后续 supervisor 使用 detached background process，不再依赖对话回合的 PTY 生命周期。

internal test 与 official1000 在中断前后均未访问。
