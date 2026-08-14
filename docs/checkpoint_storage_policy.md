# 训练权重存储约定

## 统一目录

Phase 训练产生的 checkpoint 统一保存到：

```text
/data/yz/groundingLMM_official/checkpoints/<phase_name>/
```

仓库内的统一访问入口为：

```text
checkpoints/GLaMM_train -> /data/yz/groundingLMM_official/checkpoints
```

后续阶段的训练配置应将 `checkpoint.output_root` 直接写到上述绝对目录，或写成 `checkpoints/GLaMM_train/<phase_name>`。不得再把新的 Phase 训练权重实体写入 `/data/yz/myLISA_storage/checkpoints/`。

## 当前实体目录

统一目录当前包含：

- `phase2a_single_template`
- `phase2a_unified_baseline`
- `phase3a_phrase_grounding`
- `phase3a1_paired_control`

旧 checkpoints 根目录中的 `phase2a_single_template`、`phase2a_unified_baseline` 和 `phase3a_phrase_grounding` 仅为兼容历史配置与脚本的软链接，不是重复权重。

## 不属于 Phase 训练产物的权重

HF cache、FOCAL、GLaMM 基座、LISA 和 SIDA 等外部或基座权重继续保留在原目录，不纳入 `GLaMM_train` 的 Phase 训练权重迁移范围。
