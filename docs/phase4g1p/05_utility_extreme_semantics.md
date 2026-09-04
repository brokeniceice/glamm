# Utility extreme semantics

固定随机 source masses、seed 3407，未训练：

- `U_F=1`：adapted discounted masses、fused mass与probability bit-exact等于 official ECoLaF，`UTILITY_ONE_PARITY=PASS`。
- `U_F=0`：forensic committed contribution精确为0；fused mass与DSmP probability精确等于原 language source，`UTILITY_ZERO_SEMANTICS=PASS`。
- `U_F∈{0,.25,.5,.75,1}`：定义的 forensic effective contribution `d_F^A(1-u_F)`逐元素单调不减，`UTILITY_MONOTONICITY=PASS`。

这里的 monotonicity对象是 forensic effective committed contribution，不要求 final foreground probability单调；后者还取决于 forensic class direction与source conflict。

Absent、off、fully unsupported、vacuous不依赖 U head学习，全部 direct identity dispatch `z_L`，`VACUOUS_EXACT_P1=PASS`。完整结果见 `outputs/phase4g1p/utility_extreme_tests.json`。
