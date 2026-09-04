# Utility intervention identity

CSCU-LF 在 source posteriors/masses、aligned source features与interaction materialize后才输出独立 `U_F64`。Synthetic intervention只用另一样本的 U map替换当前 U；模型重算 adapted discount/fusion，不重构 evidence。

Before/after bit-exact hashes已逐项固定：

- `p_L64,p_F64`；
- `mass_L64,mass_F64`；
- `mass_L256,mass_F256`；
- `L64,Fctx64`；
- `aligned_F64,aligned_z_F64`。

十项全部 bit-exact，outside support utility仍为0，`UTILITY_INTERVENTION_IDENTITY=PASS`。本测试只用 synthetic data，未消费 future UTILITY-AUDIT。

Future permutation必须沿用相同 contract：先 hash sources，固定 seed 3407，只替换 U；任一 identity failure立即 fail-closed，不修改 permutation或seed后重跑。结果见 `outputs/phase4g1p/utility_intervention_identity.json`。
