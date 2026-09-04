# Candidate comparison

评分 1–5；optimization risk/complexity 按“越安全/越可控分越高”。不使用任何 performance。

| 维度 | CSCU-LF | AHBFR | standalone cross-attention |
|---|---:|---:|---:|
| direct match to G1-C failure | 5 | 3 | 4 |
| literature maturity | 4 | 5 | 4 |
| spatial mismatch sensitivity | 5 | 5 | 4 |
| cross-image mismatch sensitivity | 5 | 4 | 4 |
| P1 path preservation | 5 | 4 | 5 |
| forensic specialization preservation | 5 | 5 | 5 |
| exact P1 fallback | 5 | 5 | 5 |
| causal identifiability | 5 | 2 | 2 |
| strong-strong fusion | 5 | 4 | 4 |
| weak-strong compensation | 5 | 5 | 4 |
| weak-weak refinement | 4 | 4 | 3 |
| optimization safety | 4 | 3 | 2 |
| implementation tractability | 3 | 3 | 2 |
| paper claim clarity | 5 | 3 | 2 |
| **total / 70** | **65** | **55** | **50** |

## 判定

CSCU-LF 最直接修复两个 supported negative：intrinsic forensic uncertainty 无效、当前 fusion 不感知 spatial mismatch。它同时保留 frozen source expert、original SAM decoder、geometry/vacuous policy 与 exact fallback，并通过独立 `U_F` 获得最佳 causal testability。

AHBFR 的 feature rectification更贴近 CMX 原生用法，也可能有更强 representation capacity；但这不是足以晋升的理由。它在 `S64` 内干预，难把 improvement 分解为 utility、refinement 或 decoder effect，且可能重现 Phase4F representation dominance。

Standalone cross-attention 没有足够直接文献把 attention output 等同 conditional utility，故拒绝作为 Candidate C；它只保留为 Primary 内部、可消融的 context-exchange component。

机器可读评分见 `outputs/phase4g1r/candidate_scores.json`。
