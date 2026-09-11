# Phase 6C.1 — Native MultiSEG Restoration Preflight

## Scope

TRAIN-only implementation/preflight. No formal training, optimizer step, validation/test/OOD access, checkpoint selection, classifier change, or loss-weighting change occurred. Phase 6A is the architecture-audit source boundary.

## Restored contract

`refs[i].phrase ↔ refs[i].polygon mask ↔ <p>phrase_i</p>[SEG]` is retained in annotation order. Invalid masks remove only their own pair and record `ref_index + reason`. Token overflow removes complete suffix pairs atomically. GLaMM's native global per-mask mean remains unchanged.

## Gates

| Gate | Result |
|---|---:|
| TRAIN adapter K | [1, 2, 3] |
| phrase / SEG / mask counts | PASS |
| ordered ref indices | [[0], [0, 1], [0, 1, 2]] |
| K=1 union mask identity | PASS |
| atomic truncation retained refs | [0, 1] |
| actual empty-mask drops | 1 pair, 0 images |
| P1 K=1 SEG-state backward parity | PASS |
| P1 K=1 mask-loss backward parity | PASS |
| R1 K / T / offsets | [1, 2, 3] / 6 / [0, 1, 3, 6] |
| R1 parameter copies | 0 |
| R1 weights unchanged | PASS |
| loss | native global per-mask mean |

`P1_MULTI_READY = YES`

`R1_MULTI_READY = YES`

## Boundary

This result authorizes no formal training. Per-image-balanced loss was not implemented or tested in this phase.
