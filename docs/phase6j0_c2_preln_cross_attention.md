# Phase 6J.0: C2 Pre-LN cross attention

## Protocol

C2 starts from `checkpoints/GLaMM-FullScope` with seed 3407 and the frozen RINE Q2 checkpoint used by C1. It is a new run; it does not load trained C1 or P1 weights. The historical C1 code, checkpoints and results remain untouched.

The 576 projected CLIP patch tokens provide keys and values. The projected RINE Q2 token provides one query. Separate LayerNorms precede the 512-dimensional, eight-head attention. The 512-to-4096 output projection has zero weights and bias, so the fused token equals the C1 token at initialization. There is no post-attention LayerNorm, gate or FFN. The fused token is appended after the unchanged CLIP patch tokens.

Training uses GPUs 1 and 2, a global batch of 20, 500 optimizer steps per epoch, 10 epochs, and validation-total-loss checkpoint selection. The historical C1 run used one GPU, so a C2-versus-C1 result is **not a strictly matched single-variable causal comparison**. The frozen data, prompt, loss weights and selector are retained.

## Gates and checkpoint protection

1. Unit preflight checks step-zero equality with C1's token and initial gradients.
2. Distributed training preflight runs one optimizer step and saves a preflight checkpoint.
3. Distributed validation preflight then runs two validation batches per rank through the full validator, writing `c2_validation_preflight.json`. Formal training begins only after this passes.
4. Before **each** full-validation pass, a resumable checkpoint with optimizer and scheduler state is saved in `checkpoints/phase6j0_c2/pre_validation/last/`. This ensures a validation exception cannot erase all progress from the preceding epoch. Successful validation still drives the normal `last` and `best` checkpoints.

Training diagnostics record residual ratio, fused-token norm, attention entropy and maximum attention mass, and Q/K/V/output-projection and LayerNorm gradient norms. Zero Q/K/V gradients at the first step are expected while the output projection is zero; they should become nonzero afterward.

## Current status

The two-GPU training and validation preflight passed on 2026-09-26: one optimizer step completed, the two-rank parameter synchronization check passed, a resumable preflight checkpoint was written, and the existing validator produced finite metrics from two batches per rank. All seven historical C1 selected initialization groups have identical hashes in C2's initialization audit. The first-step C2 output projection had nonzero gradients; Q/K/V and both LayerNorm gradients were zero as expected with a zero-initialized output projection.

Formal training was interrupted when competing GPU workloads caused GPU 1 to run out of memory after logging step 3741/5000. The last complete checkpoint is epoch 7, step 3500. The failed-run logs remain under `outputs/phase6j0_c2/recovery/`. A recovery preflight on GPUs 1 and 2 restored that checkpoint, completed one optimizer step with the original per-device batch 10, and passed a two-batch-per-rank validation check. Formal training resumed from step 3500 with the original batch, global batch, optimizer and scheduler. The logged but uncheckpointed steps 3501–3741 were archived and removed from the active metrics stream before replay.

After all 5000 steps and the training-time internal-validation selector pass the completion gates, the detached continuation runs the internal test set, SynthScars Official1000 G0 localization, and classification OOD on AIGI-Holmes/GenImage/LOKI/RAISE998. The extra internal-validation evaluation pass and C2 external OOD localization on LOKI/X-AIGD/PAL4VST are outside the revised scope. Only complete full-N results for the scoped datasets will be appended to `docs/final_evaluation_report.md`; the report will label this as a scoped evaluation. `outputs/phase6j0_c2/continuation_status.json` is the current status record. C2 has no final result until the evaluation and report gate finish.

During Official1000 on GPU 0, the rank 1 classification-OOD shard runs independently on GPU 2. Official1000 was moved from GPU 1 to GPU 0 after a verified prefix of completed predictions; the resumed evaluator keeps the same checkpoint and canonical manifest. The later OOD supervisor detects the GPU 2 worker, waits for its completion, and resumes its shard there if needed; rank 0 runs on GPU 0. Final results require the full frozen manifests, exact sample identity, and both completed shards.

Implementation and gates are in `model/c2_preln_cross_attention.py`, `scripts/phase6j0_c2_train.py`, `scripts/phase6j0_c2_finish.py`, and `configs/phase6j0_c2_preln_cross_attention.yaml`.
