# Phase6L0 R2 preflight audit

Status: **INCOMPLETE; formal training is gated.** This audit records the repository baseline and the current R2 preflight evidence. Existing uncommitted C2 and Phase6K files are preserved.

## Implementation inventory

| Existing source | Audited role | R2 decision |
| --- | --- | --- |
| `model/c2_preln_cross_attention.py`, `scripts/phase6j0_c2_evaluate.py`, `scripts/phase6j0_c2_train.py` | C2 architecture, loader, frozen/trainable split | Use only frozen C2 loader/capture |
| `scripts/phase6k0_c2_after_cache.py`, `scripts/phase6k0_c2_capture_parity.py`, `scripts/phase6k0_c2_split_cache.py`, `scripts/phase6k0_c2_after_train.py` | A/E capture definition, parity, cache/provenance and canonical IDs | Reuse the capture code path and ID source; recapture A/E/q with r_prime in one forward |
| `model/clip_forensic_adapter.py` | Historical local block and dense-head structure | Use block design experience; instantiate a new R2 block without old weights or dense head |
| `model/sam_forensic_rectifier.py`, `model/csculf.py`, `model/tf_fdg.py` | Frozen prompt/mask runtime, old Rectifier/Utility/global cross-attention | Use lightweight prompt/mask runtime structure only; no old Rectifier, Utility or 4096×576 attention |
| `scripts/phase6e2_c1_specific_r1_train.py`, `scripts/phase6g*`, `tools/phase4f.py` | Historical canonical traversal, spatial batch, frozen SAM decode and mask loss | Use canonical order, frozen source cache and final mask-loss contract; no old R1 checkpoint or module |
| `tools/phase4e1.py`, `model/pcerf.py`, `tools/phase3c1.py` | Audited coordinate/resampling and geometry functions | Use `sam_coordinates` and `geometry_for` to map CLIP crop features to the padded SAM lattice |

## Frozen C2 and capture

- Selected C2: `checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt`; measured SHA256 `4a67e6a87c453d554fa5bd6cf93329ae54c1c2853f0925dc7a63397eef27e8ce`. `outputs/phase6j0_c2/final_evaluation/protocol.json` selects epoch 7 / optimizer step 3500.
- `model/c2_preln_cross_attention.py` implements hidden width 4096, inner width 512, eight 64-D heads, and 576 patches. Its existing Phase6K capture stores `A`, `E`, the pre-output head context, and the pre-fusion input. `E` is the FP32 product of attention probability and each head's value, laid out `[B,512,24,24]`. The fused output returned by the module is the actual `r_prime` used downstream; an existing forward hook in `scripts/phase6k0_c2_capture_parity.py` demonstrates how to capture it without changing the forward.
- `model/GLaMM.py::evaluate` first runs autoregressive `generate`, then a full-sequence `super().forward` replay; `projected_seg_embeddings` are extracted from that replay. The R2 cache takes the **last** C2 attention hook output and its `last_spatial_intermediates`, so `r_prime`, A and E come from the same full replay that produced q_seg. Earlier generation-time attention calls are discarded.
- Existing Phase6K `outputs/phase6k0_c2_after/cache/capture_preflight.json` measured maximum A head-mass error `1.1920929e-07` and E/context maximum absolute error `0.0210800` (relative to maximum context magnitude `0.00107414`, with bf16 accumulation). `full_capture_parity.json` records exact fused token, raw classification logits, generated IDs, q_seg, and SAM mask on one C2 sample. R2's own fixed-sample `outputs/phase6l0_r2/cache/preflight_subset.json` now repeats exact capture-on/off parity for those outputs, including raw classification logits, and C2-native SAM exact parity.

## Canonical population and spatial sources

- Phase6K immutable cache index: `outputs/phase6k0_c2_after/cache/{train,val}.json`; payloads: `/data/yz/groundingLMM_official/cache/phase6k0_c2_after_v2/{train,train_tail,val}/shard_*.pt`. TRAIN has 8,836 images, 8,682 exactly-one-SEG valid and 154 zero-SEG invalid. DEV has 1,106 images, 1,080 valid and 26 zero-SEG invalid. Ordered ID hashes are in the index files; every shard has an indexed SHA256. The cache has `q_seg [N,256]`, `A [N,8,24,24]`, `E [N,512,24,24]`, validity and SEG count. It **does not contain `r_prime`**, so it is insufficient as an R2 cache alone. R2's single-sample forward differed from the Phase6K train cache (A max abs `0.0002231`, E max abs `0.07751`, q_seg max abs `4.0`); the two used different generation batch groupings. Hence R2 captures A/E/r_prime/q_seg afresh in the same C2 forward and only uses Phase6K for canonical ID/provenance checks. There is no C1/P1 q fallback.
- Phase3C.1 frozen spatial cache: `outputs/phase3c1_spatial_probe/cache/{sam,clip}/{train,val}/`. Its per-source `complete.json` declares the 8,836-image TRAIN and 1,106-image DEV populations, provenance and source-parameter hashes. A sampled shard has `S64 [N,256,64,64]` and raw CLIP `[N,1024,24,24]`, both bf16. The same sampled SAM/CLIP shard IDs and order agree. Full cross-cache order and source parity remain R2 gates.
- The raw CLIP source is `CLIPImageProcessor-ResizeShortest336-CenterCrop336-Rescale-Normalize:v1` with ViT-L/14@336 (`configs/phase6j0_c2_preln_cross_attention.yaml`, `scripts/phase3c1_cache.py`). The patch count is 24×24=576 and the raw width is 1024. The spatial cache was made from P1's **frozen source**; the R2 subset audit measured tensor-exact C2/P1 S64 and raw CLIP equality and matched both source module hashes to their Phase3C.1 manifests. Full ordered IDs, geometry and shard SHA still need cache-completion checks.
- `tools/phase3c1.py::geometry_for` gives the audited CLIP center crop and SAM resize/right-bottom padding. `tools/phase4e1.py::{sam_coordinates,clip_coordinates}` gives their cell centers in original-normalized coordinates. `model/pcerf.py::resample_clip_to_original_normalized` maps a CLIP feature into an *original-normalized* output grid and returns crop support. Because S64 is on SAM's **padded** lattice, an R2 bridge must use the SAM cell coordinates and CLIP crop geometry together; directly applying the original-normalized 64×64 mapper then adding to S64 would misalign non-square images. `sam_lowres_to_original_normalized` also removes SAM padding and therefore is not the desired direct `z_L -> 64×64 SAM query` operation. Native low-res SAM logits can be bilinearly resized to 64×64 for that query.

## SAM runtime and loss

- `model/sam_forensic_rectifier.py::FrozenP1SAMPath` defines the lightweight frozen prompt encoder and mask decoder. Its input `q_seg` may be `[B,256]` or `[B,1,256]`; native output is low-resolution mask logits `[B,1,256,256]`. `tools/phase4f.py::mask_loss` interpolates logits to the 1024² SAM target, then computes `2.0 * BCEWithLogits + 0.5 * soft Dice` under the current Phase4F config.
- **Critical non-identity repaired for R2:** `/data/yz/groundingLMM_official/cache/phase4f_language_preserving_rectification/p1_sam_runtime.pt` is P1, not C2. Comparing identical mask-decoder keys in the formal C2 checkpoint and this runtime found **96 of 120 tensors unequal**. C2 trained its mask decoder (`configs/phase6j0_c2_preln_cross_attention.yaml`). `scripts/phase6l0_r2_preflight.py` now constructs a C2-native runtime using the checkpoint's full decoder and the base-frozen prompt state (the prompt positional matrix is exact equal in C2). On one fixed C2 sample, its low logits equal direct C2 tensor-exact; the zero-head R2 S64 and SAM low logits also equal C2-G0 tensor-exact. The runtime is saved in `outputs/phase6l0_r2/cache/c2_sam_runtime.pt` with SHA in `preflight_subset.json`. No R2 training uses the P1 decoder.
- The C2 cache q_seg storage is `[N,256]` rather than the anticipated `[B,1,256]`; the runtime adds the singleton prompt dimension. This is a representation detail, not a fallback.
- The R2 residual head is exactly zero-initialized. Its FiLM projection weights use a tiny `1e-4` normal initialization with zero bias: an exactly zero FiLM weight would block GlobalConditioner gradients on the required second backward after a single zero-head update. Step0 SAM identity is still exact because the final residual head is zero. This resolves the spec's suggested FiLM initialization in favor of its mandatory two-backward gradient gate.
- A **synthetic GPU smoke check** using the C2-native frozen SAM runtime confirmed first-backward W_out gradient `10.2488` with upstream zero, then second-backward nonzero gradients for Evidence `0.02281`, Bridge `0.01793`, GlobalConditioner `3.90e-05`, all three adapter blocks, and W_out `9.3132`; frozen SAM gradients were all `None`. This is not the formal real-cache G6 gate, which remains pending.
- A **real first-cache sample** has now passed the two-backward audit (`outputs/phase6l0_r2/cache/real_subset_gradient_gate.json`): step0 SAM low logits exactly equal cached C2-G0; first W_out gradient norm `15.6179`, upstream zero; after one audit optimizer step, Evidence `0.12293`, Bridge `0.07728`, GlobalConditioner `1.49e-05`, all three adapter blocks, and W_out `6.62424` have nonzero gradients. Frozen SAM gradients remain `None`, and its state hash is unchanged. The trainer repeats this gate after both full caches complete.

## Gate ledger

| Gate | Current state | Required next evidence |
| --- | --- | --- |
| G1 checkpoint | PASS | Recheck SHA before any runtime work |
| G2 capture parity | R2 fixed-sample PASS | Verify full cache capture |
| G3 A/E context | R2 fixed-sample PASS with measured bf16 tolerance | Verify full cache finite and mass checks |
| G4 cache ID/order/geometry | PENDING | Full C2/R2/spatial ID and geometry check; capture `r_prime` |
| G5 step0 C2-G0 | R2 fixed-sample PASS | Verify cached TRAIN/DEV and epoch0 DEV |
| G6 gradient routing | Real fixed-sample PASS | Recheck in formal trainer on complete cache |
| G7 frozen integrity | PENDING | State hashes before/after each epoch |
| G8 finite pass | PENDING | Full batch finite check |

**Training stop:** G4-G8 have not all passed. Formal optimizer steps remain gated on complete fresh R2 caches, full canonical DEV epoch0 parity, gradient routing, and frozen-state checks.
