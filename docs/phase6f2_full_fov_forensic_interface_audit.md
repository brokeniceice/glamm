# Phase 6F.2 — Full-FOV Forensic Evidence Interface Audit

## 0. Scope and outcome

This phase is a code/architecture design audit only. It did **not** train a model, modify a checkpoint, run evaluation, or access internal test / Official1000 / OOD data. No GPU job was started.

The fixed system is `C1 + new R1`. The forensic source remains the same frozen CLIP and frozen Phase4C-A adapter. The proposed change is limited to acquiring the same forensic representation from deterministic full-FOV tiles and adapting the two existing R1 input interfaces.

**Decision: `PROCEED_FULL_FOV_GRID`.**

The lowest-risk design is:

```text
full image
  -> resize shortest side to 336 (aspect ratio preserved)
  -> deterministic 336x336 tiles along the long dimension
  -> same frozen CLIP + same frozen Phase4C-A adapter, independently per tile
  -> geometrically feather-and-normalize tile F/z into one variable-size
     full-FOV forensic grid at native 24-cells-per-336-pixels density
       |-> flatten grid + original-normalized coordinates -> existing Rectifier
       `-> resample grid to 64x64 -> existing Utility body
```

This is preferred over passing all overlapping tile tokens directly because direct concatenation is mechanically accepted by the attention code but causes overlap regions to have more keys and therefore an uncontrolled multiplicity prior. A stitched grid removes duplicates once, gives Rectifier and Utility one shared representation, and preserves the learned parameters and main computation of both paths.

## 1. Audited frozen interfaces

### 1.1 Current forensic source

Each current center crop produces:

```text
F_forensic: [B, 256, 24, 24]
z_forensic: [B,   1, 24, 24]
```

Phase6F.1 already established that the `ResizeShortest336 + CenterCrop336` support, rather than the support-boundary implementation, is the principal coverage bottleneck. Phase6F.1's frozen population contained 1,106 samples; its support-boundary audit found only a small aggregate GT-access fraction and did not support the boundary as the main explanation. This phase therefore does not redesign the backbone, adapter, Rectifier, or Utility learner.

### 1.2 Rectifier implementation

The relevant path is:

- `model/sam_forensic_rectifier.py::GeometryAwareSAMRectifier.forward`
- `model/tf_fdg.py::CrossAttentiveSemanticRectification.forward`
- `model/tf_fdg.py::GeometryAwareCrossAttention.forward`

The wrapper currently assumes a grid only because it executes:

```python
forensic = evidence.flatten(2).transpose(1, 2)
```

The attention implementation itself obtains `nk = key.shape[1]` at runtime. Its Q/K/V projections, coordinate bias, key-valid mask, softmax, and output projection contain no `nk == 576` assertion and no parameter whose shape depends on 576. Consequently, `[B, M, 256]` with variable/padded `M` is already compatible with the learned attention parameters.

The current fixed-size assumptions are peripheral:

1. The Rectifier wrapper accepts an evidence grid and flattens it, rather than accepting token-form evidence explicitly.
2. Call sites commonly construct `evidence_valid` as `[B, 576]`.
3. `semantic_support()` takes the axis-aligned min/max of all evidence coordinates. It does not currently exclude invalid padded coordinates and represents support as one rectangle.
4. Some historical corruption diagnostics explicitly use `24 * 24`; these are not in the normal forward path, but must not be reused unmodified for a variable grid.

### 1.3 Utility implementation

The deployed Utility path is centered on `model/csculf.py::CSCULF` and the equivalent materialization in `scripts/phase4g1q_conditional_utility.py::utility_forward`.

Its learned body consumes aligned 64×64 tensors:

```text
ForensicContext64
CMXJointRectification
LocalCrossExchange (7x7)
ConditionalUtilityHead
```

The center-crop dependency is concentrated before this body:

```text
F24 / z_F24 / forensic masses
  -> CSCULF._map_forensic()
  -> resample_clip_to_original_normalized(single clip_geometry)
  -> aligned 64x64 tensors + support
```

The downstream learned layers do not require a 24×24 source. They require only aligned 64×64 feature, logit/posterior, and support tensors. Thus the Utility parameters and its main network can remain bit-identical if the interface supplies correctly aligned full-FOV 64×64 inputs.

## 2. Rectifier: multi-tile feasibility

### 2.1 Can `N x 576` tokens be sent directly?

**Yes mechanically, but not as the recommended A1 representation.**

The following tensor contract is valid for the current attention parameters:

```text
forensic_tokens       [B, M, 256], M = N * 576 after padding
forensic_coordinates  [B, M, 2], original-normalized (x, y)
forensic_valid        [B, M]
SAM tokens            [B, 4096, 256]
SAM coordinates       [B, 4096, 2]
```

Overlapping tile tokens can coexist numerically. However, attention normalizes over keys. If an original region appears in two tiles, that region contributes approximately twice as many keys as a singly covered region. Coordinate bias does not remove this duplication. Direct coexistence therefore changes the evidence prior as a function of tile overlap and makes zero-shot interpretation unnecessarily difficult.

Adding confidence weights or a log multiplicity correction inside attention would alter its scoring interface. Hard ownership masks would introduce seams and discard one of two contextual views. Neither is as minimal or auditable as deterministic geometric stitching before attention.

### 2.2 Recommended Rectifier input

Construct a canonical full-FOV grid at the native CLIP-cell density after the per-tile adapter:

```text
resized image: H_r x W_r, min(H_r, W_r) = 336
full grid: H_f x W_f
H_f = round(24 * H_r / 336)
W_f = round(24 * W_r / 336)
```

Each tile's 24×24 `F_forensic` cells are placed into this grid by original resized-image coordinates and fused as described in Section 3. The resulting grid is flattened:

```text
F_full [B, 256, H_f, W_f]
  -> F_tokens [B, H_f*W_f, 256]
coordinates [B, H_f*W_f, 2]
valid [B, H_f*W_f]
```

Variable-sized samples are padded to the maximum `M` in a batch; padding must be marked false in `forensic_valid`. The Rectifier's learned LayerNorm, Q/K/V/out projections, residual projection, and learned `gamma` remain unchanged.

### 2.3 Support

Do not use the current unmasked min/max over padded evidence coordinates. Supply support explicitly from the stitched coverage raster:

```text
support_full = accumulated_weight > epsilon
support_SAM  = sample support_full at the 64x64 SAM cell centers
```

For the proposed tiling, valid content coverage is contiguous and full, so this support should be true over all valid SAM content cells. Pixels arising only from SAM right/bottom padding remain unsupported. An explicit support tensor is more exact than a coordinate bounding box and continues to work if a future acquisition protocol contains holes.

### 2.4 Required Rectifier code change

Add a backward-compatible token-form input path to `GeometryAwareSAMRectifier.forward`, or a separate thin `forward_tokens` method:

```text
legacy: evidence_grid [B,256,H,W] -> flatten internally
new:    evidence_tokens [B,M,256] + evidence_coordinates + evidence_valid
```

Also permit an optional externally supplied `semantic_support`. If absent, retain the legacy behavior for exact A0 compatibility. No Rectifier state-dict key or parameter shape changes.

## 3. Utility: minimum multi-tile interface

### 3.1 Grid choice

The recommended shared representation is the independent variable-size full-FOV forensic grid described above, not a Utility-only 64×64 canvas.

Reasons:

1. It preserves the native adapter output density along the long dimension instead of first collapsing all evidence to 64×64.
2. The same de-duplicated evidence can feed both Rectifier and Utility.
3. It isolates overlap fusion to one auditable operation.
4. Utility can still receive exactly the 64×64 tensors expected by its learned network via one deterministic resampling step.

Building 64×64 directly is a viable Utility-only shortcut, but it would leave Rectifier with a separate overlapping-token representation and two different fusion semantics. The independent full-FOV grid is therefore the smaller **system-level** interface change.

### 3.2 Overlap fusion

For every tile and every target cell, use a fixed geometry-only separable feather weight. A triangular/tent window with a small positive floor at the outer image boundary is sufficient:

```text
acc_F += w_tile * sampled_F_tile
acc_z += w_tile * sampled_z_tile
acc_w += w_tile

F_full = acc_F / max(acc_w, eps)
z_full = acc_z / max(acc_w, eps)
support = acc_w > eps
```

The same scalar `w_tile` must be used for `F_forensic` and `z_forensic`. The window is based only on distance to the tile edge and is fixed before seeing validation results.

Choice audit:

- **Geometry-weighted average (recommended):** deterministic, smooths tile-boundary artifacts, does not use labels or model confidence, and is easy to invariance-test.
- Plain average: simpler but gives low-context edge cells the same weight as tile interiors; acceptable as a unit-test reference, not the preferred A1 arm.
- Nearest / hard ownership: introduces seams and discards redundant contextual evidence.
- Confidence weighting: changes the semantics using model outputs, can amplify confident errors, and would require a new confidence definition. It is not minimal.

For `z_forensic`, average the pre-sigmoid logit using the same geometry weights. Do not average thresholded probabilities and do not let `z` determine its own fusion weight.

### 3.3 Evidential mass ordering

The frozen `ForensicEvidentialHead` is convolutional and was trained on a 24×24 tile. To preserve its learned local operating point, run it independently on each tile before stitching its masses/posteriors. Do **not** run the frozen head for the first time on an elongated stitched grid and assume equivalence at tile seams.

For Utility, maintain two synchronized stitched products:

```text
1. stitched F_forensic and z_forensic -> 64x64 -> ForensicContext64
2. per-tile frozen forensic evidence head -> masses -> stitched 64x64 masses
```

Mass fusion uses the same geometry weights followed by a per-pixel renormalization across the mass channel. Unsupported cells use the existing vacuous mass. This preserves the frozen source head while avoiding nonlinear disagreement between “average inputs then apply head” and “apply head then average outputs.”

### 3.4 Required Utility code change

Add an explicit pre-aligned interface to `CSCULF.forward`, for example:

```text
aligned_F64
aligned_z_F64
mass_F64
support64
mass_F256
support256
```

When these are supplied, bypass only `_map_forensic()` and the single `clip_geometries` branch. The following remain unchanged:

```text
ForensicContext64
CMXJointRectification
LocalCrossExchange
comparison feature construction
ConditionalUtilityHead
adapted ECoLaF fusion
all learned weights
```

Keep the legacy `f24 + z_f24 + clip_geometries` path intact and verify A0 bitwise/numerical parity. Do not encode a synthetic “giant crop geometry” to force a stitched grid through `_map_forensic`; an explicit aligned-input contract makes support and provenance auditable.

## 4. Deterministic tile protocol

### 4.1 Acquisition

1. Resize the shortest image side to 336 while preserving aspect ratio, exactly matching the current scale rule.
2. The short dimension has one tile origin at zero. Tile only along the long dimension.
3. Tile size is fixed at 336×336.
4. Fix nominal stride at **280 pixels** (56-pixel / 4-cell nominal overlap).
5. Use origins `0, 280, 560, ...`; if the final origin does not equal `L-336`, append an end-anchored origin at `L-336`. Remove exact duplicates.
6. Record resized shape, every tile box, nominal stride, actual pairwise overlap, grid shape, and coordinate transform in each sample's provenance.

Tile count is therefore:

```text
N = 1                                      if L == 336
N = 1 + ceil((L - 336) / 280)             otherwise
```

The last overlap may be larger than 56 pixels because the final tile is end anchored; this is deterministic and the feather normalization handles it. The protocol covers both extreme ends and never silently drops a suffix.

### 4.2 Square and near-square images

An image resized to exactly 336×336 uses the existing single view, and the new stitcher must reproduce the legacy 24×24 tensors within numerical tolerance when its feather normalization is applied to one tile.

If the resized long side is even slightly greater than 336, strict full-FOV semantics require more than one tile. Do not retain a center-only exception for “near square,” because that would preserve the very blind strip under study. Tile count remains variable.

### 4.3 Global view and compute cap

Do not add a separate global-center view in A1. It duplicates evidence, changes scale, and reintroduces multiplicity without addressing a missing region. The deterministic tile set is sufficient.

Do not silently cap tile count, because a cap violates full-FOV coverage. Instead, preflight the aspect-ratio/tile-count distribution on the authorized development split, preregister a maximum supported `N`, and fail explicitly if an image exceeds it. Tiles may be passed through frozen CLIP/adapter in a microbatch or stream to bound peak memory.

### 4.4 Coordinates

For a tile box `(top, left, bottom, right)` in the resized image, each 24×24 cell center maps to:

```text
x = (left + (j + 0.5) * tile_width  / 24) / resized_width
y = (top  + (i + 0.5) * tile_height / 24) / resized_height
```

These are original-normalized coordinates because aspect-preserving resize does not change the fractional position. Retain `(x,y)` ordering used by `normalized_grid()` and the current coordinate positional encoding.

## 5. What changes and what remains fixed

### 5.1 Completely unchanged parameters/modules

- C1, including LoRA, H2, FRC projector, language path, and `[SEG]` query production.
- RINE used by C1, frozen and in eval mode.
- CLIP weights and preprocessing normalization.
- Phase4C-A forensic adapter weights and architecture.
- SAM image encoder, prompt encoder, and mask decoder.
- new-R1 Rectifier learned parameters and its Q/K/V/out/residual projections and `gamma`.
- new-R1 Utility learned modules: source heads, context projections, CSCU/CMX rectification, local exchange, utility head, and fusion rule.

### 5.2 Code that must change

1. **Forensic acquisition:** introduce deterministic resize/tile extraction, tile batching, and per-tile geometry/provenance.
2. **Stitcher:** add geometry-only feather accumulation for `F`, `z`, and frozen-head masses, plus full-grid coordinates, support, and validity.
3. **Rectifier wrapper:** accept token-form/variable-grid evidence and explicit semantic support; make any fallback coordinate support calculation validity-aware.
4. **Utility wrapper:** accept pre-aligned full-FOV tensors/masses/support and bypass the single-crop `_map_forensic()` calls only.
5. **Batch collation:** pad variable token counts and carry `evidence_valid`; never infer validity from coordinate values.
6. **Diagnostics/provenance:** remove normal-path assumptions of `24*24`, record tile boxes and coverage, and assert no uncovered valid image cells.

No state-dict migration is required. The new interface must load the selected new-R1 checkpoint strictly with no missing or unexpected learned keys.

## 6. Required implementation invariants before replay

The next implementation preflight should pass all of the following without optimizer construction:

1. Single 336×336 image: stitched `F`, `z`, masses, support, Rectifier output, Utility output, and final mask match the legacy path within declared dtype tolerance.
2. Wide and tall synthetic geometries: 100% of valid resized-image cells have positive accumulated weight; SAM padding remains unsupported.
3. Tile-order permutation leaves stitched tensors and model outputs invariant within tolerance.
4. Duplicate identical tile insertion leaves normalized stitched values invariant.
5. `F`, `z`, and mass support masks are identical.
6. Fused masses are finite, nonnegative within tolerance, and sum to one on support; unsupported mass is vacuous.
7. Padded token values do not change Rectifier output when their validity is false.
8. Original-normalized coordinates are monotone and match known tile endpoints.
9. All C1, CLIP, adapter, Rectifier, Utility, and SAM parameters have `requires_grad=False`, all are in eval mode, and optimizer count is zero.
10. A0 legacy-interface replay remains unchanged after the new code is added.

## 7. Next controlled experiment

### 7.1 Arms

```text
A0 = current single-center-crop C1 + selected new R1
A1 = full-FOV tiled acquisition
     + same frozen CLIP
     + same frozen Phase4C-A adapter
     + stitched full-FOV grid
     + unchanged selected new-R1 Rectifier and Utility weights
```

### 7.2 First pass: frozen replay

Run only on the authorized internal validation population used for development:

```text
C1 frozen
CLIP frozen + eval
adapter frozen + eval
Rectifier frozen + eval
Utility frozen + eval
SAM frozen + eval
optimizer count = 0
```

Use the same canonical G0 prompt, SEG trigger, mask evaluator, population, ordering, and sample IDs for A0 and A1. Report the existing localization metrics and paired per-image deltas, with explicit stratification by Phase6F.1 coverage/GT-outside-crop fraction and by tile count. Record SEG trigger equality; the acquisition change must not alter C1 generation.

This replay is worth doing first. It directly tests whether the selected new-R1 functions are sufficiently coordinate- and acquisition-compatible without conflating coverage with optimization. A zero-shot failure does not by itself refute the coverage diagnosis, because both Rectifier key-set statistics and Utility input statistics change.

### 7.3 If frozen replay is insufficient

Do not unfreeze the adapter first. The smallest justified retraining scope is:

```text
CLIP                 frozen
Phase4C-A adapter    frozen
C1 / SAM             frozen
Rectifier            trainable under the existing new-R1 recipe
Utility              trainable under the existing new-R1 recipe
```

Both Rectifier and Utility receive distribution-shifted evidence, so retraining only one while freezing the other is not the clean default. If attribution is needed, predeclare separate Rectifier-only and Utility-only ablations; do not select among them using test/OOD. Any retraining requires a later, separately authorized phase and internal-validation-only selection.

## 8. Compute and memory risk

### 8.1 Rectifier attention

Current attention has shape approximately:

```text
[B, 8, 4096, 576]
```

With direct `N`-tile concatenation it becomes `[B,8,4096,N*576]`, so attention compute and memory grow linearly with `N`. The raw BF16 attention tensor alone is approximately:

| Evidence keys | Raw attention / image |
|---:|---:|
| 576 (current) | 36 MiB |
| 1,152 | 72 MiB |
| 1,728 | 108 MiB |
| 2,304 | 144 MiB |

This excludes logits, distance matrices, Q/K/V, output, allocator overhead, and any saved training activations. Coordinate distance is also `[B,4096,M]` and normally float32, adding about 9 MiB per 576 keys per image.

The stitched native-density grid is cheaper than raw concatenation whenever tiles overlap. For a 2:1 resized image it has roughly `24x48 = 1,152` keys, compared with 1,728 keys under the proposed three-tile raw concatenation.

For frozen replay, use image batch size 1 initially, stream/microbatch CLIP tiles, inference mode, and avoid retaining attention maps unless the diagnostic explicitly needs them. A future optional query-chunked attention implementation can reduce peak memory without changing parameters, but it is not required for the first implementation and must pass numerical parity tests.

### 8.2 CLIP and adapter

CLIP/adapter FLOPs grow approximately with tile count `N`. They need not reside in memory as one large tile batch: process a bounded tile microbatch and accumulate the stitch buffers. The frozen source makes this straightforward and avoids activation retention.

### 8.3 Utility

The learned Utility body remains 64×64, so its dominant CSCU/local-exchange memory and compute remain essentially unchanged. Extra cost is tile source-head inference, stitch accumulation, and resampling, all linear in `N` and small relative to Rectifier attention and CLIP.

## 9. Explicit answers

1. **Can Rectifier extend at low cost?** Yes. Its learned attention is token-count agnostic. A thin variable-token/explicit-support interface is required; no Rectifier parameter or architectural block must change.
2. **How should Utility support multi-tile evidence?** Stitch per-tile adapter outputs and per-tile frozen-head masses in original-normalized geometry, resample to its existing 64×64 space, and bypass only the legacy single-crop mapping. Keep the full CSCU/Utility network unchanged.
3. **Recommended full-FOV representation:** a variable-size, native-cell-density full-FOV forensic grid, produced after applying the adapter independently to every tile; flatten it for Rectifier and resample it for Utility.
4. **Recommended tile protocol:** shortest-side resize to 336, 336×336 tiles along the long dimension, nominal stride 280 / overlap 56, final tile end anchored, variable `N`, no extra global view, deterministic geometry provenance.
5. **Modules completely unchanged:** C1, RINE, CLIP, Phase4C-A adapter, SAM, all Rectifier learned blocks/weights, and all Utility learned blocks/weights.
6. **Code that must change:** acquisition/stitching, Rectifier input/support wrapper, Utility pre-aligned-input wrapper, variable-length collation, and invariance/provenance diagnostics.
7. **Is frozen replay worthwhile first?** Yes. It is the cleanest controlled test of zero-shot compatibility and must precede any retraining.
8. **Minimum controlled experiment:** A0 current center crop versus A1 frozen full-FOV grid on the same internal-validation samples under canonical G0, with paired and coverage-stratified analysis.
9. **Memory/compute risk:** CLIP cost and Rectifier attention scale approximately linearly with tile count; stitched grids remove overlap duplication. Utility's learned 64×64 cost stays nearly fixed. Start frozen replay at batch size 1 and stream tiles.

## 10. Final conclusion

```text
RECTIFIER_MULTI_TILE_LOW_COST = YES
UTILITY_PARAMETERS_CAN_REMAIN_UNCHANGED = YES
FROZEN_REPLAY_FIRST = YES
FINAL_DECISION = PROCEED_FULL_FOV_GRID
```

The direct `N x 576` token concatenation is feasible as a diagnostic implementation, but it is not the recommended controlled arm because overlap multiplicity changes attention semantics. The unified stitched full-FOV grid is the smallest auditable interface that solves coverage for both Rectifier and Utility while keeping the established `C1 + new R1` learned architecture intact.
