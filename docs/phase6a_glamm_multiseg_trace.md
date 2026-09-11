# Phase 6A — GLaMM and LEGION Native Multi-SEG Trace

## Current GLaMM native chain

The generic GCG path remains natively multi-SEG even though the unified forensic adapter uses one SEG.

| Stage | Source/function | Contract |
|---|---|---|
| raw grounding | `dataset/gcg_datasets/GranDf_gcg_ds.py:GCGBaseDataset._parse_annotations` | one label/token span and one decoded mask per grounding |
| target tagging | same file, `create_conversations/tag_caption` | inserts `<p> phrase_i </p> [SEG]` at every `tokens_positive[i]` span |
| ordering | same file, `process_data` | phrase spans, labels and masks sorted by caption start |
| dataset assertion | `dataset/segm_datasets/Semantic_Segm_ds.py` and GCG construction | SEG count must equal mask count; masks `[K,H,W]` |
| collation | `dataset/dataset.py:custom_collate_fn` | variable mask tensors remain a list per image; conversations flatten with `offset` |
| causal extraction | `model/GLaMM.py:extract_seg_predictor_hidden` | all `[SEG]` positions are found; hidden immediately before each SEG is returned as variable `[K_i,4096]` |
| projection | `model/GLaMM.py:_extract_projected_seg_predictor_hidden` | shared `text_hidden_fcs`, each image gets `[K_i,256]` |
| SAM decode | `model/GLaMM.py:_generate_and_postprocess_masks` | text prompts `[K_i,1,256]`; one image embedding is repeated by SAM; output `[K_i,H_i,W_i]` |
| training alignment | `model/GLaMM.py:_compute_loss_components` | requires `pred_count == gt_mask.shape[0]`; BCE/Dice over aligned index order |
| inference | `model/GLaMM.py:evaluate` | generated sequence may contain K SEG tokens; returns K original-resolution masks per image |
| image evaluator | `scripts/phase3a_evaluate.py:_union_logits` | max over mask logits, then threshold `>0`; equivalent binary union for image-level metrics |

The model has no `[:,0]`, first-SEG, or last-SEG reduction in the core extraction/decoder. The current mask loss, however, divides by total masks over the batch. Native GLaMM therefore gives an image with K phrases K times the mask contribution of a one-phrase image. The proposed forensic multi-SEG protocol must change this to per-phrase mean inside each image and then per-image mean.

## Official LEGION Stage-1 chain

At commit `d21535dd45f6fea509337a83095966f0b86ac924`:

1. `LegionGCGDataset._parse_annotations` iterates every `refs` entry whose `sentence` is a verbatim caption substring.
2. Each matched ref produces one phrase span, one decoded mask formed from all polygons inside that ref, and one label.
3. `GCGBaseDataset.create_conversations` inserts `<p> phrase_i </p> [SEG]` for every span, after sorting by caption position.
4. `LegionForCausalLM._create_seg_token_mask/_process_hidden_states` selects every SEG-aligned projected hidden and groups them by image using counts and offsets.
5. `_generate_and_postprocess_masks` gives SAM all K text embeddings and returns `[K,H,W]`.
6. `_compute_loss_components` truncates GT masks to predicted count when counts differ, then sums per-mask BCE/Dice and divides by total masks globally.
7. `scripts/loc_exp/infer.py:162-164` thresholds every predicted mask at `>0` and applies `torch.any(..., dim=0)` for the saved image-level mask.

Thus official LEGION allows **K = any number of caption-matched refs that survive token truncation**, not a fixed maximum in the model. In the frozen internal TRAIN conversion the observed maximum is 18; one long target had 18 refs but only 12 SEG units survive the official text limit. Official behavior silently drops unmatched phrases and truncates excess GT masks to predicted count. Those behaviors are reproducible facts, but a new P1/R1 experiment should expose rather than silently hide dropped phrase-mask pairs.

## Difference table

| Aspect | Generic/current GLaMM core | Unified P1/R1 data path | Official LEGION Stage 1 |
|---|---|---|---|
| phrase target | `<p>phrase_i</p>[SEG]` K times | joined `Target regions: phrase1; phrase2 [SEG]` | `<p>phrase_i</p>[SEG]` K times |
| GT masks | `[K,H,W]` | `[1,H,W]` all-ref union | `[K,H,W]`, ref polygon-union each |
| SEG extraction | all SEG | only one exists in target | all SEG |
| decoder | K masks | one mask | K masks |
| count mismatch | skip sample and warn in current hardened core | normally 1==1 | truncates GT to predicted K |
| mask weighting | global per-mask mean | equivalent to per-image because K=1 | global per-mask mean |
| inference output | K masks preserved | one mask | K masks preserved, official CLI unions |
| R1 integration | not part of generic core | assumes one cached q-seg/union target | N/A |

## Frozen TRAIN supervision audit

Source: `outputs/phase6a_architecture_audit/synthscars_phrase_mask_audit.json`.

| K phrases/original annotation | Count | Ratio |
|---:|---:|---:|
| 1 | 4,181 | 46.6057% |
| 2 | 2,229 | 24.8467% |
| 3 | 1,224 | 13.6440% |
| 4 | 639 | 7.1230% |
| 5+ | 698 | 7.7806% |

- 8,836 frozen Fake images select 8,971 original annotation samples.
- 19,173 phrase refs map directly to 19,173 polygon regions.
- 11,628 exact whitespace-normalized unique phrase strings; 7,545 repeated occurrences globally and 136 within-annotation duplicate occurrences.
- no phrase contains multiple polygon regions in this selected TRAIN scope; no exact decoded region is shared by two phrases.
- 608 phrase-mask pairs overlap across 480 annotations; overlap does not break ordered one-to-one supervision.
- zero empty phrases, zero missing polygon refs, one decoded empty mask.

**Recoverability gate: YES, qualified.** Ordered `phrase_i ↔ mask_i` supervision is available without manual relabeling. The one empty decoded ref must be handled by a predeclared deterministic invalid/zero-mask policy. Long-sequence phrase/SEG dropping must also update the mask list atomically; silent independent truncation is unacceptable for the proposed controlled ablation.

No validation/test/external benchmark was opened.
