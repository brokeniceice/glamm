# Phase 6A — P1/R1 Classification Trace

## 结论

P1 的正式分类不是 CLIP CLS 线性头。它读取 multimodal LLM 在固定 assistant-prefix `[CLS]` 位置的最终层 hidden state `[B,4096]`，再经过 `Linear(4096,2)`。R1 checkpoint 没有任何 classification、LLM、LoRA 或 `text_hidden_fcs` 权重；R1 分类在定义上是 P1 的 EXACT REUSE。

## 完整调用链与 shape

| Stage | Source | Function | Tensor contract | Frozen/trainable in P1 |
|---|---|---|---|---|
| RGB decode | `dataset/forensics/unified.py` | `UnifiedForensicsDataset.__getitem__` | RGB `[H,W,3]` | data operation |
| CLIP preprocessing | same | `CLIPImageProcessor.preprocess` | `[3,336,336]` | no parameters |
| fixed query construction | `dataset/dataset.py:270-318` | `custom_collate_fn` | assistant begins `[CLS]`; `[CLS]` target ignored | no parameters |
| CLIP vision | `model/llava/model/multimodal_encoder/clip_encoder.py` | `CLIPVisionTower.forward` | selected layer `-2`, CLS discarded, patch tokens `[B,576,1024]` | frozen |
| multimodal projection | `model/llava/llava_with_region_arch.py` | `encode_images` / multimodal preparation | `[B,576,4096]`, inserted at image token | frozen `mm_projector` |
| LLM | `model/GLaMM.py:465-504` via LLaVA/LLaMA | `_training_path` / `_inference_path` | final hidden `[B,L_expanded,4096]` | base frozen; LoRA q/v, embeddings and LM head trainable in P1 |
| CLS extraction | `model/GLaMM.py:399-410` | `_extract_cls_logits` | gather fixed `[CLS]` position → `[B,4096]` | hidden producer partly trainable through LoRA |
| classifier | `model/GLaMM.py:200-209` | `classification_head` | `[B,4096] → [B,2]` | trainable |
| image aggregation | `model/GLaMM.py:431-445` | `_aggregate_cls_logits` | mean valid conversation logits per image → `[N_image,2]` | differentiable |
| loss/decision | `model/GLaMM.py:447-463` | CE / argmax | CE on Real=0, Fake=1; argmax or softmax Fake probability | classifier and upstream trainable paths receive gradient |

CLIP CLS is not carried into the P1 LLM because `select_feature="patch"` removes token 0. The classifier therefore reads a prompt-conditioned LLM query representation that causally sees the image patch tokens and canonical user prompt, but is positioned before `[REAL]`/`[FAKE]` and explanation tokens.

## Gradient contract

`configs/phase3a_p1.yaml` and `train.py:set_trainable_modules/build_optimizer_parameter_groups` establish six optimizer groups:

- LoRA `q_proj/v_proj`;
- `classification_head`;
- `text_hidden_fcs`;
- SAM `mask_decoder`;
- token embeddings;
- `lm_head`.

Vision tower, `mm_projector`, SAM image encoder, region encoder and base LLM weights are frozen. Classification CE is computed directly from `[CLS]` logits, so gradients reach the 8,194-parameter classification head, the `[CLS]`/special-token embedding rows and the active LLM LoRA path. It does not use `text_hidden_fcs` or SAM; those can receive other task losses but not classification CE through this branch.

## Checkpoint evidence

P1 checkpoint:

- path: `/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt`
- SHA256: `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`
- step/epoch: `3500 / 7`
- keys: `base_model.model.classification_head.weight [2,4096]` and `.bias [2]`, BF16;
- `text_hidden_fcs`: `[4096,4096] → ReLU → [256,4096]`; this is SEG projection, not classifier input.

R1 checkpoint:

- path: `outputs/phase4hd/r1/selected_checkpoint.pt`
- SHA256: `9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5`
- top-level keys: `arm, config_sha256, epoch, optimizer, optimizer_updates, rectifier_state, schema, utility_state, validation_g0`;
- no classification/LLM/LoRA/text projection keys.

Therefore current R1 localization improvements have **not** entered classification.

## Causal order of current deployed-style inference

```text
RGB image
  ├─ CLIP preprocess → patch grid → mm_projector ─┐
  │                                               ↓
  │ canonical prompt + fixed [CLS] → LLM hidden at [CLS]
  │                                               ↓
  │                                  classification_head → Real/Fake
  │                                               ↓
  │                           generated verdict/explanation/phrase/[SEG]
  │                                               ↓
  ├─ SAM image encoder → S64 ─────────────────────┤
  └─ CLIP forensic arm → F24,z_F24 ───────────────┤
                                                  ↓
                               q_seg + preliminary z_L + CSCU utility
                                                  ↓
                         R1 rectifier/gate → adapted SAM embedding
                                                  ↓
                                           SAM final mask
```

The image-only forensic arm and image-side rectifier can be computed before the verdict, but the complete CSCU utility/gated R1 path cannot: it consumes q-seg and preliminary segmentation logits produced after a valid `[SEG]` trajectory.

## Firewall

No model inference, training, test/Official1000/external benchmark access, or checkpoint modification was performed.
