# Phase 4B-G — Global FEPN Interface Contract

## Frozen representation

Phase 4B-G uses the Phase 4A selected FEPN-v0 epoch-6 checkpoint. The exact primary tensor is:

```python
dense_features, _ = fepn.encode(normalized_rgb)
E_global = dense_features.mean(dim=(2, 3))
```

`E_global` has shape `[B,128]`. It is the input of `FEPNv0.global_head`, before both linear layers in that classification MLP. There is no activation or normalization after global average pooling. It is neither the final scalar logit nor `P(fake)`. FEPN is always in `eval()` with every parameter frozen; cached features carry stop-gradient semantics.

Input preprocessing is the Phase 4A frozen CLIPImageProcessor route: RGB, resolution 336, CLIP ViT-L/14-336 resize/center-crop/rescale/normalization, with mean `[0.48145466,0.4578275,0.40821073]` and standard deviation `[0.26862954,0.26130258,0.27577711]`.

## Projector and placement

The only projector architecture is `Linear(128,256) → GELU → Linear(256,16384)`, reshaped to `[B,4,4096]`. It has 4,243,712 parameters. There is no K, depth, pooling, learning-rate, or activation sweep.

The raw prompt and tokenizer vocabulary are unchanged. GLaMM normally replaces one raw `<image>` token with 576 CLIP-projected embeddings, a net raw-to-expanded offset of 575. Phase 4B-G appends four continuous FEPN embeddings to those visual embeddings, giving this causal sequence:

```text
[576 original visual embeddings]
[4 projected FEPN evidence embeddings]
[original canonical text embeddings]
```

The resulting net image expansion is 579. Classification and `[SEG]` predictor lookup therefore use the same dynamic 579 offset in evidence arms, while original P1 continues to use 575. Evidence is present in the initial generation forward and is retained by the normal KV cache; subsequent one-token decoding steps do not reinsert it. Labels over all 580 visual/evidence embeddings are `IGNORE_INDEX`, and the original raw text labels and token IDs are unchanged.

## Trainable boundaries

- PROJ-ONLY: projector only.
- PROJ-LORA: projector plus the original P1 LoRA tensors.
- Frozen in both: FEPN, LLM base, embeddings, LM head, original visual projector, classification head, `text_hidden_fcs`, mask decoder, grounding image encoder and SAM.

The only loss is the canonical P1 phrase-aligned language CE. Both Real and Fake receive exactly four evidence tokens. Internal test and official1000 remain sealed.
