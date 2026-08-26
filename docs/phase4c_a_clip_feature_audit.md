# Phase 4C-A exact CLIP feature audit

The matched Phase 3C.1 source is the frozen CLIP vision tower embedded in canonical P1 (`openai/clip-vit-large-patch14-336`, P1 SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`). The tower selects hidden state **-2**, removes the CLS token, and reshapes 576 patch tokens into a **1024 x 24 x 24** spatial tensor.

The image path uses `CLIPImageProcessor`: resize the shortest edge to 336, center-crop 336, rescale, and CLIP-normalize. The official annotation-derived per-image all-reference union mask follows the same geometry with nearest-neighbor interpolation. Evaluation bilinearly upsamples logits into the crop, fills unseen crop exterior with background logit -100, resizes to original resolution, and thresholds at logit 0.

Train and validation caches contain 8836 and 1106 unique Fake samples. Validation identity SHA256 is `2b12a1c0280f578278e29790d210cca338064828480d4d087e2624c6edecf589`. Frozen CLIP parameter hash is `ea25ce94579902eb0a94c9638c0277360f2b93cc156eb20f2fc4a481653afc17` before and after extraction. The historical raw linear probe is therefore an exact protocol match and is reused without retraining.
