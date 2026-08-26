# Phase 4F Gradient routing audit

```json
{
  "status": "PASS",
  "sample_ids": [
    "synthscars:train:d3d29749-b2f5-41d1-9ba4-9da51ca52ce8",
    "synthscars:train:285b8065-3235-4b12-9965-1c114ce5dc5b",
    "synthscars:train:593c9903-c327-464a-993a-a935fdd0ef2b",
    "synthscars:train:4fd3132c-0a62-4c0d-bb08-bdc7157b35bc",
    "synthscars:train:c99546b5-27c2-46c0-8e75-9d882a2bf5b9",
    "synthscars:train:b7c4c430-2ddf-4aec-bfee-aab0e351193d",
    "synthscars:train:591d731f-0e5e-47a3-96e3-093e5c168594",
    "synthscars:train:28f98ffc-d157-4056-a0c8-c64dd4e47c88"
  ],
  "arms": {
    "forensic_rect": {
      "pass": true,
      "loss": 0.5396878719329834,
      "gradients": {
        "rectification.gamma": 0.056640625,
        "rectification.semantic_norm.weight": 0.000114864444185514,
        "rectification.semantic_norm.bias": 3.784295768127777e-05,
        "rectification.forensic_norm.weight": 0.00011065367289120331,
        "rectification.forensic_norm.bias": 1.7292084919517947e-07,
        "rectification.cross_attention.q_proj.weight": 0.0032041361555457115,
        "rectification.cross_attention.q_proj.bias": 6.695494812447578e-05,
        "rectification.cross_attention.k_proj.weight": 0.003245061729103327,
        "rectification.cross_attention.k_proj.bias": 2.8038587629453104e-07,
        "rectification.cross_attention.v_proj.weight": 0.016864098608493805,
        "rectification.cross_attention.v_proj.bias": 0.0017529282486066222,
        "rectification.cross_attention.out_proj.weight": 0.017037326470017433,
        "rectification.cross_attention.out_proj.bias": 0.0030001220293343067,
        "rectification.projection.weight": 0.016373565420508385,
        "rectification.projection.bias": 0.005300696473568678
      },
      "required_groups": {
        "Wq": true,
        "Wk": true,
        "Wv": true,
        "Wo": true,
        "Projection": true,
        "gamma": true
      },
      "frozen": {
        "P1_SAM_gradient_tensor_count": 0,
        "4C_A_source_gradient_tensor_count": 0,
        "P1_SAM_hash_unchanged": true,
        "4C_A_source_hash_unchanged": true
      },
      "formal_optimizer_updates": 0
    },
    "clip_rect": {
      "pass": true,
      "loss": 0.53952556848526,
      "gradients": {
        "rectification.gamma": 0.0284423828125,
        "rectification.semantic_norm.weight": 0.00011635533883236349,
        "rectification.semantic_norm.bias": 4.5100987335899845e-05,
        "rectification.forensic_norm.weight": 0.00012158478784840554,
        "rectification.forensic_norm.bias": 1.2900235901724955e-07,
        "rectification.cross_attention.q_proj.weight": 0.003333061933517456,
        "rectification.cross_attention.q_proj.bias": 7.562320388387889e-05,
        "rectification.cross_attention.k_proj.weight": 0.003260478377342224,
        "rectification.cross_attention.k_proj.bias": 2.0201163408728462e-07,
        "rectification.cross_attention.v_proj.weight": 0.013151305727660656,
        "rectification.cross_attention.v_proj.bias": 0.0017609751084819436,
        "rectification.cross_attention.out_proj.weight": 0.013091213069856167,
        "rectification.cross_attention.out_proj.bias": 0.0031534635927528143,
        "rectification.projection.weight": 0.014030777849256992,
        "rectification.projection.bias": 0.006012028083205223
      },
      "required_groups": {
        "Wq": true,
        "Wk": true,
        "Wv": true,
        "Wo": true,
        "Projection": true,
        "gamma": true
      },
      "frozen": {
        "P1_SAM_gradient_tensor_count": 0,
        "4C_A_source_gradient_tensor_count": 0,
        "P1_SAM_hash_unchanged": true,
        "4C_A_source_hash_unchanged": true
      },
      "formal_optimizer_updates": 0
    }
  },
  "initial_rectifier_hashes": {
    "forensic_rect": "5099ffcd68b1c4aecfbbedc29cf0935adf034236f08e8788d737db940dc05cea",
    "clip_rect": "5099ffcd68b1c4aecfbbedc29cf0935adf034236f08e8788d737db940dc05cea"
  },
  "matched_initialization": true,
  "formal_optimizer_updates": 0,
  "internal_test_accessed": false,
  "official1000_accessed": false
}
```
