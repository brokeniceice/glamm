# Phase 4F Geometry scale audit

```json
{
  "status": "PASS",
  "selection_uses_train_only": true,
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
  "candidates": [
    {
      "gamma": 0.005,
      "median_residual_to_s64_norm_ratio": 0.0048358747735619545,
      "bf16_token_survival": 1.0,
      "pass": false
    },
    {
      "gamma": 0.01,
      "median_residual_to_s64_norm_ratio": 0.009671749547123909,
      "bf16_token_survival": 1.0,
      "pass": true
    },
    {
      "gamma": 0.03,
      "median_residual_to_s64_norm_ratio": 0.029014473780989647,
      "bf16_token_survival": 1.0,
      "pass": true
    },
    {
      "gamma": 0.05,
      "median_residual_to_s64_norm_ratio": 0.04836345091462135,
      "bf16_token_survival": 1.0,
      "pass": true
    },
    {
      "gamma": 0.1,
      "median_residual_to_s64_norm_ratio": 0.0967269018292427,
      "bf16_token_survival": 1.0,
      "pass": true
    }
  ],
  "selected_gamma": 0.01,
  "formal_optimizer_updates": 0,
  "validation_performance_used": false,
  "internal_test_accessed": false,
  "official1000_accessed": false
}
```
