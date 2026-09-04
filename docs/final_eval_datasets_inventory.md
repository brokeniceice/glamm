# Final Evaluation Datasets Inventory

Status: `IN PROGRESS — DATA PREPARATION ONLY; NO MODEL EVALUATION`

This document freezes the provenance, identities, raw/processed boundary, and leakage checks for the final evaluation datasets. Existing model outputs, historical reports, source code, checkpoints, and manifests are preserved.

## AIGI replacement audit

### Deleted legacy processed copy (pre-deletion record)

| Field | Value |
| --- | --- |
| Absolute path | `/data/yz/myLISA_storage/AIGC/AIGI-Holmes-Dataset` (project path: `datasets/AIGI-Holmes-Dataset`) |
| Identity | Project-created `myLISA_dataset` package containing manually selected/renamed train, validation, and test images plus derived masks and rewritten JSONL records; it is not the official pristine Holmes `TestSet.zip` |
| File count | 35,533 regular files |
| Apparent bytes | 5,557,734,432 bytes |
| Directory disk/accounted bytes | 5,560,605,728 bytes (`du -sbL`) |
| Archive | `myLISA_dataset.tar.gz`, 2,734,431,397 bytes, SHA256 `402c2d8d5c2946455ced96b97d15ac9329a09e6a50a9111fc1256a52d7c16f4d` |
| Historical test manifest | `datasets/AIGI-Holmes-Dataset/dataset/test.jsonl`, 1,731 rows (870 Real + 861 Fake by the frozen loader rule), 5,659,473 bytes |
| Other embedded splits | `train.jsonl` 13,848 rows; `val.jsonl` 1,730 rows; multiple `_clean`, `_CLIP`, mini/test derivatives |
| Image directories | `0_real`: 8,700 files; `1_fake`: 8,609 files |
| Derived annotation directories | `masks`: 8,609; `masks_ori`: 8,610; `masks_t`: 989 files |
| Historical references retained | All `outputs/`, `docs/`, scripts, evaluation results, and checkpoints referencing the legacy data |
| Deletion authorization | Explicit user instruction in the final-evaluation data-preparation phase |

The old copy cannot serve as the pristine official benchmark because its contents mix project-defined train/validation/test selections, renamed images, transformed/derived masks, and rewritten supervision records. Historical results remain historical only; the old `AIGI-test` is not a final main benchmark.

### Official replacement source

Pending verified download and integrity audit. The pinned source is the author release `zzy0123/AIGI-Holmes-Dataset`, revision `3e856ce5ed44ac3b578bf36434829ea42953be02`, file `TestSet.zip` only (40,457,899,107 bytes). `SFTDATA.jsonl` and `dataset_huggingface.zip` are excluded.

## Inventory

Pending completion of official downloads, frozen manifests, and overlap audit.

## Benchmark independence audit

Pending. No model inference or performance metric is part of this audit.

## Final frozen evaluation protocol

Pending final integrity and leakage gates.
