# Phase 5B-0 — FakeShield Official Deployment Audit

## 0. Scope and stop boundary

This audit freezes the public FakeShield source, inventories the released weights, and traces the official deployment path. It does **not** run model forward/inference, train, evaluate a dataset, access R1/internal/LOKI data, or compute metrics. Phase 5A-2 continued independently on GPUs 1/2 during this CPU/network-only audit.

Audit date: 2026-09-01 UTC.

## 1. Frozen official source

| Field | Frozen value |
|---|---|
| Checkout | `external/FakeShield_official/` |
| Remote | `https://github.com/zhipeixu/FakeShield.git` |
| Branch | `main` |
| Commit | `b22717d7d524cbf617b8841d488d1edf56ceb7f3` |
| Commit date | 2026-02-21 18:46:34 +0800 |
| Final status | clean (`git status --porcelain` empty) |

No official source file was edited. The repository itself contains tracked `__pycache__` files; those are upstream content and do not mean the checkout is dirty.

Primary upstream references: [official repository](https://github.com/zhipeixu/FakeShield), [official model repository](https://huggingface.co/zhipeixu/fakeshield-v1-22b), and [paper](https://arxiv.org/abs/2410.02761).

## 2. Released checkpoint audit

### 2.1 Identity and local placement

| Field | Value |
|---|---|
| HF repository | `zhipeixu/fakeshield-v1-22b` |
| Frozen revision | `db861d30d9dc842fde855d95170e24a7a86d5a8a` |
| HF last modified | 2025-04-25T15:41:37Z |
| Gated/private | no/no |
| API-reported total | 44,157,589,488 bytes (44.16 GB decimal) |
| Local target | `/data/yz/myLISA_storage/checkpoints/phase5b0_fakeshield/models/fakeshield-v1-22b_db861d30d9dc842fde855d95170e24a7a86d5a8a/` |

The project-level `checkpoints` symlink resolves to `/data/yz/myLISA_storage/checkpoints`; no large checkpoint or HF cache was placed on the root filesystem.

Local verification completed after download:

- exact remote/local path-set equality: `24/24`, with no missing or extra files;
- local total: exactly `44,157,589,488` bytes;
- per-file byte-size mismatches: `0`;
- all 11 HF LFS objects (six DTE shards, two MFLM shards, DTG, and the two tokenizer copies) were streamed through SHA-256 locally;
- LFS SHA-256 mismatches against the frozen HF API metadata: `0`.

This verifies the downloaded bytes; it is not a model-load or inference claim.

### 2.2 Complete HF file inventory

| Path | Bytes | Role |
|---|---:|---|
| `.gitattributes` | 1,519 | repository metadata |
| `README.md` | 1,441 | model card |
| `DTG.pth` | 94,378,783 | ResNet-50 three-domain classifier |
| `DTE-FDM/config.json` | 1,395 | LLaVA-Llama 13B config |
| `DTE-FDM/generation_config.json` | 154 | generation config |
| `DTE-FDM/model.safetensors.index.json` | 79,096 | six-shard index |
| `DTE-FDM/model-00001-of-00006.safetensors` | 4,978,265,728 | merged model shard |
| `DTE-FDM/model-00002-of-00006.safetensors` | 4,970,422,160 | merged model shard |
| `DTE-FDM/model-00003-of-00006.safetensors` | 4,970,422,184 | merged model shard |
| `DTE-FDM/model-00004-of-00006.safetensors` | 4,933,701,432 | merged model shard |
| `DTE-FDM/model-00005-of-00006.safetensors` | 4,933,722,144 | merged model shard |
| `DTE-FDM/model-00006-of-00006.safetensors` | 2,522,262,456 | merged model shard |
| `DTE-FDM/special_tokens_map.json` | 552 | tokenizer metadata |
| `DTE-FDM/tokenizer.model` | 499,723 | SentencePiece tokenizer |
| `DTE-FDM/tokenizer_config.json` | 936 | tokenizer config |
| `MFLM/added_tokens.json` | 137 | includes `[SEG]`, `<p>`, `</p>`, `<bbox>`, `<point>` |
| `MFLM/config.json` | 1,785 | GLaMM 7B config |
| `MFLM/generation_config.json` | 176 | generation config |
| `MFLM/pytorch_model.bin.index.json` | 96,091 | two-shard index |
| `MFLM/pytorch_model-00001-of-00002.bin` | 9,976,691,902 | merged model shard |
| `MFLM/pytorch_model-00002-of-00002.bin` | 6,776,538,784 | merged model shard |
| `MFLM/special_tokens_map.json` | 438 | tokenizer metadata |
| `MFLM/tokenizer.model` | 499,723 | SentencePiece tokenizer |
| `MFLM/tokenizer_config.json` | 749 | tokenizer config |

Index-level checks:

- DTE-FDM index: 758 keys, declared tensor bytes `27,308,693,504`; it includes LLM embeddings/lm-head, MM projector, and CLIP vision-tower keys.
- MFLM index: 968 keys, declared tensor bytes `16,752,883,392`; it includes LLM embeddings/lm-head, MM projector, text-to-mask projection, and 594 `grounding_encoder` keys (SAM image encoder, prompt encoder, and mask decoder).
- Neither index contains `lora` or `adapter` keys. The public directories contain neither adapter configs nor non-LoRA side weights.
- The inference launcher passes each subdirectory directly to `from_pretrained(..., model_base=None)`. Merge scripts exist for training/export, but are not part of public-checkpoint inference.
- In issue [#19](https://github.com/zhipeixu/FakeShield/issues/19#issuecomment-2696301128), the repository owner explicitly confirms that the HF weights are LoRA-finetuned and already merged, and can be used directly for testing or subsequent training.

Therefore DTE-FDM and MFLM are released as merged HF-style model directories. Issue [#63](https://github.com/zhipeixu/FakeShield/issues/63) points the CLI at the HF repository root and consequently reports missing config/tokenizer files; the official launcher correctly uses the `DTE-FDM/` and `MFLM/` subdirectories (`scripts/cli_demo.sh:9,17`).

### 2.3 Additional models and SAM

There is **no separate LLaVA-13B or GLaMM base-model merge requirement** for the released weights. There is, however, one runtime model dependency:

- Both configs name `openai/clip-vit-large-patch14-336`; both loaders initialize this vision tower by name. Strict offline inference therefore needs that HF model in cache or an equivalent launcher-level local resolution. This is an additional vision encoder, not a missing language-model base. Issue [#51](https://github.com/zhipeixu/FakeShield/issues/51) independently shows offline loading failing at this CLIP dependency.

A previously verified local copy is already available from Phase 5A-1 at revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` under `/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. Phase 5B-1 can mount/cache that exact asset rather than download another copy, while recording the cross-phase reuse explicitly.

Official README preparation also requests:

| SAM field | Frozen value |
|---|---|
| Repository | `ybelkada/segment-anything` |
| Revision audited | `7790786db131bcdc639f24a915d9f2c331d843ee` |
| File | `checkpoints/sam_vit_h_4b8939.pth` |
| Bytes | 2,564,550,879 |
| SHA-256 | `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` |
| Local file | `/data/yz/myLISA_storage/checkpoints/phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth` |

The local SAM file was checked independently: its size and SHA-256 both match the frozen repository metadata.

Important source-level nuance: the released MFLM index already contains the complete `grounding_encoder`, and `MFLM/cli_demo.py` does not pass `vision_pretrained` when constructing MFLM. Thus the separate SAM file is an official preparation/training/export dependency, but the published merged MFLM inference path appears to restore SAM from MFLM itself. Phase 5B-1 should retain the official SAM file and log whether it is actually opened; it must not silently claim that the demo consumed it.

## 3. Actual model structure and inference chain

The source implements the following chain:

```text
input image
  ├─> DTG: ResNet-50, 3 classes only
  │     0=AIGC inpainting, 1=DeepFake, 2=Photoshop
  │     (domain prior, NOT real/fake classification)
  │
  └─> DTE-FDM: LLaVA-v1.5-13B + fixed question + DTG domain prefix
          └─> free-form detection decision + tampered-area description
              + forensic/semantic explanation
                  └─> exact English substring gate
                      "has not been tampered with" => stop/no mask
                      otherwise => MFLM
                          └─> image + complete DTE-FDM text
                              [batch test additionally adds a localization request]
                                  └─> MFLM/GLaMM generation
                                      └─> each generated [SEG] hidden state
                                          └─> text projection -> SAM prompt encoder
                                              -> SAM mask decoder -> original-size logits
```

### 3.1 DTG and classification

- `DTE-FDM/llava/serve/cli.py:23-66` constructs a torchvision ResNet-50 and replaces its head with three outputs.
- DTG has no authentic/real class. Every image, including a real image, is forcibly assigned one of the three manipulation domains. DTG therefore conditions DTE-FDM but does not itself supply the final binary decision.
- `cli.py:126-132` prepends the predicted domain sentence to one fixed detection/explanation question.
- DTE-FDM produces no structured probability or class field. The final detection result is embedded in natural-language `outputs`.
- `MFLM/cli_demo.py:245-248` and `MFLM/test.py:257-260` define “real/no localization” by case-sensitive substring matching for `has not been tampered with`. Any alternative wording is routed to localization. This is a brittle text gate and must be logged explicitly in a future smoke test.

### 3.2 Explanation and its relationship to localization

- DTE-FDM jointly generates the binary conclusion, tampered-region description, and judgment basis. The supplied playground outputs demonstrate this two-part prose format.
- Localization **does depend on the generated DTE-FDM text**. The single-image demo passes the complete text directly; the folder/test path prefixes it with “This photo has been tampered...” and appends one randomly selected localization question (`MFLM/test.py:262-267`).
- Therefore this is not image-only localization. DTE classification can suppress localization entirely, and DTE explanation/region text conditions MFLM when localization is attempted.
- MFLM does not parse a structured phrase field from DTE-FDM. It receives the entire prose string as prompt context.

### 3.3 `[SEG]`, hidden representations, SAM, and masks

- `[SEG]` is token ID 32004 in `MFLM/added_tokens.json`; `<p>`/`</p>` are present in the inherited GLaMM tokenizer, but FakeShield's tamper-segmentation training answers are generic variants such as `It is [SEG].` (`MFLM/dataset/utils/utils.py:110-114`). The official FakeShield path does **not** require a visible `<p>phrase</p>[SEG]` output.
- `GLaMM.py:290-313` generates text and selects hidden states at all generated `[SEG]` positions.
- `GLaMM.py:213-246` projects every selected hidden state to a SAM text prompt, uses `multimask_output=False`, and returns one original-resolution mask logit per `[SEG]` token.
- Binary threshold semantics are logit `> 0`; SAM declares `mask_threshold = 0.0`, while the demo/test helpers also apply `curr_mask > 0` explicitly.
- Model-level output count is therefore zero masks when no `[SEG]`, otherwise one mask per `[SEG]`.

Mask post-processing differs across official entry points:

1. `MFLM/cli_demo.py:123-152` overlays all selected masks for visualization but writes the returned binary `seg_mask` from only the **last** mask, not their union.
2. `MFLM/test.py:137-158` intends a union (`final_mask[curr_mask > 0] = 255`) but treats the outer per-image list as though each item were an individual mask. For one image with multiple `[SEG]`, the item is actually an `N_seg × H × W` tensor; the helper can retain an unsqueezed multi-channel array and is not a reliable multi-mask union implementation.
3. Both `inference()` functions can return an unbound `seg_mask` when generated text has no `[SEG]`; future wrappers must record this official failure rather than tune generation.

These are deployment bugs, not authorization to edit core code in Phase 5B-0.

## 4. Official inference entry points

### 4.1 Single-image `scripts/cli_demo.sh`

Inputs and outputs:

| Stage | Input | Output |
|---|---|---|
| DTG/DTE-FDM | one local image path; fixed prompt; checkpoint paths | one JSON object: `{"image": ..., "outputs": ...}` (named `.jsonl`, but no newline and only one record) |
| MFLM | that JSON object and original image | mask image saved as `MFLM_OUTPUT/<input basename>`; generated MFLM text is returned internally but no final JSON is written |

Launcher problems:

- All default paths are relative placeholders. The README/launcher default uses `playground/image/...`, while the checkout contains `playground/images/`; the specific default image is absent.
- `python -m llava.serve.cli` requires the editable DTE-FDM install or `PYTHONPATH=$repo/DTE-FDM`.
- The script mutates the live environment: Transformers 4.37.2 for DTE-FDM, then 4.28.0 for MFLM (`scripts/cli_demo.sh:6,14`). It suppresses installation errors/output.
- GPU is hard-coded to physical device 0.
- DTE's loader advertises HTTP image support, but DTG separately calls `PIL.Image.open(image_path)`, so a URL input fails before generation.
- There is no structured final result JSON combining detection, explanation, MFLM text, SEG count, and mask path.

### 4.2 Folder/test path

`scripts/test.sh` + `DTE-FDM/llava/eval/model_vqa.py` + `MFLM/test.py` support multi-record JSONL/folder-style processing. Question rows require at least `image` and `text`. The DTE stage writes one `{"image", "outputs"}` object per line. MFLM loops over those records and writes masks by basename. The supplied `playground/eval_jsonl.py` is not portable: it contains an upstream `/data03/xzp/...` absolute dataset path.

This folder path is not a clean batch API: it has the multi-mask issue above, random choice among three localization requests, hard-coded CUDA operations, and no completion/result manifest.

## 5. Environment and deployment audit

### 5.1 Declared native stack

An independent Conda environment named `fakeshield` was created at `/data/yz/myLISA_storage/conda_envs/fakeshield` and verified as Python 3.9.25 (pip 26.0.1). It currently contains only the isolated base Python/pip toolchain. The full native FakeShield dependency stack was deliberately **not** installed in Phase 5B-0: doing so would require choosing between the incompatible official DTE/MFLM Transformers stacks before the deployment route is frozen, and no inference is authorized in this phase. R1 and LEGION environments were not modified.

The README declares Python 3.9, PyTorch 1.13.0, CUDA 11.6, and source-built MMCV v1.4.7 with `MMCV_WITH_OPS=1`. The root requirements additionally pin, among others:

- Transformers 4.28.0 and Tokenizers 0.13.3;
- Flash-Attn 2.3.6;
- PEFT 0.5.0;
- DeepSpeed 0.12.5;
- Accelerate 0.25.0.

DTE-FDM's own `pyproject.toml` pins torch 1.13.0/torchvision 0.14.0/Transformers 4.28.0 but uses unpinned PEFT and bitsandbytes; its optional train group pins DeepSpeed 0.12.6, not the root's 0.12.5. The DTE checkpoint config reports Transformers 4.36.0, and the official demo installs 4.37.2 for that phase. One static Conda environment therefore cannot reproduce both official phases without the launcher's runtime package swap.

Known native-environment reports include:

- MMCV extension build/import failures ([#24](https://github.com/zhipeixu/FakeShield/issues/24), [#64](https://github.com/zhipeixu/FakeShield/issues/64)); #64 reports manual `MMCV_WITH_OPS=1 FORCE_CUDA=1` source compilation as the successful workaround.
- CUDA 11.6 versus Flash-Attn 2.3.6 availability concerns ([#41](https://github.com/zhipeixu/FakeShield/issues/41), [#71](https://github.com/zhipeixu/FakeShield/issues/71)).
- CLIP absolute/offline/device-map loading failures ([#14](https://github.com/zhipeixu/FakeShield/issues/14), [#25](https://github.com/zhipeixu/FakeShield/issues/25), [#26](https://github.com/zhipeixu/FakeShield/issues/26), [#51](https://github.com/zhipeixu/FakeShield/issues/51), [#56](https://github.com/zhipeixu/FakeShield/issues/56)). Current frozen HF config now names the public `openai/clip-vit-large-patch14-336`, so the old author-local `/data03/...` config defect is not present in revision `db861d30...`.
- Missing module/PYTHONPATH and stale image-path failures ([#28](https://github.com/zhipeixu/FakeShield/issues/28), [#45](https://github.com/zhipeixu/FakeShield/issues/45), [#62](https://github.com/zhipeixu/FakeShield/issues/62)).
- Unused DTE vision-tower keys ([#27](https://github.com/zhipeixu/FakeShield/issues/27), [#72](https://github.com/zhipeixu/FakeShield/issues/72)); the current loader reconstructs/reloads CLIP from its configured tower, explaining why embedded vision keys can be reported unused.
- OOM/abnormal allocation reports including a nominal 95 GB GPU ([#30](https://github.com/zhipeixu/FakeShield/issues/30), [#31](https://github.com/zhipeixu/FakeShield/issues/31), [#73](https://github.com/zhipeixu/FakeShield/issues/73)).
- An unresolved scale-sensitivity question ([#74](https://github.com/zhipeixu/FakeShield/issues/74)).
- A public-checkpoint paper-reproduction gap report remains open ([#20](https://github.com/zhipeixu/FakeShield/issues/20)); therefore deployment success must not be equated with exact paper-table reproduction.

Positive/official reproduction evidence is limited but not absent: in [#14](https://github.com/zhipeixu/FakeShield/issues/14), one user reports getting past the initial deployment problem after downloading the CLIP vision tower and correcting paths/dependency versions; the repository owner acknowledges that the project still needs improvement. In [#37](https://github.com/zhipeixu/FakeShield/issues/37), the owner states that they did not find a good CUDA 12.1 solution because of MMCV-Full constraints. In [#45](https://github.com/zhipeixu/FakeShield/issues/45), the owner confirms that saved DTE-FDM output is the MFLM input and says non-GPU Mac deployment is not expected to work.

No issue was found that demonstrates a complete, independently documented, end-to-end reproduction with the frozen public revision. Several users reached individual load/runtime stages, but the issue record is dominated by environment, path, CLIP, MMCV, memory, and checkpoint-warning reports.

### 5.2 Docker audit

The exact published `v1.0` tags exist remotely:

| Image | v1.0 status | Manifest/config evidence | Compressed size |
|---|---|---|---:|
| `zhipeixu/mflm:v1.0` | exists, linux/amd64 | config `sha256:da6009d1300c8439c520536a7add876faeab3e17e2184260b8160393320f4e82` | 14,635,104,755 bytes |
| `zhipeixu/dte-fdm:v1.0` | exists, linux/amd64 | config `sha256:2f9f37baebe32ff7266bdf37054349a71f327e7959b680500dda01fc410f86eb` | 15,856,191,871 bytes |

The README's pull commands use `:v1.0`, but its subsequent run commands use `:latest`. Neither repository currently publishes a `latest` tag, matching issue [#67](https://github.com/zhipeixu/FakeShield/issues/67). A future launch must use the frozen `:v1.0` tags explicitly.

Docker CLI is installed on this machine, but user `yz` cannot access `/var/run/docker.sock`; no image was pulled or container started in Phase 5B-0. The manifests were audited remotely without the daemon.

### 5.3 Recommendation

**Recommendation: A, official Docker, using two explicit `v1.0` images**, with a repository-external wrapper that mounts source, weights, image, CLIP cache, and output paths and runs DTE and MFLM sequentially. This best preserves the two incompatible dependency stacks and follows the README's paper-reproduction recommendation.

The isolated `fakeshield` Python 3.9 environment is retained as a fallback/launcher-inspection environment, not as evidence that the full native dependency stack works. Native Conda is second choice because the official script itself swaps Transformers versions and MMCV/Flash-Attn have multiple unresolved compatibility reports.

Before Phase 5B-1, Docker socket authorization is required from the machine operator; that is an environment permission, not a reason to modify FakeShield core code.

## 6. Final deployment assessment

| Question | Answer |
|---|---|
| 官方代码完整 | **基本完整，但 demo/test launcher 有路径、环境切换、no-SEG 和 multi-mask 缺陷** |
| pretrained weights 完整 | **是；DTE-FDM、MFLM、DTG 均有完整公开文件** |
| DTE-FDM 可用 | **权重与入口齐全；需正确子目录、CLIP cache、兼容环境/launcher** |
| MFLM 可用 | **权重与入口齐全；需 CLIP、兼容环境，并处理官方输出边界缺陷** |
| DTG 可用 | **是；ResNet-50 三域分类器完整** |
| 需要 SAM | **官方准备要求是；独立文件用于训练/export，merged MFLM inference 已内含 grounding encoder，需 smoke 记录实际文件访问** |
| 需要额外 base model | **不需要额外 LLaVA/GLaMM language base；需要额外 `openai/clip-vit-large-patch14-336` vision tower/cache** |
| 需要 merge | **否；公开 DTE/MFLM 均为已合并 checkpoint** |
| detection 可直接运行 | **不是零配置；补齐路径、PYTHONPATH/安装、CLIP 和兼容环境后具备条件** |
| explanation 可直接运行 | **与 detection 同一路径，具备条件但尚未 forward 验证** |
| localization 可直接运行 | **具备模型条件，但官方 launcher/no-SEG/multi-mask/output 缺陷需要外部 wrapper；尚未 forward 验证** |
| 推荐 Docker / Conda | **Docker：两个明确的 `:v1.0` 镜像；Conda 仅备用** |
| 是否具备单图 smoke test 条件 | **有条件具备：权重、代码和入口齐全；仍需 Docker socket 权限、CLIP/SAM 本地资产校验和外部 launcher wrapper** |

Final classification:

> **Case B — 权重完整，但需额外 CLIP/SAM/launcher 与可用 Docker 权限后才能可靠运行。**

This is not Case A because the official public demo is not zero-configuration and contains reproducibility defects. It is not Case C because all three released model components and their configs/tokenizers are present. It is not Case D because no retraining is required to attempt public-checkpoint inference.

## 7. Phase 5B-1 handoff boundary

Phase 5B-1 may perform exactly one non-benchmark single-image smoke test only after separately authorizing/solving Docker access. It should freeze:

1. source commit and all model/container revisions;
2. exact official prompt and temperature (DTE default temperature 0.2 is stochastic);
3. DTG domain label, raw DTE text, substring-gate decision, raw MFLM text, SEG count, all raw masks, and saved official mask;
4. actual files opened for CLIP and SAM;
5. GPU/peak memory and dependency versions for each of the two stages;
6. no-SEG or empty-mask as a terminal smoke outcome, with no tuning or resampling.

STOP: no inference or benchmark was started in Phase 5B-0.
