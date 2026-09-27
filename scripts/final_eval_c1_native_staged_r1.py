#!/usr/bin/env python3
"""Frozen C1-center classification and C1-native staged R1 localization OOD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/final_evaluation/c1_native_staged_r1"
NEW_R1 = ROOT / "outputs/phase6e3_c1_native_staged/joint/selected_checkpoint.pt"
SELECTOR = ROOT / "outputs/phase6e3_c1_native_staged/joint/selector.json"
OFFICIAL_RESULT = ROOT / "outputs/phase6e3_c1_native_staged/official1000_results.json"
C1_SHA = "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff"
CLASS_SOURCE = ROOT / "outputs/phase6d5_full_classification_ood"
CENTER_SOURCE = ROOT / "outputs/phase6d6_decision_boundary_disentanglement/phase6d6_decision_boundary_disentanglement.json"
FUSION_PARAMETERS = ROOT / "outputs/phase6d5_decision_integration/fusion_parameters.json"
OFFICIAL_R1 = ROOT / "outputs/phase6e3_c1_native_staged/official1000_records.jsonl"
DATASETS = ("synthscars", "loki", "xaigd", "pal4vst")
CLASS_DATASETS = ("aigi_holmes", "genimage", "loki", "raise998")


def now():
    return datetime.now(timezone.utc).isoformat()


def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_r1_sha():
    selector = json.loads(SELECTOR.read_text())
    require(selector["status"] == "COMPLETE" and selector["c1_sha256"] == C1_SHA, "C1-native selector drift")
    require(not selector["official1000_used_for_selection"] and not selector["external_ood_used_for_selection"], "selection firewall drift")
    require(sha(NEW_R1) == selector["selected_checkpoint_sha256"], "C1-native selected checkpoint drift")
    return selector["selected_checkpoint_sha256"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))
    os.replace(temp, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preflight():
    import numpy as np
    from scripts import final_eval_localization_supervisor as final
    from scripts.phase6d6_decision_boundary_disentanglement import hard_metrics
    selected_sha = new_r1_sha()
    source = json.loads((CLASS_SOURCE / "results.json").read_text())
    require(source["status"] == "COMPLETE" and source["checkpoint_sha256"] == C1_SHA, "C1 classification result drift")
    require(json.loads((CLASS_SOURCE / "worker_status.json").read_text())["status"] == "COMPLETE", "C1 classification worker incomplete")
    centered = json.loads(CENTER_SOURCE.read_text())
    parameters = json.loads(FUSION_PARAMETERS.read_text())
    require(centered["status"] == "COMPLETE", "C1-center reference incomplete")
    require(centered["provenance"]["fusion_parameters"]["sha256"] == sha(FUSION_PARAMETERS), "C1-center normalization provenance drift")
    center_mean = float(parameters["normalization"]["c1"]["mean"])
    require(centered["protocol"]["normalization"]["c1"]["mean"] == center_mean, "C1-center threshold drift")
    classification = {}
    for name in CLASS_DATASETS:
        manifest = ROOT / {
            "aigi_holmes": "datasets/AIGI-Holmes/manifests/eval_manifest.jsonl",
            "genimage": "datasets/GenImage/manifests/eval_manifest.jsonl",
            "loki": "datasets/LOKI/manifests/classification_eval_manifest.jsonl",
            "raise998": "datasets/RAISE/manifests/eval_manifest.jsonl",
        }[name]
        entry = source["manifests"][name]
        require(sha(manifest) == entry["sha256"], f"{name} classification manifest drift")
        marker = json.loads((CLASS_SOURCE / "raw" / f"{name}.complete.json").read_text())
        require(marker["status"] == "COMPLETE" and marker["count"] == entry["count"] and marker["manifest_sha256"] == entry["sha256"] and marker["checkpoint_sha256"] == C1_SHA, f"{name} classification marker drift")
        prediction = CLASS_SOURCE / "raw" / f"{name}.jsonl"
        observed = rows(prediction)
        expected = rows(manifest)
        require(len(observed) == len(expected) == entry["count"], f"{name} classification count drift")
        require([r["sample_id"] for r in observed] == [r["sample_id"] for r in expected], f"{name} classification order drift")
        prediction_sha = sha(prediction)
        require(centered["provenance"]["full_ood_scores"][name]["sha256"] == prediction_sha, f"{name} C1-center input drift")
        y = np.asarray([r["label"] for r in observed], dtype=np.int64)
        margin = np.asarray([r["c1_margin"] for r in observed], dtype=np.float64)
        recomputed = hard_metrics(y, margin, center_mean)
        reference = centered["datasets"][name]["C1-center"]
        for key, value in recomputed.items():
            actual = reference[key]
            require((value is None and actual is None) or (value is not None and abs(value - actual) < 1e-10), f"{name} C1-center metric drift: {key}")
        primary = {**reference, "auprc": source["datasets"][name]["arms"]["C1"]["auprc"]}
        per_generator = {}
        for generator, arms in centered["datasets"][name].get("per_generator", {}).items():
            per_generator[generator] = {**arms["C1-center"],
                "auprc": source["datasets"][name]["per_generator"][generator]["C1"]["auprc"]}
        classification[name] = {"execution": "EXACT_REUSE_C1_WITH_FROZEN_TRAIN_CENTER", "primary_arm": "C1-center",
            "manifest_sha256": entry["sha256"], "prediction_sha256": prediction_sha, "n": entry["count"],
            "threshold_raw_margin": center_mean, "metrics": primary, "per_generator": per_generator,
            "raw_C1_diagnostic": source["datasets"][name]["arms"]["C1"]}
    manifests = {}
    for name in DATASETS:
        manifest_rows, meta = final.manifest_info(name)
        manifests[name] = meta
        require(len(manifest_rows) == meta["n"], f"{name} localization count drift")
    assert_official_reuse()
    data = {"schema": "c1_native_staged_r1_ood_protocol_v1", "status": "FROZEN", "created_at_utc": now(),
        "C1_sha256": C1_SHA, "new_R1_sha256": selected_sha, "classification": classification,
        "classification_primary": "C1-center", "classification_threshold_raw_margin": center_mean,
        "classification_center_source": {"path": str(CENTER_SOURCE.resolve()), "sha256": sha(CENTER_SOURCE),
            "normalization_parameters_sha256": sha(FUSION_PARAMETERS)},
        "localization_manifests": manifests, "localization_conditions": {"synthscars": "G0", "loki": "G1", "xaigd": "G1", "pal4vst": "G1"},
        "classification_policy": "C1-center: raw C1 logit margin > frozen internal-TRAIN mean; new R1 changes only the post-[SEG] mask path",
        "localization_policy": "full N; missing [SEG] scores 0; multiple masks pixelwise logit max; mask logit > 0",
        "no_training_or_threshold_tuning": True}
    path = OUT / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("C1_sha256", "new_R1_sha256", "localization_manifests", "localization_conditions"):
            require(previous[key] == data[key], f"protocol drift: {key}")
        require(previous.get("classification_primary") == "C1-center" and previous.get("classification_threshold_raw_margin") == center_mean,
            "C1-center protocol drift")
    else:
        write_json(path, data)
    return data


def assert_official_reuse():
    from scripts import final_eval_localization_supervisor as final
    source = rows(OFFICIAL_R1)
    old = rows(ROOT / "outputs/phase6e2_c1_specific_r1/official1000_new_r1_records.jsonl")
    manifest, _ = final.manifest_info("synthscars")
    require(len(source) == len(old) == len(manifest) == 1000, "Official1000 count drift")
    require(all(a["sample_id"] == b["sample_id"] == c["sample_id"] and a["tp"] + a["fn"] == b["tp"] + b["fn"] for a, b, c in zip(source, old, manifest)), "Official1000 identity/GT drift")
    official = json.loads(OFFICIAL_RESULT.read_text())
    require(official["status"] == "COMPLETE" and official["records_sha256"] == sha(OFFICIAL_R1), "Official1000 C1-native record drift")


def localization_dataset(name, tokenizer, config):
    if name == "loki":
        from scripts.phase3a1_loki_g0 import LokiLocalizationDataset
        from scripts import final_eval_localization_supervisor as final
        data = LokiLocalizationDataset(ROOT / "datasets/LOKI/legion_localization/manifest.jsonl",
            config["model"]["vision_tower"], int(config["model"]["image_size"]))
        official, _ = final.manifest_info(name)
        require(all(a["sample_id"] == b["sample_id"] and a["image_path"] == b["image_path"] and a["mask_path"] == b["derived_union_mask_path"] for a, b in zip(data.rows, official)), "LOKI manifest/mask drift")
        return data
    from scripts.final_eval_localization_supervisor import ExternalArtifactDataset
    return ExternalArtifactDataset(name, tokenizer, config["model"]["vision_tower"], int(config["model"]["image_size"]))


def worker(name, device_name):
    import numpy as np
    import torch
    import yaml
    from model.llava import conversation as conversation_lib
    from model.pcerf import sam_lowres_to_original_normalized
    from scripts import final_eval_localization_supervisor as final
    from scripts import phase4ha_utility_gated_rectification as ha
    from scripts import phase4hc_direct_utility_arms as hc
    from scripts import phase4hd_rectifier_unfreeze_control as hd
    from scripts.phase2a_final_evaluate import load_model
    from scripts.phase3c1_cache import clip_grid
    from scripts.phase6e1_c1_old_r1_transfer import C1_CFG_PATH, C1_CKPT
    from tools.phase3c1 import geometry_for
    from tools.phase3f_aogd import core_model
    from tools.phase4c_b import inverse_sam_logits
    from tools.phase4e1 import clip_coordinates, sam_coordinates
    from tools.phase4f import load_evidence_source, load_rectifier, load_sam_runtime
    from eval.forensics_eval import GLaMMForensicsBackend

    require(name in ("loki", "xaigd", "pal4vst"), "unsupported fresh dataset")
    preflight()
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    device = torch.device(device_name); torch.cuda.set_device(device)
    folder = OUT / "localization" / name
    status = folder / "worker_status.json"
    state = {"status": "RUNNING", "dataset": name, "device": device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(), "started_at_utc": now()}
    write_json(status, state)
    try:
        config = yaml.safe_load(C1_CFG_PATH.read_text())
        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
        model, tokenizer, meta = load_model(config, C1_CKPT, device, expected_step=2500, expected_epoch=5)
        require(meta["checkpoint_sha256"] == C1_SHA, "loaded C1 checkpoint drift")
        model.eval().requires_grad_(False); model.rine_conditioner.eval().requires_grad_(False)
        backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
            use_mm_start_end=True, max_new_tokens=400)
        utility, _ = hc.load_utility("a2", device)
        from scripts.phase6e3_c1_native_staged import c1_gamma
        rectifier = load_rectifier(hd.CFG, c1_gamma(), device)
        selected = torch.load(NEW_R1, map_location="cpu", weights_only=False)
        require(selected["c1_sha256"] == C1_SHA, "new R1 C1 linkage drift")
        utility.load_state_dict(selected["utility_state"], strict=True)
        rectifier.load_state_dict(selected["rectifier_state"], strict=True)
        utility.eval().requires_grad_(False); rectifier.eval().requires_grad_(False)
        sam = load_sam_runtime(hd.CFG, device).eval().requires_grad_(False)
        source = load_evidence_source(hd.CFG, "forensic_rect", device).eval().requires_grad_(False)
        dataset = localization_dataset(name, tokenizer, config)
        core = core_model(model); vision = model.get_model().get_vision_tower()
        path = folder / "predictions.jsonl"
        existing = rows(path) if path.exists() else []
        expected_ids = [r["sample_id"] for r in dataset.rows]
        require([r["sample_id"] for r in existing] == expected_ids[:len(existing)], "resume prefix drift")
        with path.open("a") as handle:
            for ordinal in range(len(existing), len(dataset)):
                sample = dataset[ordinal]; sid = sample["sample_id"]
                target = sample["masks"].bool().any(0)
                with torch.no_grad():
                    output = backend.generate_localization_batch([sample], provide_gt_fake=True,
                        generation_mode="unified_prompt_gt_fake_prefix")[0]
                    q_values = output.get("projected_seg_embeddings")
                    seg_count = output["generated_token_ids"].count(model.seg_token_idx)
                    valid = bool(output["seg_triggered"] and output.get("pred_mask") is not None
                        and q_values is not None and q_values.ndim == 2 and len(q_values) == seg_count and seg_count > 0)
                    if valid:
                        raw = model.get_grounding_encoder_embs(sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16))
                        tokens, _ = vision(sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16))
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            forensic = source(clip_grid(tokens), return_features=True)
                        h, w = target.shape
                        sg, cg = geometry_for("sam", (h, w)), geometry_for("clip", (h, w))
                        s64 = sam_lowres_to_original_normalized(raw, sg, output_hw=(64, 64)).to(torch.bfloat16)
                        f24 = forensic["F_forensic"].to(torch.bfloat16)
                        zf = forensic["logits"].to(torch.bfloat16)
                        sc = sam_coordinates(sg, grid=64)[None].to(device)
                        cc = clip_coordinates(cg, grid=24)[None].to(device)
                        tokens_valid = torch.ones(1, 576, dtype=torch.bool, device=device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            p4f = rectifier(raw, f24, sc, cc, tokens_valid)
                        logits = []
                        for q in q_values:
                            qone = q[None].to(device=device, dtype=torch.bfloat16)
                            with torch.autocast(device_type=device.type, enabled=False):
                                low0 = sam(qone, raw.to(torch.bfloat16))
                            zl = sam_lowres_to_original_normalized(low0, sg, output_hw=(256, 256)).to(torch.bfloat16)
                            batch = {"S64": s64, "q_seg": qone, "z_L": zl,
                                "F24": f24, "z_F24": zf, "clip_geometries": [cg],
                                "valid_g0": torch.ones(1, dtype=torch.bool, device=device),
                                "forensic_present": torch.ones(1, dtype=torch.bool, device=device),
                                "forensic_vacuous": torch.zeros(1, dtype=torch.bool, device=device),
                                "forensic_off": torch.zeros(1, dtype=torch.bool, device=device)}
                            u = hc.utility_forward(utility, batch)
                            gate = ha.gate_to_sam_grid(u["U"], sc) * p4f["support"].reshape(1, 1, 64, 64).float()
                            adapted = ha.gated_embedding(raw, p4f["image_embeddings"], gate)
                            with torch.autocast(device_type=device.type, enabled=False):
                                low = sam(qone, adapted.to(torch.bfloat16))
                            logits.append(inverse_sam_logits(low, sg))
                        binary = torch.stack(logits).amax(0).gt(0).cpu().numpy()
                        status_name = "OK"
                    else:
                        binary = np.zeros(tuple(target.shape), dtype=bool)
                        status_name = "NO_VALID_QSEG" if seg_count else "NO_SEG"
                    metric = final.metric_from_binary(binary, target.cpu().numpy())
                    record = final.enforce_failure_policy({"sample_id": sid, "ordinal": ordinal,
                        **metric, "status": status_name, "seg_triggered": bool(output["seg_triggered"]),
                        "seg_count": seg_count, "valid_q_seg": valid,
                        "has_pred_mask": output.get("pred_mask") is not None,
                        "generated_text": output.get("generated_text", ""),
                        "generation_mode": "unified_prompt_gt_fake_prefix",
                        "generator/source": dataset.rows[ordinal].get("generator/source"),
                        "artifact_categories": dataset.rows[ordinal].get("artifact_categories"),
                        "gt_foreground_pixels": int(target.sum()), "gt_is_empty": not bool(target.any())})
                handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
                if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(dataset):
                    print(json.dumps({"dataset": name, "done": ordinal + 1, "total": len(dataset)}), flush=True)
        final_rows = rows(path)
        require([r["sample_id"] for r in final_rows] == expected_ids, "final identity/order drift")
        write_json(folder / "results.json", {"status": "COMPLETE", "dataset": name,
            "C1_sha256": C1_SHA, "new_R1_sha256": new_r1_sha(),
            "manifest_sha256": sha(final.MANIFESTS[name]), "predictions_sha256": sha(path),
            "inference_condition": "G1", "metrics": final.summarize(final_rows)})
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["updated_at_utc"] = now(); write_json(status, state)


def finalize():
    from scripts import final_eval_localization_supervisor as final
    protocol = preflight()
    source = rows(OFFICIAL_R1)
    synth = []
    for ordinal, row in enumerate(source):
        synth.append(final.enforce_failure_policy({**row, "ordinal": ordinal,
            "status": "OK" if row["valid_q_seg"] else "NO_VALID_QSEG",
            "gt_foreground_pixels": row["tp"] + row["fn"], "gt_is_empty": False}))
    synth_path = OUT / "localization/synthscars/predictions.jsonl"
    write_jsonl(synth_path, synth)
    localization = {"synthscars": {"execution": "EXACT_REUSE_PHASE6E3_OFFICIAL1000", "inference_condition": "G0",
        "predictions_sha256": sha(synth_path), "metrics": final.summarize(synth)}}
    official = json.loads(OFFICIAL_RESULT.read_text())
    require(abs(localization["synthscars"]["metrics"]["mean_foreground_iou"] - official["metrics"]["mean_foreground_iou"]) < 1e-12, "Official1000 metric drift")
    for name in ("loki", "xaigd", "pal4vst"):
        folder = OUT / "localization" / name
        status = json.loads((folder / "worker_status.json").read_text())
        result = json.loads((folder / "results.json").read_text())
        prediction_path = folder / "predictions.jsonl"
        prediction = rows(prediction_path)
        manifest, meta = final.manifest_info(name)
        require(status["status"] == result["status"] == "COMPLETE", f"{name} incomplete")
        require([r["sample_id"] for r in prediction] == [r["sample_id"] for r in manifest], f"{name} identity drift")
        require(result["manifest_sha256"] == meta["sha256"] and result["predictions_sha256"] == sha(prediction_path), f"{name} provenance drift")
        require(result["metrics"] == final.summarize(prediction), f"{name} metric drift")
        localization[name] = {"execution": "RUN_NEW", "inference_condition": "G1",
            "predictions_sha256": sha(prediction_path), "metrics": result["metrics"]}
    result = {"schema": "c1_native_staged_r1_final_ood_v1", "status": "COMPLETE", "classification_primary": "C1-center",
        "generated_at_utc": now(), "protocol": protocol, "classification": protocol["classification"],
        "localization": localization, "C1_sha256": C1_SHA, "new_R1_sha256": new_r1_sha()}
    write_json(OUT / "results.json", result)
    report_lines = [
        "# C1 原生分阶段 R1：Official1000 与外部 OOD",
        "",
        f"生成时间：`{result['generated_at_utc']}`。C1 checkpoint SHA256 `{C1_SHA}`；C1 原生分阶段 R1 SHA256 `{result['new_R1_sha256']}`。",
        "",
        "## 分类 OOD：最终 C1-center",
        "",
        f"**主分类口径是 C1-center**：使用 C1 的 H2 分类 logit margin，并以已冻结的 internal-TRAIN 均值 `{protocol['classification_threshold_raw_margin']:.9f}` 为阈值，`margin > mean` 判 Fake。等价地，使用冻结 mean/std 标准化后的 `z_C > 0`。new R1 仅修改 `[SEG]` 之后的定位路径。复用已完成的 Phase6D.5 C1 逐样本分类推理，并与 Phase6D.6 的 C1-center 数值逐项复核；四份结果与当前 final_evaluation manifest 的 SHA256、样本数和顺序匹配。raw C1（margin > 0）仅作为诊断保留在结果 JSON，不进入下表。未在 OOD 上调阈值，也没有重复 GPU 推理。",
        "",
        "| 数据集 | N | Accuracy | Precision | Fake recall | TNR | FPR | F1 | ROC-AUC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in result["classification"].items():
        m = entry["metrics"]
        fmt = lambda key: "N/A" if m.get(key) is None else f"{m[key]:.6f}"
        report_lines.append(f"| {name} | {m['n']} | {fmt('accuracy')} | {fmt('precision')} | {fmt('fake_recall')} | {fmt('tnr')} | {fmt('fpr')} | {fmt('f1')} | {fmt('roc_auc')} | {fmt('auprc')} |")
    report_lines.extend(["", "RAISE998 只有 Real，主要看 TNR/FPR；ROC-AUC 和 AUPRC 不适用。GenImage 的八个生成器分组 C1-center 指标完整保存在结果 JSON 中。ROC-AUC/AUPRC 沿用同一原始 margin 的排序结果；正比例线性中心化不改变排序。", "",
        "## 定位 OOD", "", "SynthScars 复用本轮 Phase6E.3 C1 原生分阶段 R1 的 Official1000 G0，逐图 ID、顺序和 GT 前景像素数与最终评测清单一致。其余三组是本轮新推理：known-Fake G1 canonical unified prompt，不输入 GT 解释或定位短语。",
        "", "| 数据集 | 模式 | N | 空 GT | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"])
    for name, entry in result["localization"].items():
        m = entry["metrics"]
        report_lines.append(f"| {name} | {entry['inference_condition']} | {m['n']} | {m['empty_gt_images']} | {m['mean_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {m['global_foreground_iou']:.6f} | {m['global_foreground_f1']:.6f} |")
    report_lines.extend(["", "无 `[SEG]` 或无有效 query 的样本按 full-N 计零；多个预测 mask 在原图上以 logit 最大值取并集，固定阈值为 0。X-AIGD 的 `labels=[]` 和 PAL4VST 的官方空 GT 都保留；LOKI 的 GT 是 box union。不同 GT 语义与 G0/G1 模式的绝对值不直接横比。", ""])
    leak = json.loads((ROOT / "outputs/final_eval_datasets/leakage_audit/summary.json").read_text())
    override = json.loads((ROOT / "outputs/final_eval_datasets/leakage_override.json").read_text())
    report_lines.extend(["## 数据重叠与使用边界", "",
        f"沿用 final_evaluation 的泄漏审计：`{leak['status']}`；覆盖记录：`{override['status']}`。AIGI-Holmes 与内部训练集存在已记录的 779 个 exact SHA256 重叠，因此该行应连同此限制披露。",
        "", "分类结果是 C1-center 的冻结后处理复用，不是 new R1 带来的分类增益。定位结果按各数据集固定口径报告；没有在这些 OOD 集上选择 checkpoint、训练参数或阈值。",
        "", "复核入口：[结果 JSON](../outputs/final_evaluation/c1_native_staged_r1/results.json)、[冻结协议](../outputs/final_evaluation/c1_native_staged_r1/protocol.json)、[评测脚本](../scripts/final_eval_c1_native_staged_r1.py)。", ""])
    (ROOT / "docs/c1_native_staged_r1_final_ood.md").write_text("\n".join(report_lines))
    print(json.dumps({"status": "COMPLETE", "localization": {n: v["metrics"]["mean_foreground_iou"] for n, v in localization.items()}}))


def supervisor():
    preflight()
    status_path = OUT / "supervisor_status.json"
    status = {"status": "RUNNING", "started_at_utc": now(), "pid": os.getpid(),
        "gpu_assignments": {"0": ["loki", "pal4vst"], "2": ["xaigd"]}}
    write_json(status_path, status)
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = {}
    try:
        for gpu, datasets in ((0, ("loki", "pal4vst")), (2, ("xaigd",))):
            logfile = (logs / f"gpu{gpu}.log").open("a", buffering=1)
            commands = [
                [sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--dataset", name, "--device", "cuda:0"]
                for name in datasets
            ]
            # The shell only sequences fixed argument lists; dataset names are
            # validated constants above, and each worker has its own output.
            cmd = " && ".join(shlex.join(call) for call in commands)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            jobs[gpu] = (subprocess.Popen(["/bin/bash", "-lc", cmd], cwd=ROOT, env=env,
                stdout=logfile, stderr=subprocess.STDOUT), logfile)
        status["worker_pids"] = {str(gpu): value[0].pid for gpu, value in jobs.items()}
        write_json(status_path, status)
        while True:
            codes = {gpu: value[0].poll() for gpu, value in jobs.items()}
            if any(code not in (None, 0) for code in codes.values()):
                raise RuntimeError(f"localization worker failed: {codes}")
            if all(code == 0 for code in codes.values()):
                break
            time.sleep(20)
        status["stage"] = "finalize"; write_json(status_path, status)
        finalize()
        status["status"] = "COMPLETE"
    except BaseException as error:
        status.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        for process, _ in jobs.values():
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for process, logfile in jobs.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            logfile.close()
        status["updated_at_utc"] = now(); write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "worker", "finalize", "supervisor"), required=True)
    parser.add_argument("--dataset", choices=("loki", "xaigd", "pal4vst"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "preflight":
        print(json.dumps({"status": "PASS", "classification": list(preflight()["classification"])}, ensure_ascii=False))
    elif args.mode == "worker":
        require(args.dataset is not None, "worker needs --dataset")
        worker(args.dataset, args.device)
    elif args.mode == "finalize":
        finalize()
    else:
        supervisor()


if __name__ == "__main__":
    main()
