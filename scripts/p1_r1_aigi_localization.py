#!/usr/bin/env python3
"""Frozen P1/R1 AIGI-test localization comparison (G1/Phrase/TF).

This is an external evaluation only.  It never reads internal test or
official1000 and does not train, select, tune, or mutate a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    CANONICAL_UNIFIED_QUESTION,
    UnifiedForensicsDataset,
)
from eval.forensics_eval import GLaMMForensicsBackend
from model.SAM.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4g1q_conditional_utility as q
from scripts import phase4ha_utility_gated_rectification as ha
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase2a_final_evaluate import load_model
from scripts.phase3c1_cache import clip_grid
from tools.phase3f_aogd import core_model
from tools.phase3c1 import geometry_for
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import clip_coordinates, compare, sam_coordinates, summarize, tensor_state_sha256
from tools.phase4f import load_evidence_source, load_sam_runtime

SEED = 3407
P1 = Path("/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt")
P1_SHA = "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"
R1 = ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt"
AIGI = ROOT / "datasets/AIGI-Holmes-Dataset/dataset/test.jsonl"
AIGI_ROOT = ROOT / "datasets/AIGI-Holmes-Dataset"
OUT = ROOT / "outputs/p1_r1_aigi_localization"
CACHE = Path("/data/yz/groundingLMM_official/cache/p1_r1_aigi_localization")
PROTOCOL = OUT / "protocol.json"
RESULTS = OUT / "results.json"
P1_CFG = yaml.safe_load((ROOT / "configs/phase3a_p1.yaml").read_text())
P4F_CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())
MODES = ("g1",)
PHRASE = "forged area"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def read_fake_rows() -> list[dict]:
    raw = [json.loads(x) for x in AIGI.read_text().splitlines() if x.strip()]
    rows = []
    for ordinal, row in enumerate(raw):
        if not row.get("mask"):
            continue
        image = (AIGI_ROOT / str(row["images"][0]).removeprefix("./")).resolve()
        mask = (AIGI_ROOT / str(row["mask"]).removeprefix("./")).resolve()
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError((image, mask))
        rows.append({**row, "sample_id": f"aigi_test:{ordinal:06d}", "image_path": str(image), "mask_path": str(mask), "ordinal": ordinal})
    if len(raw) != 1731 or len(rows) != 861 or len({r["sample_id"] for r in rows}) != 861:
        raise RuntimeError("AIGI-test population drift")
    return rows


def explanation(row: dict) -> str:
    text = str(row["response"]).strip()
    text = re.sub(r"^This is a fake image\s*\[SEG\]\.?\s*", "", text, count=1, flags=re.I)
    text = " ".join(text.split())
    if not text:
        raise RuntimeError(f"empty AIGI explanation: {row['sample_id']}")
    return text


class AIGIDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.global_processor = CLIPImageProcessor.from_pretrained(P1_CFG["model"]["vision_tower"])
        self.transform = ResizeLongestSide(int(P1_CFG["model"]["image_size"]))

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise OSError(row["sample_id"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); h, w = image.shape[:2]
        if mask.shape != (h, w): raise RuntimeError(f"mask geometry drift: {row['sample_id']}")
        resized = self.transform.apply_image(image)
        manifest = {**row, "class_label": 1, "forensics_domain": "fake", "explanation": explanation(row)}
        return {
            "image_path": row["image_path"],
            "global_enc_image": self.global_processor.preprocess(image, return_tensors="pt")["pixel_values"][0],
            "grounding_enc_image": UnifiedForensicsDataset.grounding_enc_processor(torch.from_numpy(resized).permute(2, 0, 1).contiguous()),
            "bboxes": None, "conversations": [],
            "masks": torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0),
            "label": torch.full((h, w), UnifiedForensicsDataset.IGNORE_LABEL, dtype=torch.long),
            "resize": resized.shape[:2], "questions": [CANONICAL_UNIFIED_QUESTION],
            "sampled_classes": ["synthetic artifact"], "cls_label": 1, "seg_valid": True,
            "sample_id": row["sample_id"], "source": "AIGI-test", "content_category": None,
            "manifest_row": manifest, "prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
            "prompt_sha256": CANONICAL_PROMPT_SHA256, "target_protocol": "aigi_adapted_phrase_v1",
            "localization_field": {"raw_phrases": [PHRASE], "normalized_training_phrase": PHRASE,
                                   "construction_rule": "verbatim localization noun phrase from AIGI query"},
        }


def freeze_protocol() -> dict:
    rows = read_fake_rows()
    contract = {
        "schema": "p1_r1_aigi_localization_g1_only_protocol_v2", "status": "FROZEN_BEFORE_INFERENCE",
        "population": {"source_jsonl": str(AIGI), "source_sha256": file_sha256(AIGI), "all": 1731,
                       "real_excluded": 870, "fake_with_nonempty_mask": 861,
                       "sample_ids_sha256": canonical_hash([r["sample_id"] for r in rows])},
        "models": {"P1": {"checkpoint": str(P1), "sha256": file_sha256(P1)},
                   "R1": {"checkpoint": str(R1), "sha256": file_sha256(R1), "phase4hd_selected_epoch": 9}},
        "prompt": {"user": CANONICAL_UNIFIED_QUESTION, "sha256": CANONICAL_PROMPT_SHA256,
                   "G1": "canonical user plus assistant continuation prefix [FAKE], then greedy generation",
                   "excluded_by_user_correction": ["Phrase", "TF"]},
        "metric": {"target": "AIGI released pixel mask", "threshold": "mask logit > 0", "primary": "per-image mean foreground IoU",
                   "also": ["mean foreground F1", "global foreground IoU", "global foreground F1"], "invalid_G1": "IoU=0 empty prediction"},
        "generation": {"seed": SEED, "do_sample": False, "num_beams": 1, "max_new_tokens": 400, "batch": 4},
        "comparison": "P1 and R1 share the exact G1 sample order, generated trajectory/q_seg, image features, masks, geometry and threshold",
        "prohibited": {"training": False, "selection": False, "tuning": False, "internal_test_accessed": False, "official1000_accessed": False},
    }
    if contract["models"]["P1"]["sha256"] != P1_SHA: raise RuntimeError("P1 hash drift")
    state = torch.load(R1, map_location="cpu", weights_only=False)
    if state.get("arm") != "r1" or state.get("epoch") != 9: raise RuntimeError("R1 provenance drift")
    contract["contract_sha256"] = canonical_hash(contract)
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text())
        if old != contract: raise RuntimeError("existing protocol drift")
    else: dump(PROTOCOL, contract)
    return contract


def load_p1(device):
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_model(P1_CFG, P1, device, expected_step=3500, expected_epoch=7)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
    return model, backend, meta


def capture(core, call):
    holder = []; original = core._extract_projected_seg_predictor_hidden
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        holder.append([x.detach().to(torch.bfloat16).cpu() for x in result[0]])
        return result
    core._extract_projected_seg_predictor_hidden = wrapped
    try: output = call()
    finally: core._extract_projected_seg_predictor_hidden = original
    vectors = holder[-1] if holder else []
    return output, vectors


def source_hashes(model, source):
    return {"p1": tensor_state_sha256(model.state_dict()), "forensic": tensor_state_sha256(source.state_dict())}


def build_shared(model, dataset, device):
    dest = CACHE / "shared"; complete = dest / "complete.json"; dest.mkdir(parents=True, exist_ok=True)
    source = load_evidence_source(P4F_CFG, "forensic_rect", device); before = source_hashes(model, source)
    paths = []
    with torch.no_grad():
        for start in range(0, len(dataset), 16):
            end = min(start + 16, len(dataset)); path = dest / f"shard_{start:06d}_{end:06d}.pt"
            if path.exists(): paths.append(path); continue
            payload = {k: [] for k in ("sample_ids", "Sraw64", "S64", "F24", "z_F24", "sam_geometries", "clip_geometries")}
            for index in range(start, end):
                sample = dataset[index]; h, w = sample["masks"].shape[-2:]
                sam_g, clip_g = geometry_for("sam", (h, w)), geometry_for("clip", (h, w))
                sraw = model.get_grounding_encoder_embs(sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16))
                tokens, _ = model.get_model().get_vision_tower()(sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16))
                raw_clip = clip_grid(tokens)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16): forensic = source(raw_clip, return_features=True)
                payload["sample_ids"].append(sample["sample_id"]); payload["Sraw64"].append(sraw[0].cpu())
                payload["S64"].append(sam_lowres_to_original_normalized(sraw, sam_g, output_hw=(64,64))[0].to(torch.bfloat16).cpu())
                payload["F24"].append(forensic["F_forensic"][0].to(torch.bfloat16).cpu())
                payload["z_F24"].append(forensic["logits"][0].to(torch.bfloat16).cpu())
                payload["sam_geometries"].append(sam_g); payload["clip_geometries"].append(clip_g)
            for k in ("Sraw64", "S64", "F24", "z_F24"): payload[k] = torch.stack(payload[k])
            tmp = path.with_suffix(".pt.tmp"); torch.save(payload, tmp); tmp.replace(path); paths.append(path)
            print(json.dumps({"stage":"AIGI_SHARED","done":end,"total":len(dataset)}), flush=True)
    after = source_hashes(model, source)
    if before != after: raise RuntimeError("shared-cache frozen source mutation")
    manifest = {"status":"COMPLETE","n":len(dataset),"shards":len(paths),"source_hash_before":before,"source_hash_after":after,
                "files":[{"path":str(p),"sha256":file_sha256(p)} for p in paths]}
    dump(complete, manifest); return manifest


def content(mode, sample):
    if mode == "phrase": return f"[FAKE] Target regions: {PHRASE} [SEG]"
    if mode == "tf": return f"[FAKE] {sample['manifest_row']['explanation']}\nTarget regions: {PHRASE} [SEG]"
    raise ValueError(mode)


def output_metric(sample, output):
    pred = output.get("pred_mask")
    target = torch.as_tensor(sample["masks"]).bool().any(0)
    logits = torch.full(target.shape, -torch.inf) if pred is None else torch.as_tensor(pred).detach().float().cpu().amax(0)
    return metric_record(sample["sample_id"], logits, target)


def build_context(mode, model, backend, dataset, start_bound=0, end_bound=None):
    dest = CACHE / mode; complete = dest / "complete.json"; dest.mkdir(parents=True, exist_ok=True)
    core = core_model(model); paths=[]; before=tensor_state_sha256(model.state_dict()); shard_size=4 if mode=="g1" else 16
    end_bound = len(dataset) if end_bound is None else min(int(end_bound), len(dataset))
    start_bound = int(start_bound)
    if start_bound % shard_size or (end_bound != len(dataset) and end_bound % shard_size):
        raise ValueError("worker bounds must align to context shard size")
    for start in range(start_bound, end_bound, shard_size):
        end=min(start+shard_size,end_bound); path=dest/f"shard_{start:06d}_{end:06d}.pt"
        if path.exists(): paths.append(path); continue
        samples=[dataset[i] for i in range(start,end)]; records=[]; values=[]; valid=[]
        if mode=="g1":
            outputs,vectors=capture(core,lambda:backend.generate_localization_batch(samples,provide_gt_fake=True,generation_mode="unified_prompt_gt_fake_prefix"))
            if len(vectors)!=len(samples): vectors=[]
            for i,(sample,output) in enumerate(zip(samples,outputs)):
                qv=vectors[i][0] if vectors and vectors[i].shape[0]==1 else None
                values.append(torch.zeros(256,dtype=torch.bfloat16) if qv is None else qv); valid.append(qv is not None)
                records.append({**output_metric(sample,output),"valid_q_seg":qv is not None,"seg_triggered":bool(output.get("seg_triggered")),
                                "generated_token_ids":output.get("generated_token_ids",[]),"generated_text":output.get("generated_text")})
        else:
            for sample in samples:
                batch=backend._batch(sample,content(mode,sample),question=CANONICAL_UNIFIED_QUESTION)
                output,vectors=capture(core,lambda b=batch:backend._model_forward_with_optional_evidence(b,None))
                qv=vectors[0][0] if len(vectors)==1 and vectors[0].shape[0]==1 else None
                pred=output["pred_masks"][0]; mapped={"pred_mask":pred if pred is not None and pred.shape[0] else None}
                values.append(torch.zeros(256,dtype=torch.bfloat16) if qv is None else qv); valid.append(qv is not None)
                records.append({**output_metric(sample,mapped),"valid_q_seg":qv is not None,"context_token_ids":batch["input_ids"][0].cpu().tolist()})
        payload={"mode":mode,"sample_ids":[s["sample_id"] for s in samples],"q_seg":torch.stack(values),"valid":torch.tensor(valid),"p1_records":records}
        tmp=path.with_suffix(".pt.tmp"); torch.save(payload,tmp); tmp.replace(path); paths.append(path)
        print(json.dumps({"stage":f"AIGI_{mode.upper()}","done":end,"total":len(dataset),"valid":sum(valid)}),flush=True)
    after=tensor_state_sha256(model.state_dict())
    if before!=after: raise RuntimeError("P1 mutation during context cache")
    all_ids=[]; nvalid=0
    for p in sorted(dest.glob("shard_*.pt")):
        x=torch.load(p,map_location="cpu",weights_only=False); all_ids+=x["sample_ids"]; nvalid+=int(x["valid"].sum())
    status = "COMPLETE" if len(all_ids) == len(dataset) else "PARTIAL"
    manifest={"status":status,"mode":mode,"n":len(all_ids),"valid":nvalid,"invalid":len(all_ids)-nvalid,"worker_range":[start_bound,end_bound],
              "p1_hash_before":before,"p1_hash_after":after,
              "files":[{"path":str(p),"sha256":file_sha256(p)} for p in sorted(dest.glob("shard_*.pt"))]}; dump(complete,manifest); return manifest


def load_mode(mode):
    ids=[]; qv=[]; valid=[]; records=[]
    for path in sorted((CACHE/mode).glob("shard_*.pt")):
        x=torch.load(path,map_location="cpu",weights_only=False); ids+=x["sample_ids"]; qv.append(x["q_seg"]); valid.append(x["valid"]); records+=x["p1_records"]
    return ids,torch.cat(qv),torch.cat(valid),records


def evaluate_r1(device, dataset):
    utility,rectifier,_=hd.load_common(device); state=torch.load(R1,map_location="cpu",weights_only=False)
    utility.load_state_dict(state["utility_state"],strict=True); rectifier.load_state_dict(state["rectifier_state"],strict=True)
    utility.eval().requires_grad_(False); rectifier.eval().requires_grad_(False); sam=load_sam_runtime(P4F_CFG,device)
    frozen={"utility":tensor_state_sha256(utility.state_dict()),"rectifier":tensor_state_sha256(rectifier.state_dict()),"sam":tensor_state_sha256(sam.state_dict())}
    shared_paths=sorted((CACHE/"shared").glob("shard_*.pt")); result={}
    for mode in MODES:
        ids,qseg,valid,p1_records=load_mode(mode); q_by={sid:i for i,sid in enumerate(ids)}; records=[]; gates=[]
        for sp in shared_paths:
            x=torch.load(sp,map_location="cpu",weights_only=False)
            for local,sid in enumerate(x["sample_ids"]):
                qi=q_by[sid]; sample=dataset[qi]; target=torch.as_tensor(sample["masks"]).bool().any(0)
                if not bool(valid[qi]):
                    records.append({"sample_id":sid,"foreground_iou":0.,"foreground_f1":0.,"tp":0,"fp":0,"fn":int(target.sum()),"valid_q_seg":False}); continue
                raw=x["Sraw64"][local:local+1].to(device); s64=x["S64"][local:local+1].to(device); f24=x["F24"][local:local+1].to(device); zf=x["z_F24"][local:local+1].to(device)
                qone=qseg[qi:qi+1].to(device); sg=x["sam_geometries"][local]; cg=x["clip_geometries"][local]
                with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False): low_p1=sam(qone.to(torch.bfloat16),raw.to(torch.bfloat16))
                zl=sam_lowres_to_original_normalized(low_p1,sg,output_hw=(256,256)).to(torch.bfloat16)
                batch={"S64":s64,"q_seg":qone,"z_L":zl,"F24":f24,"z_F24":zf,"clip_geometries":[cg],"valid_g0":torch.ones(1,dtype=torch.bool,device=device),
                       "forensic_present":torch.ones(1,dtype=torch.bool,device=device),"forensic_vacuous":torch.zeros(1,dtype=torch.bool,device=device),"forensic_off":torch.zeros(1,dtype=torch.bool,device=device)}
                with torch.no_grad():
                    uout=hc.utility_forward(utility,batch); evidence=x["F24"][local:local+1].to(device)
                    sc=sam_coordinates(sg,grid=64)[None].to(device); cc=clip_coordinates(cg,grid=24)[None].to(device)
                    ev_valid=torch.ones(1,576,dtype=torch.bool,device=device)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16): p4f=rectifier(raw,evidence,sc,cc,ev_valid)
                    gate=ha.gate_to_sam_grid(uout["U"],sc)*p4f["support"].reshape(1,1,64,64).float(); adapted=ha.gated_embedding(raw,p4f["image_embeddings"],gate)
                    with torch.autocast(device_type=device.type,enabled=False): low=sam(qone.to(torch.bfloat16),adapted.to(torch.bfloat16))
                logits=inverse_sam_logits(low,sg); row=metric_record(sid,logits,target); row["valid_q_seg"]=True; records.append(row); gates.append(gate[p4f["support"].reshape(1,1,64,64).bool()].cpu())
                if len(records)%100==0: print(json.dumps({"stage":"AIGI_R1","mode":mode,"done":len(records),"total":len(ids)}),flush=True)
        if [r["sample_id"] for r in records]!=ids: raise RuntimeError("R1 order drift")
        gate=torch.cat(gates).float() if gates else torch.empty(0)
        result[mode]={"P1":{"metrics":summarize(p1_records),"records":p1_records},"R1":{"metrics":summarize(records),"records":records},
                      "R1_vs_P1":compare(records,p1_records,seed=SEED),"gate":{"n":gate.numel(),"mean":float(gate.mean()),"std":float(gate.std(unbiased=False)),
                      "percentiles":{str(p):float(torch.quantile(gate,p/100)) for p in (0,1,5,25,50,75,95,99,100)},"low_le_0.01":float((gate<=.01).float().mean()),"high_ge_0.99":float((gate>=.99).float().mean())}}
    after={"utility":tensor_state_sha256(utility.state_dict()),"rectifier":tensor_state_sha256(rectifier.state_dict()),"sam":tensor_state_sha256(sam.state_dict())}
    if frozen!=after: raise RuntimeError("R1 frozen source mutation")
    payload={"schema":"p1_r1_aigi_localization_results_v1","status":"COMPLETE","protocol_sha256":file_sha256(PROTOCOL),"modes":result,
             "source_integrity":{"before":frozen,"after":after,"exact":frozen==after,"P1_checkpoint_sha256":file_sha256(P1),"R1_checkpoint_sha256":file_sha256(R1)},
             "firewall":{"internal_test_accessed":False,"official1000_accessed":False},"completed_unix":time.time()}
    dump(RESULTS,payload); return payload


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("freeze","run","context","evaluate")); ap.add_argument("--device",default="cuda:1")
    ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=None); args=ap.parse_args()
    freeze_protocol()
    if args.command=="freeze": print(PROTOCOL.read_text()); return
    device=torch.device(args.device); torch.cuda.set_device(device); seed_all(); rows=read_fake_rows(); dataset=AIGIDataset(rows)
    if args.command == "evaluate":
        evaluate_r1(device,dataset); print(json.dumps({"status":"COMPLETE","results":str(RESULTS)},ensure_ascii=False),flush=True); return
    model,backend,_=load_p1(device)
    if args.command == "run": build_shared(model,dataset,device)
    elif not (CACHE / "shared/complete.json").is_file(): raise RuntimeError("shared cache must be complete before parallel context workers")
    for mode in MODES: build_context(mode,model,backend,dataset,args.start,args.end)
    if args.command == "run": evaluate_r1(device,dataset)
    print(json.dumps({"status":"COMPLETE","results":str(RESULTS)},ensure_ascii=False),flush=True)


if __name__=="__main__": main()
