#!/usr/bin/env python3
"""Phase 6D.5 frozen C1/RINE decision integration."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import CLIPImageProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model

OUT = ROOT / "outputs/phase6d5_decision_integration"
DOC = ROOT / "docs/phase6d5_decision_integration.md"
MODELS = {
    "c0": (ROOT/"configs/phase6d3_c0_h2_p1.yaml", ROOT/"checkpoints/phase6d3_c0/diagnostics/epoch_05/mp_rank_00_model_states.pt"),
    "c1": (ROOT/"configs/phase6d3_c1_rine_conditioned_p1.yaml", ROOT/"checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt"),
}
MANIFESTS = {
    "internal_train": ROOT/"outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl",
    "internal_validation": ROOT/"outputs/data_audits/unified_forensics_split_v1/val_combined.jsonl",
    "internal_test": ROOT/"outputs/data_audits/unified_forensics_split_v1/test_combined.jsonl",
    "dev_ood_2560": ROOT/"outputs/phase6d2_cls_head_capacity/dev_ood_genimage_2560.jsonl",
    "loki_2217": ROOT/"datasets/LOKI/manifests/classification_eval_manifest.jsonl",
}

def read_jsonl(path): return [json.loads(x) for x in Path(path).open() if x.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def dump(path,x):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n")
def label(row):
    value=row.get("class_label",row.get("label"))
    return int(value) if isinstance(value,(int,bool)) else int(str(value).lower()=="fake")
def image_path(row):
    candidate=row.get("image_path")
    if candidate and Path(candidate).is_file(): return str(Path(candidate).resolve())
    root=ROOT/("datasets/SynthScars" if row.get("forensics_domain")=="fake" else "datasets")
    return str((root/row["image_relpath"]).resolve())
def image(row):
    path=image_path(row);x=cv2.imread(path,cv2.IMREAD_COLOR)
    if x is None: raise OSError(path)
    return cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

def extract(model_name,device,batch_size):
    cfg_path,ckpt=MODELS[model_name]; cfg=yaml.safe_load(cfg_path.read_text())
    dev=torch.device(device);torch.cuda.set_device(dev)
    conversation_lib.default_conversation=conversation_lib.conv_templates["llava_v1"]
    model,tok,meta=load_model(cfg,ckpt,dev,expected_step=2500,expected_epoch=5)
    processor=CLIPImageProcessor.from_pretrained(cfg["model"]["vision_tower"],local_files_only=True)
    backend=GLaMMForensicsBackend(model,tok,device=dev,dtype=torch.bfloat16,max_new_tokens=400)
    for split,manifest in MANIFESTS.items():
        src=read_jsonl(manifest); path=OUT/"raw"/model_name/f"{split}.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
        old=read_jsonl(path) if path.exists() else []; done={x["sample_id"] for x in old}
        pending=[x for x in src if x["sample_id"] not in done]
        with path.open("a") as h, ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(pending),batch_size):
                part=pending[start:start+batch_size];ims=list(pool.map(image,part))
                pixels=processor(images=ims,return_tensors="pt")["pixel_values"]
                samples=[{"image_path":image_path(r),"global_enc_image":p,"grounding_enc_image":None,
                    "bboxes":None,"conversations":[""],"masks":None,"label":None,"resize":None,
                    "questions":[],"sampled_classes":[],"cls_label":label(r),"seg_valid":False,
                    "sample_id":r["sample_id"],"source":r.get("source",r.get("generator/source")),
                    "content_category":r.get("content_category",r.get("generator/source")),"manifest_row":r}
                    for r,p in zip(part,pixels)]
                batch=backend._batch_many(samples,"",question=UNIFIED_FORENSICS_QUESTION);batch["grounding_enc_images"]=None
                with torch.inference_mode(): out=model.model_forward(**batch)
                margins=(out["cls_logits"][:,1]-out["cls_logits"][:,0]).float().cpu().tolist()
                rine=(model._last_rine_binary_logits.cpu().tolist() if model_name=="c1" else [None]*len(part))
                for r,m,rr in zip(part,margins,rine):
                    h.write(json.dumps({"sample_id":r["sample_id"],"label":label(r),"margin":m,
                        "rine_margin":rr,"image_path":image_path(r)},ensure_ascii=False)+"\n")
                h.flush()
                if (start//batch_size+1)%50==0: print(json.dumps({"model":model_name,"split":split,"done":len(done)+min(start+batch_size,len(pending)),"total":len(src)}),flush=True)
        got=read_jsonl(path)
        if [x["sample_id"] for x in got] != [x["sample_id"] for x in src]: raise RuntimeError(f"identity/order mismatch {model_name} {split}")
        dump(OUT/"raw"/model_name/f"{split}.complete.json",{"status":"COMPLETE","count":len(got),"manifest_sha256":sha(manifest),"checkpoint_sha256":meta["checkpoint_sha256"]})

def basic(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float);p=s>0
    tp=int(((p==1)&(y==1)).sum());tn=int(((p==0)&(y==0)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
    prec=tp/max(1,tp+fp);rec=tp/max(1,tp+fn);tnr=tn/max(1,tn+fp)
    fpr, tpr, _=roc_curve(y,s)
    at={f"recall_at_fpr_{int(q*100)}pct":float(tpr[fpr<=q].max()) if np.any(fpr<=q) else 0.0 for q in (.01,.05,.10)}
    return {"n":len(y),"accuracy":float((p==y).mean()),"roc_auc":float(roc_auc_score(y,s)),"fake_recall":rec,"tnr":tnr,"fpr":1-tnr,"f1":2*prec*rec/max(1e-30,prec+rec),"tp":tp,"tn":tn,"fp":fp,"fn":fn,**at}
def dist(y,s):
    ans={};y=np.asarray(y);s=np.asarray(s,float)
    for name,v in [("real",0),("fake",1)]:
        a=s[y==v];q=np.percentile(a,[5,25,50,75,95])
        ans[name]={"n":len(a),"mean":float(a.mean()),"std":float(a.std()),"median":float(np.median(a)),"p5":float(q[0]),"p25":float(q[1]),"p50":float(q[2]),"p75":float(q[3]),"p95":float(q[4]),"fraction_score_gt_0":float((a>0).mean())}
    return ans
def complement(y,a,b):
    y=np.asarray(y);ca=(np.asarray(a)>0)==y;cb=(np.asarray(b)>0)==y
    def one(mask):
        return {"n":int(mask.sum()),"both_correct":int((ca&cb&mask).sum()),"first_only_correct":int((ca&~cb&mask).sum()),"second_only_correct":int((~ca&cb&mask).sum()),"both_wrong":int((~ca&~cb&mask).sum()),"disagreement_rate":float(((ca!=cb)&mask).sum()/max(1,mask.sum()))}
    return {"overall":one(np.ones(len(y),bool)),"real":one(y==0),"fake":one(y==1),"pearson":float(pearsonr(a,b).statistic),"spearman":float(spearmanr(a,b).statistic)}

def finalize():
    data={}
    for split in MANIFESTS:
        c0=read_jsonl(OUT/"raw/c0"/f"{split}.jsonl");c1=read_jsonl(OUT/"raw/c1"/f"{split}.jsonl")
        if [x["sample_id"] for x in c0]!=[x["sample_id"] for x in c1]:raise RuntimeError(split)
        data[split]={"ids":[x["sample_id"] for x in c1],"y":np.array([x["label"] for x in c1]),"c0":np.array([x["margin"] for x in c0]),"c1":np.array([x["margin"] for x in c1]),"rine":np.array([x["rine_margin"] for x in c1])}
    tr=data["internal_train"]
    norm={k:{"mean":float(tr[k].mean()),"std":float(tr[k].std())} for k in ("c1","rine")}
    def z(d,k):return (d[k]-norm[k]["mean"])/norm[k]["std"]
    val=data["internal_validation"];alphas=[]
    for alpha in np.arange(0,1.0001,.1):
        s=alpha*z(val,"rine")+(1-alpha)*z(val,"c1");m=basic(val["y"],s);alphas.append({"alpha":round(float(alpha),1),**m})
    chosen=sorted(alphas,key=lambda x:(-x["roc_auc"],-x["accuracy"],abs(x["alpha"]-.5)))[0]["alpha"]
    X=np.column_stack([z(tr,"rine"),z(tr,"c1")]);stack=LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,random_state=3407).fit(X,tr["y"])
    params={"normalization":norm,"alpha_grid":[round(float(x),1) for x in np.arange(0,1.0001,.1)],"alpha_validation_results":alphas,"selected_alpha":chosen,"stack":{"feature_order":["sR","sC"],"weights":stack.coef_[0].tolist(),"bias":float(stack.intercept_[0]),"fit_split":"internal_train"}}
    dump(OUT/"fusion_parameters.json",params)
    result={"schema":"phase6d5_decision_integration_v1","status":"COMPLETE","protocol":{"threshold":0,"alpha_selector":"validation ROC-AUC, then Accuracy, then closest to 0.5","stack_fit":"internal TRAIN only, logistic no hidden layer","ood_used_for_selection":False},"fusion_parameters":params,"splits":{},"d0":{}}
    for split,d in data.items():
        sC=z(d,"c1");sR=z(d,"rine");scores={"C1":d["c1"],"RINE":d["rine"],"D1-fixed":.5*sC+.5*sR,"D1-alpha":chosen*sR+(1-chosen)*sC,"D1-stack":stack.decision_function(np.column_stack([sR,sC]))}
        result["splits"][split]={k:basic(d["y"],s) for k,s in scores.items()}
        result["d0"][split]={"score_distributions":{k:dist(d["y"],d[k]) for k in ("c0","c1","rine")},"metrics":{k:basic(d["y"],d[k]) for k in ("c0","c1","rine")},"c1_vs_rine":complement(d["y"],d["c1"],d["rine"])}
        path=OUT/"raw_scores"/f"{split}.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w") as h:
            for i,sid in enumerate(d["ids"]):h.write(json.dumps({"sample_id":sid,"label":int(d["y"][i]),"c0_margin":float(d["c0"][i]),"c1_margin":float(d["c1"][i]),"rine_margin":float(d["rine"][i]),**{k:float(v[i]) for k,v in scores.items() if k.startswith("D1")}})+"\n")
    held=["internal_test","dev_ood_2560","loki_2217"]
    def beats(base):
        for arm in ("D1-fixed","D1-alpha","D1-stack"):
            pairs=[(result["splits"][s][arm],result["splits"][s][base]) for s in held]
            if all(a["accuracy"]>=b["accuracy"] and a["roc_auc"]>=b["roc_auc"] for a,b in pairs) and any(a["accuracy"]>b["accuracy"] or a["roc_auc"]>b["roc_auc"] for a,b in pairs):return True
        return False
    dev=result["splits"]["dev_ood_2560"];lok=result["splits"]["loki_2217"]
    gaps=[dev["RINE"]["roc_auc"]-dev["C1"]["roc_auc"],lok["RINE"]["roc_auc"]-lok["C1"]["roc_auc"]]
    shift="YES" if all(g<=.02 for g in gaps) and all(result["splits"][s]["C1"]["fake_recall"]<result["splits"][s]["RINE"]["fake_recall"] for s in ("dev_ood_2560","loki_2217")) else ("NO" if all(g>.05 for g in gaps) else "MIXED")
    loses="YES" if all(g>.02 for g in gaps) else ("NO" if all(g<=.02 for g in gaps) else "MIXED")
    fusion_arms=["D1-fixed","D1-alpha","D1-stack"]
    best=max(fusion_arms,key=lambda a:np.mean([result["splits"][s][a]["roc_auc"] for s in held]))
    bC,bR=beats("C1"),beats("RINE")
    result["decisions"]={"C1_DECISION_SHIFT_DOMINANT":shift,"C1_LOSES_DISCRIMINATIVE_INFO":loses,"FROZEN_FUSION_BEATS_C1":"YES" if bC else "NO","FROZEN_FUSION_BEATS_RINE":"YES" if bR else "NO","BEST_FROZEN_DECISION_ARM":best,"ALLOW_DUAL_PATH_TRAINING_NEXT":"YES" if bC and bR else "NO"}
    dump(OUT/"results.json",result);dump(OUT/"score_distribution_statistics.json",result["d0"])
    lines=["# Phase 6D.5 — C1/RINE Decision Integration","","Status: **COMPLETE**. All backbones and heads were frozen. Alpha used internal validation only; logistic stacking used internal TRAIN only.","","## Frozen-arm metrics","","| Split | Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 | R@FPR1% | R@FPR5% | R@FPR10% |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ["internal_validation","internal_test","dev_ood_2560","loki_2217"]:
        for arm,m in result["splits"][split].items():lines.append(f"| {split} | {arm} | {m['accuracy']:.6f} | {m['roc_auc']:.6f} | {m['fake_recall']:.6f} | {m['tnr']:.6f} | {m['fpr']:.6f} | {m['f1']:.6f} | {m['recall_at_fpr_1pct']:.6f} | {m['recall_at_fpr_5pct']:.6f} | {m['recall_at_fpr_10pct']:.6f} |")
    lines += ["","## Frozen parameters","",f"- selected alpha: `{chosen}`",f"- stack weights `[wR,wC]`: `{params['stack']['weights']}`; bias: `{params['stack']['bias']}`","","## Final decisions",""]+[f"- `{k} = {v}`" for k,v in result["decisions"].items()]
    DOC.write_text("\n".join(lines)+"\n")

def supervisor(batch_size):
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/"protocol.json",{"schema":"phase6d5_protocol_v1","models":{k:{"config":str(v[0]),"checkpoint":str(v[1]),"checkpoint_sha256":sha(v[1])} for k,v in MODELS.items()},"manifests":{k:{"path":str(v),"sha256":sha(v),"count":len(read_jsonl(v))} for k,v in MANIFESTS.items()},"devices":{"c0":"cuda:0","c1_plus_rine":"cuda:1"},"batch_size":batch_size,"selection":{"alpha":"internal_validation_only","stack":"internal_train_only"}})
    ps=[]
    for name,dev in [("c0","cuda:0"),("c1","cuda:1")]:
        log=(OUT/f"extract_{name}.log").open("a");p=subprocess.Popen([sys.executable,__file__,"--mode","extract","--model",name,"--device",dev,"--batch-size",str(batch_size)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"});ps.append((p,log))
    codes=[]
    for p,h in ps:codes.append(p.wait());h.close()
    if any(codes):raise SystemExit(f"extractors failed {codes}")
    finalize()

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["supervisor","extract","finalize"],required=True);p.add_argument("--model",choices=["c0","c1"]);p.add_argument("--device");p.add_argument("--batch-size",type=int,default=8);a=p.parse_args()
    if a.mode=="extract":extract(a.model,a.device,a.batch_size)
    elif a.mode=="finalize":finalize()
    else:supervisor(a.batch_size)
if __name__=="__main__":main()
