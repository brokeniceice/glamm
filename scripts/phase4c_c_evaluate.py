#!/usr/bin/env python3
"""Matched reproduction and frozen evidence interventions using the Phase 4C-B evaluator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from model.evidence_reader import EvidenceReader
from scripts.phase4c_b_train import evaluate, load_p1, pair_paths, q_index
from tools.phase4c_b import dump, load_source_model, load_spatial_shard, rows, source_feature, summarize


def cli():
    p=argparse.ArgumentParser(); p.add_argument("--arm",choices=("clip_reader","forensic_reader"),required=True)
    p.add_argument("--mode",choices=("baseline","interventions"),required=True); p.add_argument("--device",default="cuda:0")
    return p.parse_args()


def write_records(path, values):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("".join(json.dumps(x)+"\n" for x in values))


def evidence_cache(cfg,bcfg,arm,source,device,pairs):
    root=Path(cfg["experiment"]["cache_root"])/"evidence"; root.mkdir(parents=True,exist_ok=True); path=root/f"{arm}.pt"
    if path.exists():
        state=torch.load(path,map_location="cpu"); return {sid:state["features"][i] for i,sid in enumerate(state["sample_ids"])}
    ids=[]; values=[]
    with torch.no_grad():
        for cp,_ in pairs:
            clip=load_spatial_shard(cp,"clip"); raw=clip["features"].to(device=device,dtype=torch.float32)
            feature=source_feature(source,raw,arm).float().cpu()
            ids += [r["sample_id"] for r in clip["records"]]; values += list(feature)
    torch.save({"schema":"phase4c_c_frozen_evidence_v1","arm":arm,"sample_ids":ids,
                "features":torch.stack(values),"dtype":"torch.float32","shape":[256,24,24]},path)
    return {sid:values[i] for i,sid in enumerate(ids)}


def contexts(cfg,bcfg):
    bcache=Path(bcfg["experiment"]["cache_root"]); ccache=Path(cfg["experiment"]["cache_root"])
    return {"G0":q_index(bcache/"validation","G0"),"phrase_repair":q_index(ccache/"validation","phrase_repair"),
            "phrase_only":q_index(bcache/"validation","phrase_only"),"tf_full_context":q_index(bcache/"validation","tf_full_context")}


def reference_metrics(bout,arm,query):
    name={"G0":"g0_metrics.json","phrase_only":"phrase_only_metrics.json","tf_full_context":"tf_full_metrics.json"}[query]
    return json.loads((bout/name).read_text())[arm]


def main():
    a=cli(); cfg=yaml.safe_load((ROOT/"configs/phase4c_c_evidence_attribution.yaml").read_text()); bcfg=yaml.safe_load((ROOT/cfg["phase4c_b"]["config"]).read_text())
    out=ROOT/cfg["experiment"]["output_root"]; bout=ROOT/cfg["phase4c_b"]["output_root"]; device=torch.device(a.device); torch.cuda.set_device(device)
    model,core=load_p1(bcfg,device); source=load_source_model(bcfg,a.arm,device); source._phase4c_b_arm=a.arm
    state=torch.load(Path(cfg["phase4c_b"]["clip_reader_checkpoint"] if a.arm=="clip_reader" else cfg["phase4c_b"]["forensic_reader_checkpoint"]),map_location="cpu")
    reader=EvidenceReader().to(device); reader.load_state_dict(state["reader"]); reader.eval()
    pairs=pair_paths(ROOT/bcfg["frozen_spatial_cache"]["phase3c1_root"],"val"); evidence=evidence_cache(cfg,bcfg,a.arm,source,device,pairs); qs=contexts(cfg,bcfg)
    common=json.loads((out/"common_population.json").read_text())["sample_ids"]; common_set=set(common)
    if a.mode=="baseline":
        result={"status":"PASS","arm":a.arm,"selected_epoch":state["epoch"],"tolerance":1e-4,"queries":{}}
        for query in ("G0","phrase_repair","phrase_only","tf_full_context"):
            include=None if query!="phrase_repair" else common
            metrics,records,low,_,diag=evaluate(reader,source,core,qs[query],pairs,device,evidence_by_id=evidence,include_ids=include)
            root=out/"evaluation"/a.arm/query; write_records(root/"matched_full.jsonl",records); dump(root/"matched_full_metrics.json",metrics)
            common_records=[r for r in records if r["sample_id"] in common_set]; write_records(root/"matched.jsonl",common_records); dump(root/"matched_metrics.json",summarize(common_records))
            row={"metrics":metrics,"common_metrics":summarize(common_records),"diagnostics":diag}
            if query!="phrase_repair":
                ref=reference_metrics(bout,a.arm,query); error=abs(metrics["mean_foreground_iou"]-ref["mean_foreground_iou"])
                row.update({"phase4c_b_reference":ref["mean_foreground_iou"],"absolute_error":error,"reproduced":error<=1e-4})
                if error>1e-4: result["status"]="FAIL"
            result["queries"][query]=row
        dump(out/f"baseline_reproduction_{a.arm}.json",result)
        if result["status"]!="PASS": raise RuntimeError(f"{a.arm} baseline reproduction failed")
        print(json.dumps(result,indent=2)); return
    pre=json.loads((out/"preflight_manifest.json").read_text())
    if pre["status"]!="PASS": raise RuntimeError("PASS preflight required before interventions")
    mapping=json.loads((out/"phase4c_c_image_permutation.json").read_text())["mapping"]
    perm=torch.tensor(json.loads((out/"spatial_shuffle_permutation.json").read_text())["permutation"],device=device)
    def transform(condition):
        if condition=="cross_image": return lambda sid,x:evidence[mapping[sid]].to(device=device,dtype=torch.float32).unsqueeze(0)
        if condition=="spatial_shuffle": return lambda sid,x:x.flatten(2).transpose(1,2)[:,perm,:]
        if condition=="global_repeat": return lambda sid,x:x.flatten(2).transpose(1,2).mean(1,keepdim=True).expand(-1,576,-1)
        if condition=="zero": return lambda sid,x:torch.zeros_like(x)
        return None
    summary={"status":"COMPLETE","arm":a.arm,"population_n":len(common),"queries":{}}
    for query in ("G0","phrase_repair","phrase_only","tf_full_context"):
        qroot=out/"evaluation"/a.arm/query; summary["queries"][query]={"matched":json.loads((qroot/"matched_metrics.json").read_text())}
        for condition in ("cross_image","spatial_shuffle","global_repeat","zero"):
            predpath=qroot/f"{condition}.jsonl"
            if predpath.exists() and sum(1 for _ in predpath.open())==len(common):
                records=rows(predpath); metrics=summarize(records); diag=None
            else:
                metrics,records,low,_,diag=evaluate(reader,source,core,qs[query],pairs,device,evidence_by_id=evidence,evidence_transform=transform(condition),include_ids=common)
                write_records(predpath,records); dump(qroot/f"{condition}_metrics.json",metrics)
                if query=="G0" and condition in ("cross_image","spatial_shuffle","global_repeat","zero"):
                    sp=Path(cfg["experiment"]["cache_root"])/"spatial"/a.arm/query; sp.mkdir(parents=True,exist_ok=True)
                    torch.save({"sample_ids":[r["sample_id"] for r in records],"low_res_logits":low},sp/f"{condition}.pt")
            summary["queries"][query][condition]={"metrics":metrics,"diagnostics":diag}
            print(json.dumps({"arm":a.arm,"query":query,"condition":condition,"mean_iou":metrics["mean_foreground_iou"]}),flush=True)
    dump(out/f"intervention_{a.arm}.json",summary); print(json.dumps({"status":"COMPLETE","arm":a.arm},indent=2))


if __name__=="__main__":main()
