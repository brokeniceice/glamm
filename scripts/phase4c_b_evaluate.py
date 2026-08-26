#!/usr/bin/env python3
"""Evaluate selected readers on cached fresh G0, Phrase-Only, or TF-Full trajectories."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.evidence_reader import EvidenceReader
from scripts.phase4c_b_train import evaluate,load_p1,pair_paths,q_index
from tools.phase4c_b import diagnostic,dump,load_source_model,rows,summarize


def cli():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=("G0","phrase_only","tf_full_context"),required=True);p.add_argument("--device",default="cuda:0");return p.parse_args()


def context_records(cache,mode):
 result=[]
 for p in sorted((cache/"validation"/mode).glob("shard_*.pt")):result+=torch.load(p,map_location="cpu")["records"]
 return result


def baseline_records(records):
 return [{k:r[k] for k in ("sample_id","foreground_iou","foreground_f1","tp","fp","fn")} for r in records]


def write_records(path,records):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")


def main():
 a=cli();cfg=yaml.safe_load((ROOT/"configs/phase4c_b_evidence_reader.yaml").read_text());out=ROOT/cfg["experiment"]["output_root"];cache=Path(cfg["experiment"]["cache_root"]);device=torch.device(a.device);torch.cuda.set_device(device)
 complete=json.loads((cache/"validation"/a.mode/"complete.json").read_text());
 if complete["status"]!="COMPLETE":raise RuntimeError("context cache incomplete")
 source_root=ROOT/cfg["frozen_spatial_cache"]["phase3c1_root"];pairs=pair_paths(source_root,"val");ctx=context_records(cache,a.mode);p1=baseline_records(ctx);write_records(out/"evaluation"/a.mode/"P1"/"predictions.jsonl",p1);dump(out/"evaluation"/a.mode/"P1"/"metrics.json",summarize(p1))
 if a.mode=="G0":
  result={"P1":summarize(p1)}
  for arm in ("clip_reader","forensic_reader"):
   pred=rows(out/"evaluation/G0"/arm/"selected_predictions.jsonl");result[arm]=summarize(pred)
  dump(out/"g0_metrics.json",result);print(json.dumps(result,indent=2));return
 model,core=load_p1(cfg,device);q=q_index(cache/"validation",a.mode);result={"P1":summarize(p1)}
 for arm in ("clip_reader","forensic_reader"):
  reader=EvidenceReader().to(device);state=torch.load(Path(cfg["experiment"]["checkpoint_root"])/arm/"selected.pt",map_location="cpu");reader.load_state_dict(state["reader"]);reader.eval();source=load_source_model(cfg,arm,device);source._phase4c_b_arm=arm
  metrics,pred,low,attn,diag=evaluate(reader,source,core,q,pairs,device,save_attention=True);write_records(out/"evaluation"/a.mode/arm/"predictions.jsonl",pred);dump(out/"evaluation"/a.mode/arm/"metrics.json",metrics);torch.save({"sample_ids":[r["sample_id"] for r in pred],"low_res_logits":low,"attention":attn},out/"evaluation"/a.mode/arm/"selected_spatial.pt");result[arm]={**metrics,"selected_epoch":state["epoch"],"beta":float(reader.beta.detach()),"diagnostics":diag}
 dump(out/("phrase_only_metrics.json" if a.mode=="phrase_only" else "tf_full_metrics.json"),result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()

