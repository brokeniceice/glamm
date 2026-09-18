#!/usr/bin/env python3
"""Continue completed core LEGION-match evaluation on one selected physical GPU."""
from __future__ import annotations
import argparse,json
import numpy as np
from pathlib import Path

from legion_retrained_match_evaluate_pipeline import (
    ROOT, OUT, STATUS, now, atomic, build_internal_fake_manifest,
    run_pair, summarize_shared, rows, sha,
)


def completed_extra(dataset: str, expected_n: int) -> dict:
    root=OUT/"localization"/"legion_retrained"/dataset
    result_path=root/"results.json";prediction_path=root/"predictions.jsonl"
    if not result_path.exists() or not prediction_path.exists():
        raise RuntimeError(f"missing completed {dataset} artifact")
    result=json.loads(result_path.read_text());prediction_rows=rows(prediction_path)
    if result.get("status")!="COMPLETE" or result.get("metrics",{}).get("n")!=expected_n or len(prediction_rows)!=expected_n:
        raise RuntimeError(f"incomplete completed-artifact claim for {dataset}")
    manifest_rows=rows(Path(result["manifest"]["path"]));expected_ids=[str(x["sample_id"]) for x in manifest_rows]
    if [str(x["sample_id"]) for x in prediction_rows]!=expected_ids:
        raise RuntimeError(f"completed {dataset} sample order drift")
    tp=sum(int(x["tp"]) for x in prediction_rows);fp=sum(int(x["fp"]) for x in prediction_rows);fn=sum(int(x["fn"]) for x in prediction_rows);tn=sum(int(x["tn"]) for x in prediction_rows)
    recomputed={"mean_foreground_iou":float(np.mean([x["foreground_iou"] for x in prediction_rows])),"mean_foreground_f1":float(np.mean([x["foreground_f1"] for x in prediction_rows])),"global_foreground_iou":float(tp/max(1,tp+fp+fn)),"global_foreground_f1":float(2*tp/max(1,2*tp+fp+fn))}
    stored=result["metrics"]
    if any(abs(recomputed[k]-float(stored[k]))>1e-12 for k in recomputed) or stored.get("confusion_pixels")!={"tp":tp,"fp":fp,"fn":fn,"tn":tn}:
        raise RuntimeError(f"completed {dataset} metric drift")
    current_sha=sha(prediction_path)
    if result.get("source_artifact_sha256")!=current_sha:
        result["source_artifact_sha256"]=current_sha;result["completed_artifact_revalidated_at_utc"]=now();atomic(result_path,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--physical-gpu",type=int,default=1);args=parser.parse_args();gpu=args.physical_gpu
    state={"status":"RUNNING","stage":f"resume_after_localization_core_gpu{gpu}_only",
           "rescheduled":True,"remaining_gpu":gpu,"started_at_utc":now()}; atomic(STATUS,state)
    internal=build_internal_fake_manifest()
    official=ROOT/"outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"
    loki=ROOT/"datasets/LOKI/legion_localization/manifest.jsonl"
    # Validate the two already completed core artifacts before continuing.
    core={"official1000":summarize_shared("official1000",official),
          "internal_test":summarize_shared("internal_test",internal)}
    loki_result=summarize_shared("loki229",loki)
    extra=str(ROOT/"scripts/legion_retrained_match_localization_extra.py")
    xaigd_result=completed_extra("xaigd",2419)
    run_pair([("pal4vst",gpu,[str(Path('/home/yz/miniconda3/envs/legion/bin/python')),extra,
                            "--dataset","pal4vst","--device","cuda:0"])],state,f"localization_ood_pal4vst_gpu{gpu}")
    localization={**core,"loki229":loki_result,
        "xaigd":xaigd_result,
        "pal4vst":json.loads((OUT/"localization/legion_retrained/pal4vst/results.json").read_text())}
    classification={name:json.loads((OUT/"classification"/name/"results.json").read_text())
                    for name in ("internal","aigi_holmes","genimage","loki","raise998")}
    atomic(OUT/"results.json",{"status":"COMPLETE","model":"legion_retrained_match",
                               "classification":classification,"localization":localization,
                               "rescheduling":{"completed_gpu0_core":"internal_test",
                                               "all_remaining_physical_gpu":gpu}})
    state.update(status="COMPLETE",stage="STOP",completed_at_utc=now()); atomic(STATUS,state)


if __name__=="__main__": main()
