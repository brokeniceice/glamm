#!/usr/bin/env python3
"""Verify that frozen batch-8 replay tokens equal canonical batch-1 greedy decoding."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from tools.utils import IMAGE_TOKEN_INDEX


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--samples",type=int,default=12)
    cli=p.parse_args(); config=yaml.safe_load((ROOT/"configs/phase3b_b1_generated_replay.yaml").read_text())
    cache=[json.loads(x) for x in (ROOT/"outputs/phase3b_generated_replay/cache/train_fake_p1_g0.jsonl").read_text().splitlines() if x]
    chosen=[cache[round(i*(len(cache)-1)/(cli.samples-1))] for i in range(cli.samples)]
    conversation_lib.default_conversation=conversation_lib.conv_templates["llava_v1"]
    device=torch.device(cli.device); torch.cuda.set_device(device)
    model,tokenizer,_=load_model(config,Path(config["source"]["checkpoint"]),device,
                                 expected_step=config["source"]["optimizer_step"],expected_epoch=config["source"]["epoch"])
    backend=GLaMMForensicsBackend(model,tokenizer,device=device,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=400)
    dataset=UnifiedForensicsDataset(ROOT/config["data"]["manifest_dir"],tokenizer,config["model"]["vision_tower"],
        split="train",datasets_root=config["data"]["datasets_root"],synthscars_root=config["data"]["synthscars_root"],
        image_size=config["model"]["image_size"],target_protocol="phrase_aligned")
    by_id={r["sample_id"]:i for i,r in enumerate(dataset.rows)}; results=[]
    for expected in chosen:
        sample=dataset[by_id[expected["sample_id"]]]
        batch=backend._batch_many([sample],"",question=UNIFIED_FORENSICS_QUESTION)
        with torch.no_grad():
            seq=model.generate(images=batch["global_enc_images"],input_ids=batch["input_ids"],bboxes=batch["bboxes"],
                max_new_tokens=400,num_beams=1,output_hidden_states=False,return_dict_in_generate=True,
                output_scores=False,use_cache=True).sequences[0,batch["input_ids"].shape[1]:]
        eos=seq.eq(tokenizer.eos_token_id).nonzero(as_tuple=False).flatten() if tokenizer.eos_token_id is not None else []
        if len(eos): seq=seq[:int(eos[0])+1]
        actual=[int(x) for x in seq.cpu().tolist()]; exact=actual==expected["generated_token_ids"]
        results.append({"sample_id":sample["sample_id"],"exact":exact,"batch8_length":len(expected["generated_token_ids"]),"batch1_length":len(actual)})
        if not exact: raise RuntimeError(f"batch-size generation divergence: {sample['sample_id']}")
    out={"status":"PASS","policy":"frozen_batch8_equals_canonical_batch1_greedy","num_checked":len(results),"results":results}
    path=ROOT/"outputs/phase3b_generated_replay/audit/cache_batch_identity.json"
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(out,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
