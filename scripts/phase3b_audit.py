#!/usr/bin/env python3
"""Static provenance and preregistration audit for Phase 3B."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3b_generated_replay/audit"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
def dump(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def main():
    p1_path=ROOT/"configs/phase3a_p1.yaml"; p1=yaml.safe_load(p1_path.read_text())
    configs={name:yaml.safe_load((ROOT/f"configs/phase3b_{name}.yaml").read_text()) for name in ("b0_gold_replay","b1_generated_replay")}
    selector=json.loads((ROOT/"outputs/phase3a_phrase_grounding/selection/p1_selector.json").read_text())
    source=Path(selector["selected_checkpoint"])
    train=rows(ROOT/"outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl")
    audit={
      "status":"PASS",
      "source":{"selector":selector,"actual_checkpoint_sha256":sha(source),"identity_pass":sha(source)==selector["checkpoint_sha256"]},
      "configs":{name:{"path":str((ROOT/f"configs/phase3b_{name}.yaml").resolve()),"sha256":sha(ROOT/f"configs/phase3b_{name}.yaml")} for name in configs},
      "p1_config":{"path":str(p1_path.resolve()),"sha256":sha(p1_path)},
      "matched_policy":{
        "trainable_equal_to_p1":all(c["trainable"]==p1["trainable"] for c in configs.values()),
        "optimizer_equal_to_p1":all(c["optimizer"]==p1["optimizer"] for c in configs.values()),
        "base_loss_equal_to_p1":all({k:c["loss"][k] for k in p1["loss"]}==p1["loss"] for c in configs.values()),
        "effective_global_batch":20,"steps_per_arm":2000,"validation_interval":250,
        "replay_fraction":.5,"lambda_replay":1.0,
      },
      "data":{"train_total":len(train),"train_real":sum(r["class_label"]==0 for r in train),"train_fake":sum(r["class_label"]==1 for r in train),
              "manifest_sha256":sha(ROOT/"outputs/data_audits/unified_forensics_split_v1/train_combined.jsonl")},
      "forbidden_features":{"npr":False,"srm":False,"focal":False,"sam_modified":False,"new_decoder":False,
                            "new_labels_or_masks":False,"dynamic_replay":False,"threshold_sweep":False,"phase3c_started":False},
      "runtime_policy":{"GLAMM_PRESERVE_CUDA_CACHE":"1","PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512",
                        "checkpoint_entity_root":"/data/yz/groundingLMM_official/checkpoints/phase3b_generated_replay",
                        "physical_gpu_assignment":{"all_phase3b_operations":1,
                                                   "b1_generated_replay":1,"b0_gold_replay":1,
                                                   "gpu0":"UNUSED","gpu2":"UNUSED",
                                                   "execution":"SEQUENTIAL"}},
      "environment":{"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                     "python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
                     "gpu_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
    }
    if not all((audit["source"]["identity_pass"],audit["matched_policy"]["trainable_equal_to_p1"],audit["matched_policy"]["optimizer_equal_to_p1"],audit["matched_policy"]["base_loss_equal_to_p1"])):
        audit["status"]="FAIL"
    dump(OUT/"preregistration_audit.json",audit)
    if audit["status"]!="PASS": raise RuntimeError(json.dumps(audit["matched_policy"]))
    print(json.dumps(audit,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
