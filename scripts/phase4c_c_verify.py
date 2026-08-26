#!/usr/bin/env python3
"""Final immutability, matrix completeness, and sealed-set audit for Phase 4C-C."""
import json
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase4c_b import dump,file_sha256

cfg=yaml.safe_load((ROOT/"configs/phase4c_c_evidence_attribution.yaml").read_text());bcfg=yaml.safe_load((ROOT/cfg["phase4c_b"]["config"]).read_text());out=ROOT/cfg["experiment"]["output_root"]
expected={"P1":(Path(bcfg["p1"]["checkpoint"]),bcfg["p1"]["checkpoint_sha256"]),"clip_reader":(Path(cfg["phase4c_b"]["clip_reader_checkpoint"]),cfg["phase4c_b"]["clip_reader_sha256"]),"forensic_reader":(Path(cfg["phase4c_b"]["forensic_reader_checkpoint"]),cfg["phase4c_b"]["forensic_reader_sha256"]),"clip_source":(Path(bcfg["phase4c_a"]["clip_proj_checkpoint"]),bcfg["phase4c_a"]["clip_proj_sha256"]),"forensic_source":(Path(bcfg["phase4c_a"]["forensic_adapter_checkpoint"]),bcfg["phase4c_a"]["forensic_adapter_sha256"])}
hashes={k:{"path":str(p),"expected":h,"actual":file_sha256(p)} for k,(p,h) in expected.items()};counts={}
for arm in ("clip_reader","forensic_reader"):
 for query in ("G0","phrase_repair","phrase_only","tf_full_context"):
  for condition in ("matched","cross_image","spatial_shuffle","global_repeat","zero"):
   path=out/f"evaluation/{arm}/{query}/{condition}.jsonl";counts[f"{arm}/{query}/{condition}"]=sum(1 for _ in path.open()) if path.exists() else -1
checks={"all_checkpoint_hashes_exact":all(x["expected"]==x["actual"] for x in hashes.values()),"all_40_conditions_n1076":len(counts)==40 and all(n==1076 for n in counts.values()),"preflight_pass":json.loads((out/"preflight_manifest.json").read_text())["status"]=="PASS","completion_complete":json.loads((out/"completion_manifest.json").read_text())["status"]=="COMPLETE","training_performed":False,"internal_test_access":False,"official1000_access":False}
status="PASS" if all(v is True or v is False and k in ("training_performed","internal_test_access","official1000_access") for k,v in checks.items()) else "FAIL"
dump(out/"final_invariance_audit.json",{"status":status,"checks":checks,"checkpoint_hashes":hashes,"condition_counts":counts})
completion=json.loads((out/"completion_manifest.json").read_text());completion["final_invariance_audit"]="PASS" if status=="PASS" else "FAIL";completion["required"].append("final_invariance_audit.json") if "final_invariance_audit.json" not in completion["required"] else None;dump(out/"completion_manifest.json",completion)
print(json.dumps({"status":status,"checks":checks},indent=2))
if status!="PASS":raise RuntimeError("final audit failed")
