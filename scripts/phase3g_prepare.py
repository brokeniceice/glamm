#!/usr/bin/env python3
"""Freeze Phase 3G reuse provenance before any preflight or training."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import torch, yaml
ROOT=Path(__file__).resolve().parents[1]; CFG=ROOT/"configs/phase3g_raw4096_aogd.yaml"
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def dump(path,v): path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(v,indent=2)+"\n")
def main():
 c=yaml.safe_load(CFG.read_text()); out=ROOT/c["experiment"]["output_root"]
 source=Path(c["source"]["checkpoint"]); schedule=ROOT/c["phase3f_reuse"]["schedule"]
 rollout=ROOT/c["phase3f_reuse"]["rollout_cache"]; teacher=Path(c["phase3f_reuse"]["teacher_cache"])
 teacher_manifest=ROOT/c["phase3f_reuse"]["teacher_cache_manifest"]
 checks={"P1_hash_exact":sha(source)==c["source"]["checkpoint_sha256"],"rollout_hash_exact":sha(rollout)==c["phase3f_reuse"]["rollout_cache_sha256"],
  "schedule_exists":schedule.is_file(),"teacher_cache_exists":teacher.is_file(),"teacher_manifest_exists":teacher_manifest.is_file(),
  "canonical_prompt_exact":c["prompt"]["user_question"]=="Determine whether this image is authentic and explain the forensic evidence."}
 sch=json.loads(schedule.read_text()); checks.update({"schedule_logical_hash_exact":sch["sha256"]==c["phase3f_reuse"]["schedule_sha256"],
  "schedule_18000":len(sch["sample_ids"])==18000,"schedule_population_17672":len(set(sch["sample_ids"]))==17672})
 manifest=json.loads(teacher_manifest.read_text()); checks["teacher_file_hash_matches_manifest"]=sha(teacher)==manifest["sha256"]
 state=torch.load(teacher,map_location="cpu"); values=state["vectors"]
 checks.update({"teacher_source_P1_exact":state["source_checkpoint_sha256"]==c["source"]["checkpoint_sha256"],"teacher_count_8690":len(values)==8690,
  "all_raw_4096_present":all(tuple(v["raw_4096d"].shape)==(4096,) for v in values.values()),
  "all_projected_256_present":all(tuple(v["projected_256d"].shape)==(256,) for v in values.values())})
 status="PASS" if all(checks.values()) else "FAIL"
 dump(out/"phase3f_reuse_manifest.json",{"status":status,"checks":checks,"phase3f_gate_preserved":"GATE_AUTONOMOUS_ORACLE_DISTILLATION_NOT_LEARNABLE",
  "post_hoc_motivation":True,"only_core_method_change":"256D_post_projection_cosine_to_4096D_raw_hidden_cosine"})
 dump(out/"teacher_4096_cache_manifest.json",{"status":"REUSED_PHASE3F_CACHE" if status=="PASS" else "INVALID","path":str(teacher),"sha256":sha(teacher),
  "count":len(values),"dimension":4096,"dtype":str(next(iter(values.values()))["raw_4096d"].dtype),"teacher_P1_sha256":state["source_checkpoint_sha256"],
  "prompt_protocol":"canonical_authoritative_TF_full",
  "hidden_extraction":"hidden_at_token_immediately_preceding_last_valid_SEG_causal_predictor_before_text_hidden_fcs"})
 dump(out/"rollout_reuse_audit.json",{"status":status,"path":str(rollout),"sha256":sha(rollout),"source":"Phase3B canonical batch1 P1 cache",
  "refresh":"NEVER","same_as_Phase3F":True})
 dump(out/"rollout_integrity_audit.json",{"status":status,"expected_sha256":c["phase3f_reuse"]["rollout_cache_sha256"],"step_0":"PASS","step_500":"PENDING","step_1000":"PENDING","step_4500":"PENDING"})
 (out/"training").mkdir(parents=True,exist_ok=True); shutil.copy2(schedule,out/"training/schedule.json")
 dump(out/"experiment_manifest.json",{"status":"AUTHORIZED_PREPARED","phase":"3G","method":"P3G-RAW4096-AOGD","post_hoc_mechanism_test":True,
  "final_representation_distillation_experiment":True,"internal_test_used":False,"official1000_used":False})
 dump(out/"initial_checkpoint_manifest.json",{"path":str(source),"sha256":sha(source),"optimizer_step":3500,"epoch":7})
 dump(out/"training_config.json",c)
 dump(out/"selector_protocol.json",{"baseline_step":0,"trained_candidate_steps":[500,1000,2000,3000,4000,4500],"primary":c["selector"]["primary"],
  "tie_breaker":c["selector"]["tie_breaker"],"direct_batch_size":1,"canonical_prompt":True,"test_used":False})
 route=ROOT/"docs/paper_convergence_route.md"; marker="## Phase 3G — Raw 4096D AOGD"
 text=route.read_text()
 if marker not in text: route.write_text(text.rstrip()+f"\n\n{marker}\n\n- AUTHORIZED — FINAL POST-HOC REPRESENTATION TARGET TEST.\n- Motivation is post hoc from Phase 3F sample-level correlations; it was not preregistered before Phase 3F.\n- Phase 3F remains `GATE_AUTONOMOUS_ORACLE_DISTILLATION_NOT_LEARNABLE`.\n")
 if status!="PASS": raise RuntimeError("Phase3G reuse audit failed")
 print(json.dumps({"status":status,"teacher_count":len(values),"schedule_exposures":len(sch["sample_ids"]),"checks":checks},indent=2))
if __name__=="__main__": main()
