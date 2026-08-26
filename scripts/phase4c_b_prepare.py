#!/usr/bin/env python3
"""Freeze Phase 4C-B protocol and audit every reused artifact before execution."""
from __future__ import annotations
import json,os,sys
from collections import Counter
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from model.evidence_reader import EvidenceReader
from tools.phase4c_b import canonical_hash,dump,file_sha256,rows,tensor_hash


def main():
 cfg=yaml.safe_load((ROOT/"configs/phase4c_b_evidence_reader.yaml").read_text()); out=ROOT/cfg["experiment"]["output_root"]; cache=Path(cfg["experiment"]["cache_root"]); ckroot=Path(cfg["experiment"]["checkpoint_root"])
 out.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True); ckroot.mkdir(parents=True,exist_ok=True)
 link=out/"cache"
 if not link.exists(): os.symlink(cache,link,target_is_directory=True)
 elif link.resolve()!=cache.resolve(): raise RuntimeError("unexpected Phase 4C-B cache target")
 p1=Path(cfg["p1"]["checkpoint"]); cp=Path(cfg["phase4c_a"]["clip_proj_checkpoint"]); fa=Path(cfg["phase4c_a"]["forensic_adapter_checkpoint"])
 p4ca=ROOT/cfg["phase4c_a"]["output_root"]; gate=json.loads((p4ca/"route_gate.json").read_text()); comp=json.loads((p4ca/"completion_manifest.json").read_text())
 phase3c1=ROOT/cfg["frozen_spatial_cache"]["phase3c1_root"]
 cache_complete={f"{s}_{x}":json.loads((phase3c1/f"cache/{s}/{x}/complete.json").read_text()) for s in ("clip","sam") for x in ("train","val")}
 rollout_path=ROOT/cfg["rollout"]["cache"]; rollout_manifest=json.loads((ROOT/cfg["rollout"]["manifest"]).read_text()); replay=rows(rollout_path)
 eligible=[r for r in replay if r["replay_eligible"]]; excluded=[r for r in replay if not r["replay_eligible"]]
 checks={"phase4c_a_complete":comp["status"]=="COMPLETE","phase4c_a_gate":gate["gate"]=="GATE_CLIP_ANCHORED_FORENSIC_ADAPTER_LEARNABLE","p1_hash":file_sha256(p1)==cfg["p1"]["checkpoint_sha256"],"clip_proj_hash":file_sha256(cp)==cfg["phase4c_a"]["clip_proj_sha256"],"forensic_adapter_hash":file_sha256(fa)==cfg["phase4c_a"]["forensic_adapter_sha256"],"rollout_hash":file_sha256(rollout_path)==cfg["rollout"]["cache_sha256"],"rollout_batch1":rollout_manifest["generation_batch_size"]==1,"rollout_canonical_identity":rollout_manifest["canonical_historical_batch_size_identity"] is True,"raw_fake_count":len(replay)==8836,"eligible_count":len(eligible)==8690,"excluded_count":len(excluded)==146,"excluded_only_missing_seg":Counter(r["seg_count"] for r in excluded)==Counter({0:146}),"all_spatial_caches_complete":all(v["status"]=="COMPLETE" and v["source_parameter_hash_exact"] for v in cache_complete.values()),"clip_sam_train_ids_match":cache_complete["clip_train"]["sample_ids_sha256"]==cache_complete["sam_train"]["sample_ids_sha256"]==cfg["frozen_spatial_cache"]["clip_train_ids_sha256"],"clip_sam_val_ids_match":cache_complete["clip_val"]["sample_ids_sha256"]==cache_complete["sam_val"]["sample_ids_sha256"]==cfg["frozen_spatial_cache"]["clip_val_ids_sha256"]}
 for arm,path in (("clip_reader",cp),("forensic_reader",fa)):
  state=torch.load(path,map_location="cpu"); checks[f"{arm}_epoch4"]=int(state["epoch"])==4
 status="PASS" if all(checks.values()) else "FAIL"
 if status!="PASS": raise RuntimeError(f"Phase4C-B prepare failed: {[k for k,v in checks.items() if not v]}")
 reader=EvidenceReader(); reader_hash=tensor_hash(reader.state_dict().items()); count=sum(p.numel() for p in reader.parameters())
 dump(out/"experiment_manifest.json",{"status":"PREPARED","phase":"Phase 4C-B","name":cfg["experiment"]["name"],"trigger":"Phase 4C-A positive result","checks":checks,"hard_stop_after_completion":True,"internal_test_access":False,"official1000_access":False})
 dump(out/"initial_checkpoint_manifest.json",{"P1":{"path":str(p1),"sha256":cfg["p1"]["checkpoint_sha256"],"step":3500,"epoch":7},"clip_proj":{"path":str(cp),"sha256":cfg["phase4c_a"]["clip_proj_sha256"],"selected_epoch":4},"forensic_adapter":{"path":str(fa),"sha256":cfg["phase4c_a"]["forensic_adapter_sha256"],"selected_epoch":4}})
 dump(out/"p1_manifest.json",{"status":"FROZEN","checkpoint":str(p1),"sha256":cfg["p1"]["checkpoint_sha256"],"all_parameters_trainable":False,"role":"original best-supported P1"})
 dump(out/"clip_proj_manifest.json",{"status":"FROZEN","checkpoint":str(cp),"sha256":cfg["phase4c_a"]["clip_proj_sha256"],"representation":"selected F0 after 1024-to-256 base projection","shape":[256,24,24]})
 dump(out/"forensic_adapter_manifest.json",{"status":"FROZEN","checkpoint":str(fa),"sha256":cfg["phase4c_a"]["forensic_adapter_sha256"],"representation":"selected F_forensic after three local residual blocks","shape":[256,24,24]})
 dump(out/"reader_architecture.json",{"architecture":"single-layer 4-head cross-attention","embed_dim":256,"num_heads":4,"head_dim":64,"dropout":0.0,"query_norm":"LayerNorm","source_norm":"LayerNorm","additional_positional_embedding":False,"fusion":"q_final=q_seg+beta*q_evidence","beta_init":0.0,"trainable_parameter_count":count,"forbidden_components_present":False})
 dump(out/"training_config.json",{"frozen_before_results":True,**cfg["training"],"optimizer":cfg["optimizer"],"loss":cfg["loss"],"selector":cfg["selector"]})
 dump(out/"fairness_manifest.json",{"status":"FROZEN_PENDING_RUNTIME_CONFIRMATION","reader_initial_state_sha256":reader_hash,"same_architecture":True,"same_parameter_count":True,"same_initialization_seed":3407,"same_beta_init":True,"same_optimizer_scheduler_order_loss_budget":True,"only_core_variable":"CLIP-PROJ F0 versus Forensic Adapter F_forensic"})
 dump(out/"rollout_reuse_audit.json",{"status":"PASS","source":str(rollout_path.resolve()),"sha256":cfg["rollout"]["cache_sha256"],"protocol":cfg["rollout"]["protocol"],"canonical_prompt":cfg["prompt"],"generation_batch_size":1,"raw_fake_count":len(replay),"eligible_count":len(eligible),"eligible_rate":len(eligible)/len(replay),"excluded_count":len(excluded),"exclusion_reasons":{"missing_SEG_predictor":146},"no_quality_filtering_beyond_required_valid_SEG":True})
 dump(out/"feature_cache_manifest.json",{"status":"SOURCE_CACHES_AUDITED_QSEG_PENDING","storage_root":str(cache),"reused_phase3c1":cache_complete,"train_q_seg":"PENDING","validation_fresh_contexts":{"G0":"PENDING","phrase_only":"PENDING","tf_full_context":"PENDING"},"cache_hash_invariance_required":True})
 route=ROOT/"docs/fepn_design_route.md"; marker="## Version 0.6 — Phase 4C-B authorization"
 text=route.read_text()
 if marker not in text:
  route.write_text(text.rstrip()+f"\n\n{marker}\n\n- Phase 4C-A outcome: `GATE_CLIP_ANCHORED_FORENSIC_ADAPTER_LEARNABLE`. Adapter exceeded CLIP-PROJ and RAW CLIP with paired-bootstrap lower bounds above zero; residual-block attribution was supported.\n- Phase 4C-B is AUTHORIZED directly by the Phase 4C-A positive result.\n- Current working hypothesis: a language-derived grounding query may improve autonomous localization by actively retrieving task-aligned spatial forensic evidence.\n- Only the matched single-layer Evidence Reader and zero-initialized residual beta are trainable; P1, CLIP, Adapter and SAM remain frozen.\n",encoding="utf-8")
 print(json.dumps({"status":status,"eligible":len(eligible),"excluded":len(excluded),"reader_parameters":count,"reader_init_hash":reader_hash},indent=2))
if __name__=="__main__": main()

