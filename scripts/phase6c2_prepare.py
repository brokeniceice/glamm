#!/usr/bin/env python3
"""Freeze Phase 6C.2 L2/L3 contracts before the first optimizer step."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase6c2_multiseg_training'
P1=Path('/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt')
R1=ROOT/'outputs/phase4hd/r1/selected_checkpoint.pt'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def flatten(x,p=''):
 out={}
 if isinstance(x,dict):
  for k,v in x.items():out.update(flatten(v,f'{p}.{k}' if p else k))
 else:out[p]=x
 return out
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
 if OUT.exists() and any(OUT.iterdir()): raise RuntimeError('Phase6C2 output already exists; refuse overwrite')
 old=yaml.safe_load((ROOT/'configs/phase3a_p1.yaml').read_text());new=yaml.safe_load((ROOT/'configs/phase6c2_l2_p1_multi.yaml').read_text())
 a,b=flatten(old),flatten(new);diff={k:{'P1':a.get(k),'L2':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
 allowed={'experiment.name','experiment.runtime_output_dir','experiment.experiment_type','checkpoint.output_root',
  'forensics.target_protocol','forensics.target_template','forensics.multiseg_style','forensics.mask_target',
  'mask_loss.reduction','validation.record_native_pair_diagnostics'}
 unexpected=sorted(set(diff)-allowed)
 if unexpected:raise RuntimeError(f'L2 config changes outside target/admin scope: {unexpected}')
 if sha(P1)!='fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326':raise RuntimeError('P1 hash drift')
 if sha(R1)!='9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5':raise RuntimeError('R1 hash drift')
 c1=json.load(open(ROOT/'outputs/phase6c1_multiseg_preflight/results.json'))
 if c1['P1_MULTI_READY']!='YES' or c1['R1_MULTI_READY']!='YES':raise RuntimeError('Phase6C1 gate failed')
 contract={'schema':'phase6c2_frozen_contract_v1','status':'FROZEN_BEFORE_FIRST_OPTIMIZER_STEP',
  'arms':{'L0':{'kind':'exact_reuse','checkpoint':str(P1),'sha256':sha(P1)},
          'L1':{'kind':'exact_reuse','checkpoint':str(R1),'sha256':sha(R1)},
          'L2':{'kind':'new_training','initialization':str(P1),'initialization_sha256':sha(P1),'config':str(ROOT/'configs/phase6c2_l2_p1_multi.yaml')},
          'L3':{'kind':'new_training','base':'selected L2','import_only':'L1 utility_state + rectifier_state','forbidden_overwrite':['LLM','LoRA','text_hidden_fcs/SEG projection','classification_head','vision/backbone','L2 mask_decoder']}},
  'l2_config_diff':diff,'unexpected_config_diff':unexpected,
  'target':{'style':'append','fake':'original explanation + ordered <p>phrase_i</p>[SEG]','real':'unchanged','training_union':False,'empty_pair':'drop pair only with ref_index+reason','K0':'no localization supervision','overflow':'atomic suffix pairs only'},
  'loss':{'reduction':'global_per_mask_mean','per_image_balanced':False,'weights':{'text':1.0,'classification':1.0,'bce':2.0,'dice':.5}},
  'L2_contract':{'trainable':new['trainable'],'optimizer':new['optimizer'],'training':new['training'],'selector':new['checkpoint']},
  'L3_contract_source':{'script':'scripts/phase4hd_rectifier_unfreeze_control.py','epochs':10,'batch':8,'lr':1e-4,'weight_decay':1e-4,'scheduler':'none','selector':'validation Fake canonical G0 mean IoU; tie earlier'},
  'firewall':{'internal_train':True,'internal_validation':True,'internal_test':False,'OOD':False,'external':False}}
 dump(OUT/'protocol.json',contract);dump(OUT/'status.json',{'schema':'phase6c2_status_v1','status':'PREFLIGHT','stage':'l2_init_audit'})
if __name__=='__main__':main()
