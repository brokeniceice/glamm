#!/usr/bin/env python3
"""Frozen selected-C1 internal and DEV-OOD evaluation plus final report."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import cv2, torch, yaml
from transformers import CLIPImageProcessor

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import evaluate_detection, summarize_detection_records
from eval.forensics_eval import GLaMMForensicsBackend
from model.SAM.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model

OUT=ROOT/'outputs/phase6d3_c1'; CFG=ROOT/'configs/phase6d3_c1_rine_conditioned_p1.yaml'
CKPT=ROOT/'checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt'
DEV=ROOT/'outputs/phase6d2_cls_head_capacity/dev_ood_genimage_2560.jsonl'
PY='/home/yz/miniconda3/envs/glamm_official/bin/python'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def complete_metric(m):
 m=dict(m);m['tnr']=m['tn']/max(1,m['tn']+m['fp']);m['fpr']=m['fp']/max(1,m['tn']+m['fp']);return m

class DevDataset:
 def __init__(self, manifest, processor):
  self.rows=rows(manifest);self.processor=processor;self.transform=ResizeLongestSide(1024)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];im=cv2.cvtColor(cv2.imread(r['image_path']),cv2.COLOR_BGR2RGB);h,w=im.shape[:2]
  resized=self.transform.apply_image(im);ground=UnifiedForensicsDataset.grounding_enc_processor(torch.from_numpy(resized).permute(2,0,1).contiguous())
  label=1 if r['label']=='Fake' else 0
  return {'image_path':r['image_path'],'global_enc_image':self.processor(images=im,return_tensors='pt')['pixel_values'][0],
   'grounding_enc_image':ground,'bboxes':None,'conversations':[''],'masks':None,'label':torch.full((h,w),255,dtype=torch.long),
   'resize':resized.shape[:2],'questions':[''],'sampled_classes':[],'cls_label':label,'seg_valid':False,
   'sample_id':r['sample_id'],'source':r['generator/source'],'content_category':r['generator/source'],'manifest_row':r}

def run_internal(split, epoch, step, device):
 cmd=[PY,str(ROOT/'scripts/phase3a_evaluate.py'),'--config',str(CFG),'--checkpoint',str(CKPT),'--output-dir',str(OUT/'evaluation'/split),'--split',split,'--device',device,'--modes','detection','G0','tf_full_context','--expected-step',str(step),'--expected-epoch',str(epoch),'--generation-batch-size','1','--reset']
 subprocess.run(cmd,cwd=ROOT,check=True)

def eval_dev(config, epoch, step, device, *, output_root=OUT, checkpoint=CKPT):
 conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1'];dev=torch.device(device);torch.cuda.set_device(dev)
 model,tok,meta=load_model(config,checkpoint,dev,expected_step=step,expected_epoch=epoch)
 backend=GLaMMForensicsBackend(model,tok,device=dev,dtype=torch.bfloat16,max_new_tokens=400)
 ds=DevDataset(DEV,CLIPImageProcessor.from_pretrained(config['model']['vision_tower'],local_files_only=True))
 out=Path(output_root)/'dev_ood_2560/predictions.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:
  for i in range(len(ds)):
   rec,_=evaluate_detection([ds[i]],backend);f.write(json.dumps(rec[0],ensure_ascii=False)+'\n');f.flush()
   if (i+1)%100==0:print(f'dev-ood {i+1}/{len(ds)}',flush=True)
 metrics=summarize_detection_records(rows(out));metrics['classification_head']=complete_metric(metrics['classification_head']);dump(out.parent/'metrics.json',metrics);return metrics,meta

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:1');a=ap.parse_args()
 config=yaml.safe_load(CFG.read_text());state=torch.load(CKPT,map_location='cpu',weights_only=False);epoch=int(state['epoch']);step=int(state['optimizer_step'])
 run_internal('val',epoch,step,a.device);run_internal('test',epoch,step,a.device)
 dev,meta=eval_dev(config,epoch,step,a.device)
 val=json.load(open(OUT/'evaluation/val/summary.json'));test=json.load(open(OUT/'evaluation/test/summary.json'))
 result={'schema':'phase6d3_c1_results_v1','status':'COMPLETE','initialization_provenance':'same checkpoints/GLaMM-FullScope and seed 3407 as original P1; no trained-P1 checkpoint loaded',
  'rine_checkpoint':str((ROOT/config['forensics']['rine_checkpoint']).resolve()),'rine_checkpoint_sha256':sha(ROOT/config['forensics']['rine_checkpoint']),
  'selected_checkpoint':meta,'selector':'minimum internal validation total loss, fixed original P1 rule','internal_validation':val,'internal_test':test,
  'dev_ood_2560':{'manifest':str(DEV.resolve()),'manifest_sha256':sha(DEV),'designation':'DEV-OOD DIAGNOSTIC','metrics':dev},
  'c0_contract_frozen_before_results':config['next_c0_frozen_contract'],'external_benchmarks_run':False}
 dump(OUT/'results.json',result)
 cls=lambda s:complete_metric(s['modes']['detection']['classification_head']); loc=lambda s,m:s['modes'][m]
 md=f'''# Phase 6D.3 — C1 RINE-Conditioned P1\n\nStatus: **COMPLETE**. C1 was retrained from the same pre-P1 base initialization; no trained P1 checkpoint was used. RINE Q2 and its CLIP tower stayed frozen/eval. H2 and the prescribed two-layer GELU projector were trained jointly.\n\n- Selected checkpoint: `{meta['checkpoint']}` (epoch {epoch}, step {step}, SHA256 `{meta['checkpoint_sha256']}`)\n- RINE checkpoint SHA256: `{result['rine_checkpoint_sha256']}`\n- Selector: original P1 minimum internal-validation total loss\n- DEV-OOD manifest SHA256: `{result['dev_ood_2560']['manifest_sha256']}`\n\n## Classification\n\n| Split | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |\n|---|---:|---:|---:|---:|---:|---:|\n| internal validation | {cls(val)['accuracy']:.6f} | {cls(val)['roc_auc']:.6f} | {cls(val)['fake_recall']:.6f} | {cls(val)['tnr']:.6f} | {cls(val)['fpr']:.6f} | {cls(val)['f1']:.6f} |\n| internal test | {cls(test)['accuracy']:.6f} | {cls(test)['roc_auc']:.6f} | {cls(test)['fake_recall']:.6f} | {cls(test)['tnr']:.6f} | {cls(test)['fpr']:.6f} | {cls(test)['f1']:.6f} |\n| DEV-OOD-2560 | {dev['classification_head']['accuracy']:.6f} | {dev['classification_head']['roc_auc']:.6f} | {dev['classification_head']['fake_recall']:.6f} | {dev['classification_head']['tnr']:.6f} | {dev['classification_head']['fpr']:.6f} | {dev['classification_head']['f1']:.6f} |\n\n## Localization and generation diagnostics\n\nFull G0 generations, phrase parsing, TF predictions, spatial metrics, and failure records are under `outputs/phase6d3_c1/evaluation/val` and `test`. No external localization benchmark was run.\n\nC0 remained frozen to the preregistered matched definition in the config. Causal attribution of RINE injection is reserved for future C1-vs-C0 comparison.\n'''
 (ROOT/'docs/phase6d3_c1_rine_conditioned_p1.md').write_text(md)
if __name__=='__main__':main()
