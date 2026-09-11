#!/usr/bin/env python3
"""Compute strict online-BF16 C1-Exact logits for Phase 6B.2 internal splits."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2,torch
from torch.utils.data import DataLoader,Dataset
from transformers import CLIPImageProcessor,CLIPVisionModel
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.phase6b_cache_features import resolve_path
EXACT=ROOT/'outputs/phase6b1a_exact_stage2_control/checkpoints/checkpoint-554';CLIP=ROOT/'checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1';MAN=ROOT/'outputs/data_audits/unified_forensics_split_v1';OUT=ROOT/'outputs/phase6b2_fusion/features'
class D(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  p=resolve_path(self.rows[i]);im=cv2.imread(str(p));
  if im is None:raise OSError(p)
  return i,cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
def head():
 idx=json.loads((EXACT/'pytorch_model.bin.index.json').read_text())['weight_map'];full=torch.load(EXACT/idx['prediction_head.0.weight'],map_location='cpu');state={k.removeprefix('prediction_head.'):v for k,v in full.items() if k.startswith('prediction_head.')};del full
 m=torch.nn.Sequential(torch.nn.Linear(1024,2048),torch.nn.ReLU(),torch.nn.Linear(2048,2));m.load_state_dict(state);return m
def main():
 a=argparse.ArgumentParser();a.add_argument('--split',choices=('train','val'),required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--batch-size',type=int,default=64);x=a.parse_args();rows=[json.loads(z) for z in (MAN/f'{x.split}_combined.jsonl').read_text().splitlines() if z];proc=CLIPImageProcessor.from_pretrained(CLIP,local_files_only=True);dev=torch.device(x.device)
 clip=CLIPVisionModel.from_pretrained(CLIP,local_files_only=True).to(dev,dtype=torch.bfloat16).eval();h=head().to(dev,dtype=torch.bfloat16).eval()
 for p in list(clip.parameters())+list(h.parameters()):p.requires_grad_(False)
 def collate(batch):
  idx,images=zip(*batch);return torch.tensor(idx),proc(images=list(images),return_tensors='pt')['pixel_values']
 loader=DataLoader(D(rows),batch_size=x.batch_size,shuffle=False,num_workers=8,pin_memory=True,collate_fn=collate,persistent_workers=True);out=[];cursor=0
 with torch.inference_mode():
  for idx,pix in loader:
   if not torch.equal(idx,torch.arange(cursor,cursor+len(idx))):raise RuntimeError('order drift')
   hidden=clip(pix.to(dev,non_blocking=True,dtype=torch.bfloat16),output_hidden_states=True).hidden_states[-2][:,0];out.append(h(hidden).float().cpu()[:,[1,0]]);cursor+=len(idx)
   if cursor%1024<len(idx):print(json.dumps({'split':x.split,'done':cursor,'total':len(rows)}),flush=True)
 payload={'schema':'phase6b2_exact_c1_logits_v1','split':x.split,'sample_ids':[r['sample_id'] for r in rows],'labels':torch.tensor([int(r['class_label']) for r in rows]),'logits_real_fake':torch.cat(out),'semantics':'online BF16 CLIP and BF16 C1-Exact head; reordered Real=0 Fake=1'}
 torch.save(payload,OUT/x.split/'c1_exact_logits.pt')
if __name__=='__main__':main()
