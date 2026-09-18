#!/usr/bin/env python3
"""Run the proven staged-R1 implementation with the frozen 6G.2 fusion source."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch,torch.nn as nn
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from model.clip_forensic_adapter import CLIPSpatialArm
from scripts import phase6g1_block17_staged_r1 as base
from scripts.phase6g0_multilevel_dense_clip import cache_paths,load_shard

OUT=ROOT/'outputs/phase6g2_multilevel_attention/phase6g2b';ADAPTER=OUT/'adapter/selected.pt';CKPT=Path('/data/yz/groundingLMM_official/checkpoints/phase6g2_multilevel_attention/staged')
class PairStore:
 def __init__(self,split):
  self.values=[];self.loc={};self.ids=[];a,b=cache_paths('middle',split),cache_paths('late',split)
  if len(a)!=len(b):raise RuntimeError('block11/17 shard drift')
  for pa,pb in zip(a,b):
   x,y=load_shard(pa),load_shard(pb);ia=[r['sample_id'] for r in x['records']];ib=[r['sample_id'] for r in y['records']]
   if ia!=ib:raise RuntimeError('block11/17 ID drift')
   j=len(self.values);self.values.append((x['features'],y['features']))
   for i,s in enumerate(ia):self.ids.append(s);self.loc[s]=(j,i)
 def batch(self,ids,device):
  a=torch.stack([self.values[self.loc[s][0]][0][self.loc[s][1]] for s in ids]).to(device=device,dtype=torch.float32);b=torch.stack([self.values[self.loc[s][0]][1][self.loc[s][1]] for s in ids]).to(device=device,dtype=torch.float32);return a,b
class Pipeline(nn.Module):
 def __init__(self):super().__init__();self.fusion=CrossLayerPatchAttention(1024,8,.01);self.adapter=CLIPSpatialArm(blocks=3)
def pipeline(device):
 x=torch.load(ADAPTER,map_location='cpu',weights_only=False);m=Pipeline();m.fusion.load_state_dict(x['fusion'],strict=True);m.adapter.load_state_dict(x['adapter'],strict=True);return m.to(device).eval().requires_grad_(False)
def forensic(model,store,ids,device):
 a,b=store.batch(ids,device)
 with torch.no_grad():
  fused=model.fusion(a,b)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16):o=model.adapter(fused,return_features=True)
 return o['F_forensic'].detach(),o['logits'].detach()
def render(summary):
 c=summary['comparison'];p=c['paired'];text=f"""# Phase 6G.2 — Block11/17 Cross-Layer Attention Fusion\n\nStatus: **COMPLETE STOP**. Phase6G.2A passed before this staged R1 run. CLIP/C1/SAM remained frozen. Fusion+adapter were freshly trained together; Rectifier, Utility, and joint stages each used the matched current-new-R1 recipe and their own internal-validation selector.\n\n| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |\n|---|---:|---:|---:|---:|---:|\n| A0 block22 current new R1 | {c['A0']['mean_fg_iou']:.6f} | {c['A0']['median_fg_iou']:.6f} | {c['A0']['mean_fg_f1']:.6f} | {c['A0']['global_fg_iou']:.6f} | {c['A0']['global_fg_f1']:.6f} |\n| A1 block11+17 attention R1 | {c['A1']['mean_fg_iou']:.6f} | {c['A1']['median_fg_iou']:.6f} | {c['A1']['mean_fg_f1']:.6f} | {c['A1']['global_fg_iou']:.6f} | {c['A1']['global_fg_f1']:.6f} |\n\n- paired IoU: `{p['iou']}`\n- paired F1: `{p['f1']}`\n\n```text\n{summary['decision']}\n```\n\nNo internal test, Official1000, OOD, block22 fusion, or further architecture search was run.\n""";(ROOT/'docs/phase6g2_multilevel_attention.md').write_text(text)
def main():
 base.OUT=OUT;base.CKPT=CKPT;base.ADAPTER=ADAPTER;base.LayerStore=PairStore;base.adapter=pipeline;base.forensic=forensic;base.render=render;base.SOURCE_DESCRIPTION='legacy center-crop block11/17 per-patch cross-layer attention plus jointly trained Phase4C-A adapter';base.SUPPORTED_DECISION='MULTILEVEL_ATTENTION_R1_SUPPORTED';base.UNSTABLE_DECISION='MULTILEVEL_ATTENTION_R1_NOT_STABLY_BETTER';base.UNSUPPORTED_DECISION='MULTILEVEL_ATTENTION_R1_NOT_STABLY_BETTER';base.main()
if __name__=='__main__':main()
