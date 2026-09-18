#!/usr/bin/env python3
"""Cache frozen Full-FOV evidence for Phase 6F.4 train/validation only."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from tools.full_fov_forensic import acquire_full_fov
from tools.phase4e1 import tensor_state_sha256
from tools.phase4f import Phase4FStore,load_evidence_source

CACHE=Path('/data/yz/groundingLMM_official/cache/phase6f4_full_fov_i2')
CLIP=ROOT/'checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
SHARD=32

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)

def paths(split):
 result={};base=ROOT/hd.CFG['data']['spatial_cache_root']/'cache/clip'/split
 for p in sorted(base.glob('shard_*.pt')):
  x=torch.load(p,map_location='cpu',weights_only=False)
  for r in x['records']:result[str(r['sample_id'])]=str(Path(r['image_path']).resolve())
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=('train','val'),required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 device=torch.device(a.device);torch.cuda.set_device(device);hc.seed_all();out=CACHE/a.split;out.mkdir(parents=True,exist_ok=True)
 store=Phase4FStore(hd.CFG,a.split);ids=store.sample_ids;image_paths=paths(a.split)
 if set(ids)!=set(image_paths):raise RuntimeError('image path population drift')
 processor=CLIPImageProcessor.from_pretrained(CLIP,local_files_only=True)
 vision=CLIPVisionModel.from_pretrained(CLIP,local_files_only=True,low_cpu_mem_usage=True).to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
 adapter=load_evidence_source(hd.CFG,'forensic_rect',device)
 utility,_=hc.load_utility('a2',device);utility.eval().requires_grad_(False)
 hashes={'vision':tensor_state_sha256(vision.state_dict()),'adapter':tensor_state_sha256(adapter.state_dict()),
         'forensic_head':tensor_state_sha256(utility.forensic_source.state_dict())}
 existing=[]
 for p in sorted(out.glob('shard_*.pt')):
  x=torch.load(p,map_location='cpu',weights_only=False)
  if x['source_hashes']!=hashes:raise RuntimeError('cache source hash drift')
  existing += x['sample_ids']
 if existing!=ids[:len(existing)]:raise RuntimeError('cache resume prefix drift')
 start=len(existing)
 for begin in range(start,len(ids),SHARD):
  end=min(begin+SHARD,len(ids));records=[]
  for sid in ids[begin:end]:
   with Image.open(image_paths[sid]) as image:
    v=acquire_full_fov(image,processor,vision,adapter,utility.forensic_source,utility.temperature_f,device)
   records.append({'sample_id':sid,'resized_hw':v['geometry'].resized_hw,'tile_boxes_yxyx':v['geometry'].tile_boxes_yxyx,
     'full_grid_hw':v['geometry'].full_grid_hw,'F_full':v['F_full'][0].to(torch.bfloat16).cpu(),
     'z_full':v['z_full'][0].to(torch.bfloat16).cpu(),'mass_full':v['mass_full'][0].float().cpu(),
     'support_full':v['support_full'][0].cpu(),'coordinates':v['coordinates'].cpu()})
  payload={'schema':'phase6f4_full_fov_cache_v1','split':a.split,'start':begin,'end':end,
           'sample_ids':ids[begin:end],'source_hashes':hashes,'records':records}
  p=out/f'shard_{begin:06d}_{end:06d}.pt';t=p.with_suffix('.pt.tmp');torch.save(payload,t);os.replace(t,p)
  print(json.dumps({'stage':'FULL_FOV_CACHE','split':a.split,'done':end,'total':len(ids)}),flush=True)
 status={'schema':'phase6f4_full_fov_cache_status_v1','status':'COMPLETE','split':a.split,'count':len(ids),
         'sample_ids_sha256':__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(),'source_hashes':hashes,
         'tile_protocol':{'tile':336,'stride':280,'overlap':56,'end_anchor':True},'optimizer_count':0}
 dump(out/'status.json',status)

if __name__=='__main__':main()
