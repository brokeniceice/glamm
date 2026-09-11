#!/usr/bin/env python3
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase4e1 import summarize
AUDIT=ROOT/'outputs/phase6c2_multiseg_training/l2_epoch10_g0';CACHE=Path('/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l2_epoch10_val_g0')
complete=json.loads((AUDIT/'complete.json').read_text());parts=[]
for rank in range(2):parts.append(torch.load(CACHE/f'rank{rank:02d}.pt',map_location='cpu',weights_only=False)['records'])
records=[]
for i in range(max(map(len,parts))):
 for part in parts:
  if i<len(part):records.append(part[i])
if len(records)!=1106 or len({r['sample_id'] for r in records})!=1106:raise RuntimeError('epoch10 G0 identity drift')
result={'schema':'phase6c2_l2_epoch10_g0_v1','status':'COMPLETE','epoch':10,'optimizer_step':5000,'metrics':summarize(records),'records':records,'checkpoint_sha256':complete['L2_sha256'],'prompt':complete['canonical_prompt'],'union':'max generated mask logits; threshold >0; GT union','firewall':complete['firewall']}
(AUDIT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['metrics'],indent=2))
