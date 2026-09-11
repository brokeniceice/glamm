#!/usr/bin/env python3
"""Freeze L3 selector from the ten internal-validation canonical-G0 rows."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.phase4c_b import file_sha256
OUT=ROOT/'outputs/phase6c2_multiseg_training'; VAL=OUT/'l3_validation/g0'; CK=Path('/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l3')
rows=[]
for e in range(1,11):
 p=VAL/f'epoch_{e:02d}.json';x=json.loads(p.read_text());
 if x.get('status')!='COMPLETE' or x.get('mode')!='g0' or x['metrics']['n']!=1106:raise RuntimeError(f'epoch {e} validation incomplete')
 rows.append({'epoch':e,**x['metrics']})
best=max(rows,key=lambda x:(x['mean_foreground_iou'],-x['epoch']));src=CK/f"epoch_{best['epoch']}.pt";dst=OUT/'l3_selected_checkpoint.pt';shutil.copy2(src,dst)
result={'schema':'phase6c2_l3_selector_v1','status':'COMPLETE','primary':'internal validation Fake canonical G0 mean foreground IoU','tie_break':'earlier epoch','selected_epoch':best['epoch'],'selected_metrics':best,'selected_checkpoint':str(dst),'selected_checkpoint_sha256':file_sha256(dst),'candidates':rows,'Phrase_used':False,'TF_used':False,'internal_test_used':False,'external_used':False}
(OUT/'l3_selector.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'selected_epoch':best['epoch'],'mean_iou':best['mean_foreground_iou']}))
