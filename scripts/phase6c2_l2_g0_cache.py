#!/usr/bin/env python3
"""Cache selected-L2 canonical-G0 generated multiSEG states on internal val."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import torch,torch.distributed as dist,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dataset.forensics.unified import CANONICAL_UNIFIED_QUESTION
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import load_model
from scripts.phase3c2_p3 import dataset_for
from tools.phase4c_b import file_sha256,metric_record
L2=Path('/data/yz/groundingLMM_official/checkpoints/phase6c2_multiseg_training/l2/best/checkpoint/mp_rank_00_model_states.pt'); SHA='dbd7daa8322fe77c8bbfd80223a98ec1e6a4a2c64b4de3b09b09e592ef71d6dd'
OUT=Path('/data/yz/groundingLMM_official/cache/phase6c2_multiseg_training/l2_val_g0'); AUDIT=ROOT/'outputs/phase6c2_multiseg_training/l2_val_g0_cache'; CFG=yaml.safe_load((ROOT/'configs/phase6c2_l2_p1_multi.yaml').read_text())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--local-rank','--local_rank',type=int,default=-1)
 ap.add_argument('--checkpoint',type=Path,default=L2);ap.add_argument('--expected-epoch',type=int,default=3);ap.add_argument('--expected-step',type=int,default=1500)
 ap.add_argument('--expected-sha256',default=SHA);ap.add_argument('--output-root',type=Path,default=OUT);ap.add_argument('--audit-root',type=Path,default=AUDIT);a=ap.parse_args()
 dist.init_process_group('nccl'); rank,world=dist.get_rank(),dist.get_world_size(); local=int(os.environ.get('LOCAL_RANK',a.local_rank)); torch.cuda.set_device(local); dev=torch.device('cuda',local)
 checkpoint=a.checkpoint.resolve(); checkpoint_sha=file_sha256(checkpoint)
 if checkpoint_sha!=a.expected_sha256: raise RuntimeError('L2 drift')
 conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1']; model,tok,meta=load_model(CFG,checkpoint,dev,expected_step=a.expected_step,expected_epoch=a.expected_epoch)
 backend=GLaMMForensicsBackend(model,tok,device=dev,dtype=torch.bfloat16,use_mm_start_end=True,max_new_tokens=int(CFG['evaluation']['max_new_tokens']))
 ds=dataset_for(tok,CFG,'val'); indices=[i for i,r in enumerate(ds.rows) if int(r['class_label'])==1]; local_indices=indices[rank::world]
 records=[]; qs=[]; offsets=[0]; began=time.time()
 for begin in range(0,len(local_indices),a.batch_size):
  samples=[ds[i] for i in local_indices[begin:begin+a.batch_size]]; outputs=backend.generate_localization_batch(samples,provide_gt_fake=False,generation_mode='unified_fake_generate')
  for sample,out in zip(samples,outputs):
   qseg=out.pop('projected_seg_embeddings'); qseg=torch.empty((0,256),dtype=torch.bfloat16) if qseg is None else qseg.detach().cpu().to(torch.bfloat16)
   pred=out.pop('pred_mask'); truth=torch.as_tensor(sample['masks']).bool().any(0); logits=torch.empty((0,*truth.shape)) if pred is None else pred.detach().float().cpu()
   row=metric_record(sample['sample_id'], logits.max(0).values if len(logits) else torch.full(truth.shape,-1.),truth)
   row.update({'generated_seg_count':len(qseg),'gt_K':int(torch.as_tensor(sample['masks']).shape[0]),'generated_token_ids':out['generated_token_ids'],'seg_triggered':bool(out['seg_triggered']),'prompt_sha256':out['prompt_sha256']})
   records.append(row); qs.append(qseg); offsets.append(offsets[-1]+len(qseg))
  if len(records)%100< a.batch_size: print(json.dumps({'rank':rank,'done':len(records),'total':len(local_indices)}),flush=True)
 payload={'schema':'phase6c2_l2_val_g0_cache_v1','rank':rank,'world':world,'records':records,'offsets':torch.tensor(offsets),'q_seg':torch.cat(qs) if offsets[-1] else torch.empty((0,256),dtype=torch.bfloat16),'L2_sha256':checkpoint_sha,'epoch':a.expected_epoch,'optimizer_step':a.expected_step}
 a.output_root.mkdir(parents=True,exist_ok=True); p=a.output_root/f'rank{rank:02d}.pt'; tmp=p.with_suffix('.tmp'); torch.save(payload,tmp);tmp.replace(p)
 a.audit_root.mkdir(parents=True,exist_ok=True); (a.audit_root/f'rank{rank:02d}.json').write_text(json.dumps({'status':'COMPLETE','rank':rank,'images':len(records),'generated_masks':offsets[-1],'seconds':time.time()-began,'sha256':file_sha256(p)},indent=2)+'\n')
 dist.barrier()
 if rank==0:
  parts=[json.loads((a.audit_root/f'rank{i:02d}.json').read_text()) for i in range(world)]; (a.audit_root/'complete.json').write_text(json.dumps({'status':'COMPLETE','images':sum(x['images'] for x in parts),'generated_masks':sum(x['generated_masks'] for x in parts),'L2_sha256':checkpoint_sha,'epoch':a.expected_epoch,'optimizer_step':a.expected_step,'canonical_prompt':CANONICAL_UNIFIED_QUESTION,'firewall':{'internal_test':False,'external':False}},indent=2)+'\n')
 dist.destroy_process_group()
if __name__=='__main__':main()
