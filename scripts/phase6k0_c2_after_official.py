#!/usr/bin/env python3
"""Frozen selected Phase6K B0/A/E Official1000 matched replay (no selection)."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from model.c2_spatial_transfer import AfterProjection, utility_forward_full_input_gradient
from model.pcerf import sam_lowres_to_original_normalized
from scripts.phase3c2_p3 import dataset_for
from scripts.phase3c1_cache import clip_grid
from scripts.phase6j0_c2_evaluate import load_c2_model
from scripts.phase6k0_c2_after_train import AUDIT, C2_CKPT, provenance, runtime, require, dump, CFG
from tools.phase3c1 import geometry_for
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import clip_coordinates, sam_coordinates, summarize, compare

CKPT_CFG=ROOT/'configs/phase6j0_c2_preln_cross_attention.yaml'
MANIFEST=ROOT/'outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl'
OUT=AUDIT/'official1000'; RECORDS=OUT/'packed_records.jsonl'


def jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()] if Path(path).exists() else []


def component_logits(q, base, s64, source, sam, rect, utility, sg, cg, sc, cc, spatial, arm):
    f24=base if arm is None else arm(base,spatial)[0]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        zf=source.dense_head(f24)
        valid=torch.ones(1,576,dtype=torch.bool,device=f24.device)
        r=rect(s64,f24,sc,cc,valid)
    with torch.autocast('cuda',enabled=False):
        low0=sam(q,s64.to(torch.bfloat16))
    zl=sam_lowres_to_original_normalized(low0,sg,output_hw=(256,256)).to(torch.bfloat16)
    batch={'S64':s64,'q_seg':q,'z_L':zl,'F24':f24,'z_F24':zf,'clip_geometries':[cg],
           'valid_g0':torch.ones(1,dtype=torch.bool,device=f24.device),
           'forensic_present':torch.ones(1,dtype=torch.bool,device=f24.device),
           'forensic_vacuous':torch.zeros(1,dtype=torch.bool,device=f24.device),
           'forensic_off':torch.zeros(1,dtype=torch.bool,device=f24.device)}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        u=utility_forward_full_input_gradient(utility,batch)
    gate=__import__('scripts.phase4ha_utility_gated_rectification',fromlist=['gate_to_sam_grid']).gate_to_sam_grid(u['U'],sc)*r['support'].reshape(1,1,64,64).float()
    adapted=__import__('scripts.phase4ha_utility_gated_rectification',fromlist=['gated_embedding']).gated_embedding(s64,r['image_embeddings'],gate)
    with torch.autocast('cuda',enabled=False):
        low=sam(q,adapted.to(torch.bfloat16))
    return inverse_sam_logits(low,sg)


def run(device):
    require((AUDIT/'preflight.json').exists(), 'Phase6K preflight incomplete')
    c2,native,payload=provenance()
    selectors={name:json.loads((AUDIT/name/'selector.json').read_text()) for name in ('a','e')}
    arms={}
    for name,sel in selectors.items():
        require(sel['status']=='COMPLETE' and not sel['official1000_used_for_selection'] and
                not sel['internal_test_used_for_selection'] and not sel['external_ood_used_for_selection'],
                f'{name} selector/firewall drift')
        path=Path(sel['selected_checkpoint']);require(file_sha256(path)==sel['selected_checkpoint_sha256'],f'{name} selected SHA drift')
        state=torch.load(path,map_location='cpu',weights_only=False)
        require(state['c2_sha256']==c2['selected_checkpoint_sha256'] and
                state['native_sha256']==native['selected_checkpoint_sha256'],f'{name} payload provenance drift')
        arm=AfterProjection(name).to(device);arm.load_state_dict(state['projection_state'],strict=True)
        arms[name]=arm.eval().requires_grad_(False)
    *modules,frozen=runtime(device,payload)
    utility,rect,sam,source=modules
    cfg=yaml.safe_load(CKPT_CFG.read_text());conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1']
    model,tokenizer,meta=load_c2_model(cfg,C2_CKPT,device,expected_step=3500,expected_epoch=7)
    model.eval().requires_grad_(False)
    model.c2_cross_attention.capture_spatial_intermediates=True
    backend=GLaMMForensicsBackend(model,tokenizer,device=device,dtype=torch.bfloat16,
                                  use_mm_start_end=True,max_new_tokens=int(cfg['evaluation']['max_new_tokens']))
    full=dataset_for(tokenizer,cfg,'official1000')
    indices=[i for i,row in enumerate(full.rows) if int(row['class_label'])==1]
    expected=jsonl(MANIFEST)
    expected_ids=[str(x['sample_id']) for x in expected]
    require(len(indices)==len(expected_ids)==1000 and
            [str(full.rows[i]['sample_id']) for i in indices]==expected_ids,'Official1000 ID/order drift')
    prior=jsonl(RECORDS)
    require([x['sample_id'] for x in prior]==expected_ids[:len(prior)],'Official1000 resume prefix drift')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'status':'FROZEN','population':1000,'manifest_sha256':file_sha256(MANIFEST),
              'c2_sha256':c2['selected_checkpoint_sha256'],'native_sha256':native['selected_checkpoint_sha256'],
              'selected':{k:{'epoch':v['selected_epoch'],'sha256':v['selected_checkpoint_sha256']} for k,v in selectors.items()},
              'selector':'internal DEV only','historical_matched_canonical_control':True,
              'mask_logit_threshold':0.0,'invalid_no_seg_policy':'IoU=0','multi_seg_policy':'pixelwise logit maximum'}
    if (OUT/'protocol.json').exists():require(json.loads((OUT/'protocol.json').read_text())==protocol,'Official1000 protocol drift')
    else:dump(OUT/'protocol.json',protocol)
    vision=model.get_model().get_vision_tower()
    with torch.no_grad(),RECORDS.open('a',buffering=1) as handle:
        for ordinal,index in enumerate(indices[len(prior):],start=len(prior)):
            sample=full[index];sid=str(sample['sample_id'])
            target=torch.as_tensor(sample['masks']).bool().any(0)
            model.c2_cross_attention.last_spatial_intermediates=None
            output=backend.generate_localization_batch([sample],provide_gt_fake=False,
                generation_mode='unified_fake_generate')[0]
            generated_text=str(output.get('generated_text') or '')
            phrase_tail=generated_text.split('Target regions:',1)[1].split('[SEG]',1)[0].strip() if 'Target regions:' in generated_text else ''
            phrase_parse_success=bool(phrase_tail)
            cap=model.c2_cross_attention.last_spatial_intermediates
            require(cap is not None and cap['A'].shape==(1,8,24,24) and cap['E'].shape==(1,512,24,24),
                    f'Official1000 A/E capture drift: {sid}')
            q=output['projected_seg_embeddings'];count=0 if q is None else len(q)
            metrics={}
            if count==0:
                for name in ('b0','a','e'):
                    metrics[name]={'sample_id':sid,'foreground_iou':0.,'foreground_f1':0.,
                                   'tp':0,'fp':0,'fn':int(target.sum()),
                                   'tn':int(target.numel()-target.sum()),'valid_q_seg':False}
            else:
                raw=model.get_grounding_encoder_embs(sample['grounding_enc_image'][None].to(device=device,dtype=torch.bfloat16))
                tokens,_=vision(sample['global_enc_image'][None].to(device=device,dtype=torch.bfloat16))
                evidence=source(clip_grid(tokens),return_features=True)
                base=evidence['F_forensic'].to(torch.bfloat16)
                h,w=target.shape;sg,cg=geometry_for('sam',(h,w)),geometry_for('clip',(h,w))
                s64=sam_lowres_to_original_normalized(raw,sg,output_hw=(64,64)).to(torch.bfloat16)
                sc=sam_coordinates(sg,grid=64)[None].to(device)
                cc=clip_coordinates(cg,grid=24)[None].to(device)
                inputs={'a':cap['A'].to(device=device,dtype=torch.bfloat16).float(),
                        'e':cap['E'].to(device=device,dtype=torch.bfloat16).float()}
                logits={name:[] for name in ('b0','a','e')}
                for qone in q:
                    qone=qone[None].to(device=device,dtype=torch.bfloat16)
                    for name in logits:
                        arm=arms.get(name);spatial=inputs.get(name)
                        logits[name].append(component_logits(qone,base,s64,source,sam,rect,utility,sg,cg,sc,cc,spatial,arm))
                for name in logits:
                    merged=torch.stack(logits[name]).amax(0)
                    metrics[name]=metric_record(sid,merged,target)
                    metrics[name]['tn']=int(target.numel())-metrics[name]['tp']-metrics[name]['fp']-metrics[name]['fn']
                    metrics[name]['valid_q_seg']=True
            row={'sample_id':sid,'seg_count':count,'seg_triggered':bool(output['seg_triggered']),
                 'prompt_sha256':output['prompt_sha256'],'generated_text':generated_text,
                 'phrase_parse_success':phrase_parse_success,'metrics':metrics}
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (ordinal+1)%20==0:print(json.dumps({'stage':'OFFICIAL1000','done':ordinal+1,'total':1000}),flush=True)
    packed=jsonl(RECORDS)
    require([x['sample_id'] for x in packed]==expected_ids,'Official1000 incomplete/order drift')
    records={name:[x['metrics'][name] for x in packed] for name in ('b0','a','e')}
    result={'status':'COMPLETE','schema':'phase6k0_official1000_v1','population':1000,
            'metrics':{name:{**summarize(rows),**{key:sum(int(r[key]) for r in rows)
                       for key in ('tp','fp','fn','tn')}} for name,rows in records.items()},
            'paired':{'a_minus_b0':compare(records['a'],records['b0']),
                      'e_minus_b0':compare(records['e'],records['b0']),
                      'e_minus_a':compare(records['e'],records['a'])},
            'seg_trigger_count':sum(x['seg_triggered'] for x in packed),
            'exactly_one_seg_count':sum(x['seg_count']==1 for x in packed),
            'no_seg_count':sum(x['seg_count']==0 for x in packed),
            'multiple_seg_count':sum(x['seg_count']>1 for x in packed),
            'phrase_parse_success_count':sum(x['phrase_parse_success'] for x in packed),
            'records_sha256':file_sha256(RECORDS),'protocol':protocol}
    dump(OUT/'results.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');args=p.parse_args()
    device=torch.device(args.device);torch.cuda.set_device(device);run(device)
