#!/usr/bin/env python3
"""Read-only exact C2 fused-token and raw classification-logit capture audit."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase3c2_p3 import dataset_for
from scripts.phase6j0_c2_evaluate import load_c2_model
from scripts.phase6k0_c2_after_train import C2_CKPT,C2_PROTOCOL,AUDIT,require,dump
from tools.phase4c_b import file_sha256
from tools.phase3f_aogd import core_model


def main():
    device=torch.device('cuda:2');torch.cuda.set_device(device)
    protocol=json.loads(C2_PROTOCOL.read_text());require(file_sha256(C2_CKPT)==protocol['selected_checkpoint_sha256'],'C2 SHA drift')
    cfg=yaml.safe_load((ROOT/'configs/phase6j0_c2_preln_cross_attention.yaml').read_text())
    conversation_lib.default_conversation=conversation_lib.conv_templates['llava_v1']
    model,tokenizer,_=load_c2_model(cfg,C2_CKPT,device,expected_step=3500,expected_epoch=7)
    model.eval().requires_grad_(False)
    backend=GLaMMForensicsBackend(model,tokenizer,device=device,dtype=torch.bfloat16,
        use_mm_start_end=True,max_new_tokens=int(cfg['evaluation']['max_new_tokens']))
    dataset=dataset_for(tokenizer,cfg,'val')
    sample=next(dataset[i] for i,row in enumerate(dataset.rows) if int(row['class_label'])==1)
    attn=model.c2_cross_attention
    core=core_model(model)
    fused=[];logits=[]
    handle=attn.register_forward_hook(lambda _m,_input,output:fused.append(output.detach().cpu().clone()))
    original=core._extract_cls_logits
    def capture_logits(*args,**kwargs):
        value=original(*args,**kwargs)
        logits.append(value[0].detach().cpu().clone())
        return value
    core._extract_cls_logits=capture_logits
    def same(x,y):
        return (x is None and y is None) or (isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor) and torch.equal(x,y))
    try:
        attn.capture_spatial_intermediates=False
        old=backend.generate_localization_batch([sample],provide_gt_fake=False,generation_mode='unified_fake_generate')[0]
        require(bool(fused) and len(logits)==1,'old C2 hook missing')
        old_fused=fused[-1];old_logits=logits[-1];fused.clear();logits.clear()
        attn.capture_spatial_intermediates=True
        new=backend.generate_localization_batch([sample],provide_gt_fake=False,generation_mode='unified_fake_generate')[0]
        require(bool(fused) and len(logits)==1,'capture C2 hook missing')
        new_fused=fused[-1];new_logits=logits[-1]
        require(torch.equal(old_fused,new_fused) and torch.equal(old_logits,new_logits),
                'C2 fused token/raw classification logits changed by capture')
        require(old['generated_token_ids']==new['generated_token_ids'] and
                same(old['projected_seg_embeddings'],new['projected_seg_embeddings']) and
                same(old['pred_mask'],new['pred_mask']),'C2 q/mask changed by capture')
        dump(AUDIT/'cache/full_capture_parity.json',{'status':'PASS','sample_id':sample['sample_id'],
            'fused_token_shape':list(old_fused.shape),'raw_classification_logits_shape':list(old_logits.shape),
            'fused_token_exact':True,'classification_logits_exact':True,'q_seg_exact':True,
            'mask_exact':True,'generated_token_ids_exact':True,
            'c2_sha256':protocol['selected_checkpoint_sha256']})
    finally:
        handle.remove();core._extract_cls_logits=original

if __name__=='__main__':main()
