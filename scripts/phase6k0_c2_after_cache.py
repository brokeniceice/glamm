#!/usr/bin/env python3
"""Immutable canonical C2 G0 query and pre-sum attention cache for Phase6K."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase3c2_p3 import dataset_for
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from scripts.phase4gf_formal_localization import load_dev
from scripts.phase6j0_c2_evaluate import load_c2_model
from tools.phase4c_b import file_sha256

OUT = Path('/data/yz/groundingLMM_official/cache/phase6k0_c2_after_v2')
AUDIT = ROOT / 'outputs/phase6k0_c2_after/cache'
CKPT = ROOT / 'checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt'
CFG = ROOT / 'configs/phase6j0_c2_preln_cross_attention.yaml'
PROTOCOL = ROOT / 'outputs/phase6j0_c2/final_evaluation/protocol.json'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensors_equal(a, b, atol=0):
    if a is None or b is None:
        return a is None and b is None
    return bool(torch.allclose(a.float().cpu(), b.float().cpu(), atol=atol, rtol=0))


def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    protocol = json.loads(PROTOCOL.read_text())
    checkpoint_sha = file_sha256(CKPT)
    require(protocol['status'] == 'FROZEN' and protocol['selected_checkpoint_sha256'] == checkpoint_sha,
            'C2 selected checkpoint/protocol mismatch')
    require((protocol['selected_step'], protocol['selected_epoch']) == (3500, 7), 'C2 selector step/epoch drift')
    cfg = yaml.safe_load(CFG.read_text())
    require(cfg['architecture']['attention_inner_dim'] == 512 and cfg['architecture']['attention_heads'] == 8,
            'C2 attention configuration drift')
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_v1']
    model, tokenizer, meta = load_c2_model(cfg, CKPT, device, expected_step=3500, expected_epoch=7)
    model.eval().requires_grad_(False)
    attn = model.c2_cross_attention
    require(attn.heads == 8 and attn.head_dim == 64, 'C2 attention architecture drift')
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=int(cfg['evaluation']['max_new_tokens']))
    ds = dataset_for(tokenizer, cfg, args.split)
    fake = [i for i, row in enumerate(ds.rows) if int(row['class_label']) == 1]
    expected = train_ids() if args.split == 'train' else load_dev('g0')['sample_ids']
    ids = [str(ds.rows[i]['sample_id']) for i in fake]
    require(ids == expected, 'canonical population/order drift')
    start = args.start_index
    stop = len(expected) if args.end_index is None else args.end_index
    require(0 <= start < stop <= len(expected), 'invalid cache range')
    root = OUT / (args.output_subdir or args.split)
    root.mkdir(parents=True, exist_ok=True)
    completed = sorted(root.glob('shard_*.pt'))
    done, counts = start, []
    for path in completed:
        shard = torch.load(path, map_location='cpu', weights_only=False)
        require(shard['c2_sha256'] == checkpoint_sha and shard['sample_ids'] == expected[done:done+len(shard['sample_ids'])],
                f'cache resume drift: {path}')
        require(shard['A'].shape == (len(shard['sample_ids']), 8, 24, 24) and shard['A'].dtype == torch.float32, 'A shard shape/dtype drift')
        require(shard['E'].shape == (len(shard['sample_ids']), 512, 24, 24), 'E shard shape drift')
        counts.extend(shard['seg_count'].tolist())
        done += len(shard['sample_ids'])
    if done == 0:
        # One real sample tests the read-only capture against an unmodified forward.
        sample = [ds[fake[0]]]
        attn.capture_spatial_intermediates = False
        old = backend.generate_localization_batch(sample, provide_gt_fake=False, generation_mode='unified_fake_generate')[0]
        attn.capture_spatial_intermediates = True
        new = backend.generate_localization_batch(sample, provide_gt_fake=False, generation_mode='unified_fake_generate')[0]
        cap = attn.last_spatial_intermediates
        require(cap is not None and cap['A'].shape == (1,8,24,24) and cap['E'].shape == (1,512,24,24),
                'A/E capture missing or shape drift')
        old_q, new_q = old['projected_seg_embeddings'], new['projected_seg_embeddings']
        require(old['generated_token_ids'] == new['generated_token_ids'] and
                tensors_equal(old_q, new_q) and tensors_equal(old['pred_mask'], new['pred_mask']) and
                tensors_equal(old['cls_prob_fake'], new['cls_prob_fake']),
                'C2 no-capture vs capture output parity failed')
        a = cap['A'].float(); e = cap['E'].float()
        require(bool(torch.isfinite(a).all() and torch.isfinite(e).all()), 'A/E nonfinite')
        require(float((a.flatten(2).sum(-1)-1).abs().max()) < 1e-5, 'A head mass drift')
        ec = e.reshape(1,8,64,576).sum(-1)
        ctx_error = float((ec-cap['head_context'].float()).abs().max())
        ctx_scale = float(cap['head_context'].float().abs().max())
        ctx_relative_error = ctx_error / max(ctx_scale, 1e-12)
        require(ctx_relative_error <= 0.01, f'E context parity failed: max_abs={ctx_error}, max_relative={ctx_relative_error}')
        save_json(AUDIT/'capture_preflight.json', {
            'status':'PASS', 'c2_sha256':checkpoint_sha, 'model_meta':meta,
            'A_shape':list(a.shape), 'E_shape':list(e.shape),
            'max_A_mass_error':float((a.flatten(2).sum(-1)-1).abs().max()),
            'max_E_head_context_error':ctx_error,'max_E_head_context_relative_error':ctx_relative_error,
            'generated_token_ids_exact':True,'q_seg_exact':True,'mask_exact':True,'classification_probability_exact':True})
    attn.capture_spatial_intermediates = True
    buffer = []
    for begin in range(done, stop, args.batch_size):
        samples = [ds[i] for i in fake[begin:min(begin+args.batch_size, stop)]]
        attn.last_spatial_intermediates = None
        output = backend.generate_localization_batch(samples, provide_gt_fake=False,
                                                     generation_mode='unified_fake_generate')
        cap = attn.last_spatial_intermediates
        require(cap is not None, 'C2 capture missing')
        require(cap['A'].shape == (len(samples),8,24,24) and cap['E'].shape == (len(samples),512,24,24),
                'C2 batch capture shape drift')
        for j, (sample, result) in enumerate(zip(samples, output)):
            q = result['projected_seg_embeddings']
            q = torch.empty((0,256), dtype=torch.bfloat16) if q is None else q.detach().cpu().to(torch.bfloat16)
            buffer.append({'sample_id':str(sample['sample_id']), 'q':q,
                           'seg_count':int(q.shape[0]), 'A':cap['A'][j].to(torch.float32),
                           'E':cap['E'][j].to(torch.bfloat16),
                           'generated_token_ids':result['generated_token_ids'],
                           'prompt_sha256':result['prompt_sha256']})
        if len(buffer) >= args.shard_size or begin + len(samples) >= stop:
            n = len(buffer)
            qone = torch.full((n,256), float('nan'), dtype=torch.bfloat16)
            for j, row in enumerate(buffer):
                if row['seg_count'] == 1:
                    qone[j] = row['q'][0]
            ids = [row['sample_id'] for row in buffer]
            seg_count = torch.tensor([row['seg_count'] for row in buffer], dtype=torch.long)
            payload = {'schema':'phase6k0_c2_cache_shard_v1','split':args.split,'sample_ids':ids,
                       'q_seg':qone,'valid':seg_count.eq(1),'seg_count':seg_count,
                       'A':torch.stack([row['A'] for row in buffer]),
                       'E':torch.stack([row['E'] for row in buffer]),
                       'records':[{k:v for k,v in row.items() if k not in ('q','A','E')} for row in buffer],
                       'c2_sha256':checkpoint_sha}
            ordinal = len(sorted(root.glob('shard_*.pt')))
            path = root/f'shard_{ordinal:06d}.pt'
            tmp = path.with_suffix('.pt.tmp')
            torch.save(payload, tmp)
            os.replace(tmp, path)
            done += n
            counts.extend(seg_count.tolist())
            buffer.clear()
            print(json.dumps({'stage':'C2_CACHE','split':args.split,'done':done,'total':stop,
                              'range_start':start}), flush=True)
    require(done == stop, 'cache incomplete')
    shards = sorted(root.glob('shard_*.pt'))
    digest = hashlib.sha256('\n'.join(expected[start:stop]).encode()).hexdigest()
    summary = {'status':'COMPLETE','schema':'phase6k0_c2_cache_v1','split':args.split,
               'c2_sha256':checkpoint_sha,'n':stop-start,'range_start':start,'range_end':stop,
               'ids_sha256':digest,
               'valid_exactly_one_seg':sum(x==1 for x in counts),
               'invalid_no_seg':sum(x==0 for x in counts),
               'invalid_multiple_seg':sum(x>1 for x in counts),
               'shards':[{'path':str(p.resolve()),'sha256':file_sha256(p)} for p in shards],
               'firewall':{'internal_test':False,'official1000':False,'external_ood':False}}
    save_json(AUDIT/f'{args.summary_name or args.split}.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train','val'], required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--shard-size', type=int, default=100)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--end-index', type=int)
    parser.add_argument('--output-subdir')
    parser.add_argument('--summary-name')
    run(parser.parse_args())
