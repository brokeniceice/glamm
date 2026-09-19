#!/usr/bin/env python3
"""Finalize Phase6G.15 and render the frozen report."""
from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts import phase6g10_train_arm as g10
from scripts.phase6g15_residual_staged_optimization import OUT,G14,ARMS,configure,effective,dump
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache,phase4f_spatial_batch
from tools.phase3c1 import paired_statistics
from tools.phase4c_a import summarize_extended

def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def pair(a,b):
    return {k:paired_statistics([x[k] for x in a],[x[k] for x in b]) for k in ('foreground_iou','foreground_f1')}
def positive(stat): return stat['mean_difference']>0 and stat['bootstrap_95_ci'][0]>0

def load_models(device):
    models={}
    direct,_=__import__('scripts.phase6g14_single_matched_translator',fromlist=['build_matched_arms']).build_matched_arms(device)
    p=torch.load(G14/'arms/A1/selected_checkpoint.pt',map_location='cpu',weights_only=False); direct['A1'].load_state_dict(p['model']); models['A0']=direct['A1'].eval().requires_grad_(False)
    for arm in ARMS:
        m,_,_,_=configure(arm,device); p=torch.load(OUT/'arms'/arm/'selected_checkpoint.pt',map_location='cpu',weights_only=False);m.load_state_dict(p['model']);models[arm]=m.eval().requires_grad_(False)
    return models

def dynamics(models,device):
    fusion=g10.load_fusion(device);proj=g10.load_projection(device);p4=g10.Phase4FStore(g10.hd.CFG,'val');fs=g10.Store('val');dev=g10.load_dev('g0');cache=load_c1_cache('val',dev['sample_ids'])
    recorders={k:g10.TapRecorder(v) for k,v in models.items()}; vals={k:[] for k in models if k!='A0'}
    with torch.no_grad():
        for i,sid in enumerate(dev['sample_ids']):
            if not bool(cache['valid'][i]):continue
            s64,_,_,sc,cc=phase4f_spatial_batch(p4,[sid],device);e=g10.fused_evidence(fusion,proj,fs,[sid],device);valid=torch.ones(1,576,dtype=torch.bool,device=device)
            taps={k:recorders[k].run(s64,e,sc,cc,valid)[0] for k in models}
            for k in vals:
                r0,r=taps['A0']['R2'].float(),taps[k]['R2'].float();d0,d=taps['A0']['R6'].float(),taps[k]['R6'].float()
                vals[k].append({'sample_id':sid,'r2_cosine':float(F.cosine_similarity(r0.flatten(),r.flatten(),dim=0)),'r2_norm_ratio':float(r.norm()/r0.norm().clamp_min(1e-12)),
                    'r2_relative_difference':float((r-r0).norm()/r0.norm().clamp_min(1e-12)),'delta_cosine':float(F.cosine_similarity(d0.flatten(),d.flatten(),dim=0)),
                    'delta_norm_ratio':float(d.norm()/d0.norm().clamp_min(1e-12)),'delta_relative_difference':float((d-d0).norm()/d0.norm().clamp_min(1e-12))})
            if (i+1)%200==0: print(json.dumps({'stage':'G15_DYNAMICS','done':i+1}),flush=True)
    summary={}
    for arm,rr in vals.items():
        g10.write_rows(OUT/'dynamics'/f'{arm}_vs_A0.jsonl',rr);summary[arm]={}
        for key in rr[0]:
            if key=='sample_id':continue
            x=torch.tensor([z[key] for z in rr],dtype=torch.float64);summary[arm][key]={'n':len(x),'mean':float(x.mean()),'std':float(x.std(unbiased=False)),'median':float(x.median()),'p5':float(torch.quantile(x,.05)),'p95':float(torch.quantile(x,.95))}
    return summary

def map_comparison(models):
    maps={}; out={}
    for arm,m in models.items():
        w,b,diag=effective(m,arm if arm!='A0' else 'C1');maps[arm]=(w.detach().double().cpu(),b.detach().double().cpu());out[arm]={'spectrum':g10.spectrum(maps[arm][0]),'diagnostics':diag}
    w0,b0=maps['A0'];out['versus_A0']={}
    for arm in ARMS:
        w,b=maps[arm];out['versus_A0'][arm]={'matrix_cosine':float(F.cosine_similarity(w0.flatten(),w.flatten(),dim=0)),'frobenius_difference':float((w-w0).norm()),
          'relative_frobenius_difference':float((w-w0).norm()/w0.norm().clamp_min(1e-12)),'bias_cosine':float(F.cosine_similarity(b0,b,dim=0))}
    return out

def main():
    for a in ARMS:
        if not (OUT/'arms'/a/'summary.json').exists():raise RuntimeError(f'{a} incomplete')
    pred={'A0':rows(G14/'arms/A1/validation/selected.jsonl')};pred.update({a:rows(OUT/'arms'/a/'validation/selected.jsonl') for a in ARMS})
    ids=[x['sample_id'] for x in pred['A0']]
    if any([x['sample_id'] for x in v]!=ids for v in pred.values()):raise RuntimeError('sample order drift')
    metrics={k:summarize_extended(v) for k,v in pred.items()}; comparisons={f'{a}_minus_A0':pair(pred[a],pred['A0']) for a in ARMS}
    comparisons['A1_minus_C1']=pair(pred['A1'],pred['C1']);comparisons['A2_minus_C2']=pair(pred['A2'],pred['C2']);comparisons['A2_minus_A1']=pair(pred['A2'],pred['A1'])
    a1a0=positive(comparisons['A1_minus_A0']['foreground_iou']);a1c1=positive(comparisons['A1_minus_C1']['foreground_iou']);a2a0=positive(comparisons['A2_minus_A0']['foreground_iou']);a2c2=positive(comparisons['A2_minus_C2']['foreground_iou'])
    if a1a0 and a1c1: decision='FROZEN_BASE_RESIDUAL_OPTIMIZATION_SUPPORTED'; explains='YES'
    elif a2a0 and a2c2 and not (a1a0 and a1c1): decision='RESIDUAL_UPSTREAM_COADAPTATION_SUPPORTED';explains='YES'
    elif a1a0 or a2a0: decision='STAGED_GAIN_PRESENT_BUT_RESIDUAL_NOT_ISOLATED';explains='NO'
    else: decision='STAGED_RESIDUAL_OPTIMIZATION_NOT_SUPPORTED';explains='NO'
    device=torch.device('cuda:0');torch.cuda.set_device(device);models=load_models(device);maps=map_comparison(models);dyn=dynamics(models,device)
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines()
    preflight={'fresh_g14_A0':.17935125171212685,'historical_g10_A0':.1812439115316707,'delta':-.00189265981954385,'git_head':git_head,'dirty_entries':len(git_status),
      'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu_count':torch.cuda.device_count(),'ddp':False,
      'deterministic_flags':{'cudnn_benchmark':False,'cudnn_deterministic':True},'data_order':'seed3407+1009*global_epoch',
      'amp':'BF16 autocast Rectifier; FP32 SAM loss path','selector':'DEV G0 mean IoU, tie earlier',
      'finding':'No known forward/data/selector difference. Absolute drift is not used for conclusions; plausible BF16/CUDA nondeterminism or run variance.'}
    result={'schema':'phase6g15_final_v1','status':'COMPLETE_STOP','decision':decision,'RESIDUAL_STAGED_OPTIMIZATION_EXPLAINS_HISTORICAL_SIDE_GAIN':explains,
      'preflight_reproducibility':preflight,'metrics':metrics,'comparisons':comparisons,'effective_maps':maps,'r2_delta_dynamics':dyn,
      'budget':{'A0':'G14 direct single epochs1-10','stage1':'G14 direct single selected epoch5 from candidates1-5','stage2':'5 epochs using global data orders6-10 and restarted 5-epoch optimizer/scheduler',
                'controls':'C1/C2 isolate optimizer-scheduler restart and ownership from residual anchoring'},
      'firewall':{'utility':False,'nonlinear':False,'internal_test':False,'official1000':False,'ood':False}}
    dump(OUT/'results.json',result)
    def line(name):
        s=comparisons[name]['foreground_iou'];return f"delta={s['mean_difference']:+.6f}, CI={s['bootstrap_95_ci']}, W/T/L={s['wins']}/{s['ties']}/{s['losses']}, p={s['wilcoxon_pvalue']:.4g}"
    table='\n'.join(['| Arm | mean IoU | median IoU | mean F1 | global IoU | global F1 |','|---|---:|---:|---:|---:|---:|']+[f"| {a} | {metrics[a]['mean_foreground_iou']:.6f} | {metrics[a]['median_foreground_iou']:.6f} | {metrics[a]['mean_foreground_f1']:.6f} | {metrics[a]['global_foreground_iou']:.6f} | {metrics[a]['global_foreground_f1']:.6f} |" for a in ('A0','A1','A2','C1','C2')])
    text=f'''# Phase 6G.15 — Residual Staged Optimization Audit

## 1. Scientific question
Does base-then-zero-residual training improve the same affine correction class, rather than merely adding capacity or extra updates?

## 2. Function-class equivalence
All final corrections are `support*gamma*W_eff(R2)`. A1/A2 use `W_eff=W_base+W_residual` and can be analytically collapsed to one affine map.

## 3. Arms
- A0: reused G14 direct single, continuous 10 epochs.
- A1: selected epoch5 base frozen with all upstream; zero residual only for epochs6-10.
- A2: base frozen; upstream Rectifier and zero residual co-adapt for epochs6-10.
- C1: A1-matched ownership/restart, but directly updates the existing map.
- C2: A2-matched ownership/restart, but directly updates the existing map.

## 4. Initialization equivalence
All Stage1 arms reuse the exact G14 A1 epoch5 checkpoint. A1/A2 residual weights and biases are zero; stored Stage2 checks have exact zero error.

## 5. Update-budget control
Primary lineage is 5+5 epochs versus A0 10 epochs, with identical global epoch data orders. No extended-budget diagnostic was run. C1/C2 separate restart effects from residual anchoring.

## 6. Trainable ownership
Full parameter names/counts are in each arm `initialization.json`. A1/C1 are frozen-upstream controls; A2/C2 are upstream-adaptive controls.

## 7. DEV results
{table}

## 8. Paired statistics
- A1 vs A0: {line('A1_minus_A0')}
- A2 vs A0: {line('A2_minus_A0')}
- A2 vs A1: {line('A2_minus_A1')}
- A1 vs C1 residual-isolation control: {line('A1_minus_C1')}
- A2 vs C2 residual-isolation control: {line('A2_minus_C2')}

## 9. Optimization dynamics
Per-epoch loss, DEV IoU, gradient norms, gamma and effective-map norms are saved under each arm. Stage2 uses a fresh identical optimizer/scheduler per arm.

## 10. Residual/base magnitude analysis
```json
{json.dumps({a:json.loads((OUT/'arms'/a/'summary.json').read_text())['diagnostics'] for a in ('A1','A2')},ensure_ascii=False,indent=2)}
```

## 11. Effective-map comparison
Saved in `results.json`; it contains spectra, ranks, condition numbers and direct-versus-staged matrix similarities.

## 12. R2/Delta dynamics
Saved in `results.json` and per-sample JSONL under `dynamics/`. A1 freezes R2 by construction; A2 permits R2 co-adaptation.

## 13. Historical G11 comparison
Historical G11 main+side IoU 0.185546 is a secondary reference only. Fresh paired conclusions use A0/A1/A2/C1/C2. G14 fresh-vs-G10 absolute drift is recorded in the reproducibility section and does not select a claim.

## 14. Supported interpretation
Decision: **{decision}**. Restart-matched controls are required before assigning any staged gain specifically to residual anchoring.

## 15. Unsupported interpretation
No result establishes nonlinear capacity, Utility behavior, OOD generalization, or a mandatory two-branch inference architecture. All staged maps remain analytically collapsible.

## 16. Decision
`{decision}`

`RESIDUAL_STAGED_OPTIMIZATION_EXPLAINS_HISTORICAL_SIDE_GAIN = {explains}`

COMPLETE STOP. No nonlinear translator was started.
'''
    (ROOT/'docs/phase6g15_residual_staged_optimization.md').write_text(text,encoding='utf-8')
    print(json.dumps({'status':'COMPLETE_STOP','decision':decision,'explains':explains}),flush=True)
if __name__=='__main__':main()
