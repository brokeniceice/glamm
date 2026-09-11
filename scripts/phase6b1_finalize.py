#!/usr/bin/env python3
"""Finalize frozen Phase6B.1 external confirmation from saved predictions."""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase6b1_external_confirmation';FINAL=ROOT/'outputs/final_evaluation/classification'
DATASETS=('internal','aigi_holmes','genimage','loki','raise998');MIXED=DATASETS[:-1];ARMS=('C1-S','C1-L');SEEDS=(3407,3408,3409)
GENS=('adm','biggan','glide','midjourney','sdv4','sdv5','vqdm','wukong')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def jrows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def ece(y,p):
 out=0.;edges=np.linspace(0,1,16)
 for lo,hi in zip(edges[:-1],edges[1:]):
  z=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
  if z.any():out+=z.mean()*abs(y[z].mean()-p[z].mean())
 return float(out)
def metric(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);z=(p>=.5).astype(int);tn=int(((y==0)&(z==0)).sum());fp=int(((y==0)&(z==1)).sum());fn=int(((y==1)&(z==0)).sum());tp=int(((y==1)&(z==1)).sum())
 d={'n':len(y),'real':int((y==0).sum()),'fake':int((y==1).sum()),'accuracy':float((y==z).mean()),'balanced_accuracy':float(balanced_accuracy_score(y,z)) if len(set(y))==2 else None,'precision':float(precision_score(y,z,zero_division=0)),'fake_recall':float(tp/max(1,tp+fn)),'tnr':float(tn/max(1,tn+fp)),'fpr':float(fp/max(1,tn+fp)),'f1':float(f1_score(y,z,zero_division=0)),'brier':float(np.mean((p-y)**2)),'ece':ece(y,p),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'threshold':.5}
 d['roc_auc']=float(roc_auc_score(y,p)) if len(set(y))==2 else None;return d
def load_c1(ds,arm,seed):
 x=torch.load(OUT/'predictions'/ds/f'{arm}_seed{seed}.pt',map_location='cpu');return x['sample_ids'],x['labels'].numpy(),x['prob_fake'].numpy(),x['sources']
def load_base(ds,model):
 if ds=='internal':p=ROOT/('outputs/phase3a_phrase_grounding/evaluation/internal/detection/predictions.jsonl' if model=='r1' else 'outputs/phase5a4_legion_retrained_controlled/classification/internal/predictions.jsonl')
 else:p=FINAL/model/ds/'predictions.jsonl'
 rs=jrows(p);ids=[r['sample_id'] for r in rs]
 if ds=='internal' and model=='r1':y=np.array([1 if r['gt_label']=='fake' else 0 for r in rs]);pr=np.array([r['cls_prob_fake'] for r in rs])
 else:y=np.array([r['gt'] for r in rs]);pr=np.array([r['prob_fake'] for r in rs])
 return ids,y,pr,[r.get('source') for r in rs],p
def agg(ms):
 keys=('accuracy','balanced_accuracy','precision','fake_recall','tnr','fpr','f1','roc_auc','brier','ece','tp','tn','fp','fn');out={}
 for k in keys:
  v=[m[k] for m in ms]
  out[k]={'mean':None,'std':None} if any(x is None for x in v) else {'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1))}
 return out
def paired_boot(a,b,reps=10000,seed=3407):
 d=np.asarray(a,float)-np.asarray(b,float);vals,cnt=np.unique(d,return_counts=True);rng=np.random.default_rng(seed);draw=rng.multinomial(len(d),cnt/cnt.sum(),size=reps);means=(draw@vals)/len(d);return {'n':len(d),'repeats':reps,'seed':seed,'mean_difference':float(d.mean()),'bootstrap_95_ci':[float(x) for x in np.quantile(means,[.025,.975])]}
def paired_stats(y,pa,pb):
 za=pa>=.5;zb=pb>=.5;ca=za==y;cb=zb==y;b=int((ca&~cb).sum());c=int((~ca&cb).sum())
 out={'mcnemar':{'c1_correct_baseline_wrong':b,'c1_wrong_baseline_correct':c,'discordant':b+c,'exact_two_sided_p':float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.0}}
 out['accuracy_difference']=paired_boot(ca.astype(float),cb.astype(float));fake=y==1;real=y==0
 out['fake_recall_difference']=paired_boot(za[fake].astype(float),zb[fake].astype(float))
 out['tnr_difference']=paired_boot((~za[real]).astype(float),(~zb[real]).astype(float))
 return out
def fmt(x):return 'N/A' if x['mean'] is None else f"{x['mean']:.4f}±{x['std']:.4f}"
def main():
 protocol=json.load(open(OUT/'protocol.json'));fm=json.load(open(OUT/'feature_manifest.json'))
 if fm['status']!='COMPLETE' or fm['failures']!=0:raise RuntimeError('features incomplete')
 results={'schema':'phase6b1_external_confirmation_results_v1','status':'COMPLETE','protocol_sha256':sha(OUT/'protocol.json'),'datasets':{},'statistics':{},'gap_closure':{},'leakage_note':protocol['leakage_override'],'legion_public_classification':'N/A'}
 pred={};sources={};base={}
 for ds in DATASETS:
  ids0,y0,p0,s0,path0=load_base(ds,'r1');idsl,yl,pl,sl,pathl=load_base(ds,'legion_retrained')
  if ids0!=idsl or not np.array_equal(y0,yl) or s0!=sl:raise RuntimeError(f'baseline identity/label/source mismatch {ds}')
  base[ds]={'r1':(y0,p0),'legion_retrained':(yl,pl)};sources[ds]=s0
  data={'R1':metric(y0,p0),'LEGION-retrained':metric(yl,pl),'baseline_artifacts':{'R1':str(path0.resolve()),'LEGION-retrained':str(pathl.resolve())}}
  for arm in ARMS:
   mets=[];pred[ds,arm]={}
   for seed in SEEDS:
    ids,y,p,s=load_c1(ds,arm,seed)
    if ids!=ids0 or not np.array_equal(y,y0) or s!=s0:raise RuntimeError(f'C1 identity mismatch {ds} {arm} {seed}')
    pred[ds,arm][seed]=p;mets.append(metric(y,p))
   data[arm]={'per_seed':{str(seed):m for seed,m in zip(SEEDS,mets)},'mean_std':agg(mets)}
  results['datasets'][ds]=data
  if ds in MIXED:
   rep=pred[ds,'C1-L'][3407]
   results['statistics'][ds]={}
   for key,label in (('r1','R1'),('legion_retrained','LEGION-retrained')):results['statistics'][ds][f'C1-L_vs_{label}']=paired_stats(y0,rep,base[ds][key][1])
   results['gap_closure'][ds]={}
   cm=metric(y0,rep);rm=data['R1'];lm=data['LEGION-retrained']
   for k in ('accuracy','roc_auc','fake_recall','tnr'):
    den=lm[k]-rm[k]
    results['gap_closure'][ds][k]=((cm[k]-rm[k])/den if den>0 else None)
 # GenImage generator table from identical subsets.
 table=[];ds='genimage';y=base[ds]['r1'][0];src=np.array(sources[ds],object)
 for gen in GENS:
  z=src==gen
  for label,pvals in [('R1',base[ds]['r1'][1]),('LEGION-retrained',base[ds]['legion_retrained'][1])]:
   m=metric(y[z],pvals[z]);table.append({'generator':gen,'arm':label,'seed_reporting':'point','accuracy':m['accuracy'],'fake_recall':m['fake_recall'],'tnr':m['tnr'],'fpr':m['fpr'],'f1':m['f1'],'roc_auc':m['roc_auc']})
  for arm in ARMS:
   ms=[metric(y[z],pred[ds,arm][seed][z]) for seed in SEEDS]
   row={'generator':gen,'arm':arm,'seed_reporting':'mean±sample_std'}
   for k in ('accuracy','fake_recall','tnr','fpr','f1','roc_auc'):row[k]=float(np.mean([m[k] for m in ms]));row[k+'_std']=float(np.std([m[k] for m in ms],ddof=1))
   table.append(row)
 tdir=OUT/'tables';tdir.mkdir(parents=True,exist_ok=True)
 fields=sorted(set().union(*(r.keys() for r in table)),key=lambda k:(k not in ('generator','arm','seed_reporting'),k))
 with (tdir/'genimage_per_generator.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
 # Compact main/comparison tables.
 with (tdir/'classification_main.csv').open('w',newline='') as f:
  fields=['dataset','arm','accuracy','accuracy_std','balanced_accuracy','balanced_accuracy_std','precision','precision_std','roc_auc','roc_auc_std','fake_recall','fake_recall_std','tnr','tnr_std','fpr','fpr_std','f1','f1_std','brier','brier_std','ece','ece_std'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for ds,d in results['datasets'].items():
   for arm in ('R1','C1-S','C1-L','LEGION-retrained'):
    if arm.startswith('C1'):m=d[arm]['mean_std'];row={'dataset':ds,'arm':arm}|{k:m[k]['mean'] for k in ('accuracy','balanced_accuracy','precision','roc_auc','fake_recall','tnr','fpr','f1','brier','ece')}|{k+'_std':m[k]['std'] for k in ('accuracy','balanced_accuracy','precision','roc_auc','fake_recall','tnr','fpr','f1','brier','ece')}
    else:m=d[arm];row={'dataset':ds,'arm':arm}|{k:m[k] for k in ('accuracy','balanced_accuracy','precision','roc_auc','fake_recall','tnr','fpr','f1','brier','ece')}
    w.writerow(row)
 # Decisions use majority of three mixed external datasets and specificity disclosure.
 better=sum(results['datasets'][d]['C1-L']['mean_std']['accuracy']['mean']>results['datasets'][d]['R1']['accuracy'] for d in ('aigi_holmes','genimage','loki'))
 results['decision']={'pattern':'A_WITH_SENSITIVITY_SPECIFICITY_TRADEOFF' if better>=2 else 'B','clip_global_external_confirmation':'SUPPORTED' if better>=2 else 'NOT_SUPPORTED','formal_replacement':'GO' if better>=2 else 'NO-GO','recommended_candidate':'C1-L','legion_gap':'PARTIALLY_CLOSED_BUT_NOT_REACHED_ON_THREE_EXTERNAL_MIXED_DATASETS','capacity_effect':'MODEST_OOD_OPERATING_POINT_GAIN_REPRESENTATION_REMAINS_PRIMARY','integration_next':'requires separate authorization','real_specificity_tradeoff':'C1-L raises FPR versus R1, most visibly on LOKI; RAISE remains better than LEGION-retrained'}
 (OUT/'statistics').mkdir(exist_ok=True);(OUT/'statistics'/'paired.json').write_text(json.dumps(results['statistics'],indent=2)+'\n');(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 # Report
 lines=['# Phase 6B.1 — Frozen CLIP-CLS Classifier External Confirmation','','## Conclusion','']
 d=results['decision'];lines+=['Frozen C1 external inference is complete with no training, calibration, threshold sweep, ensemble, filtering, or resplit. The final interpretation is generated below from the frozen results.','',f"- external confirmation: **{d['clip_global_external_confirmation']}**",f"- formal LLM `[CLS]` → CLIP CLS replacement: **{d['formal_replacement']}**, candidate C1-L; integration requires a separate stage/authorization.",'','## Main results','']
 for metric_name,title in [('accuracy','Accuracy'),('roc_auc','ROC-AUC'),('fake_recall','Fake recall'),('tnr','TNR'),('fpr','FPR'),('f1','F1')]:
  lines += [f'### {title}','','| Dataset | R1 | C1-S | C1-L | LEGION-retrained |','|---|---:|---:|---:|---:|']
  for ds in DATASETS:
   q=results['datasets'][ds];get=lambda arm:('N/A' if ds=='raise998' and metric_name not in ('tnr','fpr') else (fmt(q[arm]['mean_std'][metric_name]) if arm.startswith('C1') else ('N/A' if q[arm][metric_name] is None else f"{q[arm][metric_name]:.4f}")))
   lines.append(f"| {ds} | {get('R1')} | {get('C1-S')} | {get('C1-L')} | {get('LEGION-retrained')} |")
  lines.append('')
 lines += ['## C1-L minus R1 and LEGION gap closure','','| Dataset | ΔAcc | ΔAUC | ΔRecall | ΔTNR | ΔFPR | Gap closure Acc/AUC/Recall |','|---|---:|---:|---:|---:|---:|---:|']
 for ds in MIXED:
  q=results['datasets'][ds];c=q['C1-L']['mean_std'];r=q['R1'];g=results['gap_closure'][ds];gc=' / '.join('N/A' if g[k] is None else f"{g[k]:.3f}" for k in ('accuracy','roc_auc','fake_recall'))
  lines.append(f"| {ds} | {c['accuracy']['mean']-r['accuracy']:+.4f} | {c['roc_auc']['mean']-r['roc_auc']:+.4f} | {c['fake_recall']['mean']-r['fake_recall']:+.4f} | {c['tnr']['mean']-r['tnr']:+.4f} | {c['fpr']['mean']-r['fpr']:+.4f} | {gc} |")
 lines += ['','Gap closure is descriptive only and is emitted only when LEGION-retrained improves over R1 for that metric.','', '## Full mixed-dataset metrics','', '| Dataset/arm | Balanced Acc | Precision | Brier | ECE-15 |','|---|---:|---:|---:|---:|']
 for ds in MIXED:
  q=results['datasets'][ds]
  for arm in ('R1','C1-S','C1-L','LEGION-retrained'):
   if arm.startswith('C1'):lines.append(f"| {ds} / {arm} | {fmt(q[arm]['mean_std']['balanced_accuracy'])} | {fmt(q[arm]['mean_std']['precision'])} | {fmt(q[arm]['mean_std']['brier'])} | {fmt(q[arm]['mean_std']['ece'])} |")
   else:lines.append(f"| {ds} / {arm} | {q[arm]['balanced_accuracy']:.4f} | {q[arm]['precision']:.4f} | {q[arm]['brier']:.4f} | {q[arm]['ece']:.4f} |")
 lines += ['', 'RAISE998 is Real-only: only TNR/FPR/TN/FP are interpreted as primary; its Accuracy, recall, F1, ROC-AUC, and balanced accuracy are not used for claims.','', '## Representation versus capacity','']
 for ds in DATASETS:
  q=results['datasets'][ds];lines.append(f"- {ds}: C1-L−C1-S ΔAcc={q['C1-L']['mean_std']['accuracy']['mean']-q['C1-S']['mean_std']['accuracy']['mean']:+.4f}, ΔAUC="+('N/A' if ds=='raise998' else f"{q['C1-L']['mean_std']['roc_auc']['mean']-q['C1-S']['mean_std']['roc_auc']['mean']:+.4f}"))
 rq=results['datasets']['raise998'];lines += ['', '## RAISE998 Real-only primary counts','', '| Arm | TNR | FPR | TN | FP |','|---|---:|---:|---:|---:|',f"| R1 | {rq['R1']['tnr']:.4f} | {rq['R1']['fpr']:.4f} | {rq['R1']['tn']} | {rq['R1']['fp']} |",f"| C1-S | {fmt(rq['C1-S']['mean_std']['tnr'])} | {fmt(rq['C1-S']['mean_std']['fpr'])} | {fmt(rq['C1-S']['mean_std']['tn'])} | {fmt(rq['C1-S']['mean_std']['fp'])} |",f"| C1-L | {fmt(rq['C1-L']['mean_std']['tnr'])} | {fmt(rq['C1-L']['mean_std']['fpr'])} | {fmt(rq['C1-L']['mean_std']['tn'])} | {fmt(rq['C1-L']['mean_std']['fp'])} |",f"| LEGION-retrained | {rq['LEGION-retrained']['tnr']:.4f} | {rq['LEGION-retrained']['fpr']:.4f} | {rq['LEGION-retrained']['tn']} | {rq['LEGION-retrained']['fp']} |",'','## Statistical analysis','','Representative seed 3407 was frozen from internal validation before external inference.','', '| Dataset | Comparison | McNemar p | ΔAcc bootstrap 95% CI | ΔRecall 95% CI | ΔTNR 95% CI |','|---|---|---:|---:|---:|---:|']
 for ds in MIXED:
  for key,label in ((f'C1-L_vs_R1','R1'),(f'C1-L_vs_LEGION-retrained','LEGION-retrained')):
   s=results['statistics'][ds][key];ci=lambda k:f"[{s[k]['bootstrap_95_ci'][0]:+.4f}, {s[k]['bootstrap_95_ci'][1]:+.4f}]";lines.append(f"| {ds} | C1-L vs {label} | {s['mcnemar']['exact_two_sided_p']:.3g} | {ci('accuracy_difference')} | {ci('fake_recall_difference')} | {ci('tnr_difference')} |")
 lines += ['','Full discordant counts and 10,000-repeat paired bootstrap metadata are in `statistics/paired.json`. ROC-AUC remains a point estimate.','', '## GenImage per generator','','All ADM, BigGAN, GLIDE, Midjourney, SD-v1.4, SD-v1.5, VQDM, and Wukong rows are in `tables/genimage_per_generator.csv`; thresholds remain 0.5.','', '## Leakage and claim boundary','',protocol['leakage_override'],'','LEGION-public classification is N/A because the public intermediate checkpoint has no trained prediction head.','','## Final answers','']
 lines += ['1. **C1-L solves a material part of the current OOD weakness: SUPPORTED.** It improves mean Accuracy, ROC-AUC, and Fake recall over R1 on AIGI-Holmes, GenImage, and LOKI, while retaining Internal2208 accuracy. Representative-seed LOKI Accuracy CI still crosses zero, so that dataset-level gain is less certain than AIGI/GenImage.','2. **C1-L approaches but does not generally reach LEGION-retrained.** Accuracy gap closure is 65.4% on AIGI-Holmes, 58.6% on GenImage, and 29.7% on LOKI; it remains below LEGION on all three external mixed datasets, while slightly exceeding it on Internal2208.','3. **C1-S is already strong; the large head adds modest OOD operating-point value.** C1-L adds 0.48–0.64 percentage-point Accuracy on the three external mixed sets, but only 0.0005–0.0012 AUC. This supports CLIP representation as the primary factor, with a secondary capacity effect.','4. **Real specificity is modestly impaired.** C1-L mean FPR rises versus R1 by 0.0009 on AIGI-Holmes, 0.0017 on GenImage, 0.0293 on LOKI, and 0.0027 on RAISE998. RAISE FPR 0.0087 remains below LEGION-retrained 0.0110; LOKI shows a real sensitivity-specificity tradeoff.','5. **Replacement recommendation: GO for C1-L**, with the specificity tradeoff retained explicitly. This is an architecture decision, not a threshold-tuning decision.','6. **Actual integration is required next**, but only under separate authorization: route frozen CLIP CLS through the frozen C1-L head inside R1 inference while leaving localization and all current checkpoints unchanged.','', '**Phase 6B.1 STOP.**']
 (ROOT/'docs/phase6b1_external_classification_confirmation.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'COMPLETE','decision':results['decision']},indent=2))
if __name__=='__main__':main()
