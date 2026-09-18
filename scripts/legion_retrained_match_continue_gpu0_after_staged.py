#!/usr/bin/env python3
"""Resume only unfinished LEGION-match evaluations on physical GPU 0."""
from __future__ import annotations
import json
from pathlib import Path

from legion_retrained_match_evaluate_pipeline import (
    ROOT,OUT,STATUS,atomic,build_internal_fake_manifest,now,run_pair,shared_job,
    summarize_shared,
)

def complete(path: Path) -> bool:
    try:return json.loads(path.read_text()).get('status')=='COMPLETE'
    except Exception:return False

def main():
    state={'status':'RUNNING','stage':'resume_after_phase6f4_staged_complete',
           'remaining_gpu':0,'started_at_utc':now()};atomic(STATUS,state)
    internal=build_internal_fake_manifest()
    official=ROOT/'outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl'
    loki=ROOT/'datasets/LOKI/legion_localization/manifest.jsonl'
    core={'official1000':summarize_shared('official1000',official),
          'internal_test':summarize_shared('internal_test',internal)}
    if not complete(OUT/'localization/loki229/results.json'):
        run_pair([shared_job('loki229',0,loki,'loki_bbox_union',229)],state,'localization_ood_loki_gpu0')
    if not complete(OUT/'localization/legion_retrained/xaigd/results.json'):
        extra=str(ROOT/'scripts/legion_retrained_match_localization_extra.py')
        run_pair([('xaigd',0,[str(Path('/home/yz/miniconda3/envs/legion/bin/python')),extra,
                              '--dataset','xaigd','--device','cuda:0'])],state,'localization_ood_xaigd_gpu0')
    if not complete(OUT/'localization/legion_retrained/pal4vst/results.json'):
        extra=str(ROOT/'scripts/legion_retrained_match_localization_extra.py')
        run_pair([('pal4vst',0,[str(Path('/home/yz/miniconda3/envs/legion/bin/python')),extra,
                                '--dataset','pal4vst','--device','cuda:0'])],state,'localization_ood_pal4vst_gpu0')
    localization={**core,'loki229':summarize_shared('loki229',loki),
        'xaigd':json.loads((OUT/'localization/legion_retrained/xaigd/results.json').read_text()),
        'pal4vst':json.loads((OUT/'localization/legion_retrained/pal4vst/results.json').read_text())}
    classification={name:json.loads((OUT/'classification'/name/'results.json').read_text())
                    for name in ('internal','aigi_holmes','genimage','loki','raise998')}
    atomic(OUT/'results.json',{'status':'COMPLETE','model':'legion_retrained_match',
        'classification':classification,'localization':localization,
        'rescheduling':{'phase6f4_staged_completion_gate':True,
                        'unfinished_remainder_physical_gpu':0}})
    state.update(status='COMPLETE',stage='STOP',completed_at_utc=now());atomic(STATUS,state)

if __name__=='__main__':main()
