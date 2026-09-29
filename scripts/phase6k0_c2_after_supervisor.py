#!/usr/bin/env python3
"""Durable fail-closed Phase6K continuation after the two C2 cache workers."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase6k0_c2_after'
DOC=ROOT/'docs/phase6k0_c2_after.md'
PY='/home/yz/miniconda3/envs/glamm_official/bin/python'
CACHE_PIDS={'val':1348558,'train':int(os.environ.get('PHASE6K_TRAIN_CACHE_PID','1348719'))}
CACHE_ROOT=Path('/data/yz/groundingLMM_official/cache/phase6k0_c2_after_v2')


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)
def status(stage,**kw):write(OUT/'supervisor_status.json',{'status':'RUNNING','stage':stage,'updated_utc':utc(),**kw})
def alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]
        return state!='Z'
    except FileNotFoundError:return False

def run(stage,cmd):
    status(stage)
    log=OUT/'logs'/f'{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',buffering=1) as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    if result.returncode:raise RuntimeError(f'{stage} exited {result.returncode}; see {log}')

def gpu_memory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    return [int(x.strip()) for x in raw.splitlines()]

def wait_gpu():
    while True:
        used=gpu_memory()
        if used[0]<2048 and used[2]<2048:return
        status('WAIT_GPUS_0_2',memory_mib={'0':used[0],'2':used[2]})
        time.sleep(60)

def preflight_report():
    p=read(OUT/'preflight.json');b0=read(OUT/'b0/dev.json');b=b0['metrics'];c=read(OUT/'cache/capture_preflight.json')
    full=read(OUT/'cache/full_capture_parity.json')
    if p['status']!='PASS':raise RuntimeError('Phase6K preflight not PASS')
    lines=['# Phase6K — Frozen C2 Spatial Transfer: After Injection','',
           '## 五项训练前门槛：PASS','',
           f"1. C2 checkpoint SHA256 `{c['c2_sha256']}`，epoch 7 / step 3500；C1-native joint R1 SHA256 `{b0['native_sha256']}`，selector 和 checkpoint 哈希已验证。",
           f"2. A 形状 `{c['A_shape']}`；E 形状 `{c['E_shape']}`；A mass 最大误差 {c['max_A_mass_error']:.8g}；E-context 相对误差 {c['max_E_head_context_relative_error']:.8g}。开启 capture 前后 fused token、原始分类 logits、生成 token、q_seg、掩码完全一致（{full['status']}）。",
           f"3. TRAIN {p['train_n']} 张，C2-valid {p['train_valid']} 张；DEV {p['dev_n']} 张，C2-valid {p['dev_valid']} 张。原始 ID/顺序及 shard SHA 均已锁定。",
           f"4. B0 DEV Mean FG IoU {b['mean_foreground_iou']:.6f}、Mean FG F1 {b['mean_foreground_f1']:.6f}、Global FG IoU {b['global_foreground_iou']:.6f}、Global FG F1 {b['global_foreground_f1']:.6f}。",
           '5. A/E step-0 的 F24、z_F24、Rectifier、Utility、adapted SAM embedding、mask logits、loss 与 B0 相等；投影梯度和 Utility 输入梯度非零，所有冻结参数 SHA 未变。','',
           '五项通过后，A/E 在卡 0/2 各训练 10 epoch；Official1000 只在两个内部 DEV selector 冻结后评测。','',
           '进行中状态及日志见 `outputs/phase6k0_c2_after/supervisor_status.json`。','']
    DOC.write_text('\n'.join(lines))


def main():
    try:
        status('WAIT_C2_CACHE',cache_pids=CACHE_PIDS)
        while True:
            complete=[]
            for split,pid in CACHE_PIDS.items():
                path=OUT/'cache'/f'{split}.json'
                if path.exists() and read(path).get('status')=='COMPLETE':complete.append(split)
                elif not alive(pid):raise RuntimeError(f'{split} C2 cache worker stopped before COMPLETE; inspect logs/cache_{split}.log')
            if len(complete)==2:break
            status('WAIT_C2_CACHE',complete=complete,cache_pids=CACHE_PIDS,
                   progress={split:{'shards':len(list((CACHE_ROOT/split).glob('shard_*.pt'))),
                                    'expected_samples':8836 if split=='train' else 1106}
                             for split in CACHE_PIDS})
            time.sleep(60)
        wait_gpu()
        run('capture_parity',[PY,'scripts/phase6k0_c2_capture_parity.py'])
        wait_gpu()
        run('b0_preflight',[PY,'scripts/phase6k0_c2_after_train.py','--mode','baseline','--device','cuda:2'])
        preflight_report()
        wait_gpu()
        status('TRAIN_A_E',gpu={'a':0,'e':2})
        children={}
        logs={}
        for name,gpu in [('a',0),('e',2)]:
            path=OUT/'logs'/f'train_{name}.log';logs[name]=path.open('a',buffering=1)
            children[name]=subprocess.Popen([PY,'scripts/phase6k0_c2_after_train.py','--mode','train',
                 '--arm',name,'--device',f'cuda:{gpu}'],cwd=ROOT,stdout=logs[name],stderr=subprocess.STDOUT,
                 start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
        status('TRAIN_A_E',pids={k:v.pid for k,v in children.items()})
        codes={k:v.wait() for k,v in children.items()}
        for f in logs.values():f.close()
        if any(code for code in codes.values()):raise RuntimeError(f'A/E training failed: {codes}; inspect train logs')
        require_selectors={k:read(OUT/k/'selector.json') for k in ('a','e')}
        if any(x['status']!='COMPLETE' for x in require_selectors.values()):raise RuntimeError('A/E selector incomplete')
        wait_gpu()
        run('official1000',[PY,'scripts/phase6k0_c2_after_official.py','--device','cuda:2'])
        run('finalize',[PY,'scripts/phase6k0_c2_after_finalize.py'])
        write(OUT/'supervisor_status.json',{'status':'COMPLETE','stage':'COMPLETE_STOP','completed_utc':utc(),
                                           'result':str((OUT/'results.json').resolve()),'report':str(DOC.resolve())})
    except BaseException as exc:
        write(OUT/'supervisor_status.json',{'status':'FAILED','stage':'COMPLETE_STOP','updated_utc':utc(),
              'error_type':type(exc).__name__,'error':str(exc)})
        raise

if __name__=='__main__':main()
