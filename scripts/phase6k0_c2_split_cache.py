#!/usr/bin/env python3
"""Finish Phase6K TRAIN C2 cache on GPUs 0/2 without changing canonical order."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase4hb_progressive_unfreezing_stage1 import train_ids
from tools.phase4c_b import file_sha256

OUT = ROOT / 'outputs/phase6k0_c2_after'
CACHE = Path('/data/yz/groundingLMM_official/cache/phase6k0_c2_after_v2')
PY = '/home/yz/miniconda3/envs/glamm_official/bin/python'
HEAD_PID = 1348719
CUT = 5200
SHARD_SIZE = 100


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def status(stage, **kw):
    write(OUT/'cache/split_controller_status.json',
          {'status': 'RUNNING', 'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **kw})


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]
        return state != 'Z'
    except FileNotFoundError:
        return False


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def main():
    ids = train_ids()
    require(len(ids) == 8836, 'TRAIN population drift')
    require(alive(HEAD_PID), 'GPU0 head cache worker is not running')
    require(not (OUT/'cache/train.json').exists(), 'TRAIN cache already finalized')
    head = CACHE/'train'
    tail = CACHE/'train_tail'
    tail.mkdir(parents=True, exist_ok=True)
    require(not list(tail.glob('shard_*.pt')), 'unexpected existing tail shards')
    log_path = OUT/'logs/cache_train_gpu2.log'
    with log_path.open('a', buffering=1) as log:
        worker = subprocess.Popen(
            [PY, 'scripts/phase6k0_c2_after_cache.py', '--split', 'train',
             '--device', 'cuda:2', '--batch-size', '4', '--start-index', str(CUT),
             '--end-index', str(len(ids)), '--output-subdir', 'train_tail',
             '--summary-name', 'train_tail'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'})
        write(OUT/'cache/split_assignment.json', {
            'head_gpu': 0, 'head_pid': HEAD_PID, 'head_range': [0, CUT],
            'tail_gpu': 2, 'tail_pid': worker.pid, 'tail_range': [CUT, len(ids)],
            'already_completed_at_split': len(list(head.glob('shard_*.pt')))*SHARD_SIZE,
            'tail_log': str(log_path.resolve())})
        status('CACHE_PARALLEL', head_pid=HEAD_PID, tail_pid=worker.pid,
               head_end=CUT, tail_start=CUT)
        head_stopped = False
        while True:
            head_shards = len(list(head.glob('shard_*.pt')))
            if not head_stopped and head_shards >= CUT//SHARD_SIZE:
                if alive(HEAD_PID):
                    os.kill(HEAD_PID, signal.SIGTERM)
                    for _ in range(30):
                        if not alive(HEAD_PID):
                            break
                        time.sleep(1)
                    require(not alive(HEAD_PID), 'GPU0 worker did not stop at split boundary')
                head_stopped = True
            if not head_stopped:
                require(alive(HEAD_PID), f'GPU0 worker exited before {CUT}')
            code = worker.poll()
            require(code is None or code == 0, f'GPU2 tail worker exited {code}; see {log_path}')
            if head_stopped and code == 0:
                break
            status('CACHE_PARALLEL', head_shards=head_shards,
                   tail_shards=len(list(tail.glob('shard_*.pt'))),
                   head_pid=HEAD_PID, tail_pid=worker.pid, head_stopped=head_stopped)
            time.sleep(15)

    tail_status = json.loads((OUT/'cache/train_tail.json').read_text())
    require(tail_status['status'] == 'COMPLETE' and tail_status['range_start'] == CUT and
            tail_status['range_end'] == len(ids), 'GPU2 tail summary incomplete')
    paths = [head/f'shard_{i:06d}.pt' for i in range(CUT//SHARD_SIZE)]
    paths += [Path(s['path']) for s in tail_status['shards']]
    require(all(p.exists() for p in paths), 'split shard missing')
    offset = 0
    counts = [0, 0, 0]
    sha = None
    specs = []
    for path in paths:
        shard = torch.load(path, map_location='cpu', weights_only=False)
        these_ids = shard['sample_ids']
        require(these_ids == ids[offset:offset+len(these_ids)], f'cache ID order drift: {path}')
        require(shard['A'].dtype == torch.float32 and shard['E'].dtype == torch.bfloat16,
                f'cache dtype drift: {path}')
        sha = shard['c2_sha256'] if sha is None else sha
        require(shard['c2_sha256'] == sha, f'C2 checkpoint drift: {path}')
        require(len(these_ids) == len(shard['seg_count']), f'cache record count drift: {path}')
        for n in shard['seg_count'].tolist():
            counts[0 if n == 1 else 1 if n == 0 else 2] += 1
        offset += len(these_ids)
        specs.append({'path': str(path.resolve()), 'sha256': file_sha256(path)})
    require(offset == len(ids) and sha == tail_status['c2_sha256'], 'split cache coverage/SHA drift')
    summary = {
        'status': 'COMPLETE', 'schema': 'phase6k0_c2_cache_v1', 'split': 'train',
        'c2_sha256': sha, 'n': len(ids),
        'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'valid_exactly_one_seg': counts[0], 'invalid_no_seg': counts[1],
        'invalid_multiple_seg': counts[2], 'shards': specs,
        'split_gpu': {'0': [0, CUT], '2': [CUT, len(ids)]},
        'firewall': {'internal_test': False, 'official1000': False, 'external_ood': False}}
    write(OUT/'cache/train.json', summary)
    write(OUT/'cache/split_controller_status.json', {
        'status':'COMPLETE', 'stage':'CACHE_MERGED', 'n':len(ids),
        'head_shards_used':CUT//SHARD_SIZE, 'tail_shards_used':len(tail_status['shards'])})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(OUT/'cache/split_controller_status.json',
              {'status':'FAILED', 'error_type':type(exc).__name__, 'error':str(exc)})
        raise
