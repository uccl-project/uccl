#!/usr/bin/env python3
"""NCCL-only calibration/validation using the existing eight-L4 figure harness."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'build/nccl_tuned_testbeds12_20261115'
BINARY = OUT / 'uccl_lite_compact_collectives'
p = argparse.ArgumentParser()
p.add_argument('stage', choices=['pilot', 'pilot2', 'calibrate', 'refine', 'evaluate', 'trace'])
a = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
ENV = {k: os.environ[k] for k in ['HOME', 'PATH', 'LANG'] if k in os.environ}
ENV.update(LD_LIBRARY_PATH=str(ROOT/'build/nccl-2.28.9/lib')+':/usr/local/cuda-12.9/lib64',
           CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', NCCL_NET='Socket',
           NCCL_NET_PLUGIN='none', NCCL_IB_DISABLE='1', NCCL_DEBUG='WARN')
SM = {'NCCL_SHM_USE_CUDA_MEMCPY':'0','NCCL_P2P_USE_CUDA_MEMCPY':'0'}
CE = {'NCCL_SHM_USE_CUDA_MEMCPY':'1','NCCL_P2P_USE_CUDA_MEMCPY':'1',
      'NCCL_PROTO':'Simple'}

def active():
    return {int(s) for s in subprocess.check_output([
        'nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
        if s.strip().isdigit() and Path('/proc',s.strip()).exists()}

def config(engine, **settings):
    return dict(engine=engine, environment={**(CE if engine=='ce' else SM), **settings})

def channels(n):
    return {'NCCL_MIN_NCHANNELS':str(n),'NCCL_MAX_NCHANNELS':str(n)}

configs={}
for mode in ['gpu','shm']:
    configs[mode]={
        'sm_default':config('sm'),
        'sm_simple':config('sm',NCCL_PROTO='Simple'),
        'sm_ch8':config('sm',**channels(8)),
        'sm_buffer1m_ch4':config('sm',NCCL_BUFFSIZE=str(1<<20),**channels(4)),
        'ce_send':config('ce',NCCL_SHM_MEMCPY_MODE='1'),
        'ce_recv':config('ce',NCCL_SHM_MEMCPY_MODE='2'),
        'ce_both':config('ce',NCCL_SHM_MEMCPY_MODE='3'),
        'ce_send_ch8':config('ce',NCCL_SHM_MEMCPY_MODE='1',**channels(8)),
    }
for key in ['sm_default','sm_simple','ce_send','ce_recv','ce_both']:
    cfg=configs['gpu'][key]
    configs['gpu'][key+'_p2p']=dict(engine=cfg['engine'],environment={**cfg['environment'],'NCCL_P2P_LEVEL':'SYS'})

for mode in ['gpu','shm']:
    for key in list(configs[mode]):
        if key.startswith('ce_') or key=='sm_default':
            cfg=configs[mode][key]
            configs[mode][key+'_affinity']=dict(engine=cfg['engine'],environment={**cfg['environment'],'NCCL_IGNORE_CPU_AFFINITY':'1'})

jobs=[]
if a.stage=='pilot':
    for mode in ['gpu','shm']:
        for name in ['sm_default','ce_send','ce_recv','ce_both']+(['sm_default_p2p','ce_recv_p2p'] if mode=='gpu' else []):
            jobs.append(dict(name=f'pilot_{mode}_{name}',testbed=mode,config=name,op='ag',begin=1024,end=1<<30))
elif a.stage=='pilot2':
    for mode in ['gpu','shm']:
        for name in ['sm_default_affinity','ce_send_affinity','ce_recv_affinity','ce_both_affinity']+(['ce_recv_p2p_affinity'] if mode=='gpu' else []):
            jobs.append(dict(name=f'pilot2_{mode}_{name}',testbed=mode,config=name,op='ag',begin=1024,end=1<<30))
elif a.stage=='calibrate':
    (OUT/'candidates.json').write_text(json.dumps(configs,indent=2)+'\n')
    for mode in ['gpu','shm']:
        keys=[k for k,c in configs[mode].items() if c['engine']=='sm' or k.endswith('_affinity')];random.Random(202611150 if mode=='gpu' else 202611151).shuffle(keys)
        for key in keys:
            for op in ['ar','ag','rs']:
                jobs.append(dict(name=f'cal_{mode}_{key}_{op}',testbed=mode,config=key,op=op,begin=1024,end=1<<30))
elif a.stage=='refine':
    configs=json.loads((OUT/'candidates.json').read_text())
    jobs=json.loads((OUT/'refine_jobs.json').read_text())
else:
    configs=json.loads((OUT/'candidates.json').read_text())
    segments=json.loads((OUT/'segments.json').read_text())
    if a.stage=='evaluate':
        schedule=json.loads((OUT/'schedule.json').read_text())
        groups={}
        for row in schedule:
            key=(row['testbed'],row['op'],row['engine'],row['config'])
            if key not in groups:
                groups[key]=dict(testbed=row['testbed'],op=row['op'],engine=row['engine'],config=row['config'],begin=1024,end=row['bytes'],plot_sizes=[])
            groups[key]['end']=max(groups[key]['end'],row['bytes'])
            groups[key]['plot_sizes'].append(row['bytes'])
        validations=list(groups.values())
        for repeat in range(2):
            # Every configuration starts at 1 KiB, even when only its large
            # sizes were selected. Prefix samples are retained, not selected.
            ordered=validations if repeat==0 else list(reversed(validations))
            jobs += [dict(s,name=f'eval_{repeat}_{i:03d}',run=repeat) for i,s in enumerate(ordered)]
    else:
        trace_groups={}
        for s in segments:
            key=(s['testbed'],s['config'],s['op'])
            if key not in trace_groups:
                trace_groups[key]=dict(s,name=f'trace_{len(trace_groups):03d}',begin=1024)
            trace_groups[key]['end']=max(trace_groups[key]['end'],s['end'])
        jobs=list(trace_groups.values())

if a.stage in ['evaluate','trace']:
    fingerprints={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                  for name in ['candidates.json','schedule.json','segments.json']}
    inputs=OUT/(a.stage+'_inputs.json')
    if inputs.exists():
        assert json.loads(inputs.read_text())==fingerprints,'frozen selection changed'
    else:inputs.write_text(json.dumps(fingerprints,indent=2)+'\n')

record_path=OUT/(a.stage+'_runs.json')
records=json.loads(record_path.read_text()) if record_path.exists() else []
completed={r['name'] for r in records}
with open('/tmp/uccl-lite-integration-wxz-l4.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if active():raise RuntimeError('GPU host occupied')
    if not (OUT/'hardware.json').exists():
        info={}
        for key,cmd in [('gpu',['nvidia-smi','-q']),('topology',['nvidia-smi','topo','-m']),('cpu',['lscpu']),('compiler',['/usr/local/cuda-12.9/bin/nvcc','--version'])]:
            info[key]=subprocess.check_output(cmd,text=True)
        info['binary_sha256']=hashlib.sha256(BINARY.read_bytes()).hexdigest()
        lib=(ROOT/'build/nccl-2.28.9/lib/libnccl.so.2').resolve()
        info['nccl_library']=str(lib);info['nccl_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
        info['source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (OUT/'hardware.json').write_text(json.dumps(info,indent=2)+'\n')
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest()==json.loads((OUT/'hardware.json').read_text())['binary_sha256']
    for idx,job in enumerate(jobs,1):
        if job['name'] in completed:continue
        setting=configs[job['testbed']][job['config']]
        rejected=next((r for r in records if r.get('timed_out') and
                       configs[r['testbed']][r['config']]['environment']==setting['environment']),None)
        if rejected and a.stage in ['calibrate','refine']:
            records.append(dict(**job,environment=setting['environment'],correctness_pass=False,
                                returncode=None,timed_out=False,skipped=True,
                                reason='same setting already stalled; excluded from baseline',
                                rejection_evidence=rejected['name']))
            record_path.write_text(json.dumps(records,indent=2)+'\n')
            print(f'{idx}/{len(jobs)} SKIP {job["name"]}: same unstable setting',flush=True)
            continue
        if active():raise RuntimeError('External GPU work detected')
        env={**ENV,**configs[job['testbed']][job['config']]['environment']}
        if job['testbed']=='shm':env.update(NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0')
        warm,reps,iters={'pilot':(1,1,1),'pilot2':(1,1,1),'calibrate':(3,3,3),'refine':(3,3,3),'evaluate':(5,3,10),'trace':(1,1,1)}[a.stage]
        if a.stage in ['pilot','pilot2','trace']:
            env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,ENV,TUNING,P2P,SHM',NCCL_DEBUG_FILE=str(OUT/(job['name']+'_debug.log')))
        cmd=[str(BINARY),'--gpus','8','--op',job['op'],'--variants','nccl',
             '--begin',str(job['begin']),'--end',str(job['end']),
             '--warmup',str(warm),'--repeats',str(reps),'--iterations',str(iters)]
        if job['testbed']=='shm':cmd+=['--shm-only']
        print(f'{idx}/{len(jobs)} START {job["name"]}',flush=True)
        started=datetime.now(timezone.utc).isoformat();start=time.monotonic()
        timed_out=False;foreign=[]
        with (OUT/(job['name']+'.csv')).open('x') as stdout,(OUT/(job['name']+'.log')).open('x') as stderr:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            try:
                while child.poll() is None:
                    foreign=sorted(active()-{child.pid})
                    timed_out=time.monotonic()-start>180
                    if foreign or timed_out:
                        child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(1)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        rows=(OUT/(job['name']+'.csv')).read_text().splitlines()
        samples=sum(s.startswith('sample,') for s in rows)
        medians=sum(s.startswith('median,') for s in rows)
        sizes=(job['end']//job['begin']).bit_length()
        passed=child.returncode==0 and not timed_out and not foreign and samples==sizes*reps*iters and medians==sizes*reps
        rec=dict(**job,command=cmd,environment={k:v for k,v in env.items() if k.startswith(('NCCL_','CUDA_','LD_LIBRARY_PATH'))},started_utc=started,seconds=time.monotonic()-start,returncode=child.returncode,timed_out=timed_out,foreign_gpu_users=foreign,correctness_pass=passed,samples=samples,medians=medians)
        records.append(rec);record_path.write_text(json.dumps(records,indent=2)+'\n')
        print(f'DONE {job["name"]} {"PASS" if passed else "FAIL"} {rec["seconds"]:.1f}s',flush=True)
        if foreign:raise RuntimeError('External GPU work detected')
        if not passed and a.stage in ['evaluate','trace']:raise RuntimeError('Validation failed: '+job['name'])
    if active():raise RuntimeError('GPU work remains active')
