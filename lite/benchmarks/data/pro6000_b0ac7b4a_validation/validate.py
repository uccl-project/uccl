import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=False)
SRC = ROOT / 'source/experimental/lite'
ENV = {k: v for k, v in os.environ.items() if not k.startswith(('NCCL_', 'UCCL_LITE_'))}
ENV.update(CUDA_VISIBLE_DEVICES='0,1', LD_LIBRARY_PATH=CFG['cuda']+'/lib:'+CFG['nccl']+'/lib')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def active():
    rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True)
    return {int(p) for p in rows.splitlines() if p.strip().isdigit() and Path('/proc',p.strip()).exists()}

def run(name, argv, gpu=False, extra=None, timeout=1200, cwd=None):
    if gpu and active():
        raise RuntimeError('External GPU work present; no launch: '+str(active()))
    argv = list(map(str, argv))
    metadata = dict(argv=argv, environment_overrides=extra or {}, start_utc_epoch=time.time())
    (OUT/(name+'.command.json')).write_text(json.dumps(metadata,indent=2))
    started = time.monotonic()
    print('START',name,flush=True)
    with (OUT/(name+'.log')).open('w') as log:
        p = subprocess.Popen(argv, cwd=cwd or SRC, env=ENV | (extra or {}), stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            other = active()-{p.pid} if gpu else set()
            if other or time.monotonic()-started > timeout:
                p.terminate();p.wait(timeout=15)
                raise RuntimeError('Run excluded (external GPU work/timeout): '+name+' '+str(other))
            time.sleep(1)
    metadata.update(returncode=p.returncode,seconds=time.monotonic()-started)
    (OUT/(name+'.status.json')).write_text(json.dumps(metadata,indent=2))
    print('DONE',name,p.returncode,round(metadata['seconds'],1),'s',flush=True)
    if p.returncode:
        raise RuntimeError('Failed '+name)

def nccl_trace(name):
    return dict(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,ENV,GRAPH,P2P,SHM,NET',
                NCCL_DEBUG_FILE=str(OUT/(name+'.nccl.%p.log')))

def verify_transport(name):
    lines = [line for p in OUT.glob(name+'.nccl.*.log') for line in p.read_text().splitlines()
             if 'Channel ' in line and ' via ' in line]
    result = dict(default_policy=True,lines=lines,p2p_only=bool(lines) and all('via P2P' in line for line in lines))
    (OUT/(name+'.transport.json')).write_text(json.dumps(result,indent=2))
    assert result['p2p_only'], 'NCCL default P2P not established'
    print('VERIFIED',name,len(lines),'P2P edges; no SHM/NET data edges',flush=True)

def bench(name, mode, reverse=False, calibrate=False):
    binary = SRC/'build'/('uccl_lite_joint_nccl_reference' if mode=='nccl' else 'uccl_lite_joint_bench')
    warm,reps,iters = (3,3,5) if calibrate else (100,5,20)
    argv = [binary,'--backend',mode,'--gpus',2,'--begin',1024,'--end',1073741824,'--op','all',
            '--warmup',warm,'--repeats',reps,'--iters',iters,'--samples',OUT/(name+'.samples.csv'),
            '--source-identity',CFG['revision'],'--seed',314159 if calibrate else 20261115]
    if mode in ['calibrate','joint']:
        argv += ['--profile',OUT/'calibration.profile']
    if reverse:
        argv.append('--reverse')
    run(name,argv,gpu=True,extra=nccl_trace(name) if mode=='nccl' else None,timeout=2400)
    assert 'correctness: PASS' in (OUT/(name+'.log')).read_text()
    if mode=='nccl':
        verify_transport(name)
    if not calibrate:
        rows=list(csv.DictReader((OUT/(name+'.samples.csv')).open()))
        assert len(rows)==6300
        choices={r['selected_backend'] for r in rows}
        if mode=='peer': assert choices=={'peer'}
        print('ROUTES',name,sorted(choices),flush=True)

with open(CFG['lock'],'a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if active():raise RuntimeError('GPU node busy')
    assert sha(ROOT/'source.tar.gz')==CFG['source_archive_sha256']
    (ROOT/'source').mkdir()
    run('extract',['tar','-xzf',ROOT/'source.tar.gz','-C',ROOT/'source'],cwd=ROOT)
    for name,args in [('topology',['nvidia-smi','topo','-m']),('peer-access',['nvidia-smi','topo','-p2p','r']),
                      ('gpu-identity',['nvidia-smi','-q']),('cpu-identity',['lscpu']),('compiler',[CFG['cuda']+'/bin/nvcc','--version'])]:
        run(name,args)
    targets=['joint-bench','joint-nccl-reference','router-policy-check','nccl-identity-check']
    targets += ['build/uccl_lite_'+n for n in ['cpu_bf16_check','router_core_check','dispatcher_check','router_check','router_graph_check']]
    run('build',['make','-j2']+targets+['CUDA_HOME='+CFG['cuda'],'CUDA_ARCH=sm_120','CUTLASS_HOME='+CFG['cutlass'],
                                     'NCCL_HOME='+CFG['nccl'],'EXTRA_NVCC_FLAGS=-L'+CFG['cuda']+'/lib',
                                     'CXX=g++ -L'+CFG['cuda']+'/lib'])
    for name in ['router_policy_check','nccl_identity_check','cpu_bf16_check','router_core_check','dispatcher_check']:
        run(name,[SRC/'build'/('uccl_lite_'+name)])
    run('router-eager',[SRC/'build/uccl_lite_router_check','--gpus',2,'--cpu-interleave-scatter','--packet-bytes',4194304,'--delay-rank',1],gpu=True)
    run('router-graph',[SRC/'build/uccl_lite_router_graph_check','--gpus',2,'--cpu-interleave-scatter','--packet-bytes',4194304,'--large'],gpu=True)
    run('joint-selftest',[SRC/'build/uccl_lite_joint_bench','--selftest','--gpus',2,'--end',4096,'--source-identity',CFG['revision']],gpu=True)
    run('nccl-transport',[SRC/'build/uccl_lite_joint_nccl_reference','--backend','nccl','--gpus',2,'--begin',1048576,'--end',1048576,
                          '--warmup',2,'--repeats',1,'--iters',2],gpu=True,extra=nccl_trace('nccl-transport'))
    verify_transport('nccl-transport')
    files = [p for d in ['src','include','tests','benchmarks/joint'] for p in (SRC/d).rglob('*') if p.is_file()]
    files += [SRC/'Makefile',SRC/'benchmarks/cpu_router/nccl_identity.hpp',SRC/'benchmarks/data/primitives_8l4.csv']
    provenance=dict(config=CFG,source_files={str(p.relative_to(SRC)):sha(p) for p in files},
                    binaries={name:sha(SRC/'build'/name) for name in ['uccl_lite_joint_bench','uccl_lite_joint_nccl_reference']},
                    nccl_library_sha256=sha(Path(CFG['nccl'])/'lib/libnccl.so'))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    bench('calibration','calibrate',calibrate=True)
    for name,mode,reverse in [('nccl-forward','nccl',False),('peer-forward','peer',False),('joint-forward','joint',False),
                              ('joint-reverse','joint',True),('peer-reverse','peer',True),('nccl-reverse','nccl',True)]:
        bench(name,mode,reverse)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,timed_primary_invocations=37800,
                                                   calibration_profile_sha256=sha(OUT/'calibration.profile')),indent=2))
