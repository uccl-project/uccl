#!/usr/bin/env python3
"""Fresh fixed-configuration, fully checked 1-KiB–1-MiB AllGather sweep."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent
BUILD = ROOT / 'build/ag_latency_2x4l4_20261116'
MPI = '/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun'
SIZES = [1 << p for p in range(10, 21)]
CONFIGS = {'uccl': {}, 'prior': {},
           'sm': {'NCCL_P2P_USE_CUDA_MEMCPY': '0', 'NCCL_SHM_USE_CUDA_MEMCPY': '0'},
           'ce': {'NCCL_P2P_USE_CUDA_MEMCPY': '1', 'NCCL_SHM_USE_CUDA_MEMCPY': '0',
                  'NCCL_P2P_LEVEL': 'SYS', 'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple'}}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=DATA)
OUT = parser.parse_args().out.resolve()
OUT.mkdir(parents=True,exist_ok=True)
OUT.joinpath('raw').mkdir(exist_ok=True)
records = []
for repeat in range(1, 4):
    order = list(CONFIGS) if repeat != 2 else list(reversed(CONFIGS))
    for series in order:
        name = f'{repeat}_{series}'
        path = OUT / 'raw' / f'{name}.log'
        assert not path.exists(), path
        env = {k:v for k,v in os.environ.items() if not k.startswith(('NCCL_', 'UCX_', 'MSCCLPP_', 'CUDA_VISIBLE_DEVICES'))}
        env.update(CONFIGS[series], NCCL_SOCKET_IFNAME='ibp55s0f0', NCCL_IB_HCA='mlx5_0,mlx5_1',
                   CUDA_VISIBLE_DEVICES='0,1,2,3', NCCL_DEBUG='INFO')
        command = [MPI, '--host', '10.10.55.2:2,10.10.55.1:2', '-np', '4',
                   '--map-by', 'ppr:2:node', '--bind-to', 'none']
        if series == 'uccl':
            command += ['--mca','pml','ucx','--mca','coll','^hcoll']
        else:
            command += ['--mca','pml','ob1','--mca','btl','self,vader,tcp',
                        '--mca','btl_tcp_if_include','ibp55s0f0']
        for key in sorted(env):
            if key.startswith(('NCCL_', 'CUDA_VISIBLE_DEVICES')): command += ['-x',key]
        if series == 'uccl':
            command += [str(BUILD/'uccl_lite_multinode_host_allgather'), '-b','1024','-e','1048576',
                        '-w','20','-n','50','-r','1','--nccl','0','-k','262144',
                        '--download','sm','--upload','dma','--local-input','self','--early-consume','1',
                        '--gpu-workers','0','--progress','ready','--local-exchange','shared']
        else:
            command += [str(BUILD/'uccl_lite_nccl_2x4l4_baseline'),'--op','allgather',
                        '--gpus-per-process','2','-b','1024','-e','1048576','-w','20','-n','50']
        print(name + ' starting',flush=True)
        started = datetime.now(timezone.utc).isoformat()
        with path.open('w') as stream:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        entry = dict(series=series,repeat=repeat,command=command,
                     environment={k:v for k,v in env.items()if k.startswith(('NCCL_','CUDA_VISIBLE_DEVICES'))},
                     started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode)
        (OUT/'raw'/f'{name}.json').write_text(json.dumps(entry,indent=2)+'\n')
        assert result.returncode == 0,path
        values = {n:[]for n in SIZES}
        for line in path.read_text().splitlines():
            fields = line.split(',')
            if series == 'uccl' and len(fields)==8 and fields[:2]==['# sample','allgather']:
                n,value,error=int(fields[2]),float(fields[6]),int(fields[7])
            elif series != 'uccl' and len(fields)==6 and fields[:2]==['nccl','timed']:
                n,value,error=int(fields[3]),float(fields[4]),int(fields[5])
            else: continue
            assert error==0 and math.isfinite(value) and value>0,(path,line)
            values[n].append(value)
        assert all(len(v)==50 for v in values.values()),(path,{k:len(v)for k,v in values.items()})
        for n,v in values.items():
            records.append(dict(series=series,repeat=repeat,bytes=n,mean_us=statistics.mean(v)))
        print(name+' PASS',flush=True)
with (OUT/'run_means.csv').open('w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=['series','repeat','bytes','mean_us']);w.writeheader();w.writerows(records)
with (OUT/'summary.csv').open('w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=['series','bytes','median_us','min_us','max_us']);w.writeheader()
    for s in CONFIGS:
        for n in SIZES:
            v=[r['mean_us']for r in records if r['series']==s and r['bytes']==n];assert len(v)==3
            w.writerow(dict(series=s,bytes=n,median_us=statistics.median(v),min_us=min(v),max_us=max(v)))
