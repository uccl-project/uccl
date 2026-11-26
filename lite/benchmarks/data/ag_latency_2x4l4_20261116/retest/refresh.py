#!/usr/bin/env python3
"""Fresh single-size NCCL runs to investigate the 256/512-KiB spikes."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent/'refresh'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=OUT)
OUT=parser.parse_args().out.resolve()
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'raw').mkdir(parents=True,exist_ok=True)
BINARY=ROOT/'build/ag_latency_2x4l4_20261116/uccl_lite_nccl_2x4l4_baseline'
MPI='/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun'
rows=[]
for repeat in range(1,4):
    sizes=[None]
    if repeat==2:sizes.reverse()
    if repeat==3:random.Random(20261116).shuffle(sizes)
    for size in sizes:
        for mode in (('default','sm')if repeat!=2 else ('sm','default')):
            name=f'{repeat}_{mode}_full'
            env={k:v for k,v in os.environ.items()if not k.startswith(('NCCL_','UCX_','MSCCLPP_','CUDA_VISIBLE_DEVICES'))}
            env.update(NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',CUDA_VISIBLE_DEVICES='0,1,2,3',NCCL_DEBUG='INFO')
            if mode=='sm':env.update(NCCL_P2P_USE_CUDA_MEMCPY='0',NCCL_SHM_USE_CUDA_MEMCPY='0',NCCL_PROTO='Simple')
            cmd=[MPI,'--host','10.10.55.2:2,10.10.55.1:2','-np','4','--map-by','ppr:2:node','--bind-to','none','--mca','pml','ob1','--mca','btl','self,vader,tcp','--mca','btl_tcp_if_include','ibp55s0f0']
            for k in sorted(env):
                if k.startswith(('NCCL_','CUDA_VISIBLE_DEVICES')):cmd+=['-x',k]
            cmd+=[str(BINARY),'--op','allgather','--gpus-per-process','2','-b','1024','-e','1048576','-w','100','-n','200']
            log=OUT/'raw'/f'{name}.log';assert not log.exists(),log
            print(name+' starting',flush=True)
            with log.open('w')as f:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
            (OUT/'raw'/f'{name}.json').write_text(json.dumps({'command':cmd,'environment':{k:v for k,v in env.items()if k.startswith(('NCCL_','CUDA_VISIBLE_DEVICES'))},'returncode':p.returncode,'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
            assert p.returncode==0,log
            values={n:[] for n in (1<<p for p in range(10,21))}
            for line in log.read_text().splitlines():
                s=line.split(',')
                if len(s)==6 and s[:2]==['nccl','timed']:
                    assert int(s[5])==0
                    v=float(s[4]);assert math.isfinite(v)and v>0;values[int(s[3])].append(v)
            assert all(len(v)==200 for v in values.values())
            for size,v in values.items():
                row={'mode':mode,'repeat':repeat,'bytes':size,'mean_us':statistics.mean(v),'median_us':statistics.median(v),'min_us':min(v),'p95_us':sorted(v)[189],'max_us':max(v)}
                rows.append(row);print(json.dumps(row),flush=True)
with (OUT/'runs.csv').open('w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
