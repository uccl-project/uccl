#!/usr/bin/env python3
"""Protocol selection only: diagnostic logging is excluded from plotted timing."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent/'protocol_trace';OUT.mkdir(exist_ok=True)
MPI='/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun'
B=ROOT/'build/ag_latency_2x4l4_20261116/uccl_lite_nccl_2x4l4_baseline'
for mode in ('default','sm','simple'):
 for size in (131072,262144,524288,1048576):
  env={k:v for k,v in os.environ.items()if not k.startswith(('NCCL_','UCX_','MSCCLPP_','CUDA_VISIBLE_DEVICES'))}
  env.update(NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',CUDA_VISIBLE_DEVICES='0,1,2,3',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='TUNING,ENV')
  if mode=='sm':env.update(NCCL_P2P_USE_CUDA_MEMCPY='0',NCCL_SHM_USE_CUDA_MEMCPY='0')
  if mode=='simple':env['NCCL_PROTO']='Simple'
  cmd=[MPI,'--host','10.10.55.2:2,10.10.55.1:2','-np','4','--map-by','ppr:2:node','--bind-to','none','--mca','pml','ob1','--mca','btl','self,vader,tcp','--mca','btl_tcp_if_include','ibp55s0f0']
  for k in sorted(env):
   if k.startswith(('NCCL_','CUDA_VISIBLE_DEVICES')):cmd+=['-x',k]
  cmd+=[str(B),'--op','allgather','--gpus-per-process','2','-b',str(size),'-e',str(size),'-w','5','-n','10']
  log=OUT/f'{mode}_{size}.log';assert not log.exists()
  with log.open('w')as f:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
  assert p.returncode==0,log
  lines=[l for l in log.read_text().splitlines()if 'Bytes -> Algo' in l]
  assert lines,log
  print(mode,size,lines[-1],flush=True)
