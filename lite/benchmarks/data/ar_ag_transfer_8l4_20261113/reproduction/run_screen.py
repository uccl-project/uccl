from run_sweeps import *
binary=OUT/'collectives_before'
variants=[('cpu',[]),('cpu_parallel',['--gpu-launch-workers-per-node','4']),('gpu',['--gpu-progress']),('gpu_parallel',['--gpu-progress','--gpu-launch-workers-per-node','4']),('compact',['--workers-per-node','4','--transport-blocks','16'])]
for op in ['allreduce','allgather']:
 for name,extra in variants:
  run('screen_'+op+'_'+name,binary,['--main-cpu','47']+extra,op=op)
