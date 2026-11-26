from run_sweeps import *
binary=OUT/'collectives_before'
for op in ['allreduce','allgather']:
 for name,extra in [('single_cta',['--gpu-progress','--gpu-launch-workers-per-node','4','--single-cta-bytes','8192']),('per_source',['--ag-per-source','--gpu-launch-workers-per-node','4'])] if op=='allgather' else [('single_cta',['--gpu-progress','--gpu-launch-workers-per-node','4','--single-cta-bytes','8192'])]:
  run('screen_'+op+'_'+name,binary,['--main-cpu','47']+extra,op=op)
