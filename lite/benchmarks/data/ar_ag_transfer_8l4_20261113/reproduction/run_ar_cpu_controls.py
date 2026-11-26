from run_sweeps import *
variants=[('cpu_parallel',[]),('cpu_parallel_compact',['--workers-per-node','4','--transport-blocks','16'])]
for suffix,order in [('a',variants),('b',list(reversed(variants)))]:
 for name,extra in order:
  run('final_allreduce_'+name+'_'+suffix,OUT/'uccl_lite_collectives',['--main-cpu','47','--in-place','--batch-cpu-jobs','--gpu-launch-workers-per-node','4']+extra,op='allreduce',end=4194304)
