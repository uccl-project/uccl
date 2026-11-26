from run_sweeps import *
cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs','--main-cpu','47']
variants=[('previous',LITE/'build/rs_opt_20261113/uccl_lite_collectives',[]),('serial',OUT/'uccl_lite_collectives',[]),('hybrid',OUT/'uccl_lite_collectives',['--small-rs-gpu-progress-max-bytes','65536']),('parallel',OUT/'uccl_lite_collectives',['--gpu-launch-workers-per-node','4']),('hybrid_parallel',OUT/'uccl_lite_collectives',['--small-rs-gpu-progress-max-bytes','65536','--gpu-launch-workers-per-node','4'])]
for suffix,order in [('a',variants),('b',list(reversed(variants)))]:
 for name,binary,extra in order:
  run('final_'+name+'_'+suffix,binary,cap+extra,end=4194304)
