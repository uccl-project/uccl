from run_sweeps import *
cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs','--main-cpu','47','--small-rs-gpu-progress-max-bytes','65536']
for name,idle,extra in [('cadence_dense_hybrid',0,[]),('cadence_sparse_hybrid',5000,[]),('cadence_sparse_parallel',5000,['--gpu-launch-workers-per-node','4']),('cadence_dense_parallel',0,['--gpu-launch-workers-per-node','4'])]:
 ENV['UCCL_LITE_BENCH_IDLE_US']=str(idle)
 run(name,OUT/'collectives_cadence_probe',cap+extra,end=65536)
