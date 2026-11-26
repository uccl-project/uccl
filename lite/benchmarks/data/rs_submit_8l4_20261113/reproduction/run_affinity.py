from run_sweeps import *
cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs','--main-cpu','47']
for name,extra in [('affinity_serial_a',[]),('affinity_numa_a',['--gpu-launch-workers-per-node','1']),('affinity_rank_a',['--gpu-launch-workers-per-node','4']),('affinity_rank_b',['--gpu-launch-workers-per-node','4']),('affinity_numa_b',['--gpu-launch-workers-per-node','1']),('affinity_serial_b',[])]:
 run(name,OUT/'uccl_lite_collectives',cap+extra)
