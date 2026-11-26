from run_sweeps import *
cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs','--main-cpu','47']
for name,extra in [('spin_serial_a',[]),('spin_numa_a',['--gpu-launch-workers-per-node','1']),('spin_rank_a',['--gpu-launch-workers-per-node','4']),('spin_rank_b',['--gpu-launch-workers-per-node','4']),('spin_numa_b',['--gpu-launch-workers-per-node','1']),('spin_serial_b',[])]:
 run(name,OUT/'uccl_lite_collectives',cap+extra,end=65536)
