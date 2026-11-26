from run_sweeps import *
cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs']
for name,extra in [('launch_serial_a',[]),('launch_numa_a',['--gpu-launch-workers-per-node','1']),('launch_rank_a',['--gpu-launch-workers-per-node','4']),('launch_rank_b',['--gpu-launch-workers-per-node','4']),('launch_numa_b',['--gpu-launch-workers-per-node','1']),('launch_serial_b',[])]:
 run(name,OUT/'uccl_lite_collectives',cap+extra,end=4194304)
