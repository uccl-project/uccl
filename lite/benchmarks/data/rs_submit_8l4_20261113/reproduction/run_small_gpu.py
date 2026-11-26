from run_sweeps import *
cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs','--main-cpu','47']
for name,extra in [('small_cpu_a',[]),('small_gpu_flat_a',['--gpu-progress']),('small_gpu_striped_a',['--gpu-progress','--interleave-scatter']),('small_gpu_striped_b',['--gpu-progress','--interleave-scatter']),('small_gpu_flat_b',['--gpu-progress']),('small_cpu_b',[])]:
 run(name,OUT/'uccl_lite_collectives',cap+extra,end=131072)
