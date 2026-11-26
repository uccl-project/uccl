from run_sweeps import *
binary=OUT/'uccl_lite_collectives'
for op in ['allreduce','allgather']:
 base=['--main-cpu','47']+(['--in-place'] if op=='allreduce' else [])
 hybrid=['--small-ar-gpu-progress-max-bytes','8192','--small-ag-gpu-progress-max-bytes','1048576']
 for name,extra in [('batch',['--batch-cpu-jobs']),('combined',['--batch-cpu-jobs','--gpu-launch-workers-per-node','4']+hybrid),('compact_combined',['--batch-cpu-jobs','--gpu-launch-workers-per-node','4','--workers-per-node','4','--transport-blocks','16']+hybrid)]:
  run('new_'+op+'_'+name,binary,base+extra,op=op,end=4194304)
