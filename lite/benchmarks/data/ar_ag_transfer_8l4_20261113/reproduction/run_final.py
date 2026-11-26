from run_sweeps import *
before=OUT/'collectives_before'
current=OUT/'uccl_lite_collectives'
progress=['--small-ar-gpu-progress-max-bytes','8192','--small-ag-gpu-progress-max-bytes','1048576']
variants=[('previous',before,[]),('batch',current,['--batch-cpu-jobs']),('progress',current,['--batch-cpu-jobs']+progress),('combined',current,['--batch-cpu-jobs','--gpu-launch-workers-per-node','4']+progress),('compact_combined',current,['--batch-cpu-jobs','--gpu-launch-workers-per-node','4','--workers-per-node','4','--transport-blocks','16']+progress)]
for suffix,order in [('a',variants),('b',list(reversed(variants)))]:
 for op in ['allreduce','allgather'] if suffix=='a' else ['allgather','allreduce']:
  for name,binary,extra in order:
   run('final_'+op+'_'+name+'_'+suffix,binary,['--main-cpu','47']+(['--in-place'] if op=='allreduce' else [])+extra,op=op,end=4194304)
