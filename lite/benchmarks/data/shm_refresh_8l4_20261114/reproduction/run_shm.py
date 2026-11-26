from run_sweeps import *
binary=LITE/'build/ar_ag_transfer_20261113/uccl_lite_collectives'
probe_env=dict(ENV,NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH')
command=[str(binary),'--shm-only','--gpus','8','--fallback','nccl','--backend','off','--op','all','--begin','1024','--end','1024','--warmup','3','--iterations','3','--repeats','3','--main-cpu','47']
if users():raise RuntimeError('GPU busy before probe')
with (OUT/'nccl_probe.csv').open('w') as stdout,(OUT/'nccl_probe.log').open('w') as stderr:
 result=subprocess.run(command,cwd=LITE,env=probe_env,stdout=stdout,stderr=stderr,timeout=120)
(OUT/'nccl_probe.json').write_text(json.dumps(dict(command=command,environment=probe_env,returncode=result.returncode),indent=2))
if result.returncode or users():raise RuntimeError('NCCL SHM probe failed')
probe_text=(OUT/'nccl_probe.csv').read_text()+(OUT/'nccl_probe.log').read_text()
edges=[line for line in probe_text.splitlines() if 'Channel' in line and ' via ' in line]
assert edges and all('via SHM/' in line for line in edges),edges
(OUT/'transport_check.json').write_text(json.dumps(dict(channel_edges=len(edges),all_shm=True,edges=edges),indent=2))
print('Verified NCCL SHM edges:',len(edges),flush=True)
variants=[('allreduce',['--in-place','--workers-per-node','4','--transport-blocks','16','--small-ar-gpu-progress-max-bytes','8192']),('allgather',['--small-ag-gpu-progress-max-bytes','1048576']),('reducescatter',['--small-rs-workers-per-node','4','--small-rs-gpu-progress-max-bytes','65536'])]
for suffix,order in [('a',variants),('b',list(reversed(variants)))]:
 for op,extra in order:
  run('shm_'+op+'_'+suffix,binary,['--shm-only','--batch-cpu-jobs','--gpu-launch-workers-per-node','4']+extra,end=4194304,op=op)
