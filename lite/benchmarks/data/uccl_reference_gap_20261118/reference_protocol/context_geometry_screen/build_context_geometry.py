import json,subprocess
from pathlib import Path
old='build/uccl_host_spin';base=json.loads(Path(old+'/build_command.json').read_text())
variants={'uccl_sync_primary':['-DUCCL_LITE_SYNC_PRIMARY_CONTEXT'],'uccl_ag128_threads384':['-DUCCL_LITE_AG_PACKET_THREADS=384','-DUCCL_LITE_AG_PACKET_THREADS_MIN_BYTES=131072'],'uccl_ag128_threads768':['-DUCCL_LITE_AG_PACKET_THREADS=768','-DUCCL_LITE_AG_PACKET_THREADS_MIN_BYTES=131072'],'uccl_graph_single128':['-DUCCL_LITE_RS_GRAPH_SINGLE_MAX_BYTES=134217728']}
for name,extra in variants.items():
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)).replace('/mscclpp_strict_abi_medium64_check','/mscclpp_strict_abi_medium128_check') for x in base if not x.endswith('/async_host_gather_check')]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='))
 if name=='uccl_graph_single128':cmd[i]=cmd[i].replace(' -DUCCL_LITE_RS_GRAPH_SINGLE_MAX_BYTES=67108864','')
 cmd[i]+=' '+' '.join(extra)
 if name=='uccl_sync_primary':cmd+=[str(b/'nccl_lite/sync_device_check')]
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
