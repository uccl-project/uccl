import json,subprocess,shutil,hashlib
from pathlib import Path
old='build/uccl_graph_single64';base=json.loads(Path(old+'/build_command.json').read_text())
variants={'uccl_ag_fused_own':['-DUCCL_LITE_AG_FUSED_OWN'],'uccl_ag_fused_ready':['-DUCCL_LITE_AG_FUSED_OWN','-DUCCL_LITE_AG_COALESCED_FLUSH','-DUCCL_LITE_AG_READY_FLAGS'],'uccl_ar_compact_barriers':['-DUCCL_LITE_AR_COMPACT_BARRIERS'],'uccl_host_spin':['-DUCCL_LITE_HOST_SPIN_PROGRESS'],'uccl_host_spin_pin':['-DUCCL_LITE_HOST_SPIN_PROGRESS','-DUCCL_LITE_HOST_PIN_PROGRESS']}
for name,extra in variants.items():
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base if not x.endswith('/graph_queue_check')]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='));cmd[i]+=' '+' '.join(extra)
 if 'host_spin' in name:cmd+=[str(b/'nccl_lite/async_host_gather_check')]
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
