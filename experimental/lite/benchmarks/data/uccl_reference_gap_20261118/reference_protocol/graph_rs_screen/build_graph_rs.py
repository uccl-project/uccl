import json,subprocess
from pathlib import Path
old='build/uccl_shared_host_event'
base=json.loads(Path(old+'/build_command.json').read_text())
variants={'uccl_graph_descriptor_cache':['-DUCCL_LITE_GRAPH_DESCRIPTOR_CACHE'],'uccl_rs_graph_sm_input':['-DUCCL_LITE_RS_GRAPH_SM_INPUT'],'uccl_rs8_remote_first':['-DUCCL_LITE_RS8M_REMOTE_FIRST']}
for name,extra in variants.items():
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base if not x.endswith('/async_host_gather_check')]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='))
 if name.startswith('uccl_ar256_'):cmd[i]=cmd[i].replace(' -DUCCL_LITE_AR256_THREADS=384','')
 cmd[i]+=' '+' '.join(extra)
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
