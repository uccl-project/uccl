import json,subprocess
from pathlib import Path
old='build/uccl_graph_descriptor_cache';base=json.loads(Path(old+'/build_command.json').read_text())
variants={'uccl_graph_binding_cache':['-DUCCL_LITE_GRAPH_BINDING_CACHE'],'uccl_rs_medium64':['-DUCCL_LITE_RS_MEDIUM_BYTES=67108864'],'uccl_graph_banks_combo':['-DUCCL_LITE_GRAPH_BINDING_CACHE','-DUCCL_LITE_RS_MEDIUM_BYTES=67108864','-DUCCL_LITE_RS8M_REMOTE_FIRST']}
for name,extra in variants.items():
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)).replace('/mscclpp_strict_abi_serial_check','/mscclpp_strict_abi_medium64_check') for x in base]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='))
 if any('RS_MEDIUM_BYTES=' in x for x in extra):cmd[i]=cmd[i].replace(' -DUCCL_LITE_RS_MEDIUM_BYTES=33554432','')
 cmd[i]+=' '+' '.join(extra)
 cmd+=['NCCL_STRICT_EXTRA_LIBS=-lcuda',str(b/'nccl_lite/graph_queue_check')]
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
