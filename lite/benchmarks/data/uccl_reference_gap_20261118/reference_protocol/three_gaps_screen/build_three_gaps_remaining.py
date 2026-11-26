import json,subprocess
from pathlib import Path
old='build/uccl_shared_host_event'
base=json.loads(Path(old+'/build_command.json').read_text())
variants={'uccl_ag128_pairs':['-DUCCL_LITE_AG128_PAIR_EXCHANGE'],'uccl_ar256_threads512':['-DUCCL_LITE_AR256_THREADS=512'],'uccl_ar256_after_flush256':['-DUCCL_LITE_AR256_THREADS=256'],'uccl_rs8m_blocks16':['-DUCCL_LITE_RS8M_BLOCKS=16'],'uccl_rs8m_blocks32':['-DUCCL_LITE_RS8M_BLOCKS=32']}
for name,extra in variants.items():
 if name in ['uccl_ag128_pairs','uccl_ar256_threads512']:continue
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base if not x.endswith('/async_host_gather_check')]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='))
 if name.startswith('uccl_ar256_'):cmd[i]=cmd[i].replace(' -DUCCL_LITE_AR256_THREADS=384','')
 cmd[i]+=' '+' '.join(extra)
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
