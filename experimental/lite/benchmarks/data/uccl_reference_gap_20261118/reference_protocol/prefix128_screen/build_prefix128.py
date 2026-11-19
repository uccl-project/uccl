import json,subprocess,hashlib,shutil
from pathlib import Path
old='build/uccl_ar256_banks8';base=json.loads(Path(old+'/build_command.json').read_text())
variants={'uccl_rs128_prefix':['-DUCCL_LITE_RS128_PREFIX_BANKS'],'uccl_rs128_prefix_primary':['-DUCCL_LITE_RS128_PREFIX_BANKS','-DUCCL_LITE_SYNC_PRIMARY_CONTEXT']}
for name,extra in variants.items():
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='));cmd[i]+=' '+' '.join(extra)
 cmd+=[str(b/'nccl_lite/layout_queue_check')]
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
