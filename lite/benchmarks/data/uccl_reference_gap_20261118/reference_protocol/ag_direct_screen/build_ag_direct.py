import json,subprocess,hashlib,shutil
from pathlib import Path
old='build/uccl_host_spin';base=json.loads(Path(old+'/build_command.json').read_text())
for path,digest in json.loads(Path('build/ag_direct_expected_sources.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
for name,extra in {'uccl_ag_direct128':['-DUCCL_LITE_AG_DIRECT128'],'uccl_ag_direct_primary':['-DUCCL_LITE_AG_DIRECT128','-DUCCL_LITE_SYNC_PRIMARY_CONTEXT']}.items():
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)).replace('/mscclpp_strict_abi_medium64_check','/mscclpp_strict_abi_medium128_check') for x in base if not x.endswith('/async_host_gather_check')]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='));cmd[i]+=' '+' '.join(extra)
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for path in json.loads(Path('build/ag_direct_expected_sources.json').read_text()):
  p=Path(path);q=b/'sources'/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 shutil.copy2('build/ag_direct_expected_sources.json',b/'sources.json')
