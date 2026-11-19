import json,subprocess,hashlib,shutil
from pathlib import Path
old='build/uccl_ag_direct128';base=json.loads(Path(old+'/build_command.json').read_text())
for path,digest in json.loads(Path('build/ar256_banks_expected_sources.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
for name,blocks in [('uccl_ar256_banks8',8),('uccl_ar256_banks4',4),('uccl_ar256_banks2',2)]:
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='));cmd[i]+=f' -DUCCL_LITE_AR256_MEDIUM_BANKS -DUCCL_LITE_AR256_RING_BLOCKS={blocks}'
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for path in json.loads(Path('build/ar256_banks_expected_sources.json').read_text()):
  p=Path(path);q=b/'sources'/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 shutil.copy2('build/ar256_banks_expected_sources.json',b/'sources.json')
