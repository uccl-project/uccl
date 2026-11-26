import json,subprocess,hashlib,shutil
from pathlib import Path
old='build/uccl_query_primary';base=json.loads(Path(old+'/build_command.json').read_text())
for name,consumers in [('uccl_medium_cons24',24),('uccl_medium_cons32',32)]:
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base if not x.endswith('/layout_queue_check')]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='));cmd[i]+=f' -DUCCL_LITE_HOST_MEDIUM_CONSUMERS={consumers}'
 cmd+=[str(b/'nccl_lite/async_host_gather_check')]
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for p in [Path('src/multinode/mscclpp_backend.cu'),Path('src/multinode/device_collectives.cuh'),Path('src/multinode/async_host_gather.cuh')]:
  q=b/'sources'/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 (b/'sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/multinode/mscclpp_backend.cu'),Path('src/multinode/device_collectives.cuh'),Path('src/multinode/async_host_gather.cuh')]},indent=2)+'\n')
 (b/'run_component.py').write_text(Path('build/uccl_shared_host_event/run_component.py').read_text().replace('uccl_shared_host_event',name))
