import json,os,subprocess,sys,time,hashlib,shutil
from pathlib import Path
state_path=Path('build/host_dispatch_state.json');state={'pid':os.getpid(),'stage':'build','started':time.time()}
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 assert json.loads(Path('build/rs16_stream_state.json').read_text()).get('exit_code')==0
 for name in ['uccl_fixed_dispatch','uccl_driver_graph']:
  base=Path('build')/name;state['candidate']=name;save()
  with (base/'build.log').open('w') as log:subprocess.run(json.loads((base/'build_command.json').read_text()),stdout=log,stderr=subprocess.STDOUT,check=True)
  sources={}
  for name in ['src/multinode/mscclpp_backend.cu','src/multinode/device_collectives.cuh','src/multinode/async_host_gather.cuh','tests/nccl_mscclpp/strict_abi.cu']:
   p=Path(name);target=base/'sources'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  (base/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
 state['stage']='ABI234, graph queue48 and54 focused jobs';save()
 subprocess.run([sys.executable,'-B','build/check_host_dispatch.py'],check=True)
 state.update(stage='complete',exit_code=0)
except BaseException:
 state.update(stage='failed',exit_code=1);raise
finally:state['finished']=time.time();save()
