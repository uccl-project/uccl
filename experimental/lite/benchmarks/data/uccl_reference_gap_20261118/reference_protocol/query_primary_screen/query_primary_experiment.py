import json,subprocess,sys,hashlib,shutil,os,time
from pathlib import Path
base=Path('build/uccl_query_primary')
state={'pid':os.getpid(),'stage':'build','started':time.time()}
def save(): (base/'experiment_state.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 cmd=json.loads((base/'build_command.json').read_text())
 with (base/'build.log').open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 sources={}
 for path in ['src/multinode/mscclpp_backend.cu','src/multinode/device_collectives.cuh','src/multinode/async_host_gather.cuh']:
  p=Path(path);q=base/'sources'/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);sources[path]=hashlib.sha256(p.read_bytes()).hexdigest()
 (base/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
 state['stage']='correctness and focused comparison';save()
 subprocess.run([sys.executable,'-B','build/check_query_primary.py'],check=True)
 state.update(stage='complete',exit_code=0)
except BaseException:
 state.update(stage='failed',exit_code=1);raise
finally:state['finished']=time.time();save()
