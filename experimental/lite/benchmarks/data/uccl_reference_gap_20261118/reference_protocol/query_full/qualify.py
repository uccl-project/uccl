import subprocess,sys,json,hashlib,time,os
from pathlib import Path
base=Path(__file__).resolve().parent
library=base/'nccl_lite/libuccl_lite_nccl_strict.so'
sha=hashlib.sha256(library.read_bytes()).hexdigest()
state={'pid':os.getpid(),'started':time.time(),'candidate_sha256':sha,'scope':'AG/AR/RS, 1 KiB–1 GiB, both layouts, unchanged nccl-tests plus full-retirement paired reference; zero native fallback','jobs':[]}
(base/'qualification_state.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 for script in ['measure_all.py','paired.py']:
  assert hashlib.sha256(library.read_bytes()).hexdigest()==sha
  state['jobs'].append({'script':script,'status':'running'})
  (base/'qualification_state.json').write_text(json.dumps(state,indent=2)+'\n')
  print('Starting',script,flush=True)
  r=subprocess.run([sys.executable,'-B',str(base/script)])
  state['jobs'][-1].update(status='complete',exit_code=r.returncode)
  (base/'qualification_state.json').write_text(json.dumps(state,indent=2)+'\n')
  assert r.returncode==0
 assert hashlib.sha256(library.read_bytes()).hexdigest()==sha
 state['exit_code']=0
except BaseException:
 state['exit_code']=1
 raise
finally:
 state['finished']=time.time()
 (base/'qualification_state.json').write_text(json.dumps(state,indent=2)+'\n')
