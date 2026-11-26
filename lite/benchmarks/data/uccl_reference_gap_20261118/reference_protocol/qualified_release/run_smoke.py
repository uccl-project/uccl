import json,os,subprocess,sys,importlib.util,hashlib,shlex,time
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_default_release';stage=base/'abi';measured=root/'build/uccl_rs16_graph512';state={'pid':os.getpid(),'stage':'default-library stock nccl-tests smoke','started':time.time(),'runs':[]}
def save(): (base/'smoke_state.json').write_text(json.dumps(state,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('validation',root/'tests/nccl_mscclpp/run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
script=['#!/usr/bin/env bash','set -euo pipefail','cd '+shlex.quote(str(root))]
for key in ['UCCL_LITE_MSCCLPP_CONTROL_ROOT','NCCL_SOCKET_IFNAME','NCCL_IB_HCA','MSCCLPP_SOCKET_IFNAME','MSCCLPP_IBV_MODE']:
 script.append('export '+key+'='+shlex.quote(env[key]))
script.append('export LD_LIBRARY_PATH=/home/zhongjie/third_party/nccl/build/lib:${LD_LIBRARY_PATH:-}')
commands=[]
for op in ['ag','ar','rs']:
 command=json.loads((measured/f'{op}_measure/manifest.json').read_text())['command'];command=[x.replace(str(measured/'abi'),str(stage)) for x in command];commands.append(command);script.append(shlex.join(command))
(base/'reproduce_nccl_tests.sh').write_text('\n'.join(script)+'\n')
save()
try:
 for op,num,command in zip(['ag','ar','rs'],[1,3,2],commands):
  log=base/f'smoke_{op}.log';state['runs'].append({'operation':op,'command':command,'state':'running'});save()
  with log.open('w') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  assert r.returncode==0,(op,r.returncode)
  result=v.checked_strict(log.read_text(),num);assert set(result['rows_us'])=={1<<p for p in range(10,31)}
  state['runs'][-1].update(state='complete',exit_code=0,result=result);save()
 state.update(stage='complete',exit_code=0,scope='one unchanged nccl-tests sweep per operation on the canonical default library; correctness/routing smoke, not a replacement performance campaign')
except BaseException:
 state.update(stage='failed',exit_code=1);raise
finally:state['finished']=time.time();save()
