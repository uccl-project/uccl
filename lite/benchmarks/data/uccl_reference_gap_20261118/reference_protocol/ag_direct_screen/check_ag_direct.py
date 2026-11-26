import subprocess,sys,json,os,time,re
from pathlib import Path
root=Path.cwd();state_path=root/'build/check_ag_direct_state.json';state={'pid':os.getpid(),'stage':'starting','started':time.time()}
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for name in ['uccl_ag_direct128','uccl_ag_direct_primary']:
  base=root/'build'/name
  if 'host_spin' in name:
   state.update(stage='standalone component',candidate=name);save()
   subprocess.run([sys.executable,'-B',str(base/'run_component.py')],check=True)
  state.update(stage='mixed ABI',candidate=name);save()
  cmd=[sys.executable,'-B','tests/nccl_mscclpp/run.py','--hosts','10.10.55.2,10.10.55.1','--iface','ibp55s0f0','--mpirun','/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun','--tests','build/nccl-tests-mpi','--real-nccl','/home/zhongjie/third_party/nccl/build/lib/libnccl.so.2','--library',str(base/'nccl_lite/libuccl_lite_nccl_strict.so'),'--out',str(base/'abi'),'--max-bytes','1073741824','--abi-only','--abi-extended','--abi-boundary512','--abi-serial-packet','--abi-medium64','--abi-medium128','--abi',str(base/'nccl_lite/mscclpp_strict_abi_medium128_check'),'--job-timeout','180']
  print(name,'ABI',flush=True);subprocess.run(cmd,check=True)
  if name=='uccl_sync_primary':
   subprocess.run([sys.executable,'-B','build/check_sync_device.py'],check=True)
  if name.endswith('_wrap'):
   text=(base/'abi/abi.log').read_text();assert sorted(map(int,re.findall(r'UCCL Lite LL reset: rank=(\d+)',text)))==sorted(list(range(8))*2)
 state['stage']='18-job raw AG comparison';save()
 subprocess.run([sys.executable,'-B','build/screen_ag_direct.py'],check=True)
 state.update(stage='complete',exit_code=0)
except BaseException:
 state.update(stage='failed',exit_code=1);raise
finally:state['finished']=time.time();save()
