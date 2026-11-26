import json,os,subprocess,sys,time,hashlib,importlib.util
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_policy_medium24';base.mkdir(exist_ok=False)
state={'pid':os.getpid(),'stage':'waiting for full medium24 qualification','started':time.time()}
def save(): (base/'state.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('identity',root/'build/elf_runtime_identity.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
def runtime(path):return identity.inspect(path)['runtime_sha256']
save()
try:
 while True:
  try:cmd=Path('/proc/3674729/cmdline').read_bytes()
  except FileNotFoundError:break
  if b'uccl_medium_cons24/controller.py' not in cmd:break
  time.sleep(2)
 q=json.loads((root/'build/uccl_medium_cons24/controller_state.json').read_text());assert q.get('exit_code')==0,q
 original=root/'build/uccl_medium_cons24'
 for name,digest in json.loads((original/'sources.json').read_text()).items():assert sha(name)==digest,name
 expected=sha(original/'nccl_lite/libuccl_lite_nccl_strict.so')
 assert expected=='b0ba7b498fddcd64d5f1788da56c3e581dfe1e3b5b90b4f9f0c8ec3239639398'
 options=['make','-j2','BUILD='+str(base.relative_to(root)),'MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl','MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx']
 targets=['nccl-mscclpp-strict']+[str((base/'nccl_lite'/n).relative_to(root)) for n in ['async_host_gather_check','network_sizes_check','network_queue_check','mscclpp_strict_abi_medium128_check']]
 command=options+['UCCL_LITE_NCCL_POLICY=medium24']+targets
 (base/'command.json').write_text(json.dumps(command,indent=2)+'\n')
 state['stage']='build explicit policy and affected targets';save()
 with (base/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 library=base/'nccl_lite/libuccl_lite_nccl_strict.so';component=base/'nccl_lite/async_host_gather_check'
 assert runtime(library)==runtime(original/'nccl_lite/libuccl_lite_nccl_strict.so'),'Loaded library code/data differ'
 assert runtime(component)==runtime(original/'nccl_lite/async_host_gather_check'),'Loaded component code/data differ'
 state['stage']='verify no-op build and policy invalidation';save()
 r=subprocess.run(command,capture_output=True,text=True,check=True);(base/'noop.log').write_text(r.stdout+r.stderr)
 assert '/bin/nvcc ' not in r.stdout and ' -shared ' not in r.stdout,'Unchanged policy rebuilt unexpectedly'
 stamp=str((base/'nccl_lite/build.policy').relative_to(root))
 subprocess.run(options+['UCCL_LITE_NCCL_POLICY=base',stamp],capture_output=True,text=True,check=True)
 assert (base/'nccl_lite/build.policy').read_text().strip()=='base'
 with (base/'restore.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert ' -c src/multinode/mscclpp_backend.cu ' in (base/'restore.log').read_text(),'Policy switch failed to rebuild backend'
 assert runtime(library)==runtime(original/'nccl_lite/libuccl_lite_nccl_strict.so') and runtime(component)==runtime(original/'nccl_lite/async_host_gather_check')
 state['stage']='ABI234 for rebuilt policy library';save()
 abi=[sys.executable,'-B','tests/nccl_mscclpp/run.py','--hosts','10.10.55.2,10.10.55.1','--iface','ibp55s0f0','--mpirun','/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun','--tests','build/nccl-tests-mpi','--real-nccl','/home/zhongjie/third_party/nccl/build/lib/libnccl.so.2','--library',str(library),'--out',str(base/'abi'),'--max-bytes','1073741824','--abi-only','--abi-extended','--abi-boundary512','--abi-serial-packet','--abi-medium64','--abi-medium128','--abi',str(base/'nccl_lite/mscclpp_strict_abi_medium128_check'),'--job-timeout','180']
 (base/'abi_command.json').write_text(json.dumps(abi,indent=2)+'\n')
 subprocess.run(abi,check=True)

 state.update(stage='complete',exit_code=0,library_sha256=sha(library),measured_library_sha256=expected,component_sha256=sha(component),byte_identical_to_measured=sha(library)==expected,runtime_identical_to_measured=True,runtime_sha256=runtime(library),abi_calls_per_rank=234,policy_switch_invalidates=True,unchanged_policy_no_rebuild=True,scope='explicit experimental policy build verification; performance/default admission separate')
except BaseException:
 state.update(stage='failed',exit_code=1);raise
finally:state['finished']=time.time();save()
