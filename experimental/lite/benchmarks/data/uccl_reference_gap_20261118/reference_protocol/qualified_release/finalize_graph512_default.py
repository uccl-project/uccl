import json,os,subprocess,sys,time,hashlib,shutil,importlib.util
from pathlib import Path
root=Path.cwd();out=root/'build/uccl_default_release';out.mkdir(exist_ok=False)
state={'pid':os.getpid(),'stage':'verify qualified input','started':time.time()}
def save(): (out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('identity',root/'build/elf_runtime_identity.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
save()
try:
 measured=root/'build/uccl_rs16_graph512';paired=json.loads((measured/'paired/raw_log_audit.json').read_text());batch=json.loads((measured/'batch_raw_log_audit.json').read_text());policy=json.loads((root/'build/uccl_policy_graph512/state.json').read_text())
 assert paired['zero_slowdown_passed'] and paired['rows']==126 and paired['runs']==27
 assert batch['slower_rows']==0 and batch['rows']==126 and batch['runs']==9
 assert policy['exit_code']==0 and policy['runtime_identical_to_measured'] and policy['abi_calls_per_rank']==234
 expected=sha(measured/'nccl_lite/libuccl_lite_nccl_strict.so');assert expected==paired['candidate_sha256']==batch['candidate_sha256']
 for name,digest in json.loads((measured/'sources.json').read_text()).items():assert sha(name)==digest,name
 backup=out/'previous_default';backup.mkdir()
 prior={}
 for name in ['libuccl_lite_nccl_strict.so','libuccl_lite_nccl_mscclpp.so','mscclpp_backend.o','build.policy','strict/libnccl.so.2']:
  source=root/'build/nccl_lite'/name
  if source.exists() or source.is_symlink():
   dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True)
   if source.is_symlink():dest.symlink_to(os.readlink(source));prior[name]={'symlink':os.readlink(source)}
   else:shutil.copy2(source,dest);prior[name]={'sha256':sha(source)}
 (backup/'manifest.json').write_text(json.dumps(prior,indent=2)+'\n')
 command=['make','-j2','nccl-mscclpp-strict','build/nccl_lite/mscclpp_strict_abi_medium128_check','MSCCLPP_IB_BUILD=/home/zhongjie/third_party/mscclpp/build-ib-pcieccl','MPICXX=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicxx']
 (out/'build_command.json').write_text(json.dumps(command,indent=2)+'\n')
 state['stage']='build canonical default without policy override';save()
 with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 library=root/'build/nccl_lite/libuccl_lite_nccl_strict.so'
 actual=identity.inspect(library);reference=identity.inspect(measured/'nccl_lite/libuccl_lite_nccl_strict.so')
 assert actual['runtime_sha256']==reference['runtime_sha256']==policy['runtime_sha256']
 (out/'runtime_identity.json').write_text(json.dumps(actual,indent=2)+'\n')
 assert (root/'build/nccl_lite/build.policy').read_text().strip()=='graph512'
 assert (root/'build/nccl_lite/strict/libnccl.so.2').resolve()==library.resolve()
 assert (root/'build/nccl_lite/libuccl_lite_nccl_mscclpp.so').resolve()==library.resolve()
 dynamic=subprocess.run(['readelf','-d',str(library)],capture_output=True,text=True,check=True).stdout
 (out/'dynamic.log').write_text(dynamic);assert 'Library soname: [libnccl.so.2]' in dynamic
 assert not any('NEEDED' in line and 'libnccl' in line for line in dynamic.splitlines())
 state['stage']='default-library ABI234 on two hosts';save()
 abi=[sys.executable,'-B','tests/nccl_mscclpp/run.py','--hosts','10.10.55.2,10.10.55.1','--iface','ibp55s0f0','--mpirun','/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun','--tests','build/nccl-tests-mpi','--real-nccl','/home/zhongjie/third_party/nccl/build/lib/libnccl.so.2','--library',str(library),'--out',str(out/'abi'),'--max-bytes','1073741824','--abi-only','--abi-extended','--abi-boundary512','--abi-serial-packet','--abi-medium64','--abi-medium128','--abi','build/nccl_lite/mscclpp_strict_abi_medium128_check','--job-timeout','180']
 (out/'abi_command.json').write_text(json.dumps(abi,indent=2)+'\n')
 subprocess.run(abi,check=True)
 state.update(stage='complete',exit_code=0,default_policy='graph512',library=str(library.relative_to(root)),library_sha256=sha(library),measured_library_sha256=expected,runtime_sha256=actual['runtime_sha256'],runtime_identical_to_measured=True,abi_calls_per_rank=234,paired_rows=126,paired_slower_rows=0,batch_rows=126,batch_slower_rows=0)
except BaseException:
 state.update(stage='failed',exit_code=1);raise
finally:state['finished']=time.time();save()
