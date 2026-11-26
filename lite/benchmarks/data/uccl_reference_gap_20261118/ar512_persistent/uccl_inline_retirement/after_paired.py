import os,time,json,subprocess,sys
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_inline_retirement';proc=Path('/proc/3485411/cmdline')
while proc.exists():
 try:command=proc.read_bytes()
 except FileNotFoundError:break
 if b'build/uccl_reference_protocol/paired.py' not in command:break
 time.sleep(5)
result=json.loads((root/'build/uccl_reference_protocol/paired_retry/result.json').read_text());assert result['exit_code']==0
print('Paired comparison completed; starting inline-retirement ABI',flush=True)
cmd=[sys.executable,'-B','tests/nccl_mscclpp/run.py','--hosts','10.10.55.2,10.10.55.1','--iface','ibp55s0f0','--mpirun','/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun','--tests','build/nccl-tests-mpi','--real-nccl','/home/zhongjie/third_party/nccl/build/lib/libnccl.so.2','--library',str(base/'nccl_lite/libuccl_lite_nccl_strict.so'),'--out',str(base/'abi'),'--max-bytes','1073741824','--abi-only','--abi-extended','--abi-boundary512','--abi','build/uccl_ar_direct512/nccl_lite/mscclpp_strict_abi_boundary_check','--job-timeout','180']
subprocess.run(cmd,check=True)
print('Starting unchanged nccl-tests full matrix',flush=True)
subprocess.run([sys.executable,'-B',str(base/'measure_all.py')],check=True)
assert json.loads((base/'full_exit.json').read_text())['exit_code']==0
(base/'validation_exit.json').write_text('{"exit_code":0}\n')
