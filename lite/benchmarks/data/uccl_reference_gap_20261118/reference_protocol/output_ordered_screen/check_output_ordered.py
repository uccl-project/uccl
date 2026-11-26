import subprocess,sys
from pathlib import Path
root=Path.cwd()
for name in ['uccl_output_ordered']:
 base=root/'build'/name
 cmd=[sys.executable,'-B','tests/nccl_mscclpp/run.py','--hosts','10.10.55.2,10.10.55.1','--iface','ibp55s0f0','--mpirun','/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun','--tests','build/nccl-tests-mpi','--real-nccl','/home/zhongjie/third_party/nccl/build/lib/libnccl.so.2','--library',str(base/'nccl_lite/libuccl_lite_nccl_strict.so'),'--out',str(base/'abi'),'--max-bytes','1073741824','--abi-only','--abi-extended','--abi-boundary512','--abi','build/uccl_ar_direct512/nccl_lite/mscclpp_strict_abi_boundary_check','--job-timeout','180']
 print(name,'ABI',flush=True);subprocess.run(cmd,check=True)
s=Path('build/retirement_screen.py').read_text().replace("out=root/'build/retirement_screen'", "out=root/'build/output_ordered_screen'")
s=s.replace("variants=['uccl_reference_protocol','uccl_quiet_retirement','uccl_inline_retirement']", "variants=['uccl_early_retirement','uccl_output_ordered']")
s=s.replace('"jobs":18', '"jobs":12')
Path('build/output_ordered_screen.py').write_text(s)
subprocess.run([sys.executable,'-B','build/output_ordered_screen.py'],check=True)
