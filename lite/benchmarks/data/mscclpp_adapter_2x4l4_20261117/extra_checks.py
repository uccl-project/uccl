import json, os, pathlib, shutil, subprocess
out=pathlib.Path(__file__).resolve().parent
manifest=json.loads((out/'manifest.json').read_text())
env=os.environ.copy();env.update(NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib')
for src,name in [('build/mscclpp_compile/nccl_lite/mscclpp_abi_check','abi_check'),('build/mscclpp_compile/uccl_lite_multinode_allreduce','standalone')]:
 shutil.copy2(src,out/name)
 subprocess.run(['rsync','-a',str(out/name),f'10.10.55.1:{out}/'],check=True)
candidate=manifest['commands'][1]['argv'];prefix=candidate[:candidate.index('--')+1]
cmd=prefix+[str(out/'abi_check')]
commands=[('abi',cmd)]
native=manifest['commands'][0]['argv'];mpi=native[:native.index('env')]
for op,flag in [('allreduce','--ar-auto'),('allgather','--ag-auto'),('reducescatter','--rs-auto')]:
 cmd=mpi+[str(out/'standalone'),'--op',op,flag,'1','-b','1024','-e',str(16<<20),'-w','1','-n','2','--check-each','1','-B','96','-T','512']
 commands.append(('standalone_'+op,cmd))
(out/'extra_commands.json').write_text(json.dumps(commands,indent=2))
for label,cmd in commands:
 print('Running',label,flush=True)
 with (out/(label+'.log')).open('w') as f:
  r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 print('exit',r.returncode,flush=True)
 if r.returncode: raise RuntimeError(label)
 print((out/(label+'.log')).read_text()[-900:],flush=True)
