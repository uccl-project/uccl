import os,json,shutil,subprocess,hashlib
from pathlib import Path
root=Path.cwd();out=root/'build/uccl_gpu_gates8m/component';out.mkdir(exist_ok=True)
binary=root/'build/uccl_gpu_gates8m/nccl_lite/async_host_gather_check'
deps=subprocess.check_output(['ldd',str(binary)],text=True);assert 'libnccl' not in deps
(out/'dependencies.txt').write_text(deps);shutil.copy2(binary,out/'async_host_gather_check')
cmd=json.loads((root/'build/goal_ar_graph_client_input/full/standalone_commands.json').read_text())[0]['argv']
idx=next(i for i,s in enumerate(cmd) if s.endswith('/standalone'));cmd=cmd[:idx]+[str(out/'async_host_gather_check'),'262144']
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
subprocess.run(['ssh','10.10.55.1','mkdir','-p',str(out)],check=True)
subprocess.run(['rsync','-a',str(out)+'/',f'10.10.55.1:{out}/'],check=True)
env=os.environ.copy();assert not env.get('LD_PRELOAD')
env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
with (out/'run.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
assert r.returncode==0, f'network size check exited {r.returncode}'
assert 'Async host gather passed: 60 calls/rank, two lifetimes' in (out/'run.log').read_text()
(out/'result.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'calls_per_rank':60,'lifetimes':2,'passed':True},indent=2)+'\n')
print('Async host gather passed: 60 calls/rank, two lifetimes')
