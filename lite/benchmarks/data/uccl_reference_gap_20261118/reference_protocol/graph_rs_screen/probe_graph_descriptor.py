import json,os,subprocess,re,importlib.util
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_graph_descriptor_cache';out=base/'batch_probe';out.mkdir(exist_ok=False)
old=root/'build/uccl_shared_host_event/abi';stage=base/'abi'
cmd=json.loads((root/'build/uccl_shared_host_event/rs_measure/manifest.json').read_text())['command'];cmd=[x.replace(str(old),str(stage)) for x in cmd]
for key,value in [('-b','16M'),('-e','64M'),('-w','5'),('-n','20')]:cmd[cmd.index(key)+1]=value
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
with (out/'run.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
assert r.returncode==0
text=(out/'run.log').read_text();spec=importlib.util.spec_from_file_location('v',root/'tests/nccl_mscclpp/run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
checked=v.checked_strict(text,2);assert set(checked['rows_us'])=={1<<p for p in range(24,27)}
counts=[tuple(map(int,x)) for x in re.findall(r'UCCL Lite graph descriptor: rank=(\d+) reused=(\d+) pending=(\d+)',text)];assert sorted(x[0] for x in counts)==list(range(8))
(out/'result.json').write_text(json.dumps({'correct':True,'scope':'async graph correctness/branch probe, not admission','descriptor_counts':counts,'checked':checked},indent=2)+'\n')
print('Async graph probe passed; descriptor counts',counts,flush=True)
