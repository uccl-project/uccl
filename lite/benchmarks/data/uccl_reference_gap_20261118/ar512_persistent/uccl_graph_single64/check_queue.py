import json,os,subprocess,re,shutil,hashlib,collections
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_graph_single64';stage=base/'abi';out=base/'queue_check';out.mkdir(exist_ok=False)
exe=stage/'graph_queue_check';shutil.copy2(base/'nccl_lite/graph_queue_check',exe)
subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
old=root/'build/uccl_reference_protocol/abi';smoke=json.loads((root/'build/uccl_reference_protocol/smoke_retry/manifest.json').read_text())
cmd=[x.replace(str(old),str(stage)) for x in smoke['commands'][0][:-6]];cmd[-1]=str(exe)
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
with (out/'run.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
text=(out/'run.log').read_text();assert r.returncode==0
assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',text)))==list(range(8))
assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',text))==[(r,0) for r in range(8)]
counts=[tuple(map(int,x)) for x in re.findall(r'UCCL Lite graph descriptor: rank=(\d+) reused=(\d+) pending=(\d+)',text)];assert sorted(x[0] for x in counts)==list(range(8)) and all(x[1]>0 and x[2]>0 for x in counts),counts
checks=[tuple(map(int,x)) for x in re.findall(r'UCCL graph queue: rank=(\d+) recycled_va=(\d+) calls=(\d+) passed',text)];assert sorted(x[0] for x in checks)==list(range(8)) and all(x[2]==48 for x in checks)
routes=collections.Counter(tuple(map(int,x)) for x in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',text));assert routes==collections.Counter((r,op,16<<20,24) for r in range(8) for op in [2,3])
bindings=[tuple(map(int,x)) for x in re.findall(r'UCCL Lite graph bindings: rank=(\d+) hits=(\d+) updates=(\d+) recycled=(\d+)',text)];assert sorted(x[0] for x in bindings)==list(range(8)) and all(x[1]>0 and x[2]>0 and x[3]>0 for x in bindings),bindings
(out/'result.json').write_text(json.dumps({'correct':True,'calls_per_rank':48,'descriptor_counts':counts,'queue_checks':checks,'binding_counts':bindings,'library_sha256':hashlib.sha256((base/'nccl_lite/libuccl_lite_nccl_strict.so').read_bytes()).hexdigest()},indent=2)+'\n')
print('Queued full-output/guard checks pass; reused and pending branches exercised on all ranks. VA reuse:',checks,flush=True)
