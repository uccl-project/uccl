import json,os,subprocess,re,shutil,collections,sys,hashlib
from pathlib import Path
root=Path.cwd();name=sys.argv[1];base=root/'build'/name;stage=base/'abi';out=base/'layout_queue';out.mkdir(exist_ok=False)
exe=stage/'layout_queue_check';shutil.copy2(base/'nccl_lite/layout_queue_check',exe)
subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
old=root/'build/uccl_reference_protocol/abi';smoke=json.loads((root/'build/uccl_reference_protocol/smoke_retry/manifest.json').read_text());cmd=[x.replace(str(old),str(stage)) for x in smoke['commands'][0][:-6]];cmd[-1]=str(exe)
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
with (out/'run.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
text=(out/'run.log').read_text();assert r.returncode==0
assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',text)))==list(range(8))
assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',text))==[(r,0) for r in range(8)]
checks=[tuple(map(int,x)) for x in re.findall(r'UCCL layout queue: rank=(\d+) recycled_va=(\d+) calls=(\d+) passed',text)];assert sorted(x[0] for x in checks)==list(range(8)) and all(x[2]==48 for x in checks)
routes=collections.Counter(tuple(map(int,x)) for x in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',text));assert routes==collections.Counter((r,op,size,calls) for r in range(8) for op in [2,3] for size,calls in [(128<<20,16),(256<<20,8)])
transitions=[tuple(map(int,x)) for x in re.findall(r'UCCL Lite prefix128: rank=(\d+) entries=(\d+) reuses=(\d+) leaves=(\d+)',text)]
if name!='uccl_ar256_banks8':assert sorted(transitions)==[(r,3,29,2) for r in range(8)],transitions
(out/'result.json').write_text(json.dumps({'passed':True,'calls_per_rank':48,'queue_checks':checks,'layout_transitions':transitions,'library_sha256':hashlib.sha256((base/'nccl_lite/libuccl_lite_nccl_strict.so').read_bytes()).hexdigest()},indent=2)+'\n')
print(name,'queued layout/output/guard checks passed; transitions',transitions,flush=True)
