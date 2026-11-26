import json,os,subprocess,re,shutil,collections
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_driver_completion';stage=base/'abi';out=base/'device_check';out.mkdir(exist_ok=False)
exe=stage/'sync_device_check';shutil.copy2(base/'nccl_lite/sync_device_check',exe)
subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
old=root/'build/uccl_reference_protocol/abi';smoke=json.loads((root/'build/uccl_reference_protocol/smoke_retry/manifest.json').read_text());cmd=[x.replace(str(old),str(stage)) for x in smoke['commands'][0][:-6]];cmd[-1]=str(exe)
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
with (out/'run.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
text=(out/'run.log').read_text();assert r.returncode==0
assert sorted(map(int,re.findall(r'UCCL sync context: rank=(\d+) cases=4 passed',text)))==list(range(8))
assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',text)))==list(range(8))
assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',text))==[(r,0) for r in range(8)]
routes=collections.Counter(tuple(map(int,x)) for x in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',text));assert routes==collections.Counter((r,3,1024,4) for r in range(8))
(out/'result.json').write_text('{"passed":true,"cases_per_rank":4}\n')
print('Same-primary, foreign-device, custom-context and restored-primary checks passed.',flush=True)
