import json,subprocess,os,re,collections,shutil
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_reference_protocol';stage=base/'abi';out=base/'smoke_retry';out.mkdir(exist_ok=False)
exe=stage/'reference_timing_check';shutil.copy2(base/'nccl_lite/reference_timing_check',exe)
subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
old=root/'build/uccl_gpu_gates8m/abi';cmd=json.loads((root/'build/uccl_gpu_gates8m/ag_measure/manifest.json').read_text())['command'];cmd=[x.replace(str(old),str(stage)) for x in cmd];cmd=cmd[:cmd.index('--')+1]+[str(exe)]
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
manifest={'scope':'per-call reference-protocol smoke; not performance admission','commands':[]}
for op,num in [('ag',1),('ar',3),('rs',2)]:
 for layout in ['out','in']:
  args=cmd+[op,layout,'1024','1048576','1','2'];manifest['commands'].append(args);print(op,layout,flush=True)
  log=out/f'{op}_{layout}.log'
  with log.open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  s=log.read_text();assert r.returncode==0 and 'UCCL reference-protocol checks passed' in s
  actual=collections.Counter(tuple(map(int,x)) for x in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',s))
  expected=collections.Counter((rank,num,1<<power,3) for rank in range(8) for power in range(10,21));assert actual==expected
  assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',s)))==list(range(8))
  assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',s))==[(r,0) for r in range(8)]
  results=re.findall(r'^result,(ag|ar|rs),(out|in),(\d+),([0-9.]+)$',s,re.M);assert len(results)==11
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'result.json').write_text('{"passed":true,"jobs":6}\n')
