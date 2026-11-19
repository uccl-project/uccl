import os,json,subprocess,importlib.util,hashlib
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_timeline16';out=base/'trace_16m';out.mkdir(exist_ok=False)
old=root/'build/uccl_ar256_threads384/abi';new=base/'abi'
cmd=json.loads((root/'build/uccl_ar256_threads384/ag_measure/manifest.json').read_text())['command']
cmd=[s.replace(str(old),str(new)) for s in cmd]
for k,v in [('-b','1K'),('-e','1G'),('-w','20'),('-n','30')]:cmd[cmd.index(k)+1]=v
cmd[1:1]=['-x','UCCL_LITE_HOST_TIMELINE_DIR','-x','UCCL_LITE_HOST_TIMELINE_BYTES']
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
subprocess.run(['ssh','10.10.55.1','mkdir','-p',str(out)],check=True)
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_HOST_TIMELINE_BYTES=str(16<<20),UCCL_LITE_HOST_TIMELINE_DIR=str(out),UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
with (out/'run.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
assert r.returncode==0
spec=importlib.util.spec_from_file_location('v',root/'tests/nccl_mscclpp/run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
checked=v.checked_strict((out/'run.log').read_text(),1);assert set(checked['rows_us'])=={1<<i for i in range(10,31)}
subprocess.run(['rsync','-a',f'10.10.55.1:{out}/',str(out)+'/'],check=True)
assert len(list(out.glob('rank_*.bin')))==8
assert all(json.loads((out/f'rank_{r}.json').read_text())['shard_bytes']==2<<20 for r in range(8))
header=Path('/home/zhongjie/.cache/uv/archive-v0/uP78hy8dMFj3jrMb/flashinfer/data/include/flashinfer/profiler.cuh')
(out/'result.json').write_text(json.dumps({'correct':True,'scope':'instrumented per-block timeline, not scored performance','flashinfer_header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),'checked':checked},indent=2)+'\n')
print('Eight GPU timelines captured, nccl-tests checks passed.')
