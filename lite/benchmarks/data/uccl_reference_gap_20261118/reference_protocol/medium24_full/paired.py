import json,subprocess,os,re,hashlib,statistics,csv,shutil,collections,time
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_medium_cons24';out=base/'paired';out.mkdir(exist_ok=False)
old=json.loads((root/'build/uccl_readme_reference/fresh/manifest.json').read_text())
refs={op:next(x['command'] for x in old['runs'] if x['label']==op+'_0') for op in ['ag','ar','rs']}
for op,cmd in refs.items():
 binary=next(Path(x) for x in cmd if '/uccl_readme_reference/'+op+'/' in x)
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==old['binary_sha256'][op]
 subprocess.run(['ssh','10.10.55.1','mkdir','-p',str(binary.parent)],check=True)
 subprocess.run(['rsync','-a',str(binary),f'10.10.55.1:{binary}'],check=True)
shim=root/'build/uccl_readme_reference/libuccl_lite_nccl_strict.so';subprocess.run(['rsync','-a',str(shim),f'10.10.55.1:{shim}'],check=True)
smoke=json.loads((root/'build/uccl_reference_protocol/smoke_retry/manifest.json').read_text());candidate=[x.replace(str(root/'build/uccl_reference_protocol/abi'),str(base/'abi')) for x in smoke['commands'][0][:-6]]
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
refenv=env.copy();refenv.update(CUDA_VISIBLE_DEVICES='0,1,2,3',MSCCLPP_HCA_DEVICES='mlx5_0,mlx5_1')
manifest={'scope':'aligned per-call preparation, fixed pointers, full retirement; original process layouts retained; reference layout is original recorded layout for both candidate layouts','reference_hashes':old['binary_sha256'],'candidate_sha256':hashlib.sha256((base/'nccl_lite/libuccl_lite_nccl_strict.so').read_bytes()).hexdigest(),'runs':[]}
try:
 for rep in range(3):
  for op,num in [('ag',1),('ar',3),('rs',2)]:
   ref=refs[op];warmup=int(ref[ref.index('-w')+1]);iters=int(ref[ref.index('-n')+1])
   for kind in (['reference','out','in'] if rep%2==0 else ['in','out','reference']):
    label=f'{op}_{rep}_{kind}';cmd=ref[:] if kind=='reference' else candidate+[op,kind,'1024','1073741824',str(warmup),str(iters)]
    print('Running',label,flush=True);start=time.time();log=out/f'{label}.log'
    manifest['runs'].append({'label':label,'command':cmd,'state':'running'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with log.open('w') as f:r=subprocess.run(cmd,env=refenv if kind=='reference' else env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
    text=log.read_text();assert r.returncode==0,(label,r.returncode)
    values=collections.defaultdict(list)
    if kind=='reference':
     for line in text.splitlines():
      if not line.startswith('# sample,'):continue
      fields=line.split(',');size=int(fields[2]);us=float(fields[-2]);err=float(fields[-1]);assert 0<=err<=1e-5;values[size].append(us)
    else:
     assert 'UCCL reference-protocol checks passed' in text
     for line in text.splitlines():
      if not line.startswith('sample,'):continue
      fields=line.split(',');assert fields[1]==op and fields[2]==kind;size=int(fields[3]);us=float(fields[5]);err=float(fields[6]);assert 0<=err<=1e-5;values[size].append(us)
     loaded=re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',text);assert sorted(map(int,loaded))==list(range(8))
     fallback=re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',text);assert sorted((int(a),int(b)) for a,b in fallback)==[(i,0) for i in range(8)]
     routes=collections.Counter(tuple(map(int,x)) for x in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',text))
     expected=collections.Counter((rank,num,1<<power,warmup+iters) for rank in range(8) for power in range(10,31));assert routes==expected
    assert set(values)=={1<<p for p in range(10,31)} and all(len(v)==iters for v in values.values()),label
    entry=manifest['runs'][-1];entry.update(state='complete',exit_code=0,wall_seconds=time.time()-start,means_us={k:statistics.mean(v) for k,v in values.items()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Completed',label,round(entry['wall_seconds'],1),'s',flush=True)
 rows=[]
 for op in ['ag','ar','rs']:
  for size in [1<<p for p in range(10,31)]:
   ref=[x['means_us'][size] for x in manifest['runs'] if x['label'].startswith(op+'_') and x['label'].endswith('_reference')]
   for layout in ['out','in']:
    val=[x['means_us'][size] for x in manifest['runs'] if x['label'].startswith(op+'_') and x['label'].endswith('_'+layout)]
    rows.append({'operation':op,'bytes':size,'layout':layout,'reference_us':statistics.median(ref),'candidate_us':statistics.median(val),'ratio':statistics.median(val)/statistics.median(ref)})
 with (out/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'result.json').write_text(json.dumps({'exit_code':0,'rows':len(rows),'slower_rows':sum(r['ratio']>1 for r in rows)},indent=2)+'\n')
except BaseException:
 (out/'failed.json').write_text(json.dumps({'failed':True,'last_label':manifest['runs'][-1]['label'] if manifest['runs'] else None})+'\n');raise
