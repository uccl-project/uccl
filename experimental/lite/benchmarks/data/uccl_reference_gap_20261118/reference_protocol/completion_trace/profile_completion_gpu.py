import json,os,subprocess,re,shutil,csv,hashlib
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_completion_gpu_trace';stage=base/'abi'
exe=stage/'reference_timing_check';shutil.copy2(base/'nccl_lite/reference_timing_check',exe)
subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
m=json.loads((root/'build/uccl_rs128_prefix/paired/manifest.json').read_text());template=next(r['command'] for r in m['runs'] if r['label']=='ag_0_out')
old=root/'build/uccl_rs128_prefix/abi';template=[x.replace(str(old),str(stage)) for x in template];template[1:1]=['-x','UCCL_LITE_COMPLETION_TRACE_DIR']
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
summary=[]
for rep in range(3):
 for layout in ['out','in']:
  out=base/f'trace_{rep}_{layout}';out.mkdir(exist_ok=False)
  subprocess.run(['ssh','10.10.55.1','mkdir','-p',str(out)],check=True)
  args=template[:-6]+['ag',layout,'1024','1073741824','20','30'];env['UCCL_LITE_COMPLETION_TRACE_DIR']=str(out)
  (out/'command.json').write_text(json.dumps(args,indent=2)+'\n');print('Trace',rep,layout,flush=True)
  with (out/'run.log').open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  text=(out/'run.log').read_text();assert r.returncode==0 and 'UCCL reference-protocol checks passed' in text
  assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',text)))==list(range(8))
  assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',text))==[(r,0) for r in range(8)]
  routes={(int(r),int(op),int(s),int(n)) for r,op,s,n in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',text)};assert routes=={(r,1,1<<p,50) for r in range(8) for p in range(10,31)}
  subprocess.run(['rsync','-a',f'10.10.55.1:{out}/',str(out)+'/'],check=True)
  traces=[]
  for rank in range(8):
   rows=list(csv.DictReader((out/f'rank_{rank}.csv').open()));assert len(rows)==1050
   assert [int(r['sequence']) for r in rows]==list(range(1,1051))
   assert all(int(r['operation'])==1 and int(r['bytes'])==1<<(10+i//50) for i,r in enumerate(rows))
   traces.append(rows)
  samples={(int(s),int(i)):float(v) for s,i,v in re.findall(r'^sample,ag,'+layout+r',(\d+),(\d+),([0-9.]+),',text,re.M)};assert len(samples)==630
  phases=['enqueue_ns','device_ns','gpu_ns','host_ns','restore_ns']
  for power in range(10,31):
   size=1<<power
   for iteration in range(30):
    index=(power-10)*50+20+iteration
    rank=max(range(8),key=lambda rank:sum(int(traces[rank][index][k]) for k in phases));row=traces[rank][index]
    result={'repeat':rep,'layout':layout,'bytes':size,'iteration':iteration,'max_trace_rank':rank,'client_us':samples[size,iteration]}
    result.update({k[:-3]+'_us':int(row[k])/1000 for k in phases});result['gpu_elapsed_us']=float(row['gpu_elapsed_us']);summary.append(result)
  (out/'result.json').write_text(json.dumps({'correct':True,'scope':'host completion phase diagnostic, not scored admission','records_per_rank':1050,'library_sha256':hashlib.sha256((base/'nccl_lite/libuccl_lite_nccl_strict.so').read_bytes()).hexdigest()},indent=2)+'\n')
with (base/'phases.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('Six full AG traces passed numerical/native/route/count checks.',flush=True)
