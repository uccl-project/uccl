import json,os,subprocess,re,shutil,csv,statistics
from pathlib import Path
root=Path.cwd();out=root/'build/output_ordered_screen';out.mkdir(exist_ok=False)
variants=['uccl_early_retirement','uccl_output_ordered']
smoke=json.loads((root/'build/uccl_reference_protocol/smoke_retry/manifest.json').read_text());template=smoke['commands'][0][:-6]
old=root/'build/uccl_reference_protocol/abi'
for name in variants:
 b=root/'build'/name;exe=b/'abi/reference_timing_check';shutil.copy2(b/'nccl_lite/reference_timing_check',exe)
 subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
manifest={'scope':'small AG per-call/full-retirement screen; unchanged 1-GiB allocation; not full admission','runs':[]}
for rep in range(3):
 for name in (variants if rep%2==0 else variants[::-1]):
  stage=root/'build'/name/'abi';cmd=[x.replace(str(old),str(stage)) for x in template]
  for layout in ['out','in']:
   label=f'{name}_{rep}_{layout}';args=cmd+['ag',layout,'1024','131072','20','30','1073741824'];print(label,flush=True)
   log=out/f'{label}.log'
   with log.open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
   s=log.read_text();assert r.returncode==0 and 'UCCL reference-protocol checks passed' in s
   assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',s)))==list(range(8))
   assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',s))==[(r,0) for r in range(8)]
   rows={int(a):float(b) for a,b in re.findall(r'^result,ag,'+layout+r',(\d+),([0-9.]+)$',s,re.M)};assert set(rows)=={1<<p for p in range(10,18)}
   routes={(int(r),int(op),int(size),int(n)) for r,op,size,n in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',s)};assert routes=={(r,1,size,50) for r in range(8) for size in rows}
   manifest['runs'].append({'variant':name,'repeat':rep,'layout':layout,'command':args,'rows_us':rows});(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows=[]
for name in variants:
 for size in [1<<p for p in range(10,18)]:
  for layout in ['out','in']:
   values=[r['rows_us'][size] for r in manifest['runs'] if r['variant']==name and r['layout']==layout];rows.append({'variant':name,'bytes':size,'layout':layout,'median_us':statistics.median(values),'min_us':min(values),'max_us':max(values)})
with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'result.json').write_text('{"passed":true,"jobs":12}\n')
