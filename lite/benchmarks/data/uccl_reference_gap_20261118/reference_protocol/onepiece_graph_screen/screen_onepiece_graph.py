import os,json,subprocess,re,shutil,csv,statistics
from pathlib import Path
root=Path.cwd();out=root/'build/onepiece_graph_screen';out.mkdir(exist_ok=False)
cases=[('rs','rs',2,16777216,67108864,10,['uccl_graph_banks_combo','uccl_graph_single32','uccl_graph_single64']),('ar','ar',3,16777216,67108864,10,['uccl_graph_banks_combo','uccl_graph_single32','uccl_graph_single64'])]
old=root/'build/uccl_reference_protocol/abi';smoke=json.loads((root/'build/uccl_reference_protocol/smoke_retry/manifest.json').read_text());template=smoke['commands'][0][:-6]
for name in sorted({v for case in cases for v in case[-1]}):
 b=root/'build'/name;exe=b/'abi/reference_timing_check';shutil.copy2(b/'nccl_lite/reference_timing_check',exe)
 subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
manifest={'scope':'single graph piece through32/64MiB; fixed1-GiB allocations; full retirement; not full admission','runs':[]}
for rep in range(3):
 for case_id,op,num,begin,end,iters,variants in cases:
  for name in (variants if rep%2==0 else variants[::-1]):
   stage=root/'build'/name/'abi';cmd=[x.replace(str(old),str(stage)) for x in template]
   for layout in ['out','in']:
    label=f'{case_id}_{name}_{rep}_{op}_{layout}';args=cmd+[op,layout,str(begin),str(end),'20',str(iters),'1073741824'];print(label,flush=True)
    log=out/f'{label}.log'
    with log.open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    text=log.read_text();assert r.returncode==0 and 'UCCL reference-protocol checks passed' in text
    assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',text)))==list(range(8))
    assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',text))==[(r,0) for r in range(8)]
    values={int(a):float(b) for a,b in re.findall(r'^result,'+op+','+layout+r',(\d+),([0-9.]+)$',text,re.M)};assert set(values)=={1<<p for p in range(begin.bit_length()-1,end.bit_length())}
    routes={(int(r),int(o),int(s),int(n)) for r,o,s,n in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',text)};assert routes=={(r,num,size,20+iters) for r in range(8) for size in values}
    manifest['runs'].append({'case':case_id,'variant':name,'repeat':rep,'operation':op,'layout':layout,'command':args,'rows_us':values});(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows=[]
for case_id,op,num,begin,end,iters,variants in cases:
 for name in variants:
  for size in [1<<p for p in range(begin.bit_length()-1,end.bit_length())]:
   for layout in ['out','in']:
    values=[r['rows_us'][size] for r in manifest['runs'] if r['case']==case_id and r['variant']==name and r['operation']==op and r['layout']==layout];rows.append({'case':case_id,'variant':name,'operation':op,'bytes':size,'layout':layout,'median_us':statistics.median(values),'min_us':min(values),'max_us':max(values)})
with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'result.json').write_text('{"passed":true,"jobs":36}\n')
