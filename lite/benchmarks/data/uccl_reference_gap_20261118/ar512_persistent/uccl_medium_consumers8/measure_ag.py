import json,os,subprocess,sys,importlib.util,statistics,csv,traceback
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_medium_consumers8/abi';out=root/'build/uccl_medium_consumers8/ag_measure';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('validation',root/'tests/nccl_mscclpp/run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
old=root/'build/goal_release_strict/final';m=json.loads((old/'manifest.json').read_text());cmd=next(x['argv'][:] for x in m['commands'] if x['label']=='all_gather_1_candidate');cmd=[x.replace(str(old),str(base)) for x in cmd]
cmd[cmd.index('-w')+1]='20';cmd[cmd.index('-n')+1]='30'
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
manifest={'scope':'candidate AG diagnostic; fresh UCCL reference uses different harness/process layout, not matched admission','command':cmd,'results':[]}
try:
 for rep in range(3):
  print('AG diagnostic',rep,flush=True)
  with (out/f'candidate_{rep}.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  assert r.returncode==0
  d=v.checked_strict((out/f'candidate_{rep}.log').read_text(),1);assert set(d['rows_us'])=={1<<i for i in range(10,31)}
  manifest['results'].append(d);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 ref={int(r['bytes']):float(r['median_us']) for r in csv.DictReader(open(root/'build/uccl_readme_reference/fresh/summary.csv')) if r['operation']=='ag'}
 rows=[]
 for size in sorted(ref):
  for i,layout in enumerate(['out-of-place','in-place']):
   values=[r['rows_us'][size][i] for r in manifest['results']];us=statistics.median(values)
   rows.append(dict(bytes=size,layout=layout,median_us=us,min_us=min(values),max_us=max(values),reference_us=ref[size],diagnostic_ratio=us/ref[size]))
 with (out/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print([r for r in rows if r['bytes'] in [8388608,1073741824]],flush=True);code=0
except BaseException:traceback.print_exc();code=1
(out/'exit.json').write_text(json.dumps({'exit_code':code})+'\n')

raise SystemExit(code)
