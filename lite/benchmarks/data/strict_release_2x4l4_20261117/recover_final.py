import json,os,sys,subprocess,importlib.util,statistics,math,hashlib,shutil
from pathlib import Path
root=Path.cwd();out=root/'build/goal_release_strict/final'
spec=importlib.util.spec_from_file_location('validation',root/'tests/nccl_mscclpp/run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
m=json.loads((out/'manifest.json').read_text());assert len(m['results'])==22
assert hashlib.sha256((out/'libuccl_lite_nccl_strict.so').read_bytes()).hexdigest()==m['library_sha256']
shutil.copy2(out/'manifest.json',out/'manifest_before_recovery.json')
label='all_reduce_6_candidate';log=(out/(label+'.log')).read_text();assert '# Collective test concluded: all_reduce_perf' in log
result=v.checked_strict(log,3);assert set(result['rows_us'])=={1<<i for i in range(10,31)}
assert all(math.isfinite(x) and x>0 for row in result['rows_us'].values() for x in row)
cmd=next(x['argv'][:] for x in m['commands'] if x['label']=='all_reduce_5_candidate');m['commands'].append({'label':label,'argv':cmd});m['results'][label]=result
m['recovery']={'reason':'controller session lost across continuation; candidate 6 process was observed live then terminal with complete log','candidate_exit_code':'not observed; complete correctness, routes and normal conclusion checked','remaining_job':'all_reduce_7_native'}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
label='all_reduce_7_native';cmd=next(x['argv'][:] for x in m['commands'] if x['label']=='all_reduce_3_native');m['commands'].append({'label':label,'argv':cmd})
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
assert r.returncode==0
m['results'][label]=v.checked((out/(label+'.log')).read_text(),3,False)
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
# Revalidate all recorded jobs with the original checker, then apply its gate.
ratios={}
for op,num in [('all_gather',1),('reduce_scatter',2),('all_reduce',3)]:
 order=['native','candidate','candidate','native']*2;runs=[]
 for i,b in enumerate(order):
  log=(out/f'{op}_{i}_{b}.log').read_text();d=v.checked_strict(log,num) if b=='candidate' else v.checked(log,num,False)
  assert set(d['rows_us'])=={1<<i for i in range(10,31)}
  runs.append(d['rows_us'])
 ratios[op]={size:[statistics.median(r[size][j] for r,b in zip(runs,order) if b=='candidate')/statistics.median(r[size][j] for r,b in zip(runs,order) if b=='native') for j in range(2)] for size in runs[0]}
(out/'candidate_over_nccl.json').write_text(json.dumps(ratios,indent=2)+'\n')
fail=[{'operation':op,'bytes':size,'layout':layout,'ratio':r} for op,rows in ratios.items() for size,pair in rows.items() for layout,r in zip(['out-of-place','in-place'],pair) if r>1+m['max_regression']]
verdict={'threshold':m['max_regression'],'passed':not fail,'rows':126,'failures':fail}
(out/'performance_verdict.json').write_text(json.dumps(verdict,indent=2)+'\n');print(verdict)
