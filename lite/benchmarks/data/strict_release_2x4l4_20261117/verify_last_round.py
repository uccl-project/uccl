import json,os,sys,subprocess,importlib.util,statistics,math,hashlib,shutil,traceback
from pathlib import Path
root=Path.cwd();base=root/'build/goal_release_strict/final';out=root/'build/goal_release_strict/verified_final';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('validation',root/'tests/nccl_mscclpp/run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
m=json.loads((base/'manifest.json').read_text());m.pop('recovery',None)
m['verification_note']='AG/RS both rounds and AR first round retained. AR second round fully repeated because the original controller lost candidate 6 exit status; all original logs retained. No timing-based selection.'
m['results']={k:x for k,x in m['results'].items() if not (k.startswith('all_reduce_') and int(k.split('_')[-2])>=4)}
m['commands']=[x for x in m['commands'] if x['label'] in m['results']]
for p in base.glob('*'):
 if p.is_file() and p.suffix in ['.json','.tsv','.txt','.py'] and p.name not in ['manifest.json','performance_verdict.json','candidate_over_nccl.json','manifest_before_recovery.json']:
  shutil.copy2(p,out/p.name)
for label in m['results']:shutil.copy2(base/(label+'.log'),out/(label+'.log'))
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
original=json.loads((base/'manifest.json').read_text())
exits={}
try:
 for i,b in enumerate(['native','candidate','candidate','native'],4):
  label=f'all_reduce_{i}_{b}';src='all_reduce_0_native' if b=='native' else 'all_reduce_1_candidate'
  cmd=next(x['argv'][:] for x in original['commands'] if x['label']==src)
  print('Running',label,flush=True)
  with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  exits[label]=r.returncode;assert r.returncode==0
  log=(out/(label+'.log')).read_text();d=v.checked_strict(log,3) if b=='candidate' else v.checked(log,3,False)
  assert set(d['rows_us'])=={1<<j for j in range(10,31)}
  m['results'][label]=d;m['commands'].append({'label':label,'argv':cmd});m['observed_retry_exit_codes']=exits
  (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 ratios={}
 for op,num in [('all_gather',1),('reduce_scatter',2),('all_reduce',3)]:
  runs=[];order=['native','candidate','candidate','native']*2
  for i,b in enumerate(order):
   log=(out/f'{op}_{i}_{b}.log').read_text();d=v.checked_strict(log,num) if b=='candidate' else v.checked(log,num,False)
   assert set(d['rows_us'])=={1<<j for j in range(10,31)}
   assert all(math.isfinite(x) and x>0 for row in d['rows_us'].values() for x in row)
   runs.append(d['rows_us'])
  ratios[op]={size:[statistics.median(r[size][j] for r,b in zip(runs,order) if b=='candidate')/statistics.median(r[size][j] for r,b in zip(runs,order) if b=='native') for j in range(2)] for size in runs[0]}
 (out/'candidate_over_nccl.json').write_text(json.dumps(ratios,indent=2)+'\n')
 fail=[{'operation':op,'bytes':size,'layout':layout,'ratio':r} for op,rows in ratios.items() for size,pair in rows.items() for layout,r in zip(['out-of-place','in-place'],pair) if r>1+m['max_regression']]
 verdict={'threshold':m['max_regression'],'passed':not fail,'rows':126,'failures':fail}
 (out/'performance_verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
 rc=2 if fail else 0;print(verdict,flush=True)
except BaseException:
 traceback.print_exc();rc=1
(out/'controller_exit.json').write_text(json.dumps({'exit_code':rc,'job_exit_codes':exits},indent=2)+'\n')
sys.exit(rc)
