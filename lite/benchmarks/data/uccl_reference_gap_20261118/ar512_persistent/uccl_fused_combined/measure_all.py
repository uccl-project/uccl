import csv,json,subprocess,sys,hashlib,traceback
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_fused_combined'
summary={'scope':'one fixed experimental library, unchanged nccl-tests, cross-harness UCCL diagnostic only','matched_admission':False,'baseline':'fresh original README UCCL; no native baseline or fallback','warmup':20,'iterations':30,'repetitions':3,'dtype':'FP32 SUM (AR/RS), copy (AG)','timing':'maximum-rank host enqueue/completion batch means','policy':'compiled policy, no best-per-size envelope','library_sha256':hashlib.sha256((base/'nccl_lite/libuccl_lite_nccl_strict.so').read_bytes()).hexdigest(),'operations':{}}
try:
 for op in ['ag','ar','rs']:
  print('Starting fixed-library',op,flush=True)
  r=subprocess.run([sys.executable,'-B',str(base/f'measure_{op}.py')]);assert r.returncode==0
  assert json.loads((base/f'{op}_measure/exit.json').read_text())['exit_code']==0
  rows=list(csv.DictReader((base/f'{op}_measure/comparison.csv').open()));assert len(rows)==42
  summary['operations'][op]={'rows':42,'slower_rows':sum(float(r['diagnostic_ratio'])>1 for r in rows),'worst_ratio':max(float(r['diagnostic_ratio']) for r in rows)}
  (base/'full_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary['operations']),flush=True);code=0
except BaseException:traceback.print_exc();code=1
(base/'full_exit.json').write_text(json.dumps({'exit_code':code})+'\n')
raise SystemExit(code)
