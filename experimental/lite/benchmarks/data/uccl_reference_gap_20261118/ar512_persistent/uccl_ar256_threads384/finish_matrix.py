import subprocess,sys,csv,json,hashlib
from pathlib import Path
base=Path.cwd()/'build/uccl_ar256_threads384'
for op in ['ag','rs']:
 subprocess.run([sys.executable,'-B',str(base/f'measure_{op}.py')],check=True)
 assert json.loads((base/f'{op}_measure/exit.json').read_text())['exit_code']==0
summary={'matched_admission':False,'scope':'one fixed binary; three full sweeps per operation; original UCCL harness conventions differ','library_sha256':hashlib.sha256((base/'nccl_lite/libuccl_lite_nccl_strict.so').read_bytes()).hexdigest(),'operations':{}}
for op in ['ag','ar','rs']:
 rows=list(csv.DictReader((base/f'{op}_measure/comparison.csv').open()));assert len(rows)==42
 summary['operations'][op]={'rows':42,'slower_rows':sum(float(r['diagnostic_ratio'])>1 for r in rows),'worst_ratio':max(float(r['diagnostic_ratio']) for r in rows)}
(base/'full_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n');(base/'full_exit.json').write_text('{"exit_code":0}\n');print(summary['operations'],flush=True)
