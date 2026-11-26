import sys,json,hashlib,csv,statistics
from pathlib import Path
sys.path.insert(0,'scripts')
import run_rs_recovery as r
root=Path.cwd(); final_build=r.runner.BUILD
original=root/'build/full_refresh_8l4_20261114'
record=json.loads((root/'benchmarks/data/full_refresh_8l4_20261114/raw/full_gpu_rs_a.json').read_text())
assert hashlib.sha256((original/'uccl_lite_compact_collectives').read_bytes()).hexdigest()==record['binary_sha256']
rows=[]
for mode,labels in (('gpu',('original','updated')),('shm',('updated','original'))):
    for label in labels:
        r.runner.BUILD=original if label=='original' else final_build
        name='original_binary_'+mode+'_'+label
        r.bench(name,['--blocks','16'],mode=mode,warmup=5,iterations=10,variants='nccl,current')
        groups={}
        for line in (r.runner.DATA/'raw'/(name+'.log')).read_text().splitlines():
            if line.startswith('median,'):
                _,n,v,rep,i,us=line.split(',')
                groups.setdefault((int(n),v),[]).append(float(us))
        for n in (1<<20,2<<20,4<<20,8<<20):
            rows.append(dict(testbed=mode,revision=label,bytes=n,uccl_us=statistics.median(groups[n,'current']),nccl_us=statistics.median(groups[n,'nccl'])))
with (r.runner.DATA/'original_binary_control.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n");w.writeheader();w.writerows(rows)
for mode in ('gpu','shm'):
    for n in (1<<20,2<<20,4<<20,8<<20):
        a=next(x for x in rows if (x['testbed'],x['bytes'],x['revision'])==(mode,n,'original'))
        b=next(x for x in rows if (x['testbed'],x['bytes'],x['revision'])==(mode,n,'updated'))
        print(mode,n,'latency reduction',100*(1-b['uccl_us']/a['uccl_us']),flush=True)
        assert b['uccl_us'] < a['uccl_us']
r.runner.BUILD=final_build
