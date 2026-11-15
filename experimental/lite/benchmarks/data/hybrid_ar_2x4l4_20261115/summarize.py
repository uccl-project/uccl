"""Require complete checked three-round AllReduce comparisons."""
import csv
from pathlib import Path
from statistics import mean, median
root = Path(__file__).resolve().parent
rows=[]
for mode in ('old','hybrid','nccl'):
    rounds={}
    for repeat in range(1,4):
        path=root/'raw'/f'final_{mode}_{repeat}.log'
        samples={}
        for line in path.read_text().splitlines():
            f=line.split(',')
            if mode=='nccl' and len(f)==6 and f[:2]==['nccl','timed']:
                size=int(f[3]);value=float(f[4]);assert int(f[5])==0,path
            elif mode!='nccl' and len(f)==6 and f[:2]==['# sample','allreduce']:
                size=int(f[2]);value=float(f[4]);assert float(f[5])<=1e-5,path
            else: continue
            samples.setdefault(size,[]).append(value)
        assert set(samples)=={1048576<<i for i in range(11)},path
        for size,values in samples.items():
            assert len(values)==10,path
            rounds.setdefault(size,[]).append(mean(values))
    for size,values in rounds.items():
        rows.append((mode,size,median(values),min(values),max(values),1.75*size/median(values)/1000))
with (root/'summary.csv').open('w') as stream:
    w=csv.writer(stream,lineterminator="\n");w.writerow(('mode','input_bytes_per_gpu','median_us','min_us','max_us','bus_GBps'));w.writerows(rows)
