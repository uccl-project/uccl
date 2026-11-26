#!/usr/bin/env python3
"""Select NCCL settings only from calibration, then freeze a validation schedule."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'build/nccl_tuned_testbeds12_20261115'
p=argparse.ArgumentParser();p.add_argument('stage',choices=['refine','schedule']);args=p.parse_args()
configs=json.loads((OUT/'candidates.json').read_text())
rows={}
records=[]
for stage in ['calibrate']+(['refine'] if args.stage=='schedule' else []):
    records += json.loads((OUT/(stage+'_runs.json')).read_text())
for run in records:
    if not run['correctness_pass']:continue
    groups={}
    for r in csv.reader((OUT/(run['name']+'.csv')).open()):
        if r and r[0]=='median':groups.setdefault(int(r[1]),[]).append(float(r[5]))
    assert len(groups)==21 and all(len(v)==3 for v in groups.values()),run['name']
    for size,medians in groups.items():rows[run['testbed'],run['op'],run['config'],size]=statistics.median(medians)

if args.stage=='refine':
    jobs=[]
    for mode in ['gpu','shm']:
        for op in ['ar','ag','rs']:
            for engine in ['ce','sm']:
                eligible=[k for k,c in configs[mode].items() if c['engine']==engine and all((mode,op,k,1<<b) in rows for b in range(10,31))]
                assert eligible,(mode,op,engine)
                base=min(eligible,key=lambda k:sum(math.log(rows[mode,op,k,1<<b]) for b in range(20,31)))
                settings=configs[mode][base]['environment']
                variants=[('b8m_c4',{'NCCL_BUFFSIZE':str(8<<20),'NCCL_MIN_NCHANNELS':'4','NCCL_MAX_NCHANNELS':'4'}),
                          ('b16m_c8',{'NCCL_BUFFSIZE':str(16<<20),'NCCL_MIN_NCHANNELS':'8','NCCL_MAX_NCHANNELS':'8'})]
                for tag,change in variants:
                    env={**settings,**change}
                    if any(c['environment']==env and (mode,op,k,1<<30) in rows for k,c in configs[mode].items()):continue
                    name=f'{engine}_{op}_{tag}'
                    configs[mode][name]=dict(engine=engine,environment=env,parent=base)
                    jobs.append(dict(name=f'refine_{mode}_{name}',testbed=mode,config=name,op=op,begin=1024,end=1<<30))
    (OUT/'candidates.json').write_text(json.dumps(configs,indent=2)+'\n')
    (OUT/'refine_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print('Refinement jobs:',len(jobs))
else:
    schedule=[];segments=[]
    switch_penalty=math.log(1.03)
    for mode in ['gpu','shm']:
        for op in ['ar','ag','rs']:
            for engine in ['ce','sm']:
                eligible=sorted(k for k,c in configs[mode].items() if c['engine']==engine and all((mode,op,k,1<<b) in rows for b in range(10,31)))
                assert eligible,(mode,op,engine)
                # Sum log relative latency, plus a 3% one-point cost for each
                # configuration switch. This suppresses noise-only fragments.
                costs={};parents=[]
                for power in range(10,31):
                    size=1<<power;best=min(rows[mode,op,k,size] for k in eligible)
                    next_cost={};parent={}
                    for k in eligible:
                        value=math.log(rows[mode,op,k,size]/best)
                        if not costs:next_cost[k]=value;parent[k]=None
                        else:
                            prev=min(costs,key=lambda q:costs[q]+(switch_penalty if q!=k else 0))
                            next_cost[k]=costs[prev]+value+(switch_penalty if prev!=k else 0);parent[k]=prev
                    costs=next_cost;parents.append(parent)
                last=min(costs,key=costs.get);path=[]
                for parent in reversed(parents):path.append(last);last=parent[last]
                path.reverse()
                for power,key in zip(range(10,31),path):
                    size=1<<power
                    schedule.append(dict(testbed=mode,op=op,engine=engine,bytes=size,config=key,calibration_us=rows[mode,op,key,size]))
                    if segments and all(segments[-1][f]==v for f,v in [('testbed',mode),('op',op),('engine',engine),('config',key)]) and segments[-1]['end']*2==size:
                        segments[-1]['end']=size
                    else:segments.append(dict(testbed=mode,op=op,engine=engine,config=key,begin=size,end=size))
    assert len(schedule)==252
    (OUT/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
    (OUT/'segments.json').write_text(json.dumps(segments,indent=2)+'\n')
    (OUT/'selection.json').write_text(json.dumps(dict(method='minimum sum log calibration latency + log(1.03) per configuration switch',validation_used=False,cases=252,segments=len(segments)),indent=2)+'\n')
    print('Frozen schedule: 252 cases,',len(segments),'segments')
with (OUT/'calibration_cases.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['testbed','op','configuration','bytes','median_us'])
    for key,value in sorted(rows.items()):w.writerow([*key,value])
