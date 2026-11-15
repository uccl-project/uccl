#!/usr/bin/env python3
"""Validate all selected CE/SM points and transport traces before plotting."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'build/nccl_tuned_testbeds12_20261115'
DATA=Path(__file__).resolve().parent
OPS={'ar':'all_reduce','ag':'all_gather','rs':'reduce_scatter'}
configs=json.loads((OUT/'candidates.json').read_text())
schedule=json.loads((OUT/'schedule.json').read_text())
selected={(r['testbed'],r['op'],r['engine'],r['bytes']):r['config'] for r in schedule}
assert len(selected)==252
records=json.loads((OUT/'evaluate_runs.json').read_text())
for stage in ['evaluate','trace']:
    for name,digest in json.loads((OUT/(stage+'_inputs.json')).read_text()).items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
measurements={};defaults={};timed=0;warmups=0
for run in records:
    assert run['correctness_pass'] and not run['timed_out'] and not run['foreign_gpu_users'],run['name']
    raw_text=(OUT/(run['name']+'.csv')).read_text()
    assert 'AR-in-place AG/RS-out-of-place NCCL=22809' in raw_text
    assert 'input-ready slowest-GPU-event timing CPU=47' in raw_text
    values={};samples={}
    for row in csv.reader((OUT/(run['name']+'.csv')).open()):
        if row and row[0]=='median':values.setdefault(int(row[1]),[]).append(float(row[5]))
        if row and row[0]=='sample':samples.setdefault(int(row[1]),[]).append(float(row[5]))
    assert all(len(v)==3 for v in values.values()) and all(len(v)==30 for v in samples.values()),run['name']
    timed+=sum(len(v) for v in samples.values());warmups+=len(values)*3*5
    for size,rounds in values.items():
        assert all(math.isfinite(v) and v>0 for v in rounds)
        us=statistics.median(rounds)
        if run['name'].startswith('default_'):
            key=(run['testbed'],run['op'],size,run['run']);assert key not in defaults;defaults[key]=us
        else:
            key=(run['testbed'],run['op'],run['engine'],size)
            if selected.get(key)!=run['config']:continue
            key=(*key,run['run']);assert key not in measurements;measurements[key]=us
assert len(measurements)==504 and len(defaults)==0,(len(measurements),len(defaults))
summary=[]
for (mode,op,engine,size),cfg in sorted(selected.items()):
    pair=[measurements[mode,op,engine,size,i] for i in range(2)];us=statistics.median(pair)
    factor=1.75 if op=='ar' else .875
    summary.append(dict(testbed=mode,collective=OPS[op],bytes=size,engine=engine,configuration=cfg,median_us=us,min_run_us=min(pair),max_run_us=max(pair),run0_us=pair[0],run1_us=pair[1],bus_gbs=factor*size/us/1000))
with (DATA/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)

transport=[]
for run in json.loads((OUT/'trace_runs.json').read_text()):
    assert run['correctness_pass'],run['name']
    text=(OUT/(run['name']+'_debug.log')).read_text()
    edges=re.findall(r'Channel\s+(\d+)(?:/\d+)?\s+:\s+(\d+)\[\d+\]\s+->\s+(\d+)\[\d+\]\s+via\s+([^\n]+)',text)
    assert edges and {int(e[1]) for e in edges}==set(range(8)),run['name']
    engine=configs[run['testbed']][run['config']]['engine']
    for _,source,dest,link in edges:
        assert link.startswith(('SHM/','P2P/')),link
        if run['testbed']=='shm':assert link.startswith('SHM/'),link
        if engine=='ce':assert '/CE' in link,link
        else:assert '/CE' not in link,link
    protocols=re.findall(r'(AllReduce|AllGather|ReduceScatter):\s+(\d+) Bytes -> Algo (\S+) proto (\S+)',text)
    assert protocols,run['name']
    if engine=='ce':assert {p[3] for p in protocols}=={'SIMPLE'},set(p[3] for p in protocols)
    transport.append(dict(testbed=run['testbed'],op=run['op'],engine=engine,configuration=run['config'],channel_edges=len(edges),transports=sorted({e[3] for e in edges}),protocols=sorted({p[3] for p in protocols}),algorithms=sorted({p[2] for p in protocols}),source_log=run['name']+'_debug.log'))
traced={(r['testbed'],r['op'],r['configuration']) for r in transport}
assert {(r['testbed'],r['op'],r['config']) for r in schedule}<=traced
(DATA/'transport.json').write_text(json.dumps(transport,indent=2)+'\n')
(DATA/'validation.json').write_text(json.dumps(dict(plotted_cases=252,validation_runs_per_case=2,timed_calls_including_prefixes=timed,checked_warmups=warmups,all_selected_configurations_traced=True,per_size_calibrated_schedule=True,validation_used_for_selection=False,all_sweeps_start_at_1kib=True),indent=2)+'\n')
for name in ['candidates.json','schedule.json','segments.json','selection.json','hardware.json','source_manifest.json','calibration_cases.csv']:
    (DATA/name).write_bytes((OUT/name).read_bytes())
print('Validated 252 CE/SM points and',len(transport),'transport configurations;',timed,'checked timed calls')
