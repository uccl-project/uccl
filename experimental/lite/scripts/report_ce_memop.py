#!/usr/bin/env python3
"""Keep eager latency and prepared-trigger latency in separate tables."""
import csv
import pathlib
import statistics
import sys
from collections import defaultdict
root=pathlib.Path(sys.argv[1])
groups=defaultdict(list)
for run in ['run1','run2','run3']:
    rounds=defaultdict(list)
    with (root/run/'1024.csv').open() as f:
        for r in csv.DictReader(f):
            rounds[(r['variant'],r['round'])].append((float(r['host_us']),float(r['arm_us'])))
    for (name,_),samples in rounds.items():
        assert len(samples)==50
        groups[name].append(tuple(statistics.mean(v[i] for v in samples) for i in [0,1]))
with (root/'summary.csv').open('w') as f:
    w=csv.writer(f)
    w.writerow(['variant','scope','median_us','min_round_mean_us','max_round_mean_us','arm_us','arm_plus_trigger_us','timed_calls'])
    for name,means in sorted(groups.items()):
        assert len(means)==9
        w.writerow([name,'prepared_trigger' if 'armed' in name else 'eager',
                    statistics.median(v[0] for v in means),min(v[0] for v in means),max(v[0] for v in means),
                    statistics.median(v[1] for v in means),statistics.median(sum(v) for v in means),450])
print((root/'summary.csv').read_text())
