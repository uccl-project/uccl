#!/usr/bin/env python3
"""Compare a strict candidate with the recorded README UCCL curves.

This diagnoses gaps, not paired-run admission: process layout, timing and
buffer cycling differ. The user target allows zero slowdown versus UCCL.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFS = {
    'all_gather': ('ag_combined_latest_2x4l4_20261117', 'combined', 'bytes', 'out-of-place; 4 processes, 2 GPUs/process'),
    'all_reduce': ('ar_small_fix_2x4l4_20261115', 'hybrid', 'input_bytes_per_gpu', 'in-place; 8 processes'),
    'reduce_scatter': ('rs_direct_medium_2x4l4_20261115', 'candidate', 'input_bytes_per_gpu', 'standalone internal layout; 8 processes'),
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    refs = {}
    for op, (directory, mode, size_key, layout) in REFS.items():
        path = ROOT / 'benchmarks/data' / directory / 'summary.csv'
        points = {int(r[size_key]): float(r['median_us']) for r in csv.DictReader(path.open()) if r['mode'] == mode}
        assert set(points) == {1 << n for n in range(10, 31)}, op
        refs[op] = (points, str(path.relative_to(ROOT)), layout)
    rows = []
    seen = set()
    for r in csv.DictReader(args.candidate.open()):
        op, size, layout = r['operation'], int(r['bytes']), r['layout']
        assert r['route'] == 'uccl_lite'
        key = (op, size, layout)
        assert key not in seen
        seen.add(key)
        points, source, ref_layout = refs[op]
        baseline, candidate = points[size], float(r['candidate_median_us'])
        rows.append(dict(operation=op, bytes=size, layout=layout,
                         reference_us=baseline, candidate_us=candidate,
                         candidate_over_uccl=candidate / baseline,
                         reference_layout=ref_layout, reference_source=source))
    assert seen == {(op, 1 << n, layout) for op in REFS for n in range(10, 31)
                    for layout in ['out-of-place', 'in-place']}
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    failures = [r for r in rows if r['candidate_over_uccl'] > 1]
    result = dict(admission=False, scope='historical diagnostic; not matched measurements',
                  target='UCCL Lite README paths, zero slowdown; native NCCL is not the acceptance baseline',
                  threshold=0, rows=len(rows), slower_rows=len(failures), failures=failures)
    (args.out / 'diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(failures)}/{len(rows)} rows slower than recorded UCCL; historical diagnostic only')
    for op in REFS:
        group = [r for r in rows if r['operation'] == op]
        print(op, 'slower:', sum(r['candidate_over_uccl'] > 1 for r in group), '/ 42',
              'worst:', round(max(r['candidate_over_uccl'] for r in group), 4))

if __name__ == '__main__':
    main()
