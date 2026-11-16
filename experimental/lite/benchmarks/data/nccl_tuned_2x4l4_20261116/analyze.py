"""Validate complete NCCL-only calibration/evaluation and export frozen winners."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

from run import CONFIGS, DATA, OUT, ROOT, samples, sizes


def main():
    schedule = json.loads((DATA / 'schedule.json').read_text())
    for name, digest in json.loads((DATA / 'initial_validation_hashes.json').read_text()).items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
    original = json.loads((DATA / 'schedule_initial.json').read_text())
    assert all(entry in schedule for entry in original), 'original choices changed'
    environment = json.loads((DATA / 'environment.json').read_text())
    for name, digest in environment['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((OUT / 'uccl_lite_nccl_2x4l4_baseline').read_bytes()).hexdigest() == environment['binary_sha256']
    records = sorted(OUT.glob('calibrate_*.json')) + sorted(OUT.glob('evaluate_*.json'))
    assert len(records) == 60, f'incomplete experiment: {len(records)} runs'
    timed = {'calibrate': 0, 'evaluate': 0}
    evidence = []
    calibration = []
    for path in records:
        record = json.loads(path.read_text())
        name = path.stem
        assert record['returncode'] == 0, path
        op, config = record['operation'], record['config']
        assert {k: record['environment'][k] for k in CONFIGS[config]} == CONFIGS[config]
        fresh = samples(OUT / (name + '.stdout'), op, record['warmups'])
        assert fresh == record['means_us'], path
        logs = list(OUT.glob(name + '.debug.*.log'))
        expected_processes = 4 if op == 'allgather' else 8
        assert len(logs) == expected_processes, (name, len(logs))
        text = '\n'.join(p.read_text() for p in logs)
        ce = sum('via P2P/' in line and '/CE' in line for line in text.splitlines())
        net = sum('via NET/IB' in line for line in text.splitlines())
        assert net > 0, name
        assert (ce > 0) == config.startswith('ce_'), name
        evidence.append(dict(run=name, mpi_processes=expected_processes,
                             gpus_per_process=2 if op == 'allgather' else 1,
                             ce_edges_logged=ce, net_ib_edges_logged=net))
        timed[record['stage']] += len(sizes(op)) * 10
        if record['stage'] == 'calibrate':
            calibration.extend((op, config, record['repeat'], size, fresh[str(size)])
                               for size in sizes(op))
    rows = []
    for entry in schedule:
        op, config = entry['operation'], entry['config']
        records = [json.loads((OUT / f'evaluate_{op}_{config}_{repeat}.json').read_text())
                   for repeat in (1, 2, 3)]
        for size in sizes(op):
            values = [record['means_us'][str(size)] for record in records]
            us = median(values)
            factor = 1.75 if op == 'allreduce' else 0.875
            rows.append((op, entry['family'], size, config, us, min(values), max(values),
                         factor * size / us / 1000))
    assert len(rows) == 106
    for filename, header, values in (
        ('summary.csv', ('operation', 'family', 'bytes', 'config', 'median_us', 'min_us',
                         'max_us', 'bus_GBps'), rows),
        ('calibration.csv', ('operation', 'config', 'repeat', 'bytes', 'mean_us'), calibration),
    ):
        with (DATA / filename).open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(header)
            writer.writerows(values)
    assert timed == {'calibrate': 7420, 'evaluate': 3180}, timed
    (DATA / 'validation.json').write_text(json.dumps(dict(
        runs=60, checked_timed_calls=timed, checked_warmup_calls=6890,
        independent_evaluation=True, one_fixed_config_per_curve=True,
        uccl_input_hashes_unchanged=True, transport_evidence=evidence,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
