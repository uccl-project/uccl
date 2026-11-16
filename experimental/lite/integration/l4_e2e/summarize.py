#!/usr/bin/env python3
"""Summarize every ABBA job, checking metadata and application results."""
import argparse
import json
from pathlib import Path
import statistics
import math


def summarize(root):
    records = {p.stem: json.loads(p.read_text()) for p in root.glob('*_node*.json')}
    assert records, 'no records'
    first = next(iter(records.values()))
    nodes = first['nodes']
    result = {'model_label': 'GPT-2-XL' if first['checkpoint_config']['n_layer']==48 else 'GPT-2-large', 'parameter_dtype': first['arguments'].get('parameter_dtype','bfloat16'), 'warmups': first['arguments']['warmup'], 'repeats_per_job': first['arguments']['repeats'], 'nodes': nodes, 'world': nodes * 4, 'shard_world': first['shard_world'], 'replica_nodes': first['replica_nodes'], 'cases': [], 'validation': {'max_loss_abs_diff': 0.0}}
    selected = next((b for b in ('packet', 'peer') if f'01_{b}_serving_node0' in records), None)
    assert selected, 'missing UCCL candidate jobs'
    result['uccl_backend'] = selected
    result['checkpoint_blocks'] = first['arguments'].get('checkpoint_blocks', False)
    for kind in ('serving', 'training'):
        jobs = []
        for phase, backend in enumerate(('nccl', selected, selected, 'nccl')):
            ranks = [records[f'{phase:02d}_{backend}_{kind}_node{node}'] for node in range(nodes)]
            for record in ranks:
                assert record['collective_oracle'] == 'PASS'
                assert record['reference_check']['four_token_kv_decode_equal']
                assert record['reference_check']['greedy_tokens_equal']
                assert record['world'] == nodes * 4
                assert record['runtime_nccl_version'] == first['runtime_nccl_version']
                assert record['gpu'] == first['gpu']
                assert record['arguments']['seed'] == first['arguments']['seed']
                assert record['arguments']['cases'] == first['arguments']['cases']
                assert record['arguments']['train_cases'] == first['arguments']['train_cases']
                assert record['arguments']['compile_blocks'] == first['arguments']['compile_blocks']
                assert record['torch'] == first['torch']
                assert record['mapped_nccl'] == first['mapped_nccl']
                assert record['trainable_parameters'] == first['trainable_parameters']
                assert record['shard_world'] == first['shard_world']
                assert record['replica_nodes'] == first['replica_nodes']
                assert record['arguments'].get('checkpoint_blocks',False) == first['arguments'].get('checkpoint_blocks',False)
                assert record['arguments'].get('parameter_dtype','bfloat16') == first['arguments'].get('parameter_dtype','bfloat16')
                assert record['checkpoint_config'] == first['checkpoint_config']
                assert record['complete_block_bytes'] == first['complete_block_bytes']
                assert len(record['mapped_nccl']) == 1
                assert record['arguments']['warmup'] == first['arguments']['warmup']
                assert record['arguments']['repeats'] == first['arguments']['repeats']
                for key in ('AG', 'RS') if kind == 'training' else ('AG',):
                    assert any(x.startswith(key + ':') for x in record['histogram_including_warmups'])
                layers = record['checkpoint_config']['n_layer']
                calls = record['arguments']['warmup'] + record['arguments']['repeats']
                ag = sum(v for k,v in record['histogram_including_warmups'].items() if k.startswith('AG:'))
                rs = sum(v for k,v in record['histogram_including_warmups'].items() if k.startswith('RS:'))
                if kind == 'training':
                    expected = layers * calls * len(record['rows'])
                    assert rs == expected
                    assert ag == expected * (2 if record['arguments'].get('checkpoint_blocks',False) else 1)
                else:
                    assert rs == 0
                    assert ag == layers * calls * sum(int(row['case'].split(':')[2]) for row in record['rows'])
                for row in record['rows']:
                    assert len(row['samples_s']) == record['arguments']['repeats']
                    assert all(math.isfinite(v) and v > 0 for v in row['samples_s'])
            jobs.append(ranks)
        for case in range(len(jobs[0][0]['rows'])):
            rows = [[r['rows'][case] for r in ranks] for ranks in jobs]
            assert all(row['case'] == rows[0][0]['case'] for phase in rows for row in phase)
            if kind == 'training' and nodes == 2 and first['replica_nodes']:
                for phase in rows:
                    assert phase[0]['parameter_sha256'] == phase[1]['parameter_sha256'], 'DP replicas diverged'
            for node in range(nodes):
                ref = rows[0][node]
                for phase in range(4):
                    row = rows[phase][node]
                    assert row['case'] == ref['case']
                    if kind == 'serving':
                        assert row['prompt_sha256'] == ref['prompt_sha256']
                        assert len(set(row['response_sha256'] + ref['response_sha256'])) == 1
                    else:
                        # BF16 reduction trees differ: retain and bound loss deviations.
                        assert len(row['losses']) == len(ref['losses'])
                        delta = max(abs(a-b) for x,y in zip(row['losses'],ref['losses']) for a,b in zip(x,y))
                        assert delta < 0.02
                        result['validation']['max_loss_abs_diff'] = max(result['validation']['max_loss_abs_diff'], delta)
            medians = [statistics.median(max(v) for v in zip(*(row['samples_s'] for row in phase))) for phase in rows]
            nccl = statistics.median([medians[0], medians[3]])
            peer = statistics.median([medians[1], medians[2]])
            result['cases'].append({'kind': kind, 'case': rows[0][0]['case'], 'nccl_ms': nccl*1000,
                'peer_ms': peer*1000, 'latency_reduction_pct': (1-peer/nccl)*100,
                'throughput_ratio': nccl/peer, 'phase_medians_ms': [x*1000 for x in medians], 'all_peer_jobs_faster': max(medians[1:3]) < min(medians[0], medians[3])})
    result['validation'].update({'record_count': len(records), 'collective_oracle': 'PASS',
        'generated_tokens_equal': True, 'dp_replicas_synchronized': nodes == 1 or first['replica_nodes'], 'reference_max_logit_abs_diff': max(r['reference_check']['max_logit_abs_diff'] for r in records.values())})
    report_names = [f"parameters_{'4l4' if nodes==1 else '2x4l4'}_node{node}.json" for node in range(nodes)]
    reports = [root.parent/name for name in report_names]
    if all(p.exists() for p in reports):
        evidence=[json.loads(p.read_text()) for p in reports]
        assert all(r['status']=='PASS' for r in evidence)
        result['validation'].update({'full_parameters': 'PASS', 'parameter_reports': report_names,
            'parameter_max_abs_diff': max(c['max_abs_diff'] for r in evidence for c in r['comparisons']),
            'parameter_max_relative_l2': max(c['relative_l2'] for r in evidence for c in r['comparisons'])})
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.data), indent=2)+'\n')
