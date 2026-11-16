"""CPU-only summary of the four fixed quality repeats, retaining invalid samples.

Usage: summarize_quality_repeats.py REPEAT_ROOT OUTPUT_JSON
Inputs are copied supervisor artifacts; no server or evaluator is launched.
Exit0 means all quality/provenance gates passed and median did not decline;
exit1 means quality-unconfirmed; exit2 means incomplete/unexpected evidence.
An invalid-answer sample contributes only to descriptive counts, never gate PASS.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/wxzheng/eat-pcieccl-vllm-pro6000-20261115-074910'
CONTROLLER_SHA = '1ba8361148b6103fd921ca982d896c3ce2012f5dababa5d84b1415a42d5fdb19'
PROTOCOL_SHA = '13bff60df9c236843196008a235428840ff39b73eb77e3b954f0e2abfb1a7116'
REFERENCE_SHA = 'cc7b980f9f4071cadac05cc35d1894e064d0b5870e751ec61f4aa16100515ae8'
PINNED = {
    'initial': ('aec0856c1657fcce6e3dca1dcf979a472938a182eab239457399a8028c53f364',
                '84d27c951f0c4bfccc377eeeffeab046932e0ebfa99a800741f90f0968c7b318'),
    'c03': ('ce751fa67339d5163171721a350c09d8edc14876a2e89cb808349ba984f75175',
            '89ebe0c1de86d93f6d92e709fcc2dbf753b46579e5fd588c8e35ae5bd1dc243b'),
}
ORDER = ('initial', 'c03', 'initial', 'c03')
GSM_ARGS = ['--num-questions', '1319', '--num-shots', '5', '--max-tokens', '256',
            '--temperature', '0', '--seed', '42', '--max-concurrency', '64', '--port', '31001']


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field: ' + key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON literal: ' + value)
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=reject)
    require(type(value) is dict, 'Expected object: ' + str(path))
    return value


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def quality_counts(q):
    for key, expected in (('num_questions', 1319), ('num_shots', 5), ('max_tokens', 256)):
        require(type(q.get(key)) is int and q[key] == expected, 'Wrong quality setting: ' + key)
    counts = {}
    for key in ('accuracy', 'invalid_rate'):
        value = q.get(key)
        require(number(value) and 0 <= value <= 1, 'Invalid quality metric: ' + key)
        count = value * 1319
        require(math.isclose(count, round(count), rel_tol=0, abs_tol=1e-8),
                'Nonintegral question count: ' + key)
        counts[key] = round(count)
    require(counts['accuracy'] + counts['invalid_rate'] <= 1319,
            'Correct and invalid question counts overlap impossibly')
    return counts['accuracy'], counts['invalid_rate']


def stage(folder, name, expected_code, argv, result):
    value = read(folder / (name + '.command.json'))
    require(type(value.get('returncode')) is int and value['returncode'] == expected_code,
            'Unexpected stage returncode: ' + name)
    require(value.get('argv') == argv, 'Unexpected stage command: ' + name)
    require(number(value.get('timeout_seconds')) and 0 < value['timeout_seconds'] <= 3600,
            'Unexpected stage budget: ' + name)
    require(number(value.get('started')) and number(value.get('finished')),
            'Invalid stage timestamps: ' + name)
    require(result['started'] <= value['started'] <= value['finished'] <= result['finished'],
            'Stage timestamps outside wrapper: ' + name)
    return value


def sample(folder, index, snapshot, reference):
    folder = Path(folder)
    rp = REMOTE + f'/quality-repeats/repeat-{index}-{snapshot}'
    tracked_names = ('result.json', 'lifecycle/gsm8k.json',
                     'lifecycle/gsm8k.command.json', 'lifecycle/gsm8k.log',
                     'quality-only.command.json', 'quality-only.log',
                     'fingerprint-before.command.json', 'fingerprint-before.json',
                     'fingerprint-after.command.json', 'fingerprint-after.json')
    artifacts = {name: digest(folder / name) for name in tracked_names
                 if (folder / name).exists()}
    def remembered(path):
        artifacts.setdefault(str(Path(path).relative_to(folder)), digest(path))
        return read(path)
    result = remembered(folder / 'result.json')
    require(result.get('snapshot') == snapshot and result.get('protocol_sha256') == PROTOCOL_SHA
            and result.get('controller_sha256') == CONTROLLER_SHA,
            'Repeat identity/protocol/controller mismatch')
    require(result.get('scored') is False and result.get('quality_only') is True,
            'Wrong repeat artifact kind')
    require(number(result.get('started')) and number(result.get('finished'))
            and result['started'] <= result['finished'], 'Invalid wrapper timestamps')
    require(not (folder / 'external-gpu-work.json').exists(), 'External GPU work recorded')
    q = remembered(folder / 'lifecycle/gsm8k.json')
    correct, invalid = quality_counts(q)
    gsm_log = (folder / 'lifecycle/gsm8k.log').read_text()
    require('Error calling vLLM' not in gsm_log,
            'Original GSM evaluator logged an API error; not a model-only invalid answer')
    failed_invalid = result.get('status') == 'failed' and invalid > 0
    require(result.get('status') == 'complete' or failed_invalid,
            'Unexpected execution failure; cannot count as a completed sample')
    command = stage(folder, 'quality-only', 1 if failed_invalid else 0,
                    [REMOTE + '/env/.venv/bin/python', REMOTE + '/controller/quality_repeat.py',
                     snapshot, rp, '--worker'], result)
    fp_argv = [REMOTE + '/env/.venv/bin/python', REMOTE + '/controller/final_fingerprint.py',
               snapshot, REMOTE + '/controller/pcieccl.toml']
    before_command = stage(folder, 'fingerprint-before', 0,
                           fp_argv + [rp + '/fingerprint-before.json'], result)
    require(before_command['finished'] <= command['started'], 'Fingerprint order reversed')
    before = remembered(folder / 'fingerprint-before.json')
    require(before.get('snapshot') == snapshot, 'Before fingerprint snapshot mismatch')
    require(before['files']['source/vllm-0.26.0/csrc/pcieccl/pcieccl.cu'] == PINNED[snapshot][0]
            and before['files']['installed/extension'] == PINNED[snapshot][1],
            'Before fingerprint source/binary mismatch')
    for key in ('frozen_files', 'versions', 'normalized_environment'):
        require(before[key] == reference['provenance'][key], 'Frozen fingerprint mismatch: ' + key)
    gsm = remembered(folder / 'lifecycle/gsm8k.command.json')
    expected_gsm = [REMOTE + '/env/.venv/bin/python',
                    REMOTE + '/source/vllm-0.26.0/tests/evals/gsm8k/gsm8k_eval.py',
                    *GSM_ARGS, '--save-results', rp + '/lifecycle/gsm8k.json']
    require(gsm.get('argv') == expected_gsm and gsm.get('protocol_sha256') == PROTOCOL_SHA
            and gsm.get('quality_only') is True, 'Original GSM command/protocol mismatch')
    require(number(gsm.get('started')) and number(q.get('timestamp'))
            and command['started'] <= gsm['started'] <= q['timestamp'] <= command['finished'],
            'GSM payload timestamp is not from this worker interval')
    if failed_invalid:
        require(result.get('error') == "RuntimeError('quality-only failed with exit code 1')",
                'Failure is not the expected completed invalid-answer gate')
        log = (folder / 'quality-only.log').read_text()
        artifacts.setdefault('quality-only.log', digest(folder / 'quality-only.log'))
        require(log.rstrip().splitlines()[-1] == 'RuntimeError: Invalid quality result field: invalid_rate',
                'Worker terminal failure is not specifically invalid_rate')
        require(not (folder / 'fingerprint-after.json').exists()
                and not (folder / 'fingerprint-after.command.json').exists(),
                'Unexpected post-fingerprint artifact on unchanged invalid-failure path')
        post_verified = False
    else:
        require(invalid == 0 and not result.get('error'), 'Completed quality gate has invalid answers/error')
        after_command = stage(folder, 'fingerprint-after', 0,
                              fp_argv + [rp + '/fingerprint-after.json', '--compare',
                                         rp + '/fingerprint-before.json'], result)
        require(command['finished'] <= after_command['started'], 'After fingerprint order reversed')
        after = remembered(folder / 'fingerprint-after.json')
        require(after == before, 'Before/after provenance changed')
        require(result.get('correct_count') == correct and result.get('quality') == q
                and result.get('quality_sha256') == digest(folder / 'lifecycle/gsm8k.json')
                and result.get('fingerprint_sha256') == digest(folder / 'fingerprint-after.json'),
                'Complete result does not match its quality/fingerprint artifacts')
        require(result.get('identity', {}).get('source_sha256') == PINNED[snapshot][0]
                and result.get('identity', {}).get('binary_sha256') == PINNED[snapshot][1],
                'Complete result snapshot mismatch')
        post_verified = True
    for name in ('quality-only.command.json', 'fingerprint-before.command.json'):
        artifacts.setdefault(name, digest(folder / name))
    if post_verified:
        artifacts.setdefault('fingerprint-after.command.json', digest(folder / 'fingerprint-after.command.json'))
    require(all(digest(folder / name) == value for name, value in artifacts.items()),
            'Input artifacts changed during summary')
    return {'index': index, 'snapshot': snapshot, 'correct': correct, 'invalid': invalid,
            'questions': 1319, 'sample_completed': True, 'gate_status': result['status'],
            'gate_pass': not failed_invalid, 'pre_fingerprint_verified': True,
            'post_fingerprint_present': post_verified, 'post_fingerprint_verified': post_verified,
            'count_interpretation': 'descriptive only; invalid gate and missing post provenance'
                                    if failed_invalid else 'valid quality/provenance sample',
            'started': result['started'], 'finished': result['finished'], 'artifact_sha256': artifacts}


def summarize(repeat_root):
    report = {'protocol': 'quality-repeat-summary-v1', 'scored': False,
              'predeclared_order': list(ORDER), 'samples': [], 'errors': [],
              'all_quality_gates_pass': False, 'quality_preserving_claim_allowed': False,
              'quality_preserving_win_claim': False, 'postprovenancelimits': [],
              'limitation': 'Raw invalid-answer counts are preserved descriptively. A favorable median cannot waive zero-invalid or missing post-run provenance.'}
    try:
        require(digest(ROOT / 'controller/quality-repeat-protocol.json') == PROTOCOL_SHA,
                'Predeclared protocol changed')
        refpath = ROOT / 'results/noise/reference-nccl-wave-01.json'
        require(digest(refpath) == REFERENCE_SHA, 'Immutable reference changed')
        reference = read(refpath)
        firsts = {}
        for name, path, expected_hash in (
            ('initial', ROOT / 'results/quality/gsm8k-pcieccl.json',
             'dedfdc725346acfab514bcadb7a36600384e30aa97a7c13289e84312aedae6e5'),
            ('c03', ROOT / 'results/final-validation/quality/gsm8k-pcieccl.json',
             '0486d3de42fde05db1cf0d733e22cfb06e2c3ce2ab606a7eb4a508c4cf2d3775')):
            require(digest(path) == expected_hash, 'First quality sample changed: ' + name)
            correct, invalid = quality_counts(read(path))
            require(invalid == 0, 'First quality sample has invalid answers')
            firsts[name] = {'correct': correct, 'invalid': invalid, 'sha256': expected_hash}
        require(firsts['initial']['correct'] == 1222 and firsts['c03']['correct'] == 1210,
                'Wrong first quality counts')
        expected_dirs = {f'repeat-{i}-{name}' for i, name in enumerate(ORDER, 1)}
        require({p.name for p in Path(repeat_root).glob('repeat-*') if p.is_dir()} == expected_dirs,
                'Missing or extra repeat directories')
        for i, name in enumerate(ORDER, 1):
            value = sample(Path(repeat_root) / f'repeat-{i}-{name}', i, name, reference)
            if report['samples']:
                require(report['samples'][-1]['finished'] <= value['started'], 'Repeat order overlaps/reverses')
            report['samples'].append(value)
        report['first_samples'] = firsts
        groups = {name: [firsts[name]['correct']] + [s['correct'] for s in report['samples']
                                                  if s['snapshot'] == name] for name in ('initial', 'c03')}
        medians = {name: statistics.median(values) for name, values in groups.items()}
        all_pass = all(s['gate_pass'] and s['post_fingerprint_verified'] for s in report['samples'])
        no_decline = medians['c03'] >= medians['initial']
        labels = {'initial': 'original', 'c03': 'best'}
        invalids = {name: [firsts[name]['invalid']] + [sample['invalid'] for sample in report['samples']
                                                     if sample['snapshot'] == name] for name in ORDER[:2]}
        limits = [f"repeat-{sample['index']}-{sample['snapshot']}: invalid-answer gate failed; "
                  "post-run fingerprint was not executed, so post-run provenance is unverified"
                  for sample in report['samples'] if not sample['post_fingerprint_verified']]
        report.update(correct_counts={labels[name]: values for name, values in groups.items()},
                      median_correct={labels[name]: value for name, value in medians.items()},
                      invalid_counts={labels[name]: values for name, values in invalids.items()},
                      median_rule_met=no_decline, postprovenancelimits=limits)
        report.update(complete_planned_sequence=True, descriptive_correct_counts=groups,
                      descriptive_medians=medians, descriptive_median_no_decline=no_decline,
                      all_quality_gates_pass=all_pass,
                      quality_preserving_claim_allowed=all_pass and no_decline,
                      status='no-observed-quality-decline' if all_pass and no_decline else 'quality-unconfirmed')
        return report, 0 if all_pass and no_decline else 1
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        report['errors'].append(str(error))
        report.update(status='incomplete-or-unexpected-evidence', complete_planned_sequence=False)
        return report, 2


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write('\n')
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repeat_root', type=Path)
    parser.add_argument('output', type=Path, nargs='?')
    args = parser.parse_args()
    if args.output is None:
        args.output = args.repeat_root / 'quality-summary.json'
    require(not args.output.exists(), 'Refusing existing summary output')
    report, code = summarize(args.repeat_root)
    report['summary_controller_sha256'] = digest(__file__)
    write_once(args.output, report)
    print(json.dumps({'status': report['status'], 'quality_preserving_claim_allowed':
                      report['quality_preserving_claim_allowed'], 'output': str(args.output)}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
