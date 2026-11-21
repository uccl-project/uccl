"""Fresh processes balance method order, with an optional routing ablation."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def compute_apps() -> str:
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                    '--format=csv,noheader'], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--mma-library', type=Path, required=True)
    parser.add_argument('--ht-library', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ctas', type=int, required=True)
    parser.add_argument('--routing-ablation', action='store_true')
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.iterdir() if p.is_file()}
    libraries = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (args.mma_library, args.ht_library)}
    (args.results/'manifest.json').write_text(json.dumps({'source': manifest, 'libraries': libraries}, indent=2)+'\n')
    sessions = 4 if args.routing_ablation else 6
    for session in range(sessions):
        assert not compute_apps(), 'wait for existing GPU work to finish'
        directory = args.results/f'session-{session}'; directory.mkdir()
        command = [sys.executable, '-u', str(source/'model_e2e.py'), '--weights', str(args.weights),
                   '--mma-library', str(args.mma_library), '--ht-library', str(args.ht_library),
                   '--ctas', str(args.ctas), '--batch-size', '64', '--concurrency', '128',
                   '--prompt-tokens', '128', '--new-tokens', '4', '--warmups', '1',
                   '--repeats', '3', '--session', str(session)]
        if args.routing_ablation:
            command.append('--routing-ablation')
        (directory/'started.json').write_text(json.dumps({'command': command, 'session': session}, indent=2)+'\n')
        started = time.monotonic()
        with (directory/'run.log').open('w') as log:
            result = subprocess.run(['timeout', '--kill-after=10s', '900s', *command],
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        reports = [json.loads(line) for line in (directory/'run.log').read_text().splitlines() if line.startswith('{')]
        receipt = {'session': session, 'exit_code': result.returncode,
                   'elapsed_s': time.monotonic()-started, 'after_gpu_apps': compute_apps(),
                   'log_sha256': hashlib.sha256((directory/'run.log').read_bytes()).hexdigest(), 'reports': reports}
        (directory/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        assert result.returncode == 0 and not receipt['after_gpu_apps'], f'failed session {session}; preserve its logs'
        assert reports[-1]['test'] == 'full_qwen_model_e2e_summary' and reports[-1]['status'] == 'PASS'
        print(json.dumps({'session': session, 'elapsed_s': receipt['elapsed_s'], 'status': 'PASS'}), flush=True)
    (args.results/'complete.json').write_text(json.dumps({'sessions': sessions, 'status': 'PASS'})+'\n')


if __name__ == '__main__':
    main()
