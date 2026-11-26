"""Transport-only follow-up; unchanged serving cases and timing implementation."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'controller'))
import reproduce as original

BaseRunner = original.ReproductionRunner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transport(folder, forced):
    paths = sorted(folder.glob('nccl.*.log'))
    rows = []
    for p in paths:
        for line in p.read_text(errors='replace').splitlines():
            if 'Channel ' in line and ' -> ' in line and ' via ' in line:
                rows.append({'file': p.name, 'line': line})
    p2p = [r for r in rows if ' via P2P/' in r['line']]
    other = [r for r in rows if ' via P2P/' not in r['line']]
    ranks = sorted({int(m.group(1)) for r in p2p if (m := re.search(r'\[(\d+)\] NCCL', r['line']))})
    result = {'forced': forced, 'channel_rows': len(rows), 'p2p_rows': len(p2p),
              'other_rows': other, 'p2p_logging_ranks': ranks, 'rows': rows,
              'log_sha256': {p.name: sha(p) for p in paths}}
    if not rows:
        raise RuntimeError('No channel transport evidence in NCCL logs')
    if forced and (other or ranks != list(range(8))):
        raise RuntimeError('Forced-P2P transport evidence incomplete or contains fallback: ' + json.dumps(result))
    original.atomic_json(folder / 'transport.json', result)
    print('[transport]', folder.name, len(p2p), 'P2P rows;', len(other), 'other rows; ranks', ranks, flush=True)
    return result


class VerifiedRunner(BaseRunner):
    async def wait_ready(self, process, log_path):
        await super().wait_ready(process, log_path)
        folder = self.config.result_dir
        transport(folder, folder.name == 'p2p')
        records = []
        for pid in sorted(original.live_gpu_pids()):
            raw = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            env = dict(x.decode().split('=', 1) for x in raw if b'=' in x)
            maps = Path(f'/proc/{pid}/maps').read_text().splitlines()
            libraries = sorted({line.split()[-1] for line in maps if 'libnccl.so' in line})
            records.append({'pid': pid, 'environment': {k: v for k, v in env.items() if k.startswith(('NCCL_', 'VLLM_'))}, 'nccl_libraries': libraries})
            records[-1]['environment_note'] = '/proc environ is a startup-memory snapshot, not a reliable current environment assertion.'
            expected_library = ROOT / 'env/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2'
            if not libraries or any(Path(p).resolve() != expected_library.resolve() for p in libraries):
                raise RuntimeError(f'Worker {pid} loaded unexpected NCCL library: {libraries}')
            worker_logs = list(folder.glob(f'nccl.*.{pid}.log'))
            if len(worker_logs) != 1:
                raise RuntimeError(f'Missing NCCL log for worker {pid}')
            log = worker_logs[0].read_text()
            if folder.name == 'p2p' and 'NCCL_P2P_LEVEL set by environment to SYS' not in log:
                raise RuntimeError(f'Worker {pid} did not report SYS P2P scope')
        if len(records) != 8:
            raise RuntimeError(f'Expected eight GPU workers, observed {len(records)}')
        original.atomic_json(folder / 'worker-runtime.json', records)


async def run(config):
    original.ReproductionRunner = VerifiedRunner
    await original.evaluate([str(config)], False)
    cfg = original.load_config(config)
    transport(cfg.result_dir, cfg.result_dir.name == 'p2p')


def main():
    mode = sys.argv[1]
    if mode not in ('p2p', 'default'):
        raise RuntimeError('Unknown mode')
    config = ROOT / 'forced-p2p' / (mode + '.toml')
    base = tomllib.loads((ROOT / 'controller/final-nccl.toml').read_text())
    actual = tomllib.loads(config.read_text())
    allowed = {'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_DEBUG_FILE'}
    if mode == 'p2p':
        allowed |= {'NCCL_P2P_DISABLE', 'NCCL_P2P_LEVEL', 'NCCL_SHM_DISABLE'}
    extra = {k: actual['inference']['environment'].pop(k) for k in allowed}
    actual['benchmark']['result-dir'] = base['benchmark']['result-dir']
    if actual != base:
        raise RuntimeError('Unexpected configuration difference')
    if mode == 'p2p' and any(extra[k] != v for k, v in {'NCCL_P2P_DISABLE':'0', 'NCCL_P2P_LEVEL':'SYS', 'NCCL_SHM_DISABLE':'1'}.items()):
        raise RuntimeError('Incorrect forced transport')
    folder = ROOT / 'forced-p2p' / mode
    if folder.exists():
        raise RuntimeError('Refusing to reuse result directory')
    folder.mkdir()
    original.atomic_json(folder / 'followup-config-audit.json', {'base': str(ROOT / 'controller/final-nccl.toml'), 'changes': extra, 'config_sha256': sha(config), 'runner_sha256': sha(Path(__file__)), 'start': time.time()})
    with open('/home/wxzheng/uccl-eval.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(config))


if __name__ == '__main__':
    main()
