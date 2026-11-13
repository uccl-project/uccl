"""Verify existing pinned checkpoint shards and create the model-driver receipt."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(weights: Path, manifest: dict) -> dict:
    shards = []
    for expected in manifest['shards']:
        name = expected['name']
        if Path(name).name != name:
            raise ValueError('Shard names must be plain filenames')
        path = weights / name
        if path.is_symlink() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Invalid shard size or symlink: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Shard checksum differs from pinned checkpoint: {name}')
        shards.append({'name': name, 'bytes': expected['bytes'], 'sha256': digest.hexdigest()})
    total = sum(row['bytes'] for row in shards)
    if total != manifest['bytes'] or list(weights.glob('*.part')):
        raise ValueError('Checkpoint is incomplete')
    return {'model': manifest['model'], 'revision': manifest['revision'],
            'bytes': total, 'shards': shards, 'status': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).with_name('checkpoint_manifest.json'))
    args = parser.parse_args()
    result = verify(args.weights, json.loads(args.manifest.read_text()))
    path = args.weights / 'download-complete.json'
    if path.exists():
        if path.is_symlink():
            raise ValueError('Existing receipt is a symlink')
        old = json.loads(path.read_text())
        if any(old[key] != result[key] for key in ('model', 'revision', 'bytes', 'shards')):
            raise ValueError('Existing receipt differs; it has been preserved')
    else:
        with path.open('x') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps({'status': 'PASS', 'shards': len(result['shards']), 'bytes': result['bytes']}))


if __name__ == '__main__':
    main()
