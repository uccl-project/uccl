"""Archive measured logs/metadata only; keep binaries and build artifacts in build/."""
import hashlib
import json
import tarfile

from run import DATA, OUT

paths = sorted(p for p in OUT.iterdir() if p.suffix in ('.json', '.stdout', '.stderr')
               or '.debug.' in p.name or p.name.endswith('driver.log'))
assert len(list(OUT.glob('calibrate_*.json'))) == 42
assert len(list(OUT.glob('evaluate_*.json'))) == 18
with tarfile.open(DATA / 'measurements.tar.gz', 'w:gz') as archive:
    for path in paths:
        archive.add(path, arcname=path.name)
checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(DATA.iterdir()) if p.is_file() and p.name != 'checksums.json'}
(DATA / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
