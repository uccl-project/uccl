#!/usr/bin/env python3
"""Compare kernel instruction encodings, ignoring changed symbol names only.

Usage: compare_sass.py BEFORE.sass AFTER.sass KERNEL [KERNEL ...]
Input: CUDA cuobjdump -sass output. Exit nonzero on an instruction mismatch.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

before, after, *kernels = sys.argv[1:]
def extract(path, kernel):
    blocks = [b for b in Path(path).read_text().split('Function : ')[1:]
              if re.search(str(len(kernel)) + re.escape(kernel) + 'E',
                           b.splitlines()[0])]
    if len(blocks) != 1:
        raise RuntimeError(f'{kernel}: found {len(blocks)} functions in {path}')
    # Both halves of every instruction, including scheduling control bits.
    words = re.findall(r'/\* (0x[0-9a-f]+) \*/', blocks[0])
    if not words:
        raise RuntimeError(f'{kernel}: no instruction encodings')
    return words

records = []
for kernel in kernels:
    a, b = extract(before, kernel), extract(after, kernel)
    digest = lambda x: hashlib.sha256('\n'.join(x).encode()).hexdigest()
    records.append(dict(kernel=kernel, before_words=len(a), after_words=len(b),
                        before_sha256=digest(a), after_sha256=digest(b),
                        identical=a == b))
print(json.dumps(records, indent=2))
sys.exit(0 if all(r['identical'] for r in records) else 1)
