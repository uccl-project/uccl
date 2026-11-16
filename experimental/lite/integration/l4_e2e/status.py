#!/usr/bin/env python3
"""Print finished cells and the last emitted cell from jobs still running."""
import json
from pathlib import Path
for root in (Path('build/l4_e2e/packet_final_4l4'),Path('build/l4_e2e/packet_final_2x4l4'),Path('build/l4_e2e/unbound_large_screen'),Path('build/l4_e2e/unbound_xl_screen')):
    print(root.name)
    for path in sorted(root.glob('*.log')):
        rows=[]
        for line in path.read_text().splitlines():
            if line.startswith('{'):
                try:
                    r=json.loads(line); rows.append((r['case'], round(r['median_s']*1000,2)))
                except (ValueError, KeyError): pass
        print(path.stem,rows)
