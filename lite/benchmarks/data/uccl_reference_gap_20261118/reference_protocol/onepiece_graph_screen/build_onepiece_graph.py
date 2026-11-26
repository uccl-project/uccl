import json,subprocess
from pathlib import Path
old='build/uccl_graph_banks_combo';base=json.loads(Path(old+'/build_command.json').read_text())
for name,maximum in [('uccl_graph_single32',33554432),('uccl_graph_single64',67108864)]:
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='));cmd[i]+=f' -DUCCL_LITE_RS_GRAPH_SINGLE_MAX_BYTES={maximum}'
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
