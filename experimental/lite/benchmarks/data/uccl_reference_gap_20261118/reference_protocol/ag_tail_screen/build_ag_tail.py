import json,subprocess,hashlib,shutil
from pathlib import Path
old='build/uccl_graph_single64';base=json.loads(Path(old+'/build_command.json').read_text())
variants={'uccl_ag_compact_barriers':['-DUCCL_LITE_AG_COMPACT_BARRIERS'],'uccl_ag_early_own':['-DUCCL_LITE_AG_EARLY_OWN'],'uccl_ag_tail_combo':['-DUCCL_LITE_AG_COMPACT_BARRIERS','-DUCCL_LITE_AG_EARLY_OWN','-DUCCL_LITE_AG_COALESCED_FLUSH'],'uccl_ag_tail_ready':['-DUCCL_LITE_AG_COMPACT_BARRIERS','-DUCCL_LITE_AG_EARLY_OWN','-DUCCL_LITE_AG_COALESCED_FLUSH','-DUCCL_LITE_AG_READY_FLAGS']}
for path,digest in json.loads(Path('build/ag_tail_expected_sources.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
for name,extra in variants.items():
 b=Path('build')/name;b.mkdir(exist_ok=False)
 cmd=[x.replace(old,str(b)) for x in base if not x.endswith('/graph_queue_check')]
 i=next(i for i,x in enumerate(cmd) if x.startswith('EXTRA_NVCC_FLAGS='));cmd[i]+=' '+' '.join(extra)
 (b/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print('Building',name,flush=True)
 with (b/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for path in json.loads(Path('build/ag_tail_expected_sources.json').read_text()):
  p=Path(path);q=b/'sources'/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 shutil.copy2('build/ag_tail_expected_sources.json',b/'sources.json')
