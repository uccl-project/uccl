from run_sweeps import *
import shlex
manifest=[]
flags=['--batch-cpu-jobs','--gpu-launch-workers-per-node','4','--small-ar-gpu-progress-max-bytes','8192','--small-ag-gpu-progress-max-bytes','1048576','--main-cpu','47']
for op in ['allreduce','allgather']:
 profile=OUT/('final_'+op+'_combined_a.profile')
 expected={}
 for line in profile.read_text().splitlines()[1:]:
  row=shlex.split(line)
  expected[int(row[3])]=int(row[7])==1 and int(row[6])>=3 and float(row[12])/float(row[9])>=1.03 and float(row[10])<float(row[11])
 changed=['--small-ar-gpu-progress-max-bytes','4096'] if op=='allreduce' else ['--small-ag-gpu-progress-max-bytes','524288']
 for name,begin,end,extra in [('known',1024,4194304,[]),('unknown',1536,1536,[]),('changed_threshold',1024,4194304,changed)]:
  if users():raise RuntimeError('GPU busy before replay')
  prefix='auto_'+op+'_'+name
  command=[str(OUT/'uccl_lite_collectives'),'--gpus','8','--op',op,'--fallback','nccl','--begin',str(begin),'--end',str(end),'--warmup','3','--iterations','5','--repeats','3','--profile',str(profile)]+flags+(['--in-place'] if op=='allreduce' else [])+extra
  with (OUT/(prefix+'.csv')).open('w') as stdout,(OUT/(prefix+'.log')).open('w') as stderr:
   result=subprocess.run(command,cwd=LITE,env=ENV,stdout=stdout,stderr=stderr,timeout=120)
  rows=list(csv.DictReader(line for line in (OUT/(prefix+'.csv')).read_text().splitlines() if not line.startswith('#')))
  assert result.returncode==0 and not users(),prefix
  decisions={int(row['bytes']):row['backend']!='fallback' for row in rows}
  if name=='known':assert decisions==expected,(decisions,expected)
  else:
   assert all(not selected for selected in decisions.values()),decisions
   assert all(row['dispatcher_router_initialized']=='0' for row in rows),rows
   assert '# PacketPipeline' not in (OUT/(prefix+'.log')).read_text()
  manifest.append(dict(name=prefix,command=command,environment=ENV,returncode=result.returncode,decisions=decisions))
  (OUT/'auto_replay.json').write_text(json.dumps(manifest,indent=2))
  print(prefix,decisions,flush=True)
