from run_sweeps import *
import shlex
profile=OUT/'final_hybrid_parallel_a.profile'
expected={}
for line in profile.read_text().splitlines()[1:]:
 r=shlex.split(line);expected[int(r[3])]=float(r[12])/float(r[9])>=1.03 and float(r[10])<float(r[11])
cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs','--main-cpu','47','--small-rs-gpu-progress-max-bytes','65536']
manifest=[]
for name,begin,end,extra in [('auto_all',1024,4194304,['--gpu-launch-workers-per-node','4']),('auto_unknown',1536,1536,['--gpu-launch-workers-per-node','4']),('auto_changed_launch',1024,4194304,[])]:
 if users():raise RuntimeError('GPU busy before replay')
 command=[str(OUT/'uccl_lite_collectives'),'--gpus','8','--op','reducescatter','--fallback','nccl','--begin',str(begin),'--end',str(end),'--warmup','3','--iterations','5','--repeats','3','--profile',str(profile),*cap,*extra]
 with (OUT/(name+'.csv')).open('w') as stdout,(OUT/(name+'.log')).open('w') as stderr:
  result=subprocess.run(command,cwd=LITE,env=ENV,stdout=stdout,stderr=stderr,timeout=120)
 rows=list(csv.DictReader(line for line in (OUT/(name+'.csv')).read_text().splitlines() if not line.startswith('#')))
 if result.returncode or users():raise RuntimeError(name+' replay failed')
 decisions={int(r['bytes']):r['backend']!='fallback' for r in rows}
 if name=='auto_all':assert decisions==expected,(decisions,expected)
 else:
  assert all(not selected for selected in decisions.values()),decisions
  assert all(r['dispatcher_router_initialized']=='0' for r in rows),rows
  assert '# PacketPipeline' not in (OUT/(name+'.log')).read_text()
 record=dict(name=name,command=command,environment=ENV,returncode=result.returncode,decisions=decisions)
 manifest.append(record);(OUT/'auto_replay.json').write_text(json.dumps(manifest,indent=2))
 print(name,decisions,flush=True)
