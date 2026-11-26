from run_sweeps import ENV,LITE,OUT,users
import subprocess,time,json
CASES=[
 ('policy','router_policy_check',[]),('dispatcher','dispatcher_check',[]),('executor','router_core_check',[]),('bf16','cpu_bf16_check',[]),('capture_guards','dispatcher_cuda_check',[]),
 ('eager_4_default','router_check',['--devices','0,1,4,5']),
 ('eager_4_numa','router_check',['--devices','0,1,4,5','--gpu-launch-workers-per-node','1','--delay-rank','3']),
 ('eager_8_rank','router_check',['--gpus','8','--gpu-launch-workers-per-node','4','--delay-rank','7','--small-rs-workers-per-node','4','--batch-cpu-jobs','--packet-bytes','131072']),
 ('eager_3_striped','router_check',['--devices','0,4,1','--gpu-launch-workers-per-node','4','--cpu-interleave-scatter','--direct-scatter-results','--batch-cpu-jobs']),
 ('eager_4_gpu','router_check',['--devices','0,1,4,5','--gpu-launch-workers-per-node','4','--gpu-progress','--interleave-scatter']),
 ('graph_4','router_graph_check',['--devices','0,1,4,5','--gpu-launch-workers-per-node','1','--elements','32771','--batch-cpu-jobs']),
 ('graph_8','router_graph_check',['--gpus','8','--gpu-launch-workers-per-node','4','--elements','65537','--packet-bytes','131072','--cpu-interleave-scatter','--batch-cpu-jobs']),
 ('oracle_4_cpu','gpu_reduction_check',['--devices','0,1,4,5','--gpu-launch-workers-per-node','1','--cpu-interleave-scatter','--batch-cpu-jobs']),
 ('oracle_8_cpu','gpu_reduction_check',['--gpus','8','--gpu-launch-workers-per-node','4','--cpu-interleave-scatter','--direct-scatter-results','--batch-cpu-jobs','--packet-bytes','131072']),
]
manifest=[]
for name,binary,extra in CASES:
 before=users()
 if before:raise RuntimeError('GPU busy: '+before)
 command=[str(OUT/('uccl_lite_'+binary)),*extra]
 t=time.monotonic()
 with (OUT/(name+'.log')).open('w') as log:
  result=subprocess.run(command,cwd=LITE,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 record=dict(name=name,command=command,returncode=result.returncode,seconds=time.monotonic()-t)
 manifest.append(record);(OUT/'correctness.json').write_text(json.dumps(manifest,indent=2))
 print(name,result.returncode,round(time.monotonic()-t,1),flush=True)
 if result.returncode:
  print((OUT/(name+'.log')).read_text()[-2000:],flush=True);raise SystemExit(result.returncode)
 print((OUT/(name+'.log')).read_text().splitlines()[-1],flush=True)
