from run_sweeps import ENV,LITE,OUT,users
import subprocess,time,json
CASES=[
 ('policy','router_policy_check',[]),
 ('dispatcher','dispatcher_check',[]),
 ('executor','router_core_check',[]),
 ('bf16','cpu_bf16_check',[]),
 ('capture_and_boundaries','dispatcher_cuda_check',[]),
 ('eager_4_batch','router_check',['--devices','0,1,4,5','--batch-cpu-jobs','--gpu-launch-workers-per-node','1','--delay-rank','3']),
 ('eager_8_batch','router_check',['--gpus','8','--batch-cpu-jobs','--gpu-launch-workers-per-node','4','--delay-rank','7']),
 ('eager_4_gpu','router_check',['--devices','0,1,4,5','--gpu-progress','--gpu-launch-workers-per-node','4']),
 ('eager_8_per_source','router_check',['--gpus','8','--ag-per-source','--gpu-launch-workers-per-node','4','--delay-rank','7']),
 ('graph_4_batch','router_graph_check',['--devices','0,1,4,5','--batch-cpu-jobs','--gpu-launch-workers-per-node','1','--elements','32771']),
 ('graph_8_per_source','router_graph_check',['--gpus','8','--ag-per-source','--gpu-launch-workers-per-node','4','--elements','32771']),
 ('oracle_4_cpu','gpu_reduction_check',['--devices','0,1,4,5','--cpu-reduce','--batch-cpu-jobs','--gpu-launch-workers-per-node','1']),
 ('oracle_4_gpu','gpu_reduction_check',['--devices','0,1,4,5','--gpu-progress','--gpu-launch-workers-per-node','4']),
]
manifest=[]
for name,binary,extra in CASES:
 if users():raise RuntimeError('GPU busy before correctness check')
 command=[str(OUT/('uccl_lite_'+binary)),*extra]
 t=time.monotonic()
 with (OUT/(name+'.log')).open('w') as log:
  result=subprocess.run(command,cwd=LITE,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 record=dict(name=name,command=command,returncode=result.returncode,seconds=time.monotonic()-t)
 manifest.append(record);(OUT/'correctness.json').write_text(json.dumps(manifest,indent=2))
 print(name,result.returncode,round(time.monotonic()-t,1),flush=True)
 if result.returncode:
  print((OUT/(name+'.log')).read_text()[-3000:],flush=True);raise SystemExit(result.returncode)
 print((OUT/(name+'.log')).read_text().splitlines()[-1],flush=True)
