from run_sweeps import *
manifest=[]
for name,extra in [('trace_spin_serial_1k',['--small-rs-workers-per-node','4','--batch-cpu-jobs']),('trace_spin_rank_1k',['--small-rs-workers-per-node','4','--batch-cpu-jobs','--gpu-launch-workers-per-node','4']),('trace_hybrid_rank_1k',['--small-rs-workers-per-node','4','--batch-cpu-jobs','--gpu-launch-workers-per-node','4','--small-rs-gpu-progress-max-bytes','65536'])]:
 if users():raise RuntimeError('GPU busy before trace')
 env=ENV.copy();env['UCCL_LITE_TIMELINE']='1'
 command=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--output='+str(OUT/name),str(OUT/'uccl_lite_collectives'),'--gpus','8','--router-only','--op','reducescatter','--begin','1024','--end','1024','--warmup','5','--iterations','10','--repeats','3','--main-cpu','47',*extra]
 with (OUT/(name+'.log')).open('w') as log:
  result=subprocess.run(command,cwd=LITE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 manifest.append(dict(name=name,command=command,environment=env,returncode=result.returncode))
 (OUT/'trace_final_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(name,result.returncode,flush=True)
 if result.returncode or users():raise RuntimeError('trace failed or GPU busy')
 subprocess.run(['/usr/local/cuda/bin/nsys','export','--type=sqlite','--output='+str(OUT/(name+'.sqlite')),str(OUT/(name+'.nsys-rep'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
