import csv,json,os,subprocess,time
from pathlib import Path
LITE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ENV={k:os.environ[k] for k in ('HOME','PATH') if k in os.environ}
ENV.update(LD_LIBRARY_PATH=str(LITE/'build/nccl-2.28.9/lib')+':/usr/local/cuda-12.9/lib64',NCCL_NET='Socket',NCCL_NET_PLUGIN='none',NCCL_IB_DISABLE='1')
def users():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
def run(name,binary,extra=(),begin=1024,end=1048576,op='reducescatter'):
 command=[str(binary),'--gpus','8','--op',op,'--fallback','nccl','--begin',str(begin),'--end',str(end),'--warmup','10','--iterations','20','--repeats','5','--main-cpu','23','--calibrate',str(OUT/(name+'.profile')),'--samples',str(OUT/(name+'_samples.csv')),*extra]
 before=users()
 if before:raise RuntimeError('GPU busy before sweep: '+before)
 t=time.monotonic()
 with (OUT/(name+'.csv')).open('w') as stdout,(OUT/(name+'.log')).open('w') as stderr:
  result=subprocess.run(command,cwd=LITE,env=ENV,stdout=stdout,stderr=stderr,timeout=180)
 after=users()
 record=dict(name=name,command=command,environment=ENV,seconds=time.monotonic()-t,returncode=result.returncode,gpu_users_before=before,gpu_users_after=after)
 (OUT/(name+'.json')).write_text(json.dumps(record,indent=2))
 if result.returncode or after:raise RuntimeError(str(record)+'\n'+(OUT/(name+'.log')).read_text()[-2000:])
 rows=list(csv.DictReader(line for line in (OUT/(name+'.csv')).read_text().splitlines() if not line.startswith('#')))
 print(name,round(time.monotonic()-t,1),[(r['bytes'],r['median_us'],r['speedup']) for r in rows if r['backend']!='fallback'],flush=True)
 return rows
if __name__=='__main__':
 cap=['--small-rs-workers-per-node','4','--batch-cpu-jobs']
 for name,binary in [('guard_control_a',LITE/'build/rs_opt_20261113/uccl_lite_collectives'),('guard_single_a',OUT/'uccl_lite_collectives'),('guard_single_b',OUT/'uccl_lite_collectives'),('guard_control_b',LITE/'build/rs_opt_20261113/uccl_lite_collectives')]:
  run(name,binary,cap)
