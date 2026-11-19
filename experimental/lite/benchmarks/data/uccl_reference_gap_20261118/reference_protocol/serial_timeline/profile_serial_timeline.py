import os,json,subprocess,re,shutil,array,statistics,csv,hashlib
from pathlib import Path
root=Path.cwd();base=root/'build/uccl_serial_timeline';stage=base/'abi'
exe=stage/'reference_timing_check';shutil.copy2(base/'nccl_lite/reference_timing_check',exe)
subprocess.run(['rsync','-a',str(exe),f'10.10.55.1:{exe}'],check=True)
old=root/'build/uccl_reference_protocol/abi'
smoke=json.loads((root/'build/uccl_reference_protocol/smoke_retry/manifest.json').read_text())
cmd=[x.replace(str(old),str(stage)) for x in smoke['commands'][0][:-6]]
cmd[1:1]=['-x','UCCL_LITE_SERIAL_TIMELINE_BYTES','-x','UCCL_LITE_SERIAL_TIMELINE_DIR']
env=os.environ.copy();assert not env.get('LD_PRELOAD');env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib:'+env.get('LD_LIBRARY_PATH',''))
names=['input_stage','packet_send_and_flush_post','own_fanout','remote_receive','remote_fanout','local_ready_and_barrier','output_copy','final_flush_wait']
summary=[]
for size in [65536,131072]:
 for layout in ['out','in']:
  out=base/f'trace_{size}_{layout}';out.mkdir(exist_ok=False)
  subprocess.run(['ssh','10.10.55.1','mkdir','-p',str(out)],check=True)
  env.update(UCCL_LITE_SERIAL_TIMELINE_BYTES=str(size),UCCL_LITE_SERIAL_TIMELINE_DIR=str(out))
  args=cmd+['ag',layout,str(size),str(size),'20','30','1073741824']
  (out/'command.json').write_text(json.dumps(args,indent=2)+'\n');print('Profiling',size,layout,flush=True)
  with (out/'run.log').open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  text=(out/'run.log').read_text();assert r.returncode==0 and 'UCCL reference-protocol checks passed' in text
  assert sorted(map(int,re.findall(r'UCCL Lite strict: rank=(\d+) native_nccl_loaded=0',text)))==list(range(8))
  assert sorted((int(a),int(b)) for a,b in re.findall(r'UCCL Lite strict: rank=(\d+) fallback=(\d+)',text))==[(r,0) for r in range(8)]
  routes={(int(r),int(op),int(s),int(n)) for r,op,s,n in re.findall(r'UCCL Lite strict route: rank=(\d+) op=(\d+) bytes=(\d+) calls=(\d+)',text)};assert routes=={(r,1,size,50) for r in range(8)}
  subprocess.run(['rsync','-a',f'10.10.55.1:{out}/',str(out)+'/'],check=True)
  rows=[]
  for rank in range(8):
   meta=json.loads((out/f'rank_{rank}.json').read_text());assert meta==dict(rank=rank,bytes=size,calls=50,words_per_call=64)
   words=array.array('Q');words.frombytes((out/f'rank_{rank}.bin').read_bytes());assert len(words)==50*64
   for call in range(50):
    trace=words[call*64:(call+1)*64];assert trace[0]==(1<<32)|1
    assert not any(trace[17:]);origin=trace[1]>>32
    for event,name in enumerate(names):
     begin,end=trace[1+2*event:3+2*event]
     assert (begin&4095)==event*4 and (end&4095)==event*4+1
     start=(begin>>32)-origin;duration=((end>>32)-(begin>>32))&0xffffffff
     rows.append(dict(rank=rank,call=call,warmup=call<20,event=name,start_us=(start&0xffffffff)/1000,duration_us=duration/1000))
  with (out/'phases.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  for rank in range(8):
   for name in names:
    values=[r['duration_us'] for r in rows if r['rank']==rank and r['event']==name and not r['warmup']]
    summary.append(dict(bytes=size,layout=layout,rank=rank,event=name,median_us=statistics.median(values),mean_us=statistics.mean(values),max_us=max(values)))
  (out/'result.json').write_text(json.dumps({'correct':True,'scope':'instrumented phase attribution, not acceptance performance','calls_per_rank':50,'header_sha256':hashlib.sha256((root/'build/flashinfer-profiler/include/flashinfer/profiler.cuh').read_bytes()).hexdigest()},indent=2)+'\n')
with (base/'timeline_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('All four serial AG profiles pass full-output, route, native and trace checks.',flush=True)
