import json,os,subprocess,shutil
from pathlib import Path
out=Path(__file__).resolve().parent;m=json.loads((out/'manifest.json').read_text());native=m['commands'][0]['argv'];prefix=native[:native.index('env')]
env=os.environ.copy();env.update(UCCL_LITE_MSCCLPP_CONTROL_ROOT='10.10.55.2',NCCL_SOCKET_IFNAME='ibp55s0f0',NCCL_IB_HCA='mlx5_0,mlx5_1',MSCCLPP_SOCKET_IFNAME='ibp55s0f0',MSCCLPP_IBV_MODE='host-no-atomic',LD_LIBRARY_PATH='/home/zhongjie/third_party/nccl/build/lib')
shutil.copy2('build/mscclpp_compile/uccl_lite_multinode_allreduce',out/'standalone');subprocess.run(['rsync','-a',str(out/'standalone'),f'10.10.55.1:{out}/'],check=True)
commands=[]
for op,flag in [('allreduce','--ar-auto'),('allgather','--ag-auto'),('reducescatter','--rs-auto')]:
 cmd=prefix+[str(out/'standalone'),'--op',op,flag,'1','-b','1024','-e',str(16<<20),'-w','1','-n','2','--check-each','1','-B','96','-T','512'];commands.append({'label':op,'argv':cmd})
 with (out/('standalone_'+op+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 assert r.returncode==0,op
 samples=[line.split(',') for line in (out/('standalone_'+op+'.log')).read_text().splitlines() if line.startswith('# sample,')];assert len(samples)==30 and all(float(x[-1])==0 for x in samples)
 print(op,'30 samples passed',flush=True)
profile=out/'local_only.tsv';shutil.copy2(out/'candidate.tsv',profile)
cmd=m['commands'][1]['argv'].copy();cmd[cmd.index('--profile')+1]=str(profile);cmd[cmd.index('-e')+1]='1024';cmd[cmd.index('-n')+1]='1';cmd[cmd.index('-w')+1]='1';commands.append({'label':'one_host_missing','argv':cmd})
with (out/'one_host_missing.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
assert r.returncode!=0 and 'profile/topology consensus failed' in (out/'one_host_missing.log').read_text()
print('one-host missing profile rejected, not routed to NCCL',flush=True)
(out/'post_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
