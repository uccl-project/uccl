"""Stress shared-buffer credits by pausing only this test's rank-1 process."""
import os, signal, subprocess, time
from pathlib import Path
root=Path.cwd(); out=root/'build/hybrid_ar_20261115'
binary=str(out/'uccl_lite_multinode_allreduce').encode()
env=os.environ.copy();env.update(HOSTS='10.10.55.2:4,10.10.55.1:4',SOCKET_IFNAME='ibp55s0f0',MPIRUN='/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun',BUILD_DIR=str(out))
proc=None; pid=None; identity=None

def interrupt(signum, frame):
    if pid and identity:
        try:
            current=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            if current==identity: os.kill(pid,signal.SIGCONT)
        except (OSError,IndexError): pass
    if proc is not None and proc.poll() is None: proc.terminate()
    raise SystemExit(128+signum)

signal.signal(signal.SIGTERM,interrupt)
signal.signal(signal.SIGINT,interrupt)
cmd=['bash','scripts/run_multinode_hybrid.sh','--op','allgather','--host-piece-bytes','131200','-b','6291968','-e','6291968','-w','2','-n','1000','--check-each','1','-c','1']
with (out/'paused_consumer.log').open('w') as stream:
    proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    pid=None; identity=None; deadline=time.monotonic()+45
    while time.monotonic()<deadline and proc.poll() is None:
        if '# sample,' in (out/'paused_consumer.log').read_text():
            for path in Path('/proc').iterdir():
                if not path.name.isdigit(): continue
                try:
                    args=(path/'cmdline').read_bytes().split(b'\0')
                    if not args or args[0]!=binary or b'1000' not in args: continue
                    fields=(path/'environ').read_bytes().split(b'\0')
                    if b'OMPI_COMM_WORLD_RANK=1' not in fields: continue
                    pid=int(path.name);identity=(path/'stat').read_text().rsplit(')',1)[1].split()[19]
                    break
                except (OSError,IndexError): continue
            if pid: break
        time.sleep(.01)
    assert pid, 'test rank not found while live'
    pauses=0
    for _ in range(8):
        if proc.poll() is not None: break
        path=Path('/proc')/str(pid)
        try:
            if path.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]!=identity: break
            os.kill(pid,signal.SIGSTOP)
            try: time.sleep(.05)
            finally:
                if path.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==identity:
                    os.kill(pid,signal.SIGCONT)
            pauses+=1
        except FileNotFoundError: break
        time.sleep(.03)
    code=proc.wait(timeout=180)
    assert code==0,code
    assert pauses>0
    (out/'paused_consumer_status.txt').write_text(f'rank=1 pauses={pauses} duration_ms=50 exit={code}\n')
