import os,sys
rank=int(os.environ['OMPI_COMM_WORLD_RANK'])
command=sys.argv[1:]
if rank<4: command=['/usr/local/bin/nsys','profile','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--trace=cuda','--cuda-graph-trace=node','--export=sqlite','--output='+os.path.join(os.path.dirname(__file__),'rank'+str(rank)),*command]
os.execvpe(command[0],command,os.environ)
