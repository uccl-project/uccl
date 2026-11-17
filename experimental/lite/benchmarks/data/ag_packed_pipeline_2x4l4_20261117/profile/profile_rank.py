#!/usr/bin/env python3
import os
import sys
from pathlib import Path
build=Path(__file__).resolve().parent
exe=str(build/'uccl_lite_multinode_host_allgather')
if os.environ.get('OMPI_COMM_WORLD_RANK')=='0':
    nsys='/usr/local/bin/nsys'
    os.execv(nsys,[nsys,'profile','--sample=none','--cpuctxsw=none','--trace=cuda',
                  '--output='+str(build/'packed_dma_trace'),exe,*sys.argv[1:]])
os.execv(exe,[exe,*sys.argv[1:]])
