#!/usr/bin/env python

# Generate a permutation traffic matrix using full bisection bandwidth.
# python gen_pemutation.py <nodes> <conns> <flowsize> <extrastarttime>
# Parameters:
# <nodes>   number of nodes in the topology
# <conns>    number of active connections
# <flowsize>   size of the flows in bytes
# <extrastarttime>   How long in microseconds to space the start times over (start time will be random in between 0 and this time).  Can be a float.
# <randseed>   Seed for random number generator, or set to 0 for random seed

import os
import sys
import random
from pathlib import Path

#print(sys.argv)
if len(sys.argv) != 7:
    print("Usage: python gen_pemutation.py <filename> <nodes> <conns> <flowsize> <extrastarttime> <randseed>")
    sys.exit()
filename = sys.argv[1]
nodes = int(sys.argv[2])
conns = int(sys.argv[3])
flowsize = int(sys.argv[4])
extrastarttime = float(sys.argv[5])
randseed = int(sys.argv[6])

print("Nodes: ", nodes)
print("Connections: ", conns)
print("Flowsize: ", flowsize, "bytes")
print("ExtraStartTime: ", extrastarttime, "us")
print("Random Seed ", randseed)

if not os.path.exists(filename):
    os.system(r"touch {}".format(filename))

f = open(filename, "w")
print("Nodes", nodes, file=f)
print("Connections", conns, file=f)

srcs = []
dsts = []
stride = nodes // 2 

for _ in range(nodes):
    source = random.randint(0, nodes - 1)
    while source in srcs:
        source = random.randint(0, nodes - 1)

    destination = (source + stride) % nodes
    srcs.append(source)
    dsts.append(destination)


#print(srcs)
#print(dsts)


for n in range(conns):
    out = str(srcs[n]) + "->" + str(dsts[n]) + " id " + str(n+1) + " start " + str(int(extrastarttime * 1000000)) + " size " + str(flowsize)
    print(out, file=f)

f.close()

