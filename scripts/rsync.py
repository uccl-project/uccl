from shared import *
import os
import asyncio
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--node_file", help="Path to file containing list of nodes", default="node_ips/default.txt")
args = parser.parse_args()

UCCL_HOME = os.getenv("UCCL_HOME")
if not UCCL_HOME:
    raise ValueError("UCCL_HOME environment variable is not set.")

if not Path(UCCL_HOME).is_dir():
    raise FileNotFoundError(f"Local directory {UCCL_HOME} does not exist.")

print(f"Using UCCL_HOME: {UCCL_HOME}")

async def prepare_remote(node):
    await run_command(f"ssh -o StrictHostKeyChecking=no {node} 'mkdir -p {UCCL_HOME}'")

async def rsync(nodes):
    await asyncio.gather(*(prepare_remote(node) for node in nodes))
    tasks = [
        run_command(
            f"rsync -auv -e 'ssh -o StrictHostKeyChecking=no' --delete {UCCL_HOME}/ {node}:{UCCL_HOME}/"
        )
        for node in nodes
    ]
    await asyncio.gather(*tasks)

nodes = get_nodes(args.node_file)
print(f"Nodes: {nodes}")

asyncio.run(rsync(nodes))