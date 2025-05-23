from shared import *
import os
from pathlib import Path

UCCL_HOME = os.getenv("UCCL_HOME")
if not UCCL_HOME:
    raise ValueError("UCCL_HOME environment variable is not set.")

print(f"Using UCCL_HOME: {UCCL_HOME}")

async def rsync(nodes):
    tasks = [
        run_command(
            f"rsync -auv -e 'ssh -o StrictHostKeyChecking=no' --mkpath --delete {UCCL_HOME}/ {node}:{UCCL_HOME}/"
        )
        for node in nodes
    ]
    # Run all tasks in parallel
    await asyncio.gather(*tasks)


nodes = get_nodes()
print(f"Nodes: {nodes}")

asyncio.run(rsync(nodes))
