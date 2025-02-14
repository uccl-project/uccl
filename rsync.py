from shared import *
import os
from pathlib import Path

UCCL_HOME = os.getenv("UCCL_HOME")
uccl_parent_dir = Path(UCCL_HOME).parent if UCCL_HOME else None

async def rsync(nodes):
    tasks = [
        run_command(
            f"rsync -auv -e 'ssh -o StrictHostKeyChecking=no' {UCCL_HOME} {node}:{uccl_parent_dir}"
        )
        for node in nodes
    ]

    # Run all tasks in parallel
    await asyncio.gather(*tasks)


nodes = get_nodes()
print(f"Nodes: {nodes}")

asyncio.run(rsync(nodes))
