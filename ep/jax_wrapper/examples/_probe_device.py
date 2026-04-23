import os
master_port = int(os.environ["MASTER_PORT"])
jax_port = master_port + 1
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
addr = os.environ["MASTER_ADDR"]

import jax
jax.distributed.initialize(
    coordinator_address=f"{addr}:{jax_port}",
    num_processes=world,
    process_id=rank,
    local_device_ids=[local_rank],
    coordinator_bind_address=f"0.0.0.0:{jax_port}",
)

devs = jax.local_devices()
print(
    f"[rank={rank}/local={local_rank}] local_devs={devs} "
    f"ids={[getattr(d, 'id', None) for d in devs]} "
    f"local_hw_ids={[getattr(d, 'local_hardware_id', None) for d in devs]}",
    flush=True,
)
