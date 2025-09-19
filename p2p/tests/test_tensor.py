import sys
import torch

try:
    from uccl import p2p

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)

gpu_index = 0

if torch.cuda.is_available():
    torch.cuda.set_device(0)

engine = p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
metadata = engine.get_metadata()
ip, port, remote_gpu_idx = p2p.Endpoint.parse_metadata(metadata)

tensor, mr_id, ipc_id = p2p.create_tensor(0, 32, 4)
print(f"allocate tensor mr_id: {mr_id}, ipc_id: {ipc_id}")
p2p.free_tensor(tensor, mr_id, ipc_id)
