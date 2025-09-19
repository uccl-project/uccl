import sys
import torch
import gc

try:
    from uccl import utils, p2p

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)

use_gpu_id = 0
engine = p2p.Endpoint(local_gpu_idx=use_gpu_id, num_cpus=4)

tensor, tensor_id = utils.create_tensor((2, 4, 2), torch.float32, f"cuda:{use_gpu_id}")
print(f"Allocate tensor: id={tensor_id}")
t_id = utils.get_tensor_id_by_tensor(tensor)
print(f"Query tensor: id={tensor_id}")
# test auto free
del tensor
gc.collect()
# print(f"Try to free tensor manaual: id={t_id}")
# utils.free_tensor(tensor)
