import torch
import time
import sys
import multiprocessing

try:
    from uccl import p2p
    from uccl import P2PTensor

    print("✓ Successfully imported p2p")
except ImportError as e:
    print(f"✗ Failed to import p2p: {e}")
    sys.exit(1)

IPC_NAME = "test_tensor_ipc"
IPC_NAME_2 = "test_tensor_ipc2"

def writer():
    x = P2PTensor(torch.arange(10, dtype=torch.float32, device="cuda"), ipc_name=IPC_NAME)
    print(f"[Writer] Created tensor: {x}, ptr={x.data_ptr()}")

    y = P2PTensor(torch.arange(11, dtype=torch.float32, device="cuda"), ipc_name=IPC_NAME_2)
    print(f"[Writer] Created tensor: {y}, ptr={y.data_ptr()}")

    z = P2PTensor(torch.arange(11, dtype=torch.float32, device="cuda"), ipc_name=IPC_NAME_2)
    print(f"[Writer] Created tensor: {z}, ptr={z.data_ptr()}")
    

def reader():
    # this will increase refcount of shm, so we need delete shm mansual by "rm /dev/shm/p2p_ipc"
    ok = p2p.check_ipc_by_name_once(IPC_NAME) 
    # 1 g_ipc error, 2 success, 3 not found
    if ok == 1:
        print(f"[Reader] check_ipc_by_name_once: g_ipc error")
    elif ok == 2:
        print(f"[Reader] check_ipc_by_name_once: success")
    elif ok == 3:
        print(f"[Reader] check_ipc_by_name_once: not found")
    
    time.sleep(15)
    ok = p2p.check_ipc_by_name_once(IPC_NAME) 
    if ok == 1:
        print(f"[Reader] check_ipc_by_name_once: g_ipc error")
    elif ok == 2:
        print(f"[Reader] check_ipc_by_name_once: success")
    elif ok == 3:
        print(f"[Reader] check_ipc_by_name_once: not found")

    # time.sleep(5)
    # ok = p2p.check_ipc_by_name_once(IPC_NAME) 
    # if ok == 1:
    #     print(f"[Reader] check_ipc_by_name_once: g_ipc error")
    # elif ok == 2:
    #     print(f"[Reader] check_ipc_by_name_once: success")
    # elif ok == 3:
    #     print(f"[Reader] check_ipc_by_name_once: not found")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    import subprocess
    try:
        # reader api check_ipc_by_name_once will increase refcount of shm, so we need delete shm mansual by "rm /dev/shm/p2p_ipc"
        subprocess.run(['rm', '/dev/shm/p2p_ipc'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"{e}")

    # init cuda driver
    print("waitting gpu/cuda init")
    torch.randn(1, device="cuda")

    p1 = multiprocessing.Process(target=writer)
    p1.start()

    # p2 = multiprocessing.Process(target=reader)
    # p2.start()

    p1.join()
    # p2.join()
