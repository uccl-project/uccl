import os
import torch
import ukernel
import time


def build_p2p_server_dag():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # P2P params
    mini_batch = 2
    seq_len = 8
    hidden_size = 128

    tokens_per_batch = mini_batch * seq_len

    # Buffers
    recv_tokens = torch.empty(tokens_per_batch, hidden_size, device=device)

    # Parallel rule
    rule = ukernel.ParallelRule(num_tasks=1, tiles_per_task=4)

    # P2P communication DAG
    # Receive operation
    recv_op = ukernel.p2p_recv(recv_tokens, rule, peer_rank=1)

    ops = [recv_op]
    return ops, recv_tokens


def build_p2p_client_dag():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # P2P params
    mini_batch = 2
    seq_len = 8
    hidden_size = 128

    tokens_per_batch = mini_batch * seq_len

    # Buffers
    tokens = torch.randn(tokens_per_batch, hidden_size, device=device)

    # Parallel rule
    rule = ukernel.ParallelRule(num_tasks=1, tiles_per_task=4)

    # P2P communication DAG
    # Send operation
    send_op = ukernel.p2p_send(tokens, rule, peer_rank=0)

    ops = [send_op]
    return ops, tokens


def run_server():
    # Set environment variables for communicator configuration
    os.environ["UHM_EXCHANGER_SERVER_IP"] = "127.0.0.1"
    os.environ["UHM_EXCHANGER_SERVER_PORT"] = "6980"

    # Scheduler config
    cfg = ukernel.SchedulerConfig()
    cfg.gpu_id = 0
    cfg.rank = 0
    cfg.world_size = 2

    # Initialize UKernel
    ukernel.init(cfg)

    # Connect to the client (waiting for client connection)
    ukernel.accept_from(1)  # client rank is 1

    # Build P2P Server DAG
    ops, recv_tokens = build_p2p_server_dag()

    # Add operators
    for op in ops:
        ukernel.add(op)

    # Run the scheduler
    ukernel.run()
    ukernel.sync_all()

    print(f"Received tokens: {recv_tokens}")

    # Ensure that recv_tokens contains data (checking for any non-zero value in the tensor)
    if recv_tokens.numel() == 0:
        raise RuntimeError("No data received on server!")

    print("P2P Server Test Passed!")


def run_client():
    # Set environment variables for communicator configuration
    os.environ["UHM_EXCHANGER_SERVER_IP"] = "127.0.0.1"
    os.environ["UHM_EXCHANGER_SERVER_PORT"] = "6980"

    # Scheduler config
    cfg = ukernel.SchedulerConfig()
    cfg.gpu_id = 0
    cfg.rank = 1  # Client rank is 1
    cfg.world_size = 2

    # Initialize UKernel
    ukernel.init(cfg)

    # Connect to the server (making the connection)
    ukernel.connect_to(0)  # server rank is 0

    # Build P2P Client DAG
    ops, sent_tokens = build_p2p_client_dag()

    print(f"Sent tokens: {sent_tokens}")

    # Add operators
    for op in ops:
        ukernel.add(op)

    # Run the scheduler
    ukernel.run()
    ukernel.sync_all()

    print("P2P Client Test Passed!")


if __name__ == "__main__":
    # Start the server and client processes
    from multiprocessing import Process

    # Create and start server process
    server_process = Process(target=run_server)
    server_process.start()

    # Sleep for a short while to ensure server is up before client starts
    time.sleep(1)

    # Create and start client process
    client_process = Process(target=run_client)
    client_process.start()

    # Wait for both processes to finish
    server_process.join()
    client_process.join()

    print("P2P Communication Test Completed Successfully!")
