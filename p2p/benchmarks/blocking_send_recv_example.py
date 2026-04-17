#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import numpy as np

import torch

from nixl._api import nixl_agent, nixl_agent_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--mode",
        type=str,
        default="initiator",
        help="Local IP in target, peer IP (target's) in initiator",
    )
    return parser.parse_args()


def run_single_transfer(num_elements, args):
    """Run a single transfer with specified number of elements - creates fresh agent each time"""

    # Create fresh agent configuration for each transfer
    listen_port = args.port
    if args.mode != "target":
        listen_port = 0

    if args.use_cuda:
        try:
            if torch.cuda.is_available():
                torch.set_default_device("cuda:0")
            else:
                torch.set_default_device("cpu")
        except Exception as e:
            torch.set_default_device("cpu")
    else:
        torch.set_default_device("cpu")

    config = nixl_agent_config(True, True, listen_port)
    agent = nixl_agent(args.mode, config)
    plugin_list = agent.get_plugin_list()
    print(plugin_list)
    print("Plugin parameters")
    print(agent.get_plugin_mem_types("UCX"))
    print(agent.get_plugin_params("UCX"))

    print("\nLoaded backend parameters")
    print(agent.get_backend_mem_types("UCX"))
    print(agent.get_backend_params("UCX"))
    print()

    # Create tensors of specified size
    if args.mode == "target":
        tensors = [torch.zeros(num_elements, dtype=torch.float32) for _ in range(2)]
    else:
        tensors = [torch.ones(num_elements, dtype=torch.float32) for _ in range(2)]

    reg_descs = agent.register_memory(tensors)
    if not reg_descs:  # Same as reg_descs if successful
        print("Memory registration failed.")
        return None

    bandwidth = None

    # Target code
    if args.mode == "target":
        ready = False

        target_descs = reg_descs.trim()
        target_desc_str = agent.get_serialized_descs(target_descs)

        # Send desc list to initiator when metadata is ready
        while not ready:
            ready = agent.check_remote_metadata("initiator")

        agent.send_notif("initiator", target_desc_str)

        # Waiting for transfer
        while not agent.check_remote_xfer_done("initiator", b"UUID"):
            continue
    # Initiator code
    else:
        agent.fetch_remote_metadata("target", args.ip, args.port)
        agent.send_local_metadata(args.ip, args.port)

        notifs = agent.get_new_notifs()

        while len(notifs) == 0:
            notifs = agent.get_new_notifs()

        target_descs = agent.deserialize_descs(notifs["target"][0])
        initiator_descs = reg_descs.trim()

        # Ensure remote metadata has arrived from fetch
        ready = False
        while not ready:
            ready = agent.check_remote_metadata("target")

        start_time = time.time()

        xfer_handle = agent.initialize_xfer(
            "WRITE", initiator_descs, target_descs, "target", b"UUID"
        )

        if not xfer_handle:
            print("Creating transfer failed.")
            agent.deregister_memory(reg_descs)
            return None

        state = agent.transfer(xfer_handle)
        if state == "ERR":
            print("Posting transfer failed.")
            agent.deregister_memory(reg_descs)
            return None
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("Transfer got to Error state.")
                agent.deregister_memory(reg_descs)
                return None
            elif state == "DONE":
                break

        end_time = time.time()
        duration = end_time - start_time

        # Calculate bandwidth in GB/s
        size_bytes = num_elements * 4 * 2  # 4 bytes per float32, 2 tensors
        bandwidth = (size_bytes / duration) / (1024 * 1024 * 1024)

        # Verify data after read
        for i, tensor in enumerate(tensors):
            # if not torch.allclose(tensor, torch.ones(num_elements)):
            if args.mode == "target" and not torch.allclose(
                tensor, torch.ones(num_elements)
            ):
                print(f"Data verification failed for tensor {i}.")
                agent.deregister_memory(reg_descs)
                return None

        agent.release_xfer_handle(xfer_handle)
        agent.remove_remote_agent("target")
        agent.invalidate_local_metadata(args.ip, args.port)

    agent.deregister_memory(reg_descs)
    return bandwidth


# TPU0: python blocking_send_recv_example.py --ip 10.202.15.196 --port 12345 --mode initiator
# TPU1: python blocking_send_recv_example.py --ip 10.202.15.196 --port 12345 --mode target

if __name__ == "__main__":
    args = parse_args()

    msg_sizes_bytes = [
        1 * 1024,  # 1 KB
        4 * 1024,  # 4 KB
        16 * 1024,  # 16 KB
        64 * 1024,  # 64 KB
        256 * 1024,  # 256 KB
        1 * 1024 * 1024,  # 1  MB
        10 * 1024 * 1024,  # 10 MB
        100 * 1024 * 1024,  # 100 MB
    ]

    # elements per tensor for each size  (size_bytes / (4 bytes/elem * 2 tensors))
    element_counts = [size // 8 for size in msg_sizes_bytes]

    # Test sizes: number of float32 elements per tensor
    # Starting from 2K elements (8KB per tensor, 16KB total) to 256M elements (1GB per tensor, 2GB total)
    # element_counts = [2**i * 1024 for i in range(18)]  # 2K to 256M elements
    bandwidths = []

    if args.mode == "initiator":
        print("\nElements\tSize(MB)\tBandwidth(GB/s)")
        print("-" * 40)

    results = [[] for _ in range(3)]
    for num_elements in element_counts:
        bandwidth = run_single_transfer(num_elements, args)

        if args.mode == "initiator":
            size_mb = (num_elements * 4 * 2) / (
                1024 * 1024
            )  # 4 bytes per float32, 2 tensors
            if bandwidth is not None:
                print(f"{num_elements}\t\t{size_mb:.2f}\t\t{bandwidth:.2f}")
                bandwidths.append(bandwidth)
                results[0].append(num_elements)
                results[1].append(size_mb)
                results[2].append(bandwidth)
            else:
                print(f"{num_elements}\t\t{size_mb:.2f}\t\tFailed")
                break  # Stop on first failure

    # Final summary
    if args.mode == "initiator" and bandwidths:
        print(f"\nAverage Bandwidth: {np.mean(bandwidths):.2f} GB/s")
        print(f"Peak Bandwidth: {np.max(bandwidths):.2f} GB/s")

    # Print detailed results
    if args.mode == "initiator" and results[0]:
        print("\nDetailed Results:")
        print(f"{'Elements':<12} {'Size(MB)':<10} {'Bandwidth(GB/s)':<15}")
        print("-" * 40)
        for elems, size, bw in zip(results[0], results[1], results[2]):
            print(f"{elems:<12,} {size:<10.2f} {bw:<15.2f}")

    # Plot bandwidth results
    if args.mode == "initiator" and results[1]:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogx(results[1], results[2], marker="o")
        plt.xlabel("Data Size (MB)")
        plt.ylabel("Bandwidth (GB/s)")
        plt.title("NIXL Transfer Bandwidth vs Data Size")
        plt.grid(True)
        plt.savefig("bandwidth_plot.png")
        plt.close()

    print("Benchmark Complete.")
