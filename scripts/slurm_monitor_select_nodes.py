#!/usr/bin/env python3
import subprocess
import re
import sys

def get_nodes(partition, n=5):
    # Call scontrol
    out = subprocess.check_output(["scontrol", "show", "node"], text=True)

    # Split by NodeName
    blocks = out.split("NodeName=")[1:]

    nodes = []
    for block in blocks:
        name = block.split()[0]
        
        # Parse key info
        state_match = re.search(r"State=(\S+)", block)
        part_match = re.search(r"Partitions=(\S+)", block)
        load_match = re.search(r"CPULoad=([\d.]+)", block)
        cpu_match = re.search(r"CPUTot=(\d+)", block)
        
        if not (state_match and part_match and load_match and cpu_match):
            continue
        
        state = state_match.group(1)
        part = part_match.group(1)
        load = float(load_match.group(1))
        cputot = int(cpu_match.group(1))

        # Filter by partition & state
        if partition not in part:
            continue
        if any(bad in state for bad in ["DOWN", "DRAIN", "NOT_RESPONDING", "ALLOCATED"]):
            continue

        load_ratio = load / cputot if cputot > 0 else 999
        nodes.append((name, load_ratio, load))

    # Sort by load ratio
    nodes.sort(key=lambda x: x[1])
    return nodes[:n]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pick_nodes.py <PartitionName> [N]")
        sys.exit(1)

    partition = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    result = get_nodes(partition, n)
    if not result:
        print("No available nodes found")
    else:
        for node, ratio, load in result:
            print(f"{node}\tCPULoad={load}")
