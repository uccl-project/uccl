#!/usr/bin/env python3
import subprocess
import re
import sys
import argparse

def get_all_partitions():
    try:
        out = subprocess.check_output(["scontrol", "show", "partition"], text=True)
    except subprocess.CalledProcessError:
        return {}
    
    partitions = {}
    blocks = out.split("PartitionName=")[1:]
    
    for block in blocks:
        name = block.split()[0]
        state_match = re.search(r"State=(\S+)", block)
        if state_match and "UP" in state_match.group(1):
            available_nodes = count_available_nodes_in_partition(name)
            partitions[name] = available_nodes
    
    return partitions

def count_available_nodes_in_partition(partition):
    try:
        out = subprocess.check_output(["scontrol", "show", "node"], text=True)
    except subprocess.CalledProcessError:
        return 0
    
    blocks = out.split("NodeName=")[1:]
    count = 0
    
    for block in blocks:
        part_match = re.search(r"Partitions=(\S+)", block)
        state_match = re.search(r"State=(\S+)", block)
        
        if not (part_match and state_match):
            continue
        
        part = part_match.group(1)
        state = state_match.group(1)
        
        if partition in part and not any(bad in state for bad in ["DOWN", "DRAIN", "NOT_RESPONDING", "ALLOCATED"]):
            count += 1
    
    return count

def get_nodes(partition, n=5):
    try:
        out = subprocess.check_output(["scontrol", "show", "node"], text=True)
    except subprocess.CalledProcessError:
        return []
    
    blocks = out.split("NodeName=")[1:]
    nodes = []
    
    for block in blocks:
        name = block.split()[0]
        
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
        
        if partition not in part:
            continue
        if any(bad in state for bad in ["DOWN", "DRAIN", "NOT_RESPONDING", "ALLOCATED"]):
            continue
        
        load_ratio = load / cputot if cputot > 0 else 999
        nodes.append((name, load_ratio, load))
    
    nodes.sort(key=lambda x: x[1])
    return nodes[:n]

def select_best_partition():
    partitions = get_all_partitions()
    
    if not partitions:
        print("Error: No partitions found or unable to query slurm")
        sys.exit(1)
    
    sorted_partitions = sorted(partitions.items(), key=lambda x: x[1], reverse=True)
    
    best_partition = sorted_partitions[0][0]
    
    return best_partition

def main():
    parser = argparse.ArgumentParser(
        description='Select nodes with lowest load from SLURM partition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                          # Auto-select best partition, show 5 nodes
  %(prog)s --partition gpu          # Use GPU partition, show 5 nodes  
  %(prog)s --partition cpu --num_nodes 10  # Show 10 nodes from CPU partition
        '''
    )
    
    parser.add_argument('--partition', '-p', 
                       help='SLURM partition name (auto-select if not specified)')
    parser.add_argument('--num_nodes', '-n', type=int, default=5,
                       help='Number of nodes to select (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    if not args.partition:
        partition = select_best_partition()
    else:
        partition = args.partition
        if args.verbose:
            print(f"Using specified partition: {partition}")
    
    result = get_nodes(partition, args.num_nodes)
    
    if not result:
        print(f"No available nodes found in partition {partition}")
        sys.exit(1)
    else:
        for node, ratio, load in result:
            if args.verbose:
                print(f"{node}\tCPULoad={load}\tLoadRatio={ratio:.3f}")
            else:
                print(f"{node}")

if __name__ == "__main__":
    main()