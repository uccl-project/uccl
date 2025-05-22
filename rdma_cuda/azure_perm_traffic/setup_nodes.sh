#!/bin/bash

if [ ! -f "ips" ]; then
    echo "ips doesn't exist"
    exit 1
fi

commands=(
    "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    "sudo apt install ./cuda-keyring_1.1-1_all.deb"
    "sudo apt update"
    "sudo apt install cuda-toolkit -y"
    "sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev -y"
)

max_parallel=64

while IFS= read -r ip || [ -n "$ip" ]; do
    if [[ -z "$ip" || "$ip" == \#* ]]; then
        continue
    fi
    
    echo "Connecting to $ip..."
    
    (
        start_time=$(date +%s)
        
        remote_command=""
        for cmd in "${commands[@]}"; do
            remote_command+="echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] execute: $cmd\"; "
            remote_command+="$cmd; "
            remote_command+="echo; "
        done
        
        if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$ip" "$remote_command"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "$ip: setup successfully (${duration}seconds)"
        else
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "$ip: setup failed (${duration}seconds)" >&2
        fi
    ) &
    
    if (( $(jobs -p | wc -l) >= max_parallel )); then
        wait -n
    fi
    
done < "ips"

wait

echo "All done."
