# !/bin/bash

# Read lines from nodes.txt into an array
nodes=()
while IFS= read -r line || [[ -n "$line" ]]; do
    # Ignore lines starting with '#'
    [[ "$line" =~ ^# ]] && continue
    # Add the line to the array
    nodes+=("$line")
done < nodes.txt

for ip in "${nodes[@]}"; do
    echo "Syncing" $ip
    rsync -auv -e 'ssh -o StrictHostKeyChecking=no' /opt/uccl/ $ip:/opt/uccl/ &
done

wait