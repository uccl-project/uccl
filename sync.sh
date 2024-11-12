# !/bin/bash

nodes=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^# ]] && continue
    nodes+=("$line")
done < nodes.txt

for ip in "${nodes[@]}"; do
    echo "Syncing" $ip
    rsync -auv -e 'ssh -o StrictHostKeyChecking=no' /opt/uccl/ $ip:/opt/uccl/ &
done

wait