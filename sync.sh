# !/bin/bash

for ip in "$@"; do
  rsync -auv -e 'ssh -o StrictHostKeyChecking=no' /opt/uccl/ $ip:/opt/uccl/ &
done

wait