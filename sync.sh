# !/bin/bash

for ip in "$@"; do
  rsync -auv -e 'ssh -o StrictHostKeyChecking=no' ~/uccl/ $ip:~/uccl/ &
done

wait