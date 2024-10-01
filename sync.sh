# !/bin/bash

all_others=("172.31.19.147")
for other in "${all_others[@]}"; do
  rsync -auv -e 'ssh -o StrictHostKeyChecking=no' ~/uccl/ $other:~/uccl/ &
done

wait