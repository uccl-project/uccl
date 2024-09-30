# !/bin/bash

all_others=("172.31.18.130")
for other in "${all_others[@]}"; do
  rsync -auv -e 'ssh -o StrictHostKeyChecking=no' ~/xdp-tutorial/ $other:~/xdp-tutorial/ &
done

wait