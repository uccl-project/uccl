# !/bin/bash

all_others=("172.31.76.70")
for other in "${all_others[@]}"; do
  rsync -auv -e 'ssh -o StrictHostKeyChecking=no' ~/xdp-tutorial/advanced04-aws-afxdp/ $other:~/xdp-tutorial/advanced04-aws-afxdp/ &
done

wait