# !/bin/bash

# Can only use half of the queue, per ENA implementation: https://github.com/amzn/amzn-drivers/issues/240
NIC=$1
NQUEUE=$2
MTU=$3
echo "configuring ${NIC} with ${NQUEUE} queues"

sudo ethtool -L ${NIC} combined ${NQUEUE}
sudo ifconfig ${NIC} mtu ${MTU} up
sudo service irqbalance stop
sudo ethtool -C ${NIC} adaptive-rx off rx-usecs 0 tx-usecs 0
# sudo ethtool -C ${NIC} adaptive-rx on rx-usecs 20 tx-usecs 60

(let cnt=0; cd /sys/class/net/ens5/device/msi_irqs/;
for IRQ in *; do
    let CPU=$((cnt))
    let cnt=$(((cnt+1)%NQUEUE))
    echo $IRQ '->' $CPU
    echo $CPU | sudo tee /proc/irq/$IRQ/smp_affinity_list > /dev/null
done)

sudo ~/uccl/lib/xdp-tools/xdp-loader/xdp-loader unload ${NIC} --all

## run af_xdp l2fwd
## -z: zero-copy mode (without skb copy)
## -p: polling with timeout of 1ms.
# sudo ./af_xdp_user -d ens5 --filename af_xdp_kern.o -z

## for efa test
# sudo ./af_xdp_user_efa -d ens5 --filename af_xdp_kern_efa.o -z

## running tcp benchmark
# ./server_tcp
# ./client_tcp -a 172.31.22.249

