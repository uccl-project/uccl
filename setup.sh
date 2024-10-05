# !/bin/bash

# Can only use half of the queue, per ENA implementation: https://github.com/amzn/amzn-drivers/issues/240
NIC=$1
NQUEUE=$2
MTU=$3
MODE=$4
echo "configuring ${NIC} with ${NQUEUE} queues and ${MTU} MTU for ${MODE}"

echo "unloading any xdp programs"
sudo ~/uccl/lib/xdp-tools/xdp-loader/xdp-loader unload ${NIC} --all

sudo ethtool -L ${NIC} combined ${NQUEUE}
sudo ifconfig ${NIC} mtu ${MTU} up

if [ $MODE = "afxdp" ]; then
    sudo service irqbalance stop
    sudo ethtool -C ${NIC} adaptive-rx off rx-usecs 0 tx-usecs 0
elif [ $MODE = "tcp" ]; then
    sudo service irqbalance start
    # for aws ena
    sudo ethtool -C ${NIC} adaptive-rx on rx-usecs 20 tx-usecs 60
else
    echo "Invalid mode: ${MODE}"
    exit 1
fi

(let cnt=0; cd /sys/class/net/${NIC}/device/msi_irqs/;
IRQs=(*)
# Exclude the first IRQ, which is for the control plane
for IRQ in "${IRQs[@]:1}"; do
    let CPU=$((cnt))
    let cnt=$(((cnt+1)%NQUEUE))
    echo $IRQ '->' $CPU
    echo $CPU | sudo tee /proc/irq/$IRQ/smp_affinity_list > /dev/null
done)

# https://lwn.net/Articles/837010/; do not given improvements
# echo 2 | sudo tee /sys/class/net/ens6/napi_defer_hard_irqs
# echo 200000 | sudo tee /sys/class/net/ens6/gro_flush_timeout
# echo 0 | sudo tee /sys/class/net/ens6/napi_defer_hard_irqs
# echo 0 | sudo tee /sys/class/net/ens6/gro_flush_timeout

## run af_xdp l2fwd
## -z: zero-copy mode (without skb copy)
## -p: polling with timeout of 1ms.
# sudo ./af_xdp_user -d ${NIC} --filename af_xdp_kern.o -z

## for efa test
# sudo ./af_xdp_user_efa -d ${NIC} --filename af_xdp_kern_efa.o -z
