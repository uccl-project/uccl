# !/bin/bash

# Can only use half of the queue, per ENA implementation: https://github.com/amzn/amzn-drivers/issues/240
NIC=$1
NQUEUE=$2
NIRQCORE=$3
MTU=$4
MODE=$5
PLATFORM=$6
echo "configuring ${NIC} with ${NQUEUE} nic queues ${NIRQCORE} irq cores ${MTU} MTU for ${MODE} on ${PLATFORM}"

echo "unloading any xdp programs"
sudo /opt/uccl/lib/xdp-tools/xdp-loader/xdp-loader unload ${NIC} --all

sudo ethtool -L ${NIC} combined ${NQUEUE}
sudo ifconfig ${NIC} mtu ${MTU} up

if [ $MODE = "afxdp" ]; then
    sudo service irqbalance stop
    if [ $PLATFORM = "aws" ]; then
        sudo ethtool -C ${NIC} adaptive-rx off rx-usecs 0 tx-usecs 0
    elif [ $PLATFORM = "cloudlab" ]; then
        sudo ethtool -C ${NIC} adaptive-rx off adaptive-tx off rx-usecs 0 rx-frames 1 tx-usecs 0 tx-frames 1
    else
        echo "Invalid platform: ${PLATFORM}"
        exit 1
    fi
elif [ $MODE = "tcp" ]; then
    sudo service irqbalance start
    if [ $PLATFORM = "aws" ]; then
        sudo ethtool -C ${NIC} adaptive-rx on rx-usecs 20 tx-usecs 60
    elif [ $PLATFORM = "cloudlab" ]; then
        sudo ethtool -C ${NIC} adaptive-rx on adaptive-tx on rx-usecs 8 rx-frames 128 tx-usecs 8 tx-frames 128
    else
        echo "Invalid platform: ${PLATFORM}"
        exit 1
    fi
else
    echo "Invalid mode: ${MODE}"
    exit 1
fi

NCPU=$(nproc)
# Starting from 3/4 of the CPUs to avoid conflicting with nccl proxy services.
irq_start_cpu=$((NCPU / 2 + NCPU / 4))
(
    let cnt=0
    cd /sys/class/net/${NIC}/device/msi_irqs/
    IRQs=(*)
    # Exclude the first IRQ, which is for the control plane
    for IRQ in "${IRQs[@]:1}"; do
        let CPU=$(((cnt + irq_start_cpu) % NCPU))
        let cnt=$(((cnt + 1) % NIRQCORE))
        echo $IRQ '->' $CPU
        echo $CPU | sudo tee /proc/irq/$IRQ/smp_affinity_list >/dev/null
    done
)

# https://lwn.net/Articles/837010/; do not given improvements
# echo 2 | sudo tee /sys/class/net/ens5/napi_defer_hard_irqs
# echo 200000 | sudo tee /sys/class/net/ens5/gro_flush_timeout
# echo 0 | sudo tee /sys/class/net/ens5/napi_defer_hard_irqs
# echo 0 | sudo tee /sys/class/net/ens5/gro_flush_timeout

## run af_xdp l2fwd
## -z: zero-copy mode (without skb copy)
## -p: polling with timeout of 1ms.
# sudo ./af_xdp_user -d ${NIC} --filename af_xdp_kern.o -z

## for efa test
# sudo ./af_xdp_user_efa -d ${NIC} --filename af_xdp_kern_efa.o -z
