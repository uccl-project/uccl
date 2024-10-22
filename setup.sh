# !/bin/bash

# Can only use half of the queue, per ENA implementation: https://github.com/amzn/amzn-drivers/issues/240
NIC=$1
NQUEUE=$2
MTU=$3
MODE=$4
PLATFORM=$5
echo "configuring ${NIC} with ${NQUEUE} queues and ${MTU} MTU for ${MODE}"

echo "unloading any xdp programs"
sudo ~/uccl/lib/xdp-tools/xdp-loader/xdp-loader unload ${NIC} --all

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

start_cpu=3
ncpu=$(nproc)
(
    let cnt=0
    cd /sys/class/net/${NIC}/device/msi_irqs/
    IRQs=(*)
    # Exclude the first IRQ, which is for the control plane
    for IRQ in "${IRQs[@]:1}"; do
        let CPU=$(((cnt + start_cpu) % ncpu))
        let cnt=$(((cnt + 1) % NQUEUE))
        echo $IRQ '->' $CPU
        echo $CPU | sudo tee /proc/irq/$IRQ/smp_affinity_list >/dev/null
    done
)

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
