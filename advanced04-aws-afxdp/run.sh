# !/bin/bash

cd net; make clean && make -j
cd ../advanced03-AF_XDP; make clean && make -j
cd ..

# all_followers=("192.168.6.2" "192.168.6.3" "192.168.6.4" "192.168.6.5")
# for fip in "${all_followers[@]}"; do
#   rsync -auv -e 'ssh -o StrictHostKeyChecking=no' ~/afxdp/ $fip:~/afxdp/ &
# done

# wait

## Updating ena driver to the lastest version
# git clone https://github.com/amzn/amzn-drivers.git
# cd amzn-drivers/kernel/linux/ena/ && make
# sudo rmmod ena && sudo insmod ena.ko
# sudo mv ena.ko /lib/modules/6.5.0-1022-aws/kernel/drivers/net/ethernet/amazon/ena/ena.ko
## Check version by modinfo ena

# Can only use half of the queue, per ENA implementation: https://github.com/amzn/amzn-drivers/issues/240
sudo ethtool -L ens5 combined 1
sudo ifconfig ens5 mtu 3498 up
sudo ethtool -C ens5 adaptive-rx off rx-usecs 0 tx-usecs 0
# sudo ethtool -C ens5 adaptive-rx off rx-usecs 20 tx-usecs 60
sudo service irqbalance stop

## The -z flag forces zero-copy mode.  Without it, it will probably default to copy mode
## -p means using polling with timeout of 1ms.
# sudo ./af_xdp_user -d ens5 --filename af_xdp_kern.o -z

## for efa
# sudo ./af_xdp_user_efa -d ens5 --filename af_xdp_kern_efa.o -z

nqueue=8
(let cnt=0; cd /sys/class/net/ens5/device/msi_irqs/;
for IRQ in *; do
    let CPU=$((cnt*2+1))
    let cnt=$(((cnt+1)%nqueue))
    echo $IRQ '->' $CPU
    echo $CPU | sudo tee /proc/irq/$IRQ/smp_affinity_list > /dev/null
done)

# For client machines
# Start followers first. Run this on each follower client machine: ./follower [num threads] [follower ip]
# Then start leader: ./leader [num_followers] [num leader threads] [follower ip 1] [follower ip 2] ... [follower ip n]


# sudo systemctl stop irqbalance

# (let CPU=0; cd /sys/class/net/ens5/device/msi_irqs/;
#   for IRQ in *; do
#     echo $CPU | sudo tee /proc/irq/$IRQ/smp_affinity_list
#     # let CPU=$(((CPU+1)%ncpu))
# done)

# ./follower 40 192.168.6.3
# ./follower 40 192.168.6.4
# ./follower 40 192.168.6.5
# ./leader 3 40 192.168.6.3 192.168.6.4 192.168.6.5