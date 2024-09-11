# !/bin/bash

cd net; make clean && make -j
cd ../advanced03-AF_XDP; make clean && make -j
cd ..

# all_followers=("192.168.6.2" "192.168.6.3" "192.168.6.4" "192.168.6.5")
# for fip in "${all_followers[@]}"; do
#   rsync -auv -e 'ssh -o StrictHostKeyChecking=no' ~/afxdp/ $fip:~/afxdp/ &
# done

# wait

# Updating ena driver to the lastest version
git clone https://github.com/amzn/amzn-drivers.git
cd amzn-drivers/kernel/linux/ena/ && make
sudo rmmod ena && sudo insmod ena.ko
# Check version by modinfo ena

# Can only use half of the queue, per ENA implementation: https://github.com/amzn/amzn-drivers/issues/240
sudo ethtool -L enp39s0 combined 4 
sudo ifconfig enp39s0 mtu 3498 up

# The -z flag forces zero-copy mode.  Without it, it will probably default to copy mode
# -p means using polling with timeout of 1ms.
sudo ./af_xdp_user -d enp39s0 -z

# For client machines
# Start followers first. Run this on each follower client machine: ./follower [num threads] [follower ip]
# Then start leader: ./leader [num_followers] [num leader threads] [follower ip 1] [follower ip 2] ... [follower ip n]


# sudo systemctl stop irqbalance

# (let CPU=0; cd /sys/class/net/enp39s0/device/msi_irqs/;
#   for IRQ in *; do
#     echo $CPU | sudo tee /proc/irq/$IRQ/smp_affinity_list
#     # let CPU=$(((CPU+1)%ncpu))
# done)

# ./follower 40 192.168.6.3
# ./follower 40 192.168.6.4
# ./follower 40 192.168.6.5
# ./leader 3 40 192.168.6.3 192.168.6.4 192.168.6.5