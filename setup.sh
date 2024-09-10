# !/bin/bash

git submodule update --init
sudo apt update
sudo apt install clang llvm libelf-dev libpcap-dev gcc-multilib build-essential linux-tools-common linux-tools-generic m4 -y
sudo apt install linux-headers-$(uname -r)

./configure
cd lib/xdp-tools && ./configure && make -j
cd .. && make -j
cd .. && make -j
cd advanced03-AF_XDP && make -j

sudo ethtool -N ens1f1np1 rx-flow-hash udp4 fn
sudo ethtool -N ens1f1np1 flow-type udp4 action 20
# In order to use zero-copy mode, must be queue id 10-20
# Using `sudo ethtool -n ens1f1np1` to check existing rules
# Using `sudo ethtool -N ens1f1np1 delete 1022` to delete any redundant rules, which impacts ens1f1np1 receiving packets in AF_XDP.

# sudo arp -s [ip addr] [ethernet addr]