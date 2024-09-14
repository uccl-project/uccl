# !/bin/bash

# Install aws libfabric efa: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html 
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.34.0.tar.gz
tar -xf aws-efa-installer-1.34.0.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y

sudo apt-get install libtool autoconf -y
git clone -b 1.22.0amzn1.0 git@github.com:aws/libfabric.git && cd libfabric && mkdir build
./autogen.sh && ./configure --prefix=/home/ubuntu/libfabric/build
# make -j && make install
sudo mv /opt/amazon/efa/include/ /opt/amazon/efa/include_efa/
sudo cp -r include/ /opt/amazon/efa/include/
sudo cp config.h /opt/amazon/efa/include/


git submodule update --init
sudo apt update
sudo apt install clang llvm libelf-dev libpcap-dev gcc-multilib build-essential linux-tools-common linux-tools-generic m4 -y
# On arm
# sudo apt install clang llvm libelf-dev libpcap-dev build-essential linux-tools-common linux-tools-generic m4 -y
sudo apt install linux-headers-$(uname -r)

./configure
cd lib/xdp-tools && ./configure && make -j
cd .. && make -j
cd .. && make -j
cd advanced04-aws-afxdp && make -j


# sudo ethtool -N ens1f1np1 rx-flow-hash udp4 fn
# sudo ethtool -N ens1f1np1 flow-type udp4 action 20

# In order to use zero-copy mode, must be queue id 10-20
# Using `sudo ethtool -n ens1f1np1` to check existing rules
# Using `sudo ethtool -N ens1f1np1 delete 1022` to delete any redundant rules, which impacts ens1f1np1 receiving packets in AF_XDP.

# sudo arp -s [ip addr] [ethernet addr]