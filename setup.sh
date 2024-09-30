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


# Install last ena driver with reboot persistent
sudo apt-get install dkms
git clone https://github.com/amzn/amzn-drivers.git
sudo mv amzn-drivers /usr/src/amzn-drivers-2.13.0
sudo vi /usr/src/amzn-drivers-2.13.0/dkms.conf

# Paste the following and save the file:
PACKAGE_NAME="ena"
PACKAGE_VERSION="2.13.0"
CLEAN="make -C kernel/linux/ena clean"
MAKE="make -C kernel/linux/ena/ BUILD_KERNEL=${kernelver}"
BUILT_MODULE_NAME[0]="ena"
BUILT_MODULE_LOCATION="kernel/linux/ena"
DEST_MODULE_LOCATION[0]="/updates"
DEST_MODULE_NAME[0]="ena"
REMAKE_INITRD="yes"
AUTOINSTALL="yes"

sudo dkms add -m amzn-drivers -v 2.13.0
sudo dkms build -m amzn-drivers -v 2.13.0
sudo dkms install -m amzn-drivers -v 2.13.0
sudo modprobe -r ena; sudo modprobe ena


git submodule update --init
sudo apt update
sudo apt install clang llvm libelf-dev libpcap-dev gcc-multilib build-essential linux-tools-common linux-tools-generic m4 -y
# On arm
# sudo apt install clang llvm libelf-dev libpcap-dev build-essential linux-tools-common linux-tools-generic m4 -y
sudo apt install linux-headers-$(uname -r)

./autogen.sh && ./configure
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