# Unified CCL

### Building the system

```
sudo apt install clang llvm libelf-dev libpcap-dev build-essential libc6-dev-i386 linux-tools-$(uname -r) libgoogle-glog-dev libgtest-dev byobu net-tools iperf iperf3 -y
./setup_extra.sh
make
```

If you want to build nccl and nccl-tests on cloudlab ubuntu22, you need to install cuda and openmpi: 
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit -y
sudo apt install nvidia-driver-550 nvidia-utils-550 -y

sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev -y

cd nccl
make src.build -j
cp src/include/nccl_common.h build/include/

cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/opt/uccl/nccl/build -j
```

Remember to change `afxdp/transport_config.h` based on your NIC IPs and MACs. 

### Run TCP testing

```
cd afxdp; make -j "CXXFLAGS=-DAWS_ENA"
or 
cd afxdp; make -j "CXXFLAGS=-DCLOUDLAB_MLX5"

# On both server and client
./setup_nic.sh ens6 4 4 9001 tcp aws
or
./setup_nic.sh ens1f1np1 4 4 1500 tcp cloudlab

# On server, edit nodes.txt to include all node ips
python rsync.sh
./server_tcp_main

# On client
./client_tcp_main -a 192.168.6.1
```

### Run AFXDP testing

```
cd afxdp; make -j "CXXFLAGS=-DAWS_ENA"
or 
cd afxdp; make -j "CXXFLAGS=-DCLOUDLAB_MLX5"

# On both server and client
./setup_nic.sh ens6 1 1 3498 afxdp aws
or
./setup_nic.sh ens1f1np1 1 1 1500 afxdp cloudlab

# On server, edit nodes.txt to include all node ips
python rsync.sh
sudo ./server_main

# On client
sudo ./client_main
```

### Debugging the transport stack

Note that any program that leverages util_afxdp no long needs root to use AFXDP sockets.

```
sudo ./afxdp_daemon_main --logtostderr=1
./transport_test --logtostderr=1 --vmodule=transport=1,util_afxdp=1 --clientip=192.168.6.2 --test=async --verify --rand
./transport_test --logtostderr=1 --vmodule=transport=1,util_afxdp=1 --client --serverip=192.168.6.1 --test=async --verify --rand
```

### Run nccl-tests

Assume you have 4 node IPs in `nodes.txt`

```
# for TCP
python setup_all.py --target=aws_tcp
./run_nccl_test.sh tcp 4

# for AFXDP
python setup_all.py --target=aws_afxdp
./run_nccl_test.sh afxdp 4
```

### MISC setup

Install anaconda: 
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p /opt/anaconda3
source /opt/anaconda3/bin/activate
conda init
conda install paramiko -y
```

Avoiding nodes.txt polluting the repo: 
```
git update-index --skip-worktree nodes.txt
git update-index --no-skip-worktree nodes.txt
```

Iperf test: 
```
iperf -s -i 1 -P 64 -t 1000 --dualtest -B 192.168.6.1
iperf -c 192.168.6.1 -i 1 -P 64 -t 1000 --dualtest -B 192.168.6.2
```