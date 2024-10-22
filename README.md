# Unified CCL

### Building the system

```
sudo apt install clang llvm libelf-dev libpcap-dev build-essential libc6-dev-i386 linux-tools-$(uname -r) libgoogle-glog-dev -y
make
```

If you want to build nccl and nccl-tests on cloudlab ubuntu24, you need to install cuda and openmpi: 
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
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/users/yangzhou/uccl/nccl/build -j
```

### Run TCP testing

```
cd afxdp; make -j

# On both server and client
./setup.sh ens6 4 9001 tcp

# On server
./sync.sh 172.31.25.5
./server_tcp_main

# On client
./client_tcp_main -a 172.31.18.199
```

### Run AFXDP testing

```
cd afxdp; make -j

# On both server and client
./setup.sh ens6 1 3498 afxdp

# On server
./sync.sh 172.31.25.5
sudo ./server_main

# On client
sudo ./client_main
```

### Debugging the transport stack

Note that any program that leverages util_afxdp no long needs root to use AFXDP sockets.

```
sudo ./afxdp_daemon_main --logtostderr=1
./transport_test --logtostderr=1 --vmodule=transport=1,util_afxdp=1

sudo ./afxdp_daemon_main --logtostderr=1
./transport_test --client --logtostderr=1 --vmodule=transport=1,util_afxdp=1
```

### MISC setup

Install anaconda: 
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b
```