# UCCL Wheel Building

### Python Wheel Build

Run the following to build Python wheels: 
```bash
cd $UCCL_HOME
./build.sh [cuda|rocm|therock] [all|ccl_rdma|ccl_efa|p2p|ep] [py_version] [rocm_index_url]
```

Run the following to install the wheels locally: 
```bash
cd $UCCL_HOME
pip install wheelhouse-[cuda/rocm]/uccl-*.whl
```

Note that you need ARM hosts to build ARM wheels, as cross-compilation tool `qemu-user-static` cannot emulate CUDA or ROCm. 

### On Cloudlab CPU Machines

If you want to build nccl and nccl-tests on cloudlab ubuntu22, you need to install cuda and openmpi: 

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit -y
sudo apt install nvidia-driver-550 nvidia-utils-550 -y
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev -y
```
