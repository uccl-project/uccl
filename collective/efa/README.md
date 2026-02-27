# UCCL-EFA

AWS EFA support for UCCL. We are using Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04) AMI on 4x`p4d.24xlarge` AWS VMs; latest AMI and `p4de.24xlarge` should also work. 
Note that you need to enable all four EFA NICs in `p4d.24xlarge` instances. 

## Prerequisites

Using the following commands to install necessary kernel modules for EFA directly accessing GPU memory. 

```bash
# Latest version of aws-efa-installer should also work. 
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.42.0.tar.gz
tar -xf aws-efa-installer-1.42.0.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y
sudo modprobe efa_nv_peermem || true
```

Make sure you haveed install docker. Then run the following and log back in. 
```bash
sudo usermod -aG docker $USER
```

## Building NCCL and NCCL-tests

```bash
# Eg, /home/ubuntu/uccl
export UCCL_HOME=<the path of uccl>

# Build nccl-sg for UCCL (taking ~3min); assume A100 GPUs
cd $UCCL_HOME/thirdparty/nccl-sg
make src.build -j NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
cp src/include/nccl_common.h build/include/

# Optionally, if you want to run nccl-tests for the original NCCL
cd $UCCL_HOME/thirdparty/nccl
make src.build -j NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# Build nccl-tests; consider "conda deactivate" when hitting dependency errors
cd $UCCL_HOME/thirdparty/nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$UCCL_HOME/thirdparty/nccl/build -j
```

## Building EFA plugin

The easiest way is to use docker, which packs all needed external libraries into a python wheel and install into your local python env: 
```bash
cd $UCCL_HOME && bash build.sh cuda efa --install
```

The following alternative is best for development where you have installed all needed external libraries: 
<details><summary>Click me</summary>

```bash
# Build libnccl-net-efa.so
cd $UCCL_HOME/efa
make -j
```
</details>


## Runing nccl-tests for UCCL

Filling `$UCCL_HOME/scripts/node_ips/p4d.txt` with the ssh'able IP addresses of the nodes for rsync'ing all built libs and running mpi. 

```bash
cd $UCCL_HOME/scripts
python rsync.py -n node_ips/p4d.txt

# Assume four p4d.24xlarge instances each with 8 A100 GPUs. 
cd $UCCL_HOME/collective/efa
./run_nccl_test.sh ud 32
``` 

## Running UCCL for PyTorch Applications

Generally, the main environment variables to set for UCCL are: 
```bash
LD_PRELOAD=`python -c "import uccl; print(uccl.efa_nccl_path())"`
NCCL_NET_PLUGIN=`python -c "import uccl; print(uccl.efa_plugin_path())"`
NCCL_PROTO=Simple
```
UCCL currently only supports `Simple` protocol; support for `LL` and `LL128` is on the way. 

You can launch distributed ResNet training by: 
```bash
cd $UCCL_HOME/experimental/misc

# Benchmark UCCL
bash run_resnet_uccl.sh

# Benchmark NCCL
bash run_resnet_nccl.sh
```

You can also check [experimental/misc/run_ddp.sh](../experimental/misc/run_ddp.sh) for an example of running UCCL with PyTorch DDP applications. 
```bash
cd $UCCL_HOME/experimental/misc

# Run UCCL
./run_ddp.sh ud

# Run NCCL
./run_ddp.sh srd
```
Other applications such as DeepSpeed, vLLM, Megatron-LM, and PyTorch FSDP should work similarly. 


## MISC

<details><summary>Click me</summary>

### Install lastest perftest with patches to benchmark EFA NICs

```bash
pushd /tmp
git clone https://github.com/linux-rdma/perftest.git && cd perftest && git checkout c04922f
git apply $UCCL_HOME/collective/efa/perftest.patch
./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && make -j
sudo make install
popd
```

Throughput benchmark: 
```bash
ib_send_bw -d rdmap16s27 --report_gbits -x 0 -c <UD|SRD> -t 128 -Q 1 -q 32 -l 2 -s 8192 -F --use_cuda 0
ib_send_bw -d rdmap16s27 --report_gbits -x 0 -c <UD|SRD> -t 128 -Q 1 -q 32 -l 2 -s 8192 -F --use_cuda 0 <serverip>
```

Latency benchmark: 
```bash
ib_send_lat -d rdmap16s27 --report_gbits -x 0 -c <UD|SRD> -F --use_cuda 0
ib_send_lat -d rdmap16s27 --report_gbits -x 0 -c <UD|SRD> -F --use_cuda 0 <serverip>
```

### Run transport tests

```bash
./util_efa_test --logtostderr            # server
./util_efa_test --logtostderr <serverip> # client
```

```bash
./transport_test --logtostderr --test=bimq --clientip=<clientip>
./transport_test --logtostderr --test=bimq --serverip=<serverip>
```
</details>
