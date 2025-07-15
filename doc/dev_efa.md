# UCCL-EFA

AWS EFA support for UCCL. We are using Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04) AMI on 4x`p4d.24xlarge` AWS VMs; latest AMI and `p4de.24xlarge` should also work. 
Note that you need to enable all four EFA NICs in `p4d.24xlarge` instances. 

## Prerequisites

Using the following commands to install necessary kernel modules for EFA directly accessing GPU memory. 

```bash
# Latest version of aws-efa-installer should also work. 
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.34.0.tar.gz
tar -xf aws-efa-installer-1.34.0.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y
```

Make sure you haveed install docker. Then run the following and log back in. 
```bash
sudo usermod -aG docker $USER
```

## Building EFA plugin

```bash
# Eg, /home/ubuntu/uccl
export UCCL_HOME=<the path of uccl>

# Build libnccl-net-efa.so
cd $UCCL_HOME/efa
make -j

# Build nccl-sg for UCCL (taking ~3min); assume A100 GPUs
cd $UCCL_HOME/thirdparty/nccl-sg
make src.build -j NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
cp src/include/nccl_common.h build/include/

# Optionally, if you want to run nccl-tests for the original NCCL
cd $UCCL_HOME/thirdparty/nccl
make src.build -j NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# Build nccl-tests; consider "conda deactivate" when hitting dependency errors
cd $UCCL_HOME/thirdparty/nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$UCCL_HOME/thirdparty/nccl-sg/build -j
```

## Runing nccl-tests for UCCL

Filling `$UCCL_HOME/scripts/node_ips/p4d.txt` with the ssh'able IP addresses of the nodes for rsync'ing all built libs. 
Filling `$UCCL_HOME/efa/hosts` with the ssh'able IP addresses of the nodes for mpirun use. There, `slots` denotes the number of processes you want to run on each server; we currently only support 8. 

```bash
cd $UCCL_HOME/scripts
python rsync.py

# Assume four p4d.24xlarge instances each with 8 A100 GPUs. 
cd $UCCL_HOME/efa
./run_nccl_test.sh ud 32 0
``` 

## Running UCCL for PyTorch Applications

Generally, the main environment variables to set for UCCL are: 
```bash
LD_PRELOAD="${UCCL_HOME}/thirdparty/nccl-sg/build/lib/libnccl.so"
NCCL_NET_PLUGIN="${UCCL_HOME}/efa/libnccl-net-efa.so"
NCCL_PROTO=Simple
```
Currently, UCCL only supports `Simple` protocol; support for `LL` and `LL128` is on the way. 

Check [misc/run_ddp.sh](../misc/run_ddp.sh) for an example of running UCCL with PyTorch DDP applications by `cd ../misc; ./run_ddp.sh`. Other applications such as DeepSpeed, vLLM, Megatron-LM, and PyTorch FSDP should work similarly. 


## MISC

### Install lastest perftest with patches to benchmark EFA NICs

```bash
pushd /tmp
git clone https://github.com/linux-rdma/perftest.git && cd perftest && git checkout c04922f
git apply $UCCL_HOME/efa/perftest.patch
./autogen.sh && ./configure && make -j
sudo make install
popd
```

Throughput benchmark: 
```bash
ib_send_bw -d rdmap16s27 --report_gbits -x 0 -c UD -t 128 -Q 1 -q 32 -l 2 -s 8192 -F
ib_send_bw -d rdmap16s27 --report_gbits -x 0 -c UD -t 128 -Q 1 -q 32 -l 2 -s 8192 -F <serverip>
```

Latency benchmark: 
```bash
ib_send_lat -d rdmap16s27 --report_gbits -x 0 -c UD -F
ib_send_lat -d rdmap16s27 --report_gbits -x 0 -c UD -F <serverip>
```

### Run transport tests

```bash
./util_efa_test --logtostderr
./util_efa_test --logtostderr <serverip>
```

```bash
./transport_test --logtostderr --test=bimq --clientip=<clientip>
./transport_test --logtostderr --test=bimq --serverip=<serverip>
```
