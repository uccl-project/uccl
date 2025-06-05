#!/bin/bash

UCCL_HOME=/root/uccl

scp $UCCL_HOME/rdma_cuda/libnccl-net-uccl.so root@192.168.102.191:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/libnccl-net-uccl.so root@192.168.102.192:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/libnccl-net-uccl.so root@192.168.102.193:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/libnccl-net-uccl.so root@192.168.102.194:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/libnccl-net-uccl.so root@192.168.102.195:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/rdma_test root@192.168.102.191:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/rdma_test root@192.168.102.192:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/rdma_test root@192.168.102.193:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/rdma_test root@192.168.102.194:$UCCL_HOME/rdma_cuda/
scp $UCCL_HOME/rdma_cuda/rdma_test root@192.168.102.195:$UCCL_HOME/rdma_cuda/

