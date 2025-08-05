#!/bin/bash

docker rm /nixl_amd_bench || true

docker build -t nixl_amd:latest -f dockerfile .

docker run -it \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME:/root \
    --shm-size 64G \
    --name nixl_amd_bench \
    nixl_amd:latest 
