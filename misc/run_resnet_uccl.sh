#!/bin/bash

# 添加信号处理和清理函数
cleanup() {
    echo "正在清理进程..."
    
    # 在所有主机上杀掉torchrun进程
    for HOST in "${HOSTS[@]}"; do
        echo "在 $HOST 上终止进程..."
        ssh $HOST "pkill -f 'python /opt/conda/bin/torchrun'" || true
    done
    
    # 在本地杀掉torchrun进程
    pkill -f 'python /opt/conda/bin/torchrun' || true
    
    echo "清理完成"
}

# 注册信号处理程序
trap cleanup EXIT SIGINT SIGTERM

UCCL_HOME="/home/ubuntu/uccl_yang"
NV_LINK_DISABLE=1
CHANNELS=16
CHANNELS_NET_PEER=4
CHUNK_SIZE=131072
BUFFSIZE=1048576


# Environment variables for UCCL
export NCCL_LD_PRELOAD="${UCCL_HOME}/thirdparty/nccl-sg/build/lib/libnccl.so"
export NCCL_NET_PLUGIN="${UCCL_HOME}/efa/libnccl-net-efa.so"
export NCCL_DEBUG=
export NCCL_PROTO=Simple
export NCCL_P2P_DISABLE=${NV_LINK_DISABLE}
export NCCL_SHM_DISABLE=${NV_LINK_DISABLE}
export NCCL_NET_DISABLE=0
export NCCL_MAX_NCHANNELS=${CHANNELS}
export NCCL_MIN_NCHANNELS=${CHANNELS}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER}
export NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE}
export NCCL_BUFFSIZE=${BUFFSIZE}
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml
export UCCL_ENGINE_QUIET=1
export GLOG_logtostderr=0

# Additional settings for async communication
export NCCL_LAUNCH_MODE=PARALLEL

# 读取hostfile配置
HOSTFILE="./hostfile"
if [ ! -f "$HOSTFILE" ]; then
    echo "错误: hostfile不存在 ($HOSTFILE)"
    exit 1
fi

# 获取主机列表
HOSTS=($(cat $HOSTFILE))
NUM_NODES=${#HOSTS[@]}

if [ $NUM_NODES -lt 1 ]; then
    echo "错误: hostfile中没有找到主机"
    exit 1
fi

# 每个节点的GPU数量
NUM_GPUS_PER_NODE=8
# 总进程数
WORLD_SIZE=$((NUM_NODES * NUM_GPUS_PER_NODE))
# 主节点地址（hostfile中的第一个主机）
MASTER_ADDR=${HOSTS[0]}
# 主节点端口
MASTER_PORT=29500
# Python脚本路径
SCRIPT_PATH="customized_resnet_ddp_layer_reduce_async.py"
# torchrun完整路径
TORCHRUN_PATH="python /opt/conda/bin/torchrun"

echo "集群信息:"
echo "主机列表: ${HOSTS[@]}"
echo "节点数量: $NUM_NODES"
echo "每个节点的GPU数量: $NUM_GPUS_PER_NODE"
echo "总GPU数量: $WORLD_SIZE"
echo "主节点地址: $MASTER_ADDR"

# 设置环境变量
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export NUM_NODES=$NUM_NODES
export NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE

# 生成pdsh命令的主机组字符串
PDSH_HOSTS=$(IFS=,; echo "${HOSTS[*]}")

# 首先启动非rank 0的节点
for ((i=1; i<${#HOSTS[@]}; i++)); do
    HOST=${HOSTS[$i]}
    NODE_RANK=$i
    
    # 为每个主机准备的命令
    CMD="cd $(pwd) && \
    export MASTER_ADDR=$MASTER_ADDR && \
    export MASTER_PORT=$MASTER_PORT && \
    export WORLD_SIZE=$WORLD_SIZE && \
    export NODE_RANK=$NODE_RANK && \
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES && \
    export UCCL_HOME=$UCCL_HOME && \
    export NV_LINK_DISABLE=$NV_LINK_DISABLE && \
    export CHANNELS=$CHANNELS && \
    export CHANNELS_NET_PEER=$CHANNELS_NET_PEER && \
    export CHUNK_SIZE=$CHUNK_SIZE && \
    export BUFFSIZE=$BUFFSIZE && \
    export LD_PRELOAD=${NCCL_LD_PRELOAD} && \
    export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN} && \
    export NCCL_DEBUG= && \
    export NCCL_PROTO=Simple && \
    export NCCL_P2P_DISABLE=${NV_LINK_DISABLE} && \
    export NCCL_SHM_DISABLE=${NV_LINK_DISABLE} && \
    export NCCL_NET_DISABLE=0 && \
    export NCCL_MAX_NCHANNELS=${CHANNELS} && \
    export NCCL_MIN_NCHANNELS=${CHANNELS} && \
    export NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} && \
    export NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} && \
    export NCCL_BUFFSIZE=${BUFFSIZE} && \
    export NCCL_NET_GDR_LEVEL=SYS && \
    export NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml && \
    export UCCL_ENGINE_QUIET=1 && \
    export GLOG_logtostderr=0 && \
    export NCCL_LAUNCH_MODE=PARALLEL && \
    export OMP_NUM_THREADS=1 && \

    $TORCHRUN_PATH \
      --nnodes=$NUM_NODES \
      --nproc_per_node=$NUM_GPUS_PER_NODE \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      $SCRIPT_PATH \
      --batch_size 128 \
      --epochs 10 \
      --lr 0.1 2>&1 | sed 's/^/[NODE-$NODE_RANK] /'"
    
    # 使用ssh在远程主机上执行命令，将输出传回并添加节点前缀
    echo "在 $HOST (rank $NODE_RANK) 上启动torchrun..."
    ssh $HOST "$CMD" | sed "s/^/[NODE-$NODE_RANK] /" &
    
    # 存储进程PID以便稍后清理
    if [ $? -eq 0 ]; then
        echo "成功在 $HOST 上启动进程"
    else
        echo "警告: 在 $HOST 上启动进程可能失败"
    fi
done

# 最后启动rank 0节点
HOST=${HOSTS[0]}
NODE_RANK=0

echo "在 $HOST (rank $NODE_RANK) 上启动torchrun..."

# 为rank 0节点准备的命令
CMD="cd $(pwd) && \
export MASTER_ADDR=$MASTER_ADDR && \
export MASTER_PORT=$MASTER_PORT && \
export WORLD_SIZE=$WORLD_SIZE && \
export NODE_RANK=$NODE_RANK && \
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES && \
export UCCL_HOME=$UCCL_HOME && \
export NV_LINK_DISABLE=$NV_LINK_DISABLE && \
export CHANNELS=$CHANNELS && \
export CHANNELS_NET_PEER=$CHANNELS_NET_PEER && \
export CHUNK_SIZE=$CHUNK_SIZE && \
export BUFFSIZE=$BUFFSIZE && \
export LD_PRELOAD=${NCCL_LD_PRELOAD} && \
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN} && \
export NCCL_DEBUG= && \
export NCCL_PROTO=Simple && \
export NCCL_P2P_DISABLE=${NV_LINK_DISABLE} && \
export NCCL_SHM_DISABLE=${NV_LINK_DISABLE} && \
export NCCL_NET_DISABLE=0 && \
export NCCL_MAX_NCHANNELS=${CHANNELS} && \
export NCCL_MIN_NCHANNELS=${CHANNELS} && \
export NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} && \
export NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} && \
export NCCL_BUFFSIZE=${BUFFSIZE} && \
export NCCL_NET_GDR_LEVEL=SYS && \
export NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml && \
export UCCL_ENGINE_QUIET=1 && \
export GLOG_logtostderr=0 && \
export NCCL_LAUNCH_MODE=PARALLEL && \
export OMP_NUM_THREADS=1 && \
$TORCHRUN_PATH \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $SCRIPT_PATH \
  --batch_size 128 \
  --epochs 10 \
  --lr 0.1  2>&1 | sed 's/^/[NODE-$NODE_RANK] /'"

# 在当前节点是rank 0时，直接在前台执行；否则，通过ssh执行
if [ "$(hostname)" == "$HOST" ] || [ "$(hostname -i)" == "$HOST" ]; then
    # 直接在前台执行
    eval "$CMD"
else
    # 通过ssh执行
    ssh $HOST "$CMD"
fi

# 等待任意一个前台进程结束
wait -n

# 确保杀掉所有其他进程
cleanup

echo "基准测试完成！" 