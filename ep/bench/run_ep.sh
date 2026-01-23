# !/bin/bash

NNODES=${1:-2}
RANK=${2:-0}
MODE=${3:-ll} # ll, ht, ll-pressure, ht-pressure
MAIN_IP=${4:-10.1.18.53}

echo "Running nodes $NNODES, rank $RANK, mode $MODE, main IP $MAIN_IP"

if [ "$MODE" = "ll" ]; then
    torchrun --nnodes=$NNODES --nproc_per_node=8 --node_rank=$RANK \
        --master_addr=$MAIN_IP --master_port=12355 \
        test_low_latency.py --num-tokens=128 \
        --hidden=7168 --num-topk=8 --num-experts=288
elif [ "$MODE" = "ht" ]; then
    torchrun --nnodes=$NNODES --nproc_per_node=8 --node_rank=$RANK \
        --master_addr=$MAIN_IP --master_port=12355 \
        test_internode.py  --num-tokens=4096 \
        --hidden=7168 --num-topk=8 --num-experts=288 --test-ll-compatibility
elif [ "$MODE" = "ll-pressure" ]; then
    torchrun --nnodes=$NNODES --nproc_per_node=8 --node_rank=$RANK \
        --master_addr=$MAIN_IP --master_port=12355 \
        test_low_latency.py --num-tokens=128 \
        --hidden=7168 --num-topk=8 --num-experts=288 --pressure-test-mode=2 --debug-hash
elif [ "$MODE" = "ht-pressure" ]; then
    torchrun --nnodes=$NNODES --nproc_per_node=8 --node_rank=$RANK \
        --master_addr=$MAIN_IP --master_port=12355 \
        test_internode.py  --num-tokens=4096 \
        --hidden=7168 --num-topk=8 --num-experts=288 --test-ll-compatibility --pressure-test-mode=2
fi
# --log-dir=logs --redirect=3
