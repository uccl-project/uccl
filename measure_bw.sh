# /usr/bin/bash

NIC=${1:-ens6} # enp199s0 for g4.metal

# while true; do
#     tx1=$(cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data)
#     rx1=$(cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_data)
#     sleep 1
#     tx2=$(cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data)
#     rx2=$(cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_data)
#     tx_bandwidth=$(( (tx2 - tx1) * 4 ))  # Multiply by 4 to convert words to bytes
#     rx_bandwidth=$(( (rx2 - rx1) * 4 ))
#     echo "TX: $tx_bandwidth bytes/sec, RX: $rx_bandwidth bytes/sec"
# done

while true; do
    tx1=$(cat /sys/class/net/${NIC}/statistics/tx_bytes)
    rx1=$(cat /sys/class/net/${NIC}/statistics/rx_bytes)
    sleep 1
    tx2=$(cat /sys/class/net/${NIC}/statistics/tx_bytes)
    rx2=$(cat /sys/class/net/${NIC}/statistics/rx_bytes)
    tx_bandwidth=$(echo "scale=2; ($tx2 - $tx1) / 1000000000.0 * 8" | bc)
    rx_bandwidth=$(echo "scale=2; ($rx2 - $rx1) / 1000000000.0 * 8" | bc)
    echo "TX: $tx_bandwidth Gbps, RX: $rx_bandwidth Gbps"
done
