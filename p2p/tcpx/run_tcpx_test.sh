#!/bin/bash

# TCPX P2P Test Script
# Environment configuration based on successful collective/rdma/run_nccl_test_tcpx.sh

source ../../scripts/shared.sh

# Usage: ./run_tcpx_test.sh [test_type] [server_ip]
# test_type: connection|transfer
# server_ip: required for connection and transfer tests

TEST_TYPE=${1:-connection}
SERVER_IP=${2:-""}
HOSTFILE="${UCCL_HOME}/scripts/node_ips/tcpx.txt"

# Read node IPs from hostfile
if [ ! -f "$HOSTFILE" ]; then
    echo "❌ Hostfile not found: $HOSTFILE"
    echo "Please ensure UCCL_HOME environment variable is set correctly"
    exit 1
fi

NODE1_IP=$(sed -n '1p' "$HOSTFILE")
NODE2_IP=$(sed -n '2p' "$HOSTFILE")

echo "=== TCPX P2P Test Script ==="
echo "Test type: $TEST_TYPE"
echo "Hostfile: $HOSTFILE"
echo "Node 1 (server): $NODE1_IP"
echo "Node 2 (client): $NODE2_IP"

# Set environment variables - based on successful NCCL TCPX configuration
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/tcpx/lib64:$UCCL_HOME/thirdparty/nccl/build/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:$LD_LIBRARY_PATH"

# TCPX specific environment variables
export UCCL_TCPX_DEBUG=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_TCPX_RXMEM_IMPORT_USE_GPU_PCI_CLIENT=1
export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX="/run/tcpx"
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0

# Debug settings
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

# CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo ""
echo "Environment variables set:"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "  NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=$NCCL_GPUDIRECTTCPX_SOCKET_IFNAME"
echo "  NCCL_GPUDIRECTTCPX_CTRL_DEV=$NCCL_GPUDIRECTTCPX_CTRL_DEV"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Build test programs
echo ""
echo "=== Building test programs ==="
cd "$(dirname "$0")"

case $TEST_TYPE in
    "connection")
        if [ -z "$SERVER_IP" ]; then
            echo "❌ Connection test requires server IP"
            echo "Usage: $0 connection <server_ip>"
            echo "Or run on two nodes separately:"
            echo "  Node 1 ($NODE1_IP): $0 connection server"
            echo "  Node 2 ($NODE2_IP): $0 connection $NODE1_IP"
            exit 1
        fi

        echo "Building connection test..."
        if ! make test_connection; then
            echo "❌ Build failed"
            exit 1
        fi
        echo "✅ Build complete"
        echo ""

        if [ "$SERVER_IP" = "server" ]; then
            echo "=== Running connection test (server mode) ==="
            ./tests/test_connection server
        else
            echo "=== Running connection test (client mode) ==="
            echo "Connecting to server: $SERVER_IP"
            ./tests/test_connection client $SERVER_IP
        fi
        ;;

    "transfer")
        if [ -z "$SERVER_IP" ]; then
            echo "❌ Transfer test requires server IP"
            echo "Usage: $0 transfer <server_ip>"
            echo "Or run on two nodes separately:"
            echo "  Node 1 ($NODE1_IP): $0 transfer server"
            echo "  Node 2 ($NODE2_IP): $0 transfer $NODE1_IP"
            exit 1
        fi

        echo "Building GPU transfer test..."
        if ! make test_tcpx_transfer; then
            echo "❌ Build failed"
            exit 1
        fi
        echo "✅ Build complete"
        echo ""

        if [ "$SERVER_IP" = "server" ]; then
            echo "=== Running GPU transfer test (server mode) ==="
            ./tests/test_tcpx_transfer server
        else
            echo "=== Running GPU transfer test (client mode) ==="
            echo "Connecting to server: $SERVER_IP"
            ./tests/test_tcpx_transfer client $SERVER_IP
        fi
        ;;

    *)
        echo "❌ Unknown test type: $TEST_TYPE"
        echo "Supported test types:"
        echo "  connection  - Connection handshake test"
        echo "  transfer    - GPU-to-GPU transfer test"
        echo ""
        echo "Usage examples:"
        echo "  $0 connection server                    # Run on node 1 ($NODE1_IP)"
        echo "  $0 connection $NODE1_IP                 # Run on node 2 ($NODE2_IP)"
        echo "  $0 transfer server                      # Run on node 1 ($NODE1_IP)"
        echo "  $0 transfer $NODE1_IP                   # Run on node 2 ($NODE2_IP)"
        exit 1
        ;;
esac

echo ""
echo "=== Test complete ==="
