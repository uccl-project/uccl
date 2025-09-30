#!/bin/bash

# TCPX P2P 测试脚本
# 基于成功的 collective/rdma/run_nccl_test_tcpx.sh 环境配置

source ../../scripts/shared.sh

# Usage: ./run_tcpx_test.sh [test_type] [server_ip]
# test_type: device|connection|performance
# server_ip: required for connection and performance tests

TEST_TYPE=${1:-device}
SERVER_IP=${2:-""}
HOSTFILE="${UCCL_HOME}/scripts/node_ips/tcpx.txt"

# 从hostfile读取节点IP
if [ ! -f "$HOSTFILE" ]; then
    echo "❌ 主机文件不存在: $HOSTFILE"
    echo "请确保设置了正确的 UCCL_HOME 环境变量"
    exit 1
fi

NODE1_IP=$(sed -n '1p' "$HOSTFILE")
NODE2_IP=$(sed -n '2p' "$HOSTFILE")

echo "=== TCPX P2P 测试脚本 ==="
echo "测试类型: $TEST_TYPE"
echo "主机文件: $HOSTFILE"
echo "节点1 (服务器): $NODE1_IP"
echo "节点2 (客户端): $NODE2_IP"

# 设置环境变量 - 基于成功的NCCL TCPX配置
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/tcpx/lib64:$UCCL_HOME/thirdparty/nccl/build/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:$LD_LIBRARY_PATH"

# TCPX 特定环境变量
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

# 调试相关
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

# CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo ""
echo "环境变量已设置:"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "  NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=$NCCL_GPUDIRECTTCPX_SOCKET_IFNAME"
echo "  NCCL_GPUDIRECTTCPX_CTRL_DEV=$NCCL_GPUDIRECTTCPX_CTRL_DEV"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# 编译测试程序
echo ""
echo "=== 编译测试程序 ==="
cd "$(dirname "$0")"

case $TEST_TYPE in
    "device")
        echo "编译设备发现测试..."
        if ! make test_device_discovery; then
            echo "❌ 编译失败"
            exit 1
        fi
        echo "✅ 编译完成"
        echo ""
        echo "=== 运行设备发现测试 ==="
        ./tests/test_device_discovery
        ;;
    
    "connection")
        if [ -z "$SERVER_IP" ]; then
            echo "❌ 连接测试需要指定服务器IP"
            echo "用法: $0 connection <server_ip>"
            echo "或者在两个节点分别运行:"
            echo "  节点1 ($NODE1_IP): $0 connection server"
            echo "  节点2 ($NODE2_IP): $0 connection $NODE1_IP"
            exit 1
        fi
        
        echo "编译连接测试..."
        if ! make test_connection; then
            echo "❌ 编译失败"
            exit 1
        fi
        echo "✅ 编译完成"
        echo ""
        
        if [ "$SERVER_IP" = "server" ]; then
            echo "=== 运行连接测试 (服务器模式) ==="
            ./tests/test_connection server
        else
            echo "=== 运行连接测试 (客户端模式) ==="
            echo "连接到服务器: $SERVER_IP"
            ./tests/test_connection client $SERVER_IP
        fi
        ;;
    
    "transfer")
        if [ -z "$SERVER_IP" ]; then
            echo "❌ 传输测试需要指定服务器IP"
            echo "用法: $0 transfer <server_ip>"
            echo "或者在两个节点分别运行:"
            echo "  节点1 ($NODE1_IP): $0 transfer server"
            echo "  节点2 ($NODE2_IP): $0 transfer $NODE1_IP"
            exit 1
        fi

        echo "编译 GPU 传输测试..."
        if ! make test_tcpx_transfer; then
            echo "❌ 编译失败"
            exit 1
        fi
        echo "✅ 编译完成"
        echo ""

        if [ "$SERVER_IP" = "server" ]; then
            echo "=== 运行 GPU 传输测试 (服务器模式) ==="
            ./tests/test_tcpx_transfer server
        else
            echo "=== 运行 GPU 传输测试 (客户端模式) ==="
            echo "连接到服务器: $SERVER_IP"
            ./tests/test_tcpx_transfer client $SERVER_IP
        fi
        ;;

    "performance")
        if [ -z "$SERVER_IP" ]; then
            echo "❌ 性能测试需要指定服务器IP"
            echo "用法: $0 performance <server_ip>"
            echo "或者在两个节点分别运行:"
            echo "  节点1 ($NODE1_IP): $0 performance server"
            echo "  节点2 ($NODE2_IP): $0 performance $NODE1_IP"
            exit 1
        fi
        
        echo "编译性能测试..."
        if ! make test_performance; then
            echo "❌ 编译失败"
            exit 1
        fi
        echo "✅ 编译完成"
        echo ""
        
        if [ "$SERVER_IP" = "server" ]; then
            echo "=== 运行性能测试 (服务器模式) ==="
            ./tests/test_performance server
        else
            echo "=== 运行性能测试 (客户端模式) ==="
            echo "连接到服务器: $SERVER_IP"
            ./tests/test_performance client $SERVER_IP
        fi
        ;;
    
    *)
        echo "❌ 未知测试类型: $TEST_TYPE"
        echo "支持的测试类型:"
        echo "  device      - 设备发现测试"
        echo "  connection  - 连接测试"
        echo "  performance - 性能测试"
        echo ""
        echo "用法示例:"
        echo "  $0 device"
        echo "  $0 connection server                    # 在节点1 ($NODE1_IP) 运行"
        echo "  $0 connection $NODE1_IP                 # 在节点2 ($NODE2_IP) 运行"
        echo "  $0 performance server                   # 在节点1 ($NODE1_IP) 运行"
        echo "  $0 performance $NODE1_IP                # 在节点2 ($NODE2_IP) 运行"
        exit 1
        ;;
esac

echo ""
echo "=== 测试完成 ==="
