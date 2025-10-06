#!/usr/bin/env bash
# Diagnose multi-NIC configuration for TCPX

echo "=== TCPX Multi-NIC Diagnostics ==="
echo ""

echo "1. Network Interfaces:"
ip addr show | grep -E "^[0-9]+: (eth|ib)" | awk '{print $2}'
echo ""

echo "2. Environment Variables:"
echo "NCCL_GPUDIRECTTCPX_SOCKET_IFNAME = ${NCCL_GPUDIRECTTCPX_SOCKET_IFNAME:-NOT SET}"
echo "NCCL_GPUDIRECTTCPX_CTRL_DEV = ${NCCL_GPUDIRECTTCPX_CTRL_DEV:-NOT SET}"
echo "NCCL_NSOCKS_PERTHREAD = ${NCCL_NSOCKS_PERTHREAD:-NOT SET}"
echo "NCCL_SOCKET_NTHREADS = ${NCCL_SOCKET_NTHREADS:-NOT SET}"
echo ""

echo "3. TCPX Plugin Library:"
if [ -f "/usr/local/tcpx/lib64/libnccl-net.so" ]; then
    echo "✅ Found: /usr/local/tcpx/lib64/libnccl-net.so"
    ls -lh /usr/local/tcpx/lib64/libnccl-net.so
else
    echo "❌ NOT FOUND: /usr/local/tcpx/lib64/libnccl-net.so"
fi
echo ""

echo "4. LD_LIBRARY_PATH:"
echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -E "tcpx|nccl"
echo ""

echo "5. Network Connectivity Test:"
for iface in eth1 eth2 eth3 eth4; do
    if ip addr show "$iface" &>/dev/null; then
        ip_addr=$(ip addr show "$iface" | grep "inet " | awk '{print $2}')
        echo "  $iface: $ip_addr"
    else
        echo "  $iface: NOT FOUND"
    fi
done
echo ""

echo "6. Recommendation:"
echo "  - Make sure NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4"
echo "  - Make sure NCCL_DEBUG=INFO to see TCPX logs"
echo "  - Check logs for 'NET_GPUDIRECTTCPX_SOCKET_IFNAME set to'"
echo ""

