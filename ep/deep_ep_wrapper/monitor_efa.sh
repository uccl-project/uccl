#!/bin/bash
# EFA接口流量监控脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 格式化字节数为人类可读格式
format_bytes() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt 1048576 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1024}")KB"
    elif [ $bytes -lt 1073741824 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1048576}")MB"
    else
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1073741824}")GB"
    fi
}

# 显示单次统计
show_stats() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}EFA接口流量统计 - $(date)${NC}"
    echo -e "${BLUE}========================================${NC}"
    printf "%-15s %15s %15s %15s %15s %10s\n" "接口" "发送字节" "接收字节" "发送包数" "接收包数" "丢包"
    echo "--------------------------------------------------------------------------------------------------------"

    rdma statistic show | grep "^link" | while read line; do
        interface=$(echo $line | awk '{print $2}' | cut -d'/' -f1)
        tx_bytes=$(echo $line | grep -oP 'tx_bytes \K[0-9]+')
        rx_bytes=$(echo $line | grep -oP 'rx_bytes \K[0-9]+')
        tx_pkts=$(echo $line | grep -oP 'tx_pkts \K[0-9]+')
        rx_pkts=$(echo $line | grep -oP 'rx_pkts \K[0-9]+')
        rx_drops=$(echo $line | grep -oP 'rx_drops \K[0-9]+')

        tx_human=$(format_bytes $tx_bytes)
        rx_human=$(format_bytes $rx_bytes)

        if [ $rx_drops -gt 0 ]; then
            drops_color="${RED}"
        else
            drops_color="${NC}"
        fi

        printf "%-15s %15s %15s %15s %15s ${drops_color}%10s${NC}\n" \
            "$interface" "$tx_human" "$rx_human" "$tx_pkts" "$rx_pkts" "$rx_drops"
    done
    echo ""
}

# 持续监控模式
monitor_mode() {
    local interval=$1
    while true; do
        clear
        show_stats
        echo -e "${YELLOW}每 ${interval} 秒更新一次，按 Ctrl+C 停止${NC}"
        sleep $interval
    done
}

# 详细统计
detailed_stats() {
    local interface=$1
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}$interface 详细统计${NC}"
    echo -e "${BLUE}========================================${NC}"

    rdma statistic show | grep "^link $interface" | tr ' ' '\n' | grep -v "^$" | grep -v "^link" | while read line; do
        if [[ $line == *"_"* ]]; then
            key=$(echo $line | cut -d' ' -f1)
            value=$(echo $line | awk '{print $1}' | grep -oP '\d+$')
            printf "%-30s: %s\n" "$key" "$value"
        fi
    done
    echo ""
}

# 带宽计算模式
bandwidth_mode() {
    local interval=${1:-1}

    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}实时带宽监控 (${interval}秒采样间隔)${NC}"
    echo -e "${BLUE}========================================${NC}"

    # 获取初始值
    declare -A prev_tx prev_rx
    for dev in $(ls /sys/class/infiniband/ | grep rdmap); do
        stats=$(rdma statistic show | grep "^link $dev")
        prev_tx[$dev]=$(echo $stats | grep -oP 'tx_bytes \K[0-9]+')
        prev_rx[$dev]=$(echo $stats | grep -oP 'rx_bytes \K[0-9]+')
    done

    sleep $interval

    echo ""
    printf "%-15s %20s %20s\n" "接口" "发送带宽" "接收带宽"
    echo "--------------------------------------------------------"

    for dev in $(ls /sys/class/infiniband/ | grep rdmap); do
        stats=$(rdma statistic show | grep "^link $dev")
        curr_tx=$(echo $stats | grep -oP 'tx_bytes \K[0-9]+')
        curr_rx=$(echo $stats | grep -oP 'rx_bytes \K[0-9]+')

        tx_diff=$((curr_tx - prev_tx[$dev]))
        rx_diff=$((curr_rx - prev_rx[$dev]))

        tx_rate=$(awk "BEGIN {printf \"%.2f\", $tx_diff*8/$interval/1000000000}")
        rx_rate=$(awk "BEGIN {printf \"%.2f\", $rx_diff*8/$interval/1000000000}")

        printf "%-15s %17s Gbps %17s Gbps\n" "$dev" "$tx_rate" "$rx_rate"
    done
    echo ""
}

# 显示帮助
show_help() {
    echo "EFA接口流量监控工具"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -s, --stats            显示当前统计信息（默认）"
    echo "  -m, --monitor [间隔]   持续监控模式（默认间隔5秒）"
    echo "  -b, --bandwidth [间隔] 实时带宽监控（默认间隔1秒）"
    echo "  -d, --detail [接口]    显示特定接口的详细统计"
    echo "  -l, --list             列出所有EFA接口"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -s                  # 显示一次统计信息"
    echo "  $0 -m 2                # 每2秒更新一次统计"
    echo "  $0 -b 1                # 显示实时带宽（1秒采样）"
    echo "  $0 -d rdmap85s0        # 显示rdmap85s0的详细信息"
}

# 列出所有EFA接口
list_interfaces() {
    echo -e "${GREEN}可用的EFA接口:${NC}"
    ls /sys/class/infiniband/ | grep rdmap | nl
}

# 主程序
case "$1" in
    -s|--stats)
        show_stats
        ;;
    -m|--monitor)
        interval=${2:-5}
        monitor_mode $interval
        ;;
    -b|--bandwidth)
        interval=${2:-1}
        bandwidth_mode $interval
        ;;
    -d|--detail)
        if [ -z "$2" ]; then
            echo "错误: 请指定接口名称"
            echo "使用 $0 -l 查看所有接口"
            exit 1
        fi
        detailed_stats "$2"
        ;;
    -l|--list)
        list_interfaces
        ;;
    -h|--help)
        show_help
        ;;
    "")
        show_stats
        ;;
    *)
        echo "未知选项: $1"
        show_help
        exit 1
        ;;
esac
