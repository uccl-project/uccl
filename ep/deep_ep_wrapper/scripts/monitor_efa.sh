#!/bin/bash
# EFA Interface Traffic Monitoring Script

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Format bytes into human-readable units
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

# Show one-time statistics
show_stats() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}EFA Interface Statistics - $(date)${NC}"
    echo -e "${BLUE}========================================${NC}"
    printf "%-15s %15s %15s %15s %15s %10s\n" \
        "Interface" "TX Bytes" "RX Bytes" "TX Packets" "RX Packets" "Drops"
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

# Continuous monitoring mode
monitor_mode() {
    local interval=$1
    while true; do
        clear
        show_stats
        echo -e "${YELLOW}Refreshing every ${interval} seconds. Press Ctrl+C to stop.${NC}"
        sleep $interval
    done
}

# Detailed statistics for a specific interface
detailed_stats() {
    local interface=$1
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Detailed Statistics for $interface${NC}"
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

# Bandwidth calculation mode
bandwidth_mode() {
    local interval=${1:-1}

    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Real-time Bandwidth Monitor (${interval}s sampling)${NC}"
    echo -e "${BLUE}========================================${NC}"

    declare -A prev_tx prev_rx
    for dev in $(ls /sys/class/infiniband/ | grep rdmap); do
        stats=$(rdma statistic show | grep "^link $dev")
        prev_tx[$dev]=$(echo $stats | grep -oP 'tx_bytes \K[0-9]+')
        prev_rx[$dev]=$(echo $stats | grep -oP 'rx_bytes \K[0-9]+')
    done

    sleep $interval

    echo ""
    printf "%-15s %20s %20s\n" "Interface" "TX Bandwidth" "RX Bandwidth"
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

# Help message
show_help() {
    echo "EFA Interface Traffic Monitor"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -s, --stats            Show current statistics (default)"
    echo "  -m, --monitor [int]    Continuous monitoring mode (default 5s)"
    echo "  -b, --bandwidth [int]  Real-time bandwidth monitoring (default 1s)"
    echo "  -d, --detail [iface]   Show detailed statistics for an interface"
    echo "  -l, --list             List all EFA interfaces"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -s"
    echo "  $0 -m 2"
    echo "  $0 -b 1"
    echo "  $0 -d rdmap85s0"
}

# List all EFA interfaces
list_interfaces() {
    echo -e "${GREEN}Available EFA interfaces:${NC}"
    ls /sys/class/infiniband/ | grep rdmap | nl
}

# Main logic
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
            echo "Error: interface name required"
            echo "Use $0 -l to list interfaces"
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
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
