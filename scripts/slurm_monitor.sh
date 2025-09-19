#!/bin/bash

# SLURM Cluster AMD GPU & RDMA Monitor
# Monitors AMD GPU idleness and RDMA interface status across all accessible nodes in the SLURM cluster
# Uses rocm-smi for GPU monitoring and ibstat for RDMA interface monitoring
# Updates every 15 seconds

set -euo pipefail

# Configuration
REFRESH_INTERVAL=60
SSH_TIMEOUT=1  # Aggressive timeout for faster responses
GPU_IDLE_THRESHOLD=5  # GPU utilization below this percentage is considered idle
SSH_OPTIONS="-o ConnectTimeout=$SSH_TIMEOUT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes"
MAX_CONCURRENT_SSH=20  # Maximum number of concurrent SSH connections

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to get all SLURM nodes
get_slurm_nodes() {
    sinfo -N -h -o "%N" | sort -u
}

# Function to check if a node is accessible via SSH
check_ssh_connectivity() {
    local node=$1
    ssh $SSH_OPTIONS "$node" "exit" 2>/dev/null
    return $?
}

# Function to get RDMA status from a node
get_rdma_status() {
    local node=$1
    local active_rdma=0
    
    # Check rdma0 through rdma7 for active state
    for i in {0..7}; do
        rdma_state=$(ssh $SSH_OPTIONS "$node" "ibstat rdma$i 2>/dev/null | grep 'State:' | awk '{print \$NF}'" 2>/dev/null || echo "")
        if [[ "$rdma_state" == "Active" ]]; then
            active_rdma=$((active_rdma + 1))
        fi
    done
    
    echo "$active_rdma"
}

# Function to get GPU status from a node
get_gpu_status() {
    local node=$1
    local result=""
    
    # Try to get GPU info via SSH using rocm-smi
    if ssh $SSH_OPTIONS "$node" "command -v rocm-smi >/dev/null 2>&1" 2>/dev/null; then
        # Node has rocm-smi, get GPU utilization using the exact command provided
        total_utilization=$(ssh $SSH_OPTIONS "$node" "rocm-smi -u 2>/dev/null | awk '/GPU use/ {sum += \$NF} END {print sum+0}'" 2>/dev/null || echo "0")
        
        # Get individual GPU details - try multiple patterns to catch different rocm-smi output formats
        gpu_raw_output=$(ssh $SSH_OPTIONS "$node" "rocm-smi -u 2>/dev/null" 2>/dev/null || echo "")
        
        if [[ -n "$gpu_raw_output" ]]; then
            total_gpus=0
            idle_gpus=0
            
            # Parse GPU utilization - handle various rocm-smi output formats
            while IFS= read -r line; do
                # Skip empty lines
                [[ -z "$line" ]] && continue
                
                utilization=""
                
                # Try to extract any percentage from lines containing "use" or "util"
                if [[ "$line" =~ (use|util|GPU) ]] && [[ "$line" =~ ([0-9]+)% ]]; then
                    potential_util="${BASH_REMATCH[2]}"
                    
                    # Additional validation - make sure this is a GPU utilization line
                    if [[ "$line" =~ (GPU|use|util) ]]; then
                        utilization="$potential_util"
                    fi
                fi
                
                if [[ -n "$utilization" ]]; then
                    total_gpus=$((total_gpus + 1))
                    
                    if [[ "$utilization" -lt "$GPU_IDLE_THRESHOLD" ]]; then
                        idle_gpus=$((idle_gpus + 1))
                    fi
                fi
            done <<< "$gpu_raw_output"
            
            # If we didn't parse any GPUs but got total utilization, try a different approach
            if [[ $total_gpus -eq 0 ]]; then
                # Use awk to count and parse GPU use lines directly
                gpu_use_lines=$(ssh $SSH_OPTIONS "$node" "rocm-smi -u 2>/dev/null | awk '/GPU use/ {print \$NF}' | sed 's/%//'" 2>/dev/null || echo "")
                
                if [[ -n "$gpu_use_lines" ]]; then
                    while read -r util_value; do
                        if [[ "$util_value" =~ ^[0-9]+$ ]]; then
                            total_gpus=$((total_gpus + 1))
                            
                            if [[ "$util_value" -lt "$GPU_IDLE_THRESHOLD" ]]; then
                                idle_gpus=$((idle_gpus + 1))
                            fi
                        fi
                    done <<< "$gpu_use_lines"
                fi
                
                # Final fallback: count GPUs from device list if still no luck
                if [[ $total_gpus -eq 0 ]] && [[ "$total_utilization" != "0" ]]; then
                    gpu_count=$(ssh $SSH_OPTIONS "$node" "rocm-smi -i 2>/dev/null | grep -c 'GPU\\[' || echo 0" 2>/dev/null)
                    if [[ "$gpu_count" -gt 0 ]]; then
                        total_gpus=$gpu_count
                        # Estimate if all are idle based on low total utilization
                        if [[ $total_gpus -gt 0 ]]; then
                            avg_util=$((total_utilization / total_gpus))
                            if [[ $avg_util -lt $GPU_IDLE_THRESHOLD ]]; then
                                idle_gpus=$total_gpus
                            else
                                idle_gpus=0
                            fi
                            # GPU count and idle count already set above
                        fi
                    fi
                fi
            fi
            
            if [[ $total_gpus -gt 0 ]]; then
                if [[ $total_gpus -eq $idle_gpus ]]; then
                    server_status="${GREEN}IDLE${NC}"
                else
                    server_status="${RED}BUSY${NC}"
                fi
                result="$total_gpus GPUs, $idle_gpus idle (Total: ${total_utilization}%) - $server_status"
            else
                result="${YELLOW}GPUs detected but parsing failed (Total: ${total_utilization}%)${NC}"
            fi
        else
            result="${YELLOW}rocm-smi command failed${NC}"
        fi
    else
        # Check if node has AMD GPUs but no rocm-smi
        lspci_output=$(ssh $SSH_OPTIONS "$node" "lspci | grep -i amd" 2>/dev/null || echo "")
        if [[ -n "$lspci_output" ]]; then
            result="${YELLOW}Has AMD GPUs but rocm-smi not available${NC}"
        else
            result="${BLUE}No AMD GPUs detected${NC}"
        fi
    fi
    
    echo -e "$result"
}

# Function to check a single node and save results to temp files
check_node() {
    local node=$1
    local temp_dir=$2
    local result_file="${temp_dir}/${node}.result"
    local status_file="${temp_dir}/${node}.status"
    
    # Check SSH connectivity and GPU status
    if check_ssh_connectivity "$node"; then
        echo "SSH_OK" > "$status_file"
        gpu_status=$(get_gpu_status "$node")
        rdma_count=$(get_rdma_status "$node")
        echo -e "$gpu_status | RDMA: $rdma_count active" > "$result_file"
        
        # Determine if server is idle (extract from gpu_status)
        if [[ "$gpu_status" == *"- IDLE"* ]]; then
            echo "IDLE" >> "$status_file"
        elif [[ "$gpu_status" == *"- BUSY"* ]]; then
            echo "BUSY" >> "$status_file"
        elif [[ "$gpu_status" == *"GPUs detected"* ]] || [[ "$gpu_status" == *"rocm-smi not available"* ]] || [[ "$gpu_status" == *"rocm-smi command failed"* ]]; then
            echo "GPU_ERROR" >> "$status_file"  # Has GPUs but can't get status
        elif [[ "$gpu_status" == *"No AMD GPUs detected"* ]]; then
            echo "NO_GPU" >> "$status_file"    # Actually no GPUs
        else
            echo "UNKNOWN" >> "$status_file"   # Fallback
        fi
    else
        echo "SSH_FAILED" > "$status_file"
        echo -e "${RED}Cannot connect via SSH${NC}" > "$result_file"
    fi
}

# Function to display results from temp files
display_node_result() {
    local node=$1
    local temp_dir=$2
    local result_file="${temp_dir}/${node}.result"
    local status_file="${temp_dir}/${node}.status"
    
    printf "%-20s " "$node"
    
    if [[ -f "$status_file" ]]; then
        local ssh_status=$(head -n1 "$status_file")
        if [[ "$ssh_status" == "SSH_OK" ]]; then
            printf "${GREEN}[SSH OK]${NC} "
        else
            printf "${RED}[SSH FAILED]${NC} "
        fi
    else
        printf "${YELLOW}[TIMEOUT]${NC} "
    fi
    
    if [[ -f "$result_file" ]]; then
        cat "$result_file"
    else
        echo -e "${YELLOW}No result available${NC}"
    fi
}

# Function to display header
display_header() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                           SLURM CLUSTER GPU MONITOR${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}Last update: $(date)${NC}"
    echo -e "${PURPLE}Refresh interval: ${REFRESH_INTERVAL}s | GPU idle threshold: ${GPU_IDLE_THRESHOLD}%${NC}"
    echo ""
    printf "%-20s %-15s %s\n" "NODE" "STATUS" "GPU & RDMA DETAILS"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"
}

# Function to display summary with server lists
display_summary() {
    local nodes=("$@")
    local temp_dir="${nodes[-1]}"  # Last argument is temp_dir
    unset 'nodes[-1]'              # Remove temp_dir from nodes array
    
    local total_nodes=${#nodes[@]}
    local accessible_nodes=0
    local idle_servers=0
    local busy_servers=0
    local no_gpu_servers=0
    local gpu_error_servers=0
    local unknown_servers=0
    local ssh_failed=0
    
    # Arrays to store server names
    local idle_server_names=()
    local busy_server_names=()
    local no_gpu_server_names=()
    local gpu_error_server_names=()
    local unknown_server_names=()
    local failed_server_names=()
    local eight_idle_gpu_servers=()
    
    echo ""
    echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"
    
    # Count accessible nodes and their status from temp files
    for node in "${nodes[@]}"; do
        local status_file="${temp_dir}/${node}.status"
        
        if [[ -f "$status_file" ]]; then
            local ssh_status=$(head -n1 "$status_file")
            if [[ "$ssh_status" == "SSH_OK" ]]; then
                accessible_nodes=$((accessible_nodes + 1))
                
                # Check server status from second line
                if [[ $(wc -l < "$status_file") -gt 1 ]]; then
                    local server_status=$(tail -n1 "$status_file")
                    
                    # Check if this server has exactly 8 idle GPUs
                    local result_file="${temp_dir}/${node}.result"
                    if [[ -f "$result_file" ]]; then
                        local result_content=$(cat "$result_file")
                        if [[ "$result_content" =~ 8\ GPUs,\ 8\ idle ]]; then
                            # Extract RDMA count from result
                            rdma_info=""
                            if [[ "$result_content" =~ RDMA:\ ([0-9]+)\ active ]]; then
                                rdma_count="${BASH_REMATCH[1]}"
                                rdma_info=" (RDMA: $rdma_count)"
                            fi
                            eight_idle_gpu_servers+=("$node$rdma_info")
                        fi
                    fi
                    
                    case "$server_status" in
                        "IDLE")
                            idle_servers=$((idle_servers + 1))
                            idle_server_names+=("$node")
                            ;;
                        "BUSY")
                            busy_servers=$((busy_servers + 1))
                            busy_server_names+=("$node")
                            ;;
                        "GPU_ERROR")
                            gpu_error_servers=$((gpu_error_servers + 1))
                            gpu_error_server_names+=("$node")
                            ;;
                        "NO_GPU")
                            no_gpu_servers=$((no_gpu_servers + 1))
                            no_gpu_server_names+=("$node")
                            ;;
                        "UNKNOWN"|*)
                            unknown_servers=$((unknown_servers + 1))
                            unknown_server_names+=("$node")
                            ;;
                    esac
                else
                    unknown_servers=$((unknown_servers + 1))
                    unknown_server_names+=("$node")
                fi
            else
                ssh_failed=$((ssh_failed + 1))
                failed_server_names+=("$node")
            fi
        else
            ssh_failed=$((ssh_failed + 1))
            failed_server_names+=("$node")
        fi
    done
    

    
    # Display servers with exactly 8 idle GPUs
    echo ""
    if [[ ${#eight_idle_gpu_servers[@]} -gt 0 ]]; then
        echo -e "${GREEN}🎯 SERVERS WITH 8 IDLE GPUs:${NC} ${eight_idle_gpu_servers[*]}"
        echo -e "${GREEN}   Count: ${#eight_idle_gpu_servers[@]} servers${NC}"
    else
        echo -e "${YELLOW}⚠️  No servers currently have all 8 GPUs idle${NC}"
    fi
}

# Main monitoring loop
main() {
    echo -e "${CYAN}Starting SLURM Cluster GPU Monitor...${NC}"
    echo "Getting list of SLURM nodes..."
    
    # Get all SLURM nodes
    mapfile -t nodes < <(get_slurm_nodes)
    
    if [[ ${#nodes[@]} -eq 0 ]]; then
        echo -e "${RED}Error: No SLURM nodes found. Make sure SLURM is properly configured.${NC}"
        exit 1
    fi
    
    echo "Found ${#nodes[@]} nodes in the cluster."
    echo "Starting monitoring (Press Ctrl+C to stop)..."
    
    # Cleanup function
    cleanup() {
        echo -e "\n${YELLOW}Stopping monitoring...${NC}"
        # Kill any remaining background jobs
        jobs -p | xargs -r kill 2>/dev/null
        # Clean up any remaining temp directories
        find /tmp -name "tmp.*" -user "$(whoami)" -path "*/tmp.*" -type d -exec rm -rf {} + 2>/dev/null || true
        echo -e "${YELLOW}Monitoring stopped.${NC}"
        exit 0
    }
    
    # Trap Ctrl+C to exit gracefully
    trap cleanup INT TERM
    
    while true; do
        # Create temporary directory for this iteration
        local temp_dir=$(mktemp -d)
        local pids=()
        local active_jobs=0
        
        # Start concurrent checks for all nodes
        for node in "${nodes[@]}"; do
            # Limit concurrent jobs
            while [[ $active_jobs -ge $MAX_CONCURRENT_SSH ]]; do
                wait -n  # Wait for any job to complete
                active_jobs=$((active_jobs - 1))
            done
            
            # Start background job for this node
            check_node "$node" "$temp_dir" &
            pids+=($!)
            active_jobs=$((active_jobs + 1))
        done
        
        # Wait for all background jobs to complete with timeout
        local timeout_count=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid" 2>/dev/null; then
                timeout_count=$((timeout_count + 1))
            fi
        done
        
        # Clear screen only when all data is ready to display
        clear
        
        # Display header
        display_header
        
        # Display results from temp files
        for node in "${nodes[@]}"; do
            display_node_result "$node" "$temp_dir"
        done
        
        # Display summary (pass temp_dir as last argument)
        display_summary "${nodes[@]}" "$temp_dir"
        
        # Clean up temporary files
        rm -rf "$temp_dir"
        
        echo ""
        echo -e "${YELLOW}Next update in ${REFRESH_INTERVAL} seconds... (Press Ctrl+C to stop)${NC}"
        if [[ $timeout_count -gt 0 ]]; then
            echo -e "${YELLOW}Note: ${timeout_count} node(s) timed out${NC}"
        fi
        
        # Wait for next update
        sleep "$REFRESH_INTERVAL"
    done
}

# Check if SLURM commands are available
if ! command -v sinfo >/dev/null 2>&1; then
    echo -e "${RED}Error: SLURM commands not found. Make sure SLURM is installed and configured.${NC}"
    exit 1
fi

# Run the main function
main "$@"
