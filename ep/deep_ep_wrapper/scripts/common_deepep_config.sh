#!/bin/bash
# Generate DeepEP configuration file
# Usage: generate_deepep_config [output_path]
# If no path is provided, defaults to $(pwd)/deepep_config.json

generate_deepep_config() {
  local DEEPEP_CFG="${1:-$(pwd)/deepep_config.json}"
  
  cat > "$DEEPEP_CFG" <<'EOF'
{
  "normal_dispatch": {
    "num_sms": 24,
    "num_max_nvl_chunked_send_tokens": 16,
    "num_max_nvl_chunked_recv_tokens": 512,
    "num_max_rdma_chunked_send_tokens": 16,
    "num_max_rdma_chunked_recv_tokens": 512
  },
  "normal_combine": {
    "num_sms": 24,
    "num_max_nvl_chunked_send_tokens": 16,
    "num_max_nvl_chunked_recv_tokens": 512,
    "num_max_rdma_chunked_send_tokens": 16,
    "num_max_rdma_chunked_recv_tokens": 512
  }
}
EOF
  
  echo "$DEEPEP_CFG"
}
