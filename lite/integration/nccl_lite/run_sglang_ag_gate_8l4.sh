#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export UCCL_LITE_SGLANG_AG_GATE=1
export UCCL_LITE_AG_GEMM_BACKEND=${UCCL_LITE_AG_GEMM_BACKEND:-cublas}
export PYTHONPATH="$here/sglang_hooks${PYTHONPATH:+:$PYTHONPATH}"
exec "$here/run_packet_8l4.sh" "$@"
