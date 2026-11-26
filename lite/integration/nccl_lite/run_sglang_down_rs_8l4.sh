#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export UCCL_LITE_SGLANG_RS_SUM2=1
export UCCL_LITE_SGLANG_READY_COMBINE=1
export UCCL_LITE_SGLANG_READY_RS=1
export UCCL_LITE_SGLANG_DOWN_COUNTER=1
export UCCL_LITE_SGLANG_DOWN_OVERLAP=1
export PYTHONPATH="$here/sglang_hooks${PYTHONPATH:+:$PYTHONPATH}"
exec "$here/run_packet_8l4.sh" "$@"
