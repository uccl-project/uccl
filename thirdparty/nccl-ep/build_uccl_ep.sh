#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec make lib NCCL_EP_USE_UCCL_GIN="${NCCL_EP_USE_UCCL_GIN:-ON}" "$@"
