#!/bin/sh
set -eu

BIN="${1:-./test_transport_integration}"
PORT_BASE="${TRANSPORT_TEST_PORT_BASE:-16979}"
HOST_ID_OVERRIDE="${UHM_HOST_ID_OVERRIDE:-}"

host_id() {
  if [ -n "$HOST_ID_OVERRIDE" ]; then
    echo "$HOST_ID_OVERRIDE"
    return
  fi
  if [ -r /etc/machine-id ]; then
    cat /etc/machine-id
    return
  fi
  hostname
}

cleanup_ipc_shm() {
  hid="$(host_id)"
  rm -f /dev/shm/uk_t_oob_"${hid}"_l* 2>/dev/null || true
}

run_one() {
  echo "== $*"
  "$@"
}

run_pair() {
  case_name="$1"
  port="$2"
  transport="${3:-auto}"
  server_host_id="${4:-}"
  client_host_id="${5:-}"
  server_log="$(mktemp)"
  client_log="$(mktemp)"
  server_pid=""

  cleanup() {
    if [ -n "$server_pid" ]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    rm -f "$server_log" "$client_log"
  }
  trap cleanup EXIT INT TERM

  cleanup_ipc_shm
  echo "== communicator case: ${case_name} (port ${port}, transport ${transport})"
  if [ -n "$server_host_id" ]; then
    UHM_HOST_ID_OVERRIDE="$server_host_id" \
      UHM_LOCAL_IP=127.0.0.1 \
      "$BIN" communicator --role=server --case="$case_name" \
      --exchanger-port "$port" --transport "$transport" \
      >"$server_log" 2>&1 &
  else
    "$BIN" communicator --role=server --case="$case_name" \
      --exchanger-port "$port" --transport "$transport" \
      >"$server_log" 2>&1 &
  fi
  server_pid="$!"
  sleep 1

  if [ -n "$client_host_id" ]; then
    if ! UHM_HOST_ID_OVERRIDE="$client_host_id" \
      UHM_LOCAL_IP=127.0.0.1 \
      "$BIN" communicator --role=client --case="$case_name" \
      --exchanger-ip 127.0.0.1 --exchanger-port "$port" \
      --transport "$transport" >"$client_log" 2>&1; then
      cat "$server_log"
      cat "$client_log"
      return 1
    fi
  elif ! "$BIN" communicator --role=client --case="$case_name" \
      --exchanger-ip 127.0.0.1 --exchanger-port "$port" \
      --transport "$transport" >"$client_log" 2>&1; then
    cat "$server_log"
    cat "$client_log"
    return 1
  fi
  if ! wait "$server_pid"; then
    cat "$server_log"
    cat "$client_log"
    return 1
  fi
  server_pid=""

  cat "$server_log"
  cat "$client_log"
  cleanup_ipc_shm
  cleanup
  trap - EXIT INT TERM
}

run_one "$BIN" communicator-local
run_pair ipc-buffer-meta "$((PORT_BASE + 20))" ipc
run_pair exchange "$PORT_BASE"
run_pair exchange "$((PORT_BASE + 1))" ipc

if [ "${TRANSPORT_RUN_UCCL:-0}" = "1" ]; then
  run_pair exchange "$((PORT_BASE + 10))" auto \
    transport-suite-server transport-suite-client
  run_pair exchange "$((PORT_BASE + 12))" uccl
fi

echo "transport suite completed"
