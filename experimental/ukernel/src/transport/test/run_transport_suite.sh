#!/bin/sh
set -eu

BIN="${1:-./test_transport_main}"
PORT_BASE="${TRANSPORT_TEST_PORT_BASE:-16979}"

run_one() {
  echo "== $*"
  "$@"
}

run_pair() {
  case_name="$1"
  port="$2"
  transport="${3:-auto}"
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

  echo "== communicator case: ${case_name} (port ${port}, transport ${transport})"
  "$BIN" communicator --role=server --case="$case_name" --exchanger-port "$port" \
    --transport "$transport" \
    >"$server_log" 2>&1 &
  server_pid="$!"
  sleep 1

  if ! "$BIN" communicator --role=client --case="$case_name" \
    --exchanger-ip 127.0.0.1 --exchanger-port "$port" --transport "$transport" \
    >"$client_log" 2>&1; then
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
  cleanup
  trap - EXIT INT TERM
}

run_uccl_pair() {
  case_name="$1"
  port="$2"
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

  echo "== communicator UCCL case: ${case_name} (port ${port})"
  UHM_HOST_ID_OVERRIDE=transport-suite-server \
    UHM_LOCAL_IP=127.0.0.1 \
    "$BIN" communicator --role=server --case="$case_name" --exchanger-port "$port" \
    >"$server_log" 2>&1 &
  server_pid="$!"
  sleep 1

  if ! UHM_HOST_ID_OVERRIDE=transport-suite-client \
    UHM_LOCAL_IP=127.0.0.1 \
    "$BIN" communicator --role=client --case="$case_name" \
    --exchanger-ip 127.0.0.1 --exchanger-port "$port" \
    >"$client_log" 2>&1; then
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
  cleanup
  trap - EXIT INT TERM
}

run_one "$BIN" core
run_one "$BIN" communicator-local
run_one "$BIN" oob-socket
run_one "$BIN" oob-socket-meta --world-size 4
run_one "$BIN" oob-shm
run_one "$BIN" utils-host-id

run_pair basic "$PORT_BASE"
run_pair batch "$((PORT_BASE + 1))"
run_pair poll-release "$((PORT_BASE + 2))"
run_pair notifier "$((PORT_BASE + 3))"
run_pair basic "$((PORT_BASE + 4))" ipc
run_pair batch "$((PORT_BASE + 5))" ipc
run_pair poll-release "$((PORT_BASE + 6))" ipc

if [ "${TRANSPORT_RUN_UCCL:-0}" = "1" ]; then
  run_uccl_pair basic "$((PORT_BASE + 10))"
  run_uccl_pair poll-release "$((PORT_BASE + 11))"
  run_pair basic "$((PORT_BASE + 12))" uccl
fi

if [ "${TRANSPORT_RUN_REDIS:-0}" = "1" ]; then
  run_one "$BIN" oob-redis
  run_one "$BIN" oob-redis-meta --world-size 4
fi

echo "transport suite completed"
