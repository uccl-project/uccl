#!/usr/bin/env bash
# Fix Docker default-bridge egress when the host has multiple IPv4 subnets /
# default routes on one NIC: forwarded traffic from 172.17.0.0/16 can pick a
# broken next-hop, while locally originated traffic uses a working gateway.
#
# Symptom: container DNS fails (8.8.8.8 unreachable), pip cannot reach PyPI,
# but the same host resolves fine; docker run --network=host works.
#
# Usage:
#   sudo ./scripts/net/docker_bridge_policy_route.sh apply
#   sudo ./scripts/net/docker_bridge_policy_route.sh status
#   sudo ./scripts/net/docker_bridge_policy_route.sh revert
#
# Persistence (pick one):
#   - systemd: copy the ExecStart line into a oneshot service After=network-online.target
#   - Add to /etc/rc.local or a NetworkManager dispatcher if you use that
#
# Override detection:
#   TABLE=1000 PREF=100 BRIDGE_SUBNET=172.17.0.0/16 \
#   GATEWAY=38.123.21.1 EGRESS_DEV=eno8303 SNAT_IP=38.123.21.7 \
#   sudo ./scripts/net/docker_bridge_policy_route.sh apply

set -euo pipefail

TABLE="${TABLE:-1000}"
PREF="${PREF:-100}"

die() {
  echo "docker_bridge_policy_route: $*" >&2
  exit 1
}

detect_working_route() {
  local probe="${1:-8.8.8.8}"
  local line
  line="$(ip -4 route get "$probe" 2>/dev/null | head -1)" || line=""
  if [[ -z "$line" || "$line" == *"unreachable"* ]]; then
    line="$(ip -4 route get 1.1.1.1 2>/dev/null | head -1)" || line=""
  fi
  [[ -n "$line" ]] || die "cannot resolve a working IPv4 default path (try: ping -c1 8.8.8.8)"

  # Example: 8.8.8.8 via 38.123.21.1 dev eno8303 src 38.123.21.7 uid 1001
  GATEWAY="$(sed -n 's/.* via \([^ ]*\).*/\1/p' <<<"$line" | head -1)"
  EGRESS_DEV="$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$line" | head -1)"
  SNAT_IP="$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$line" | head -1)"

  [[ -n "$GATEWAY" && -n "$EGRESS_DEV" ]] ||
    die "parsed route is missing via/dev: $line"

  if [[ -z "$SNAT_IP" ]]; then
    SNAT_IP="$(ip -4 addr show dev "$EGRESS_DEV" | awk '/inet / {print $2; exit}' | cut -d/ -f1)"
  fi
  [[ -n "$SNAT_IP" ]] || die "cannot determine SNAT IP on $EGRESS_DEV"
}

bridge_subnet() {
  if [[ -n "${BRIDGE_SUBNET:-}" ]]; then
    echo "$BRIDGE_SUBNET"
    return
  fi
  if command -v docker &>/dev/null; then
    local s
    s="$(docker network inspect bridge -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}' 2>/dev/null || true)"
    if [[ -n "$s" ]]; then
      echo "$s"
      return
    fi
  fi
  echo "172.17.0.0/16"
}

apply() {
  [[ "$(id -u)" -eq 0 ]] || die "run as root (sudo)"

  detect_working_route
  local br
  br="$(bridge_subnet)"

  echo "Using bridge subnet: $br"
  echo "Policy table: $TABLE, priority: $PREF"
  echo "Egress: dev=$EGRESS_DEV gateway=$GATEWAY preferred_src=$SNAT_IP"

  ip route replace default via "$GATEWAY" dev "$EGRESS_DEV" src "$SNAT_IP" table "$TABLE"

  # Replace duplicate rule if re-running
  while ip rule del from "$br" lookup "$TABLE" 2>/dev/null; do :; done
  ip rule add pref "$PREF" from "$br" lookup "$TABLE"

  echo "Applied. Run: $0 status"
}

revert() {
  [[ "$(id -u)" -eq 0 ]] || die "run as root (sudo)"
  local br
  br="$(bridge_subnet)"
  while ip rule del from "$br" lookup "$TABLE" 2>/dev/null; do :; done
  ip route flush table "$TABLE" 2>/dev/null || true
  echo "Reverted policy routing for $br (table $TABLE)."
}

status() {
  echo "=== ip rules (first lines + table $TABLE) ==="
  ip rule list | grep "lookup $TABLE" || true
  ip rule list | head -6
  echo "=== table $TABLE ==="
  ip route show table "$TABLE" || true
  echo "=== probe: forwarded-route simulation (docker0 / container-ish src) ==="
  local h="${DOCKER_TEST_SRC:-172.17.0.2}"
  ip route get 8.8.8.8 from "$h" iif docker0 2>/dev/null || true
  if command -v docker &>/dev/null && docker info &>/dev/null; then
    echo "=== docker DNS quick test ==="
    docker run --rm alpine:3.19 sh -c \
      'wget -q -O- --timeout=4 https://pypi.org/simple/pip/ | head -c 80 && echo ... || echo FAIL'
  else
    echo "(docker not available, skipping container test)"
  fi
}

case "${1:-}" in
apply) apply ;;
revert) revert ;;
status) status ;;
*) die "usage: sudo $0 {apply|revert|status}" ;;
esac
