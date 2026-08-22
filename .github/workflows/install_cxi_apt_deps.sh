#!/usr/bin/env bash
# Install libfabric-dev and libhwloc-dev only when they are missing.
# Skips apt-get entirely when the packages or their headers are already
# present, so a full disk or a broken apt index does not fail CI.
#
# Optional env:
#   UCCL_SUDO_PASSWORD  If set, pipe it to `sudo -S`. Otherwise use `sudo -n`.

set -euo pipefail

PKGS=(libfabric-dev libhwloc-dev)

sudo_run() {
  if [ -n "${UCCL_SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "${UCCL_SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo -n "$@"
  fi
}

pkg_installed() {
  dpkg-query -W -f='${Status}\n' "$1" 2>/dev/null | grep -q '^install ok installed$'
}

headers_present() {
  [ -e /usr/include/rdma/fabric.h ] && [ -e /usr/include/hwloc.h ]
}

missing=()
for pkg in "${PKGS[@]}"; do
  if pkg_installed "$pkg"; then
    echo "${pkg} already installed"
  else
    missing+=("$pkg")
  fi
done

if [ "${#missing[@]}" -eq 0 ] || headers_present; then
  if [ "${#missing[@]}" -ne 0 ]; then
    echo "CXI headers already present; skipping apt-get for: ${missing[*]}"
  else
    echo "CXI apt deps already present; skipping apt-get"
  fi
  exit 0
fi

echo "Installing missing CXI apt deps: ${missing[*]}"
sudo_run apt-get update
sudo_run apt-get install -y "${missing[@]}"
