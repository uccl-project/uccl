#!/usr/bin/env bash
# One-time administrator setup for the GB10 SSH jump host; no build tools needed.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run this provisioning script as root' >&2; exit 1; }
key=${1:?Pass the dedicated CI SSH public key}
[[ $key == ssh-ed25519\ * && $key != *$'\n'* ]] || { echo 'Expected one Ed25519 public key' >&2; exit 1; }
ssh-keygen -lf /dev/stdin <<< "$key" >/dev/null
read -r _ key_data _ <<< "$key"
if ! id uccl >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash uccl
fi
[[ $(id -u uccl) != 0 ]] || { echo 'uccl must not be root' >&2; exit 1; }
ci_home=$(getent passwd uccl | cut -d: -f6)
install -d -o uccl -g "$(id -gn uccl)" -m 700 "$ci_home/.ssh"
touch "$ci_home/.ssh/authorized_keys"
key_entry="restrict,port-forwarding,permitopen=\"localhost:57676\",permitopen=\"localhost:57677\",command=\"/bin/false\" $key"
if ! grep -Fxq -- "$key_entry" "$ci_home/.ssh/authorized_keys"; then
  if grep -Fq -- " $key_data" "$ci_home/.ssh/authorized_keys"; then
    echo 'CI key already exists with different restrictions; review authorized_keys' >&2
    exit 1
  fi
  printf '%s\n' "$key_entry" >> "$ci_home/.ssh/authorized_keys"
fi
chown uccl:"$(id -gn uccl)" "$ci_home/.ssh/authorized_keys"
chmod 600 "$ci_home/.ssh/authorized_keys"
echo "Provisioned GB10 jump account on $(hostname): $(id uccl)"
