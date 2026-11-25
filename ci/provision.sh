#!/usr/bin/env bash
# One-time administrator setup. CI jobs themselves never invoke sudo or Docker.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run this provisioning script as root' >&2; exit 1; }
key=${1:?Pass the dedicated CI SSH public key}
[[ $key == ssh-ed25519\ * ]] || { echo 'Expected an Ed25519 public key' >&2; exit 1; }
if ! id uccl >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash uccl
fi
[[ $(id -u uccl) != 0 ]] || { echo 'uccl must not be root' >&2; exit 1; }
ci_home=$(getent passwd uccl | cut -d: -f6)
for group in video render; do
  if getent group "$group" >/dev/null; then usermod -aG "$group" uccl; fi
done
install -d -o uccl -g "$(id -gn uccl)" -m 700 "$ci_home/.ssh"
touch "$ci_home/.ssh/authorized_keys"
if ! grep -Fq -- "$key" "$ci_home/.ssh/authorized_keys"; then
  printf 'restrict %s\n' "$key" >> "$ci_home/.ssh/authorized_keys"
fi
chown uccl:"$(id -gn uccl)" "$ci_home/.ssh/authorized_keys"
chmod 600 "$ci_home/.ssh/authorized_keys"
install -d -o uccl -g "$(id -gn uccl)" -m 700 /var/lib/uccl-ci
case $(findmnt -T /var/lib/uccl-ci -no FSTYPE) in
  nfs*|cifs) echo 'CI requires local storage' >&2; exit 1;;
esac
install -d -o uccl -g "$(id -gn uccl)" -m 700 \
  /var/lib/uccl-ci/runs /var/lib/uccl-ci/cache
cat > /etc/security/limits.d/90-uccl-ci.conf <<'EOF'
uccl soft memlock unlimited
uccl hard memlock unlimited
EOF
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build pkg-config git rsync curl \
  python3-venv python3-dev libnuma-dev libgflags-dev libgtest-dev \
  libelf-dev zlib1g-dev libnl-3-dev libnl-route-3-dev libfabric-dev \
  libssl-dev libcurl4-openssl-dev \
  libibverbs-dev ibverbs-utils numactl
echo "Provisioned $(hostname): $(id uccl), local CI root /var/lib/uccl-ci"
