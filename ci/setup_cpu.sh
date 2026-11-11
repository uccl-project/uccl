#!/usr/bin/env bash
# Native dependencies for the GitHub Ubuntu CPU runner.
set -euo pipefail

# GitHub's ubuntu-24.04 sources already use this APT mirror list. A lower
# priority number is preferred; keep the runner's default mirrors as fallbacks.
if ! grep -Fq 'mirror+file:/etc/apt/apt-mirrors.txt' /etc/apt/sources.list.d/ubuntu.sources; then
  echo '::error::Expected the GitHub Ubuntu runner APT mirror-list configuration.'
  exit 1
fi
mirror=$'https://mirrors.edge.kernel.org/ubuntu/\tpriority:0'
if ! grep -Fxq "$mirror" /etc/apt/apt-mirrors.txt; then
  printf '%s\n' "$mirror" | sudo tee -a /etc/apt/apt-mirrors.txt
fi

apt_options=(-o Acquire::Retries=1
             -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15
             -o DPkg::Lock::Timeout=30)

echo 'Refreshing CPU dependency package indexes'
sudo apt-get "${apt_options[@]}" update --error-on=any
echo 'Installing CPU dependencies'
sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y \
  build-essential python3-venv libgtest-dev libibverbs-dev \
  libgflags-dev zlib1g-dev libelf-dev
