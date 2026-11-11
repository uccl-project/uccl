#!/usr/bin/env bash
# Native dependencies for the GitHub Ubuntu CPU runner.
set -euo pipefail

packages=(build-essential python3-venv libgtest-dev libibverbs-dev
          libgflags-dev zlib1g-dev libelf-dev)
apt_options=(-o Acquire::Retries=1
             -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15
             -o DPkg::Lock::Timeout=30)
fallback_dir=''

download_packages() {
  # Bound network work separately from dpkg so a mirror retry cannot interrupt
  # package installation. --error-on=any rejects incomplete/stale index updates.
  echo 'Refreshing CPU dependency package indexes (90-second deadline)'
  sudo timeout --kill-after=5s 90s apt-get "${apt_options[@]}" \
    update --error-on=any || return "$?"
  echo 'Downloading CPU dependencies (90-second deadline)'
  sudo timeout --kill-after=5s 90s apt-get "${apt_options[@]}" \
    install -y --download-only "${packages[@]}"
}

check_fallback_platform() {
  # The fallback is for the ubuntu-24.04 x86 runner, not lab provisioning.
  . /etc/os-release
  if [[ $ID != ubuntu || $VERSION_ID != 24.04 || $(dpkg --print-architecture) != amd64 ]]; then
    echo '::error::Ubuntu mirror fallback requires Ubuntu 24.04 amd64.'
    return 1
  fi
}

main() {
  trap 'if [[ -n $fallback_dir ]]; then sudo rm -rf -- "$fallback_dir"; fi' EXIT

  echo 'Using runner-configured APT repositories'
  if ! download_packages; then
    echo '::warning::CPU dependency download failed or timed out; retrying the kernel.org Ubuntu mirror.'
    check_fallback_platform
    fallback_dir=$(mktemp -d /tmp/uccl-ci-apt.XXXXXX)
    chmod 755 "$fallback_dir"
    mkdir "$fallback_dir/lists"
    cat > "$fallback_dir/ubuntu.sources" <<'EOF'
Types: deb
URIs: https://mirrors.edge.kernel.org/ubuntu
Suites: noble noble-updates noble-backports noble-security
Components: main universe restricted multiverse
Architectures: amd64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
    # Use a separate DNS domain: archive/security.ubuntu.com can fail together.
    # Isolate metadata as well as sources: stale mirror+file indexes must not
    # send the fallback's package downloads back through the broken mirror.
    apt_options+=(-o "Dir::Etc::sourcelist=$fallback_dir/ubuntu.sources"
                  -o Dir::Etc::sourceparts=-
                  -o "Dir::State::lists=$fallback_dir/lists"
                  -o Dir::Cache::pkgcache= -o Dir::Cache::srcpkgcache=)
    if ! download_packages; then
      echo '::error::CPU dependencies unavailable from both runner repositories and the kernel.org Ubuntu mirror.'
      return 1
    fi
  fi

  echo 'Installing downloaded CPU dependencies'
  sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" \
    install -y --no-download "${packages[@]}"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
