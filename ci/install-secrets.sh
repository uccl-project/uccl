#!/usr/bin/env bash
# Run locally with a GitHub token that can write this repository's Actions secrets.
set -euo pipefail
bundle=${1:?Path to the private CI credential bundle}
repo=${2:-uccl-project/uccl}
for item in 'UCCL_CI_SSH_KEY:id_ed25519' 'UCCL_CI_SSH_CONFIG:github-config' 'UCCL_CI_KNOWN_HOSTS:known_hosts'; do
  name=${item%%:*}; file=${item#*:}
  gh secret set "$name" --repo "$repo" < "$bundle/$file"
done
